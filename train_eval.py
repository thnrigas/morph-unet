#
# Train and Test for U-Net Segmentation
#

import argparse
import contextlib
import glob
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from networks.UNET import UNet
from networks.morph_block import MorphResidualUNet, MorphBankUNet, ConvBankUNet
from networks.morph_unet import MorphUNet
from networks.morph_attention import MorphAttentionUNet
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.morph_loss import MorphConsistencyLoss
from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from evaluation.evaluator import aggregate_scores, Evaluator

LABELS = config.LABELS   # {class_id: name}, read from the task's dataset.json

#
# fixed seed so runs share initialisation
#
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#
# parse a morph-bank spec "tophat:3,bottomhat:1,..." -> [(mode, k), ...]
# the integer is a disk RADIUS (survey vocabulary); SE window is k = 2r+1 (odd)
#
def parse_bank(spec):
    # "tophat:3,bottomhat:1" -> [(mode, radius), ...]. radius = disk-init radius; the SE
    # window is a shared k_max, so blocks share support but start at their own survey scale.
    out = []
    for tok in spec.split(","):
        mode, r = tok.split(":")
        out.append((mode.strip(), int(r)))
    return out


def _read_bank_spec(task, fold, root="results/explore"):
    # the survey (morph_explore.py) writes results/explore/<task>_bank.json as {fold: "mode:r,..."}
    # from that fold's TRAINING data only (no test leakage). Return the fold's spec, or None.
    path = os.path.join(root, f"{task}_bank.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        bank = json.load(f)
    return bank.get(str(fold))


def ensure_bank_spec(task, fold, dataset_dir, top_k, workers, root="results/explore"):
    # resolve --morph-bank auto: reuse the cached fold spec if present, else run the survey once
    # (which writes the cache) so `train_eval.py --morph-bank auto` is a single self-contained step.
    spec = _read_bank_spec(task, fold, root)
    if spec is None:
        print(f"[morph-bank auto] no cached spec for {task} fold {fold}; running survey ...", flush=True)
        subprocess.run([sys.executable, os.path.join("utilities", "morph_explore.py"), "survey",
                        str(dataset_dir), "--fold", str(fold), "--top-k", str(top_k),
                        "--workers", str(workers), "--out-dir", root, "--viz", "0"], check=True)
        spec = _read_bank_spec(task, fold, root)
        if spec is None:
            raise RuntimeError(f"survey produced no spec for fold {fold} (empty ranking?)")
    print(f"[morph-bank auto] {task} fold {fold} -> \"{spec}\"", flush=True)
    return spec


#
# build loaders
#
def build_loaders(args):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = (splits[args.fold]["train"], splits[args.fold]["val"], splits[args.fold]["test"])
    data_dir = str(config.PREPROCESSED_DIR)
    # npy is 2-channel float16 (image, label); every variant computes its residuals on the fly
    input_slice = (0,)
    label_slice = 1
    common = dict(target_size=args.patch_size, batch_size=args.batch_size, input_slice=input_slice,
                  label_slice=label_slice, num_processes=args.num_workers, fg_fraction=args.fg_fraction)
    cap = dict(num_batches=args.iters_per_epoch) if args.iters_per_epoch > 0 else {}
    train = None if args.test_only else NumpyDataSet(data_dir, keys=tr, **common, **cap)
    # val: full-slice like test so model selection uses foreground Dice on the real distribution,
    # not a background-dominated centre patch. --val-cases caps to a fixed (seeded) subset of volumes
    # (unbiased, cheaper); --val-batch packs full slices per forward pass to use the GPU.
    val_keys = list(vl)
    if args.val_cases and 0 < args.val_cases < len(val_keys):
        val_keys = sorted(random.Random(args.seed).sample(val_keys, args.val_cases))
    val = None if args.test_only else NumpyDataSet(data_dir, keys=val_keys, mode="test", do_reshuffle=False,
                                                   **{**common, "batch_size": args.val_batch})
    # test: full-slice inference one at a time, all slices (uncapped)
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False,
                        **{**common, "batch_size": 1})
    in_channels = len(input_slice)
    return train, val, test, in_channels

#
# run epoch
#
def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None, morph_loss=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    amp = device.type == "cuda"   # bf16 autocast on the L4 tensor cores (no-op off CUDA)
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            data = batch["data"][0].float().to(device, non_blocking=True)   # [b, c, H, W]
            target = batch["seg"][0].long().to(device, non_blocking=True)   # [b, 1, H, W]
            if train_mode:
                optimizer.zero_grad()
            with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
                pred = model(data)
                pred_softmax = F.softmax(pred, dim=1)
                loss = dice_loss(pred_softmax, target.squeeze()) + ce_loss(pred, target.squeeze())
                if morph_loss is not None:
                    loss = loss + morph_loss(pred_softmax)
            if train_mode:
                loss.backward()
                # clip grad-norm so one foreground-sparse batch with a sharp Dice gradient can't
                # spike/destabilise the epoch (tightened to 5 after residual spikes at norm 12)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

#
# full-slice validation -> global foreground Dice (TP/FP/FN accumulated over the whole val set,
# like nnU-Net's pseudo-Dice). Matches the test distribution and the reported metric, so it is a
# faithful, low-variance model-selection signal -- unlike a background-dominated patch loss.
#
def run_val_dice(model, loader, device, num_classes):
    model.eval()
    n_fg = max(num_classes - 1, 1)
    tp = torch.zeros(n_fg); fp = torch.zeros(n_fg); fn = torch.zeros(n_fg)
    amp = device.type == "cuda"
    chunk = [0]   # GPU sub-batch cap; 0 = whole loader-batch, drops (halving) after a CUDA OOM

    def _count(x, t):
        with (torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()):
            pred_lab = model(x).argmax(1)                                    # [b, H, W]
        for c in range(1, num_classes):
            p, tt, i = (pred_lab == c), (t == c), c - 1
            tp[i] += (p & tt).sum().item()
            fp[i] += (p & ~tt).sum().item()
            fn[i] += (~p & tt).sum().item()

    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float()                                 # [b, c, H, W] (cpu)
            target = batch["seg"][0][:, 0].long()                           # [b, H, W]
            b, i = data.shape[0], 0
            while i < b:                                                     # forward in OOM-safe chunks
                step = min(chunk[0] or b, b - i)
                try:
                    _count(data[i:i + step].to(device, non_blocking=True),
                           target[i:i + step].to(device, non_blocking=True))
                    i += step
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower() or step == 1:
                        raise
                    torch.cuda.empty_cache()
                    chunk[0] = max(step // 2, 1)
                    print(f"[val] CUDA OOM at chunk {step} -> retry at {chunk[0]}", flush=True)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)         # per-class global Dice
    present = (tp + fn) > 0                              # classes actually present in the val set
    mean = dice[present].mean().item() if present.any() else 0.0
    return mean, dice.tolist()

#
# test
#
def evaluate_test(model, loader, device, json_path, num_workers=1):
    model.eval()
    # accumulate per-case pred/GT as uint8 (labels are small ints). Storing int64 preds + float
    # targets for the whole full-slice test set blew up CPU RAM and got the process OOM-killed.
    pred_dict, gt_dict = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device)
            target = batch["seg"][0][:, 0].to(torch.uint8).numpy()           # [b, H, W]
            pred = model(data).argmax(1).to(torch.uint8).cpu().numpy()       # [b, H, W]
            for i, fname in enumerate(batch["fnames"]):
                pred_dict[fname[0]].append(pred[i])
                gt_dict[fname[0]].append(target[i])
    pairs = [(np.stack(pred_dict[k]), np.stack(gt_dict[k])) for k in pred_dict]   # each [Z, H, W]
    scores = aggregate_scores(pairs, evaluator=Evaluator, labels=LABELS,
        json_output_file=json_path, json_author="cv-project",
        json_task=config.TASK, num_workers=num_workers, advanced=True,
    )
    return scores


def _run_name(path):
    base = os.path.basename(path)
    for suf in ("_scores.json", ".json"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return base

#
# average each metric across all per-fold <tag>_f<fold>_scores.json
#
def fold_mean(tag):
    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    paths = sorted(glob.glob(os.path.join(results_dir, f"{tag}_f*_scores.json")))
    if not paths:
        raise SystemExit(f"no files match {tag}_f*_scores.json in {results_dir}")
    per_fold = []
    for p in paths:
        with open(p) as f:
            per_fold.append(json.load(f)["results"]["mean"])
    print(f"{tag}: mean +/- std over {len(paths)} folds "
          f"({', '.join(_run_name(p) for p in paths)})")
    agg = {}
    for label in per_fold[0]:
        agg[label] = {}
        for metric in ("Dice", "Avg. Symmetric Surface Distance"):
            vals = [fold[label].get(metric) for fold in per_fold]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m, s = float(np.mean(vals)), float(np.std(vals))
            agg[label][metric] = {"mean": m, "std": s, "n": len(vals)}
            print(f"  label {label:>10} | {metric:<32} "
                  f"{m:.4f} +/- {s:.4f}  (n={len(vals)})")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"{tag}_mean_scores.json")
    with open(out, "w") as f:
        json.dump({"tag": tag, "folds": paths, "mean": agg}, f, indent=2)
    print(f"written to {out}")
    return agg

#
# per-label Dice/ASSD deltas of each run vs the baseline
#
def compare_runs(paths):
    if len(paths) < 2:
        raise SystemExit("--compare needs at least two JSON files")
    baseline, others = paths[0], paths[1:]

    def load_mean(p):
        """Flat {label: {metric: float}}, accepting per-fold or fold-mean files."""
        actual_path = p
        if not os.path.exists(actual_path):
            results_dir = os.path.join(config.PROJECT_ROOT, "results")
            alt_path = os.path.join(results_dir, p)
            if os.path.exists(alt_path):
                actual_path = alt_path
        with open(actual_path) as f:
            d = json.load(f)
        kind = "fold-mean" if "mean" in d else "per-fold"
        raw = d["mean"] if kind == "fold-mean" else d["results"]["mean"]
        # fold-mean files nest {"mean", "std", "n"}; flatten to the mean
        flat = {label: {m: (v["mean"] if isinstance(v, dict) else v) for m, v in metrics.items()}
                for label, metrics in raw.items()}
        return kind, flat

    kinds = {p: load_mean(p)[0] for p in paths}
    if len(set(kinds.values())) > 1:
        print("WARNING: mixing per-fold and fold-mean files in one compare")
        for p, k in kinds.items():
            print(f"  {k:<10} {_run_name(p)}")

    base = load_mean(baseline)[1]
    bname = _run_name(baseline)
    print(f"baseline : {bname}")
    for other in others:
        o = load_mean(other)[1]
        oname = _run_name(other)
        print(f"\n{oname} vs {bname} (mean over test set)")
        for label in base:
            for metric in ("Dice", "Avg. Symmetric Surface Distance"):
                bv, ov = base[label].get(metric), o[label].get(metric)
                if bv is None or ov is None:
                    continue
                print(f"  label {label:>10} | {metric:<32} "
                      f"{bv:.4f} -> {ov:.4f}   delta={ov - bv:+.4f}")

#
# main
#
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag")
    p.add_argument("--tophat", action="store_true")
    p.add_argument("--bottomhat", action="store_true")
    p.add_argument("--morph-block", action="store_true")
    p.add_argument("--morph-unet",
                   choices=["none", "heavy", "balanced", "bottleneck", "deep", "full", "full_l1", "full_l2"],
                   help="replace conv stages with morphological-separable blocks (networks/morph_unet.py). "
                        "none = all-conv backbone (used with --morph-attn for the attention baseline); "
                        "deep = enc4+center+dec4; full = all stages morph (18 layers); "
                        "full_l1/full_l2 = full but with level 1 / levels 1-2 kept linear (14 / 10 layers)")
    p.add_argument("--morph-impl", choices=["fast", "paper", "convsep", "convmpm"], default="fast",
                   help="morph block: 'fast' = depthwise morph + 1x1 (efficient); "
                        "'paper' = strict full channel-mixing max-plus conv + depthwise 3x3 activation "
                        "(Setting 1, faithful, prunes the paper's way, heavier); "
                        "'convsep' = plain-conv TWIN of 'fast' -- depthwise 3x3 conv + 1x1 (no "
                        "morphology, matched params, trains much faster; the ablation control); "
                        "'convmpm' = convsep but the 1x1 channel mixer is a MORPHOLOGICAL MPM neuron "
                        "(depthwise 3x3 conv + 1x1 max-plus/min-plus channel mix -- morphology placed "
                        "where channel sparsity/pruning happens)")
    p.add_argument("--fs", type=int, default=64,
                   help="base feature width of the (morph) U-Net; channel counts are fs, 2fs, 4fs, "
                        "8fs, 16fs across the levels. Default 64. A smaller fs (e.g. 13 ~ 1/5 width) "
                        "shrinks every layer proportionally for a much faster/cheaper model")
    p.add_argument("--morph-half", action="store_true",
                   help="morph stages use HybridUnit (half channels morphological, half plain conv) "
                        "-- ~2x faster training, lower memory")
    p.add_argument("--morph-tie-mirror", action="store_true",
                   help="init encoder<->decoder mirror stages (enc1<->dec1, enc2<->dec2) with identical "
                        "structuring elements; they diverge freely during training")
    p.add_argument("--morph-attn", action="store_true",
                   help="morphological Attention-U-Net: gate skip connections with soft top-hat/bottom-hat "
                        "attention (networks/morph_attention.py)")
    p.add_argument("--morph-attn-warm", type=float, default=0.0,
                   help="warm-start the morph-attention skip gate half-active (--morph-attn). 0.0 (default) "
                        "= identity start (plain U-Net skip); 0.5 = top-hat/bottom-hat morphology influences "
                        "the skip at half strength from step 0 (analogue of the linear gate's gamma=0.5 init)")
    p.add_argument("--morph-no-conv-stem", action="store_true",
                   help="disable the default 3x3 denoising conv stem (raw input goes straight into "
                        "morphology instead of conv features)")
    p.add_argument("--morph-no-checkpoint", action="store_true",
                   help="disable gradient checkpointing on morph stages: faster, uses more VRAM, "
                        "identical maths/training. Use when the model fits (e.g. --morph-half)")
    p.add_argument("--morph-relu", action="store_true",
                   help="use plain ReLU instead of LeakyReLU everywhere (the non-linearity the "
                        "tropical-geometry / TropNNC pruning theory is derived for). Auto-appends "
                        "'_relu' to --tag so it writes to a NEW file, not the LeakyReLU one")
    p.add_argument("--finetune-from", metavar="CKPT",
                   help="warm-start the model weights from this checkpoint (a _best.pth state-dict or "
                        "a _last.pth full-state), then train fresh (new optimizer/scheduler at --lr). "
                        "Use with --morph-relu to fine-tune a LeakyReLU model into a ReLU one cheaply")
    p.add_argument("--morph-beta-warmup", type=int, default=30, metavar="EPOCHS",
                   help="ramp the LogSumExp temperature beta from --morph-beta-start up to --morph-beta "
                        "over this many epochs (softer morphology early -> denser gradients). default 30; 0 = off")
    p.add_argument("--morph-beta-start", type=float, default=2.0,
                   help="starting beta for --morph-beta-warmup")
    p.add_argument("--morph-bank", metavar="SPEC",
                   help='trainable morph bank, e.g. "tophat:3,tophat:5,bottomhat:1,gradient:2" (radii); '
                        'use "auto" to load this task/fold spec from the survey\'s results/explore/<TASK>_bank.json')
    p.add_argument("--conv-control", action="store_true",
                   help="matched conv front-end (same #channels) instead of the morph bank")
    p.add_argument("--survey-top-k", type=int, default=5,
                   help="--morph-bank auto: how many (mode,radius) the survey selects if it must run")
    p.add_argument("--survey-workers", type=int, default=min(os.cpu_count() or 1, 8),
                   help="--morph-bank auto: parallel workers for the survey if it must run")
    p.add_argument("--morph-k", type=int, default=5)
    p.add_argument("--morph-k-max", type=int, default=11,
                   help="shared SE window for the morph bank; disk-init radii sit inside it (room to grow)")
    p.add_argument("--morph-beta", type=float, default=10.0)
    p.add_argument("--morph-dropout", type=float, default=0.0,
                   help="spatial Dropout2d rate on every stage output (0 = off, the default). "
                        "e.g. 0.2 for a regularised deep/bottleneck variant")
    p.add_argument("--lin-attn", action="store_true",
                   help="train the linear-attention U-Net (networks/linear_attention.py): a "
                        "plain-conv U-Net with LINEAR cross-attention gates on the skip connections")
    p.add_argument("--lin-attn-heads", type=int, default=4,
                   help="number of linear-attention heads per skip gate (--lin-attn)")
    p.add_argument("--lin-attn-gamma-init", type=float, default=0.0,
                   help="initial value of the ReZero skip gate gamma (--lin-attn). 0.0 (default) = "
                        "skips start as the plain U-Net skip and ramp attention in; 0.5 = attention "
                        "contributes at half strength from the start")
    p.add_argument("--freeze-se", action="store_true",
                   help="freeze the SE weights (fixed structuring element = static residual)")
    p.add_argument("--morph-loss", action="store_true")
    p.add_argument("--epochs", type=int, default=config.HP["epochs"])
    p.add_argument("--patience", type=int, default=config.HP["patience"],
                   help="early stop after this many EPOCHS with no fg-Dice gain (independent of --val-every)")
    p.add_argument("--val-every", type=int, default=3,
                   help="run full-slice foreground-Dice validation every K epochs (raise to speed up)")
    p.add_argument("--val-batch", type=int, default=12,
                   help="full slices per validation forward pass; packs the GPU, auto-halves on CUDA OOM")
    p.add_argument("--val-cases", type=int, default=15,
                   help="validate on this many fixed (seeded) val volumes instead of all (unbiased, cheaper); "
                        "0 = all. Falls back to all if the val set is smaller.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=config.HP["batch_size"])
    p.add_argument("--patch-size", type=int, default=config.HP["patch_size"])
    p.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 6))
    p.add_argument("--lr", type=float, default=config.HP["lr"])
    p.add_argument("--iters-per-epoch", type=int, default=config.HP["iters_per_epoch"],
                   help="cap batches per (train/val) epoch; 0 = full pass over all slices")
    p.add_argument("--fg-fraction", type=float, default=config.HP["fg_fraction"],
                   help="fraction of train crops centred on a foreground voxel")
    p.add_argument("--se-lr-mult", type=float, default=10.0,
                   help="lr multiplier for the morphological SE weights (own param group)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    p.add_argument("--fold-mean", metavar="TAG")
    p.add_argument("--compare", nargs="+", metavar="JSON")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from <tag>_f<fold>_last.pth if it exists (survives interruption / spot preemption)")
    args = p.parse_args()

    if args.fold_mean:
        fold_mean(args.fold_mean)
        return
    if args.compare:
        compare_runs(args.compare)
        return
    if not args.tag:
        p.error("--tag is required")
    # ReLU runs write to a *_relu tag so the LeakyReLU checkpoints/scores are never overwritten
    if args.morph_relu and not args.tag.endswith("_relu"):
        args.tag += "_relu"
        print(f"[--morph-relu] output tag -> {args.tag}")

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        # let Ada use TF32 matmuls and autotune convs for the fixed patch size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader, in_channels = build_loaders(args)

    if args.lin_attn:
        # plain-conv U-Net with LINEAR cross-attention gates on the skips (networks/
        # linear_attention.py) -- the linear-attention counterpart to the morphological
        # attention net. No morphology, so the SE/beta/impl flags do not apply here.
        from networks.linear_attention import LinAttnUNet
        model = LinAttnUNet(num_classes=args.num_classes, in_channels=in_channels,
                            conv_stem=not args.morph_no_conv_stem,
                            heads=args.lin_attn_heads,
                            gamma_init=args.lin_attn_gamma_init,
                            act="relu" if args.morph_relu else "leaky").to(device)
    elif args.morph_unet:
        # morphological-separable U-Net: conv stages replaced by depthwise soft morphology
        # + 1x1 projection, per the chosen config. Its SEs end in ".se", so they pick up the
        # boosted SE lr and --freeze-se just like the bank/residual variants. --morph-attn
        # swaps in the morphological Attention-U-Net (gated skips); the other flags apply to both.
        Net = MorphAttentionUNet if args.morph_attn else MorphUNet
        net_kwargs = dict(num_classes=args.num_classes, in_channels=in_channels, fs=args.fs,
                    k=args.morph_k, beta=args.morph_beta, config=args.morph_unet,
                    half_morph=args.morph_half, tie_mirror=args.morph_tie_mirror,
                    conv_stem=not args.morph_no_conv_stem,
                    checkpoint=not args.morph_no_checkpoint, impl=args.morph_impl,
                    act="relu" if args.morph_relu else "leaky",
                    dropout=args.morph_dropout)
        if args.morph_attn:
            net_kwargs["attn_warm"] = args.morph_attn_warm   # >0 -> warm-start the skip gate half-active
        model = Net(**net_kwargs).to(device)
    elif args.morph_bank:
        # bank of trainable-SE residual channels (or a matched conv control)
        if args.morph_bank == "auto":
            args.morph_bank = ensure_bank_spec(config.TASK, args.fold, config.DATA_DIR,
                                               args.survey_top_k, args.survey_workers)
        specs = parse_bank(args.morph_bank)
        n_extra = len(specs)
        base = UNet(num_classes=args.num_classes, in_channels=1 + n_extra)
        if args.conv_control:
            model = ConvBankUNet(base, n_extra=n_extra, k=args.morph_k_max).to(device)
        else:
            model = MorphBankUNet(base, specs, k_max=args.morph_k_max, beta=args.morph_beta).to(device)
    elif args.morph_block or args.tophat or args.bottomhat:
        # trainable (--morph-block, learnable SE) or static (--tophat/--bottomhat, frozen SE)
        # residuals, both computed on the fly. under --morph-block default to top-hat.
        use_th = args.tophat or (args.morph_block and not args.bottomhat)
        use_bh = args.bottomhat
        base = UNet(num_classes=args.num_classes, in_channels=1 + use_th + use_bh)
        model = MorphResidualUNet(base, k=args.morph_k, beta=args.morph_beta,
                                  use_tophat=use_th, use_bottomhat=use_bh).to(device)
        if not args.morph_block:            # static residual -> fixed (frozen) SE
            args.freeze_se = True
    else:
        model = UNet(num_classes=args.num_classes, in_channels=in_channels).to(device)

    if args.freeze_se:                      # static: SE is a fixed structuring element
        for n, p in model.named_parameters():
            if n.endswith(".se"):
                p.requires_grad_(False)

    # warm-start: load weights from a prior checkpoint (e.g. a LeakyReLU run) into this model
    # before training. Activations carry no parameters, so a LeakyReLU state-dict loads 1:1 into
    # a ReLU model. Optimizer/scheduler are NOT restored -- training starts fresh at --lr, which
    # is the whole point of a fine-tune (vs --resume, which continues the same run). Skipped when
    # --resume is set (then we continue this tag's own _last.pth instead).
    if args.finetune_from and not args.resume:
        if not os.path.exists(args.finetune_from):
            raise SystemExit(f"--finetune-from: no such file {args.finetune_from}")
        ck = torch.load(args.finetune_from, map_location=device)
        if isinstance(ck, dict) and "model" in ck:      # a full-state _last.pth
            ck = ck["model"]
        missing, unexpected = model.load_state_dict(ck, strict=False)
        if missing or unexpected:
            print(f"  [finetune] WARNING partial load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected params")
        print(f"warm-started weights from {args.finetune_from} "
              f"(fresh optimizer/scheduler @ lr={args.lr:g})")

    # (<tag>_f<fold>_{best.pth,last.pth,scores.json}
    stem = f"{args.tag}_f{args.fold}"
    if args.lin_attn:
        mode = f"lin-attn(heads={args.lin_attn_heads},stem={not args.morph_no_conv_stem})"
    elif args.morph_unet:
        extra = ("".join(f",{t}" for t, on in
                 [("attn", args.morph_attn), ("half", args.morph_half), ("tie", args.morph_tie_mirror),
                  ("stem", not args.morph_no_conv_stem), ("bwarm", args.morph_beta_warmup > 0),
                  ("relu", args.morph_relu),
                  (f"drop={args.morph_dropout}", args.morph_dropout > 0),
                  (f"impl={args.morph_impl}", args.morph_impl != "fast")] if on))
        mode = f"morph-unet({args.morph_unet},k={args.morph_k},beta={args.morph_beta}{extra})"
    elif args.morph_bank:
        kind = "conv-control" if args.conv_control else "morph-bank"
        mode = f"{kind}([{args.morph_bank}],beta={args.morph_beta})"
    elif args.morph_block:
        res = "+".join((["tophat"] if use_th else []) + (["bottomhat"] if use_bh else []))
        mode = f"morph-block({res},k={args.morph_k},beta={args.morph_beta})"
    else:
        parts = (["tophat"] if args.tophat else []) + (["bottomhat"] if args.bottomhat else [])
        mode = "+".join(parts) if parts else "baseline"
    loss_desc = "dice+ce" + ("+morph" if args.morph_loss else "")
    epoch_len = args.iters_per_epoch if args.iters_per_epoch > 0 else len(train_loader)
    print(f"[{stem}] device={device} mode={mode} loss={loss_desc} seed={args.seed} "
          f"fold={args.fold} loader_in_ch={in_channels} patch={args.patch_size} "
          f"iters/epoch={epoch_len} max-epochs={args.epochs} (<= {epoch_len * args.epochs} updates) "
          f"fg={args.fg_fraction}")

    # foreground-only Dice (drop background channel): background is >99% of pixels and its Dice is
    # ~constant, so including it dilutes the gradient on the sparse target. CE still sees all classes.
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()
    morph_loss = MorphConsistencyLoss().to(device) if args.morph_loss else None
    # give the morphological SE weights their own (higher) lr — few, geometrically
    # important params with sparse gradients, so a larger step helps them actually move.
    # frozen SEs (static residual) are requires_grad=False -> excluded from the optimiser.
    se_named = [(n, p) for n, p in model.named_parameters() if n.endswith(".se") and p.requires_grad]
    if se_named:
        se_ids = {id(p) for _, p in se_named}
        other = [p for p in model.parameters() if p.requires_grad and id(p) not in se_ids]
        optimizer = optim.Adam([{"params": other, "lr": args.lr},
                                {"params": [p for _, p in se_named], "lr": args.lr * args.se_lr_mult}])
        print(f"  SE param group: {len(se_named)} block(s) @ lr={args.lr * args.se_lr_mult:g} "
              f"(x{args.se_lr_mult:g}); rest @ lr={args.lr:g}")
    else:
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)

    results_dir = os.path.join(config.PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    best_path = os.path.join(results_dir, f"{stem}_best.pth")
    last_path = os.path.join(results_dir, f"{stem}_last.pth")
    if not args.test_only:
        # selection metric is foreground Dice now (higher is better); best_dice tracks the best so far
        start_epoch, best_dice, best_epoch = 1, -1.0, 0
        t0, val_curve = time.time(), []   # wall-clock + (epoch, fg_dice) learning curve for the summary
        # resume from the full-state last.pth (model + optim + scheduler + counters)
        if args.resume and os.path.exists(last_path):
            ck = torch.load(last_path, map_location=device)
            if isinstance(ck, dict) and "model" in ck:
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                scheduler.load_state_dict(ck["scheduler"])
                start_epoch = ck["epoch"] + 1
                best_dice, best_epoch = ck["best_val"], ck.get("best_epoch", 0)
                print(f"resumed from epoch {ck['epoch']} (best fg-Dice={best_dice:.4f} @ ep {best_epoch}, "
                      f"{start_epoch - 1 - best_epoch}/{args.patience} epochs stale)")
        se_prev = {n: p.detach().clone() for n, p in se_named}   # to log how much the SEs move
        for epoch in range(start_epoch, args.epochs + 1):
            if args.morph_beta_warmup > 0 and hasattr(model, "set_beta"):
                # linearly ramp beta start -> target over the warmup epochs, then hold
                frac = min(1.0, (epoch - 1) / args.morph_beta_warmup)
                cur_beta = args.morph_beta_start + frac * (args.morph_beta - args.morph_beta_start)
                model.set_beta(cur_beta)
                if epoch == start_epoch or epoch <= args.morph_beta_warmup + 1:
                    print(f"  beta warmup: epoch {epoch} -> beta={cur_beta:.2f}")
            tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer, morph_loss)
            do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            if do_val:
                vd, per_class = run_val_dice(model, val_loader, device, args.num_classes)
                scheduler.step(-vd)    # scheduler minimises; we maximise Dice, so feed -Dice
                print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val_fgDice={vd:.4f}  "
                      f"[per-class {' '.join(f'{d:.3f}' for d in per_class)}]")
                val_curve.append([epoch, round(vd, 5)])
            else:
                print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  (val every {args.val_every})")
            if se_named:   # is the SE actually learning? |Δ| per block this epoch + |SE| magnitude
                deltas, norms = [], []
                for n, p in se_named:
                    deltas.append((p.detach() - se_prev[n]).norm().item())
                    norms.append(p.detach().norm().item())
                    se_prev[n] = p.detach().clone()
                print("  SE |Δ|: " + " ".join(f"{d:.3f}" for d in deltas)
                      + "   |SE|: " + " ".join(f"{v:.3f}" for v in norms))
            if do_val:
                if vd > best_dice:
                    best_dice, best_epoch = vd, epoch
                    torch.save(model.state_dict(), best_path)   # best: weights only (for eval)
                else:
                    patience_str = f"/{args.patience}" if args.patience else ""
                    print(f"  val fg-Dice did not improve from {best_dice:.4f} @ ep {best_epoch} "
                          f"(stale {epoch - best_epoch}{patience_str} epochs)")
            # last: full state so training can resume after an interruption
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_val": best_dice,
                        "best_epoch": best_epoch}, last_path)
            # early stop on epochs since last improvement (independent of --val-every)
            if args.patience and (epoch - best_epoch) >= args.patience:
                print(f"early stop: no val fg-Dice improvement for {epoch - best_epoch} epochs "
                      f"(best fg-Dice={best_dice:.4f} @ ep {best_epoch})")
                break

        # per-run training summary: convergence + cost (lets --compare show whether an arm, e.g.
        # the morph bank, reaches the same Dice in fewer epochs / less time / fewer params)
        elapsed = time.time() - t0
        front = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(("blocks.", "front.", "tophat.", "bottomhat.")))
        total = sum(p.numel() for p in model.parameters())
        # epochs to first reach 90% of the best fg-Dice — a threshold-based convergence-speed metric
        thr = 0.9 * best_dice
        ep_to_thr = next((e for e, d in val_curve if d >= thr), best_epoch)
        summary = {"best_fg_dice": round(best_dice, 5), "best_epoch": best_epoch,
                   "epochs_to_90pct_best": ep_to_thr, "stopped_epoch": epoch, "max_epochs": args.epochs,
                   "seconds": round(elapsed, 1),
                   "sec_per_epoch": round(elapsed / max(epoch - start_epoch + 1, 1), 2),
                   "updates": epoch * epoch_len, "patches": epoch * epoch_len * args.batch_size,
                   "params_total": total, "params_frontend": front, "val_curve": val_curve}
        with open(os.path.join(results_dir, f"{stem}_train.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{stem}] trained {epoch} ep in {elapsed / 60:.1f} min "
              f"({summary['sec_per_epoch']:.1f}s/ep) | best fg-Dice {best_dice:.4f} @ ep{best_epoch} "
              f"(90% @ ep{ep_to_thr}) | {total / 1e6:.2f}M params (front-end {front})")
    ckpt = best_path if os.path.exists(best_path) else last_path
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:   # last.pth is a full-state dict
        state = state["model"]
    model.load_state_dict(state)
    json_path = os.path.join(results_dir, f"{stem}_scores.json")
    scores = evaluate_test(model, test_loader, device, json_path, num_workers=args.num_workers)
    print(f"[{stem}] mean scores written to {json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')} "
              f"ASSD={md.get('Avg. Symmetric Surface Distance')}")


if __name__ == "__main__":
    main()
