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


def _read_static_spec(task, fold, root="results/explore"):
    # the survey writes results/explore/<task>_static.json as {fold: ["recontophat:3", "hdome:0.1", ...]}
    path = os.path.join(root, f"{task}_static.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        bank = json.load(f)
    return bank.get(str(fold))


def _run_survey(task, fold, dataset_dir, top_k, workers, root, n=25):
    """Run the morphology survey once (all-slices, train-split, n cases; n=0 = all cases),
    writing both _bank.json and _static.json for this fold."""
    print(f"[auto] no cached spec for {task} fold {fold}; running survey "
          f"(n={'all' if n == 0 else n} cases, all slices, train split) ...", flush=True)
    subprocess.run([sys.executable, os.path.join("utilities", "morph_explore.py"), "survey",
                    str(dataset_dir), "--fold", str(fold), "--top-k", str(top_k),
                    "--n", str(n), "--split", "train", "--all-slices",
                    "--workers", str(workers), "--out-dir", root, "--viz", "0"], check=True)


def ensure_bank_spec(task, fold, dataset_dir, top_k, workers, root="results/explore", n=0):
    # resolve --morph-bank auto: reuse the cached fold spec if present, else run the survey once
    # (which writes the cache) so `train_eval.py --morph-bank auto` is a single self-contained step.
    spec = _read_bank_spec(task, fold, root)
    if spec is None:
        _run_survey(task, fold, dataset_dir, top_k, workers, root, n)
        spec = _read_bank_spec(task, fold, root)
        if spec is None:
            raise RuntimeError(f"survey produced no spec for fold {fold} (empty ranking?)")
    print(f"[auto] trainable -> \"{spec}\"", flush=True)
    return spec


def ensure_static_channels(task, fold, dataset_dir, top_k, workers, root="results/explore", n=0):
    """Resolve static channels: read the cached static spec (or run the full survey), then run
    augment_channels.py if the augmented dir doesn't exist yet. Returns (static_dir, n_ch)
    where n_ch is the number of static filter channels (0 if none selected)."""
    specs = _read_static_spec(task, fold, root)
    if specs is None:
        # survey hasn't been run yet (ensure_bank_spec usually runs it first, but handle standalone)
        _run_survey(task, fold, dataset_dir, top_k, workers, root, n)
        specs = _read_static_spec(task, fold, root)
    if not specs:
        print("[auto] no static filters selected by survey", flush=True)
        return None, 0
    # augmented dir is PER-FOLD: folds select different filters, so a shared dir would either
    # mis-slice (wrong channel count) or train a fold on another fold's filters.
    prep_dir = str(config.PREPROCESSED_DIR)
    static_dir = prep_dir.rstrip("/") + f"_static_f{fold}"
    want_ch = 1 + len(specs) + 1                       # image + N filters + label
    npys = [f for f in os.listdir(static_dir) if f.endswith(".npy")] if os.path.isdir(static_dir) else []
    # reuse only if the cached dir already has the right channel count (else the spec changed)
    fresh = bool(npys) and np.load(os.path.join(static_dir, npys[0]), mmap_mode="r").shape[0] == want_ch
    if fresh:
        print(f"[auto] static dir exists: {static_dir} ({len(specs)} filters)", flush=True)
    else:
        print(f"[auto] precomputing static channels: {specs} -> {static_dir}", flush=True)
        augment = os.path.join(config.PROJECT_ROOT, "augment_channels.py")   # cwd-independent
        subprocess.run([sys.executable, augment,
                        "--filters"] + specs + [
                        "--src", prep_dir,
                        "--out", static_dir], check=True)
    print(f"[auto] static -> {specs}  ({len(specs)} channels)", flush=True)
    return static_dir, len(specs)


#
# build loaders
#
def build_loaders(args):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = (splits[args.fold]["train"], splits[args.fold]["val"], splits[args.fold]["test"])
    # --static-dir overrides to the augmented preprocessed dir; --static-channels sets the layout
    n_static = getattr(args, "static_channels", 0) or 0
    data_dir = getattr(args, "static_dir", None) or str(config.PREPROCESSED_DIR)
    # npy layout: [image, filt_1, ..., filt_N, label] — N = static_channels (0 for baseline)
    input_slice = tuple(range(1 + n_static))
    label_slice = 1 + n_static
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
# all per-run artefacts (checkpoints, scores, train summaries) live under
# results/<TASK>/ so runs for different MSD tasks don't collide in one flat folder.
# task_results_dir() is the WRITE target (created on demand); read_results_dir() is the
# READ source and falls back to the flat results/ for the legacy (pre-subdir) layout.
#
def task_results_dir():
    d = os.path.join(config.PROJECT_ROOT, "results", config.TASK)
    os.makedirs(d, exist_ok=True)
    return d


def read_results_dir():
    d = os.path.join(config.PROJECT_ROOT, "results", config.TASK)
    return d if os.path.isdir(d) else os.path.join(config.PROJECT_ROOT, "results")

# segmentation metrics reported by fold_mean / compare. Dice + ASSD are the MSD primaries;
# HD95 is a robust boundary metric (unlike outlier-prone raw Hausdorff); Precision/Recall
# explain *why* Dice moves (e.g. trading precision for recall). Jaccard (monotone with Dice)
# and the background-dominated rates (Accuracy/FPR/TNR/NPV) are omitted as uninformative.
SCORE_METRICS = ("Dice", "Avg. Symmetric Surface Distance", "Hausdorff Distance 95",
                 "Precision", "Recall")
# per-run training-cost / convergence stats worth averaging across folds (from <run>_train.json).
# params_* are ~constant across folds; the rest capture "does this arm cost more to reach its Dice".
TRAIN_STATS = ("best_fg_dice", "best_epoch", "epochs_to_90pct_best", "sec_per_epoch",
               "params_total", "params_frontend")


def _mean_std(vals):
    """{mean, std, n} over the non-None values, or None if there are none."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


#
# average each metric (and the training stats) across all per-fold <tag>_f<fold>_scores.json
#
def fold_mean(tag):
    results_dir = read_results_dir()   # task subdir, or flat results/ (legacy) as a fallback
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
        for metric in SCORE_METRICS:
            ms = _mean_std([fold[label].get(metric) for fold in per_fold])
            if ms is None:
                continue
            agg[label][metric] = ms
            print(f"  label {label:>10} | {metric:<32} "
                  f"{ms['mean']:.4f} +/- {ms['std']:.4f}  (n={ms['n']})")
    # training-cost / convergence summary, averaged over the same folds (sibling <run>_train.json)
    train_per_fold = []
    for p in paths:
        tpath = p.replace("_scores.json", "_train.json")
        if os.path.exists(tpath):
            with open(tpath) as f:
                train_per_fold.append(json.load(f))
    train_agg = {}
    if train_per_fold:
        print(f"  -- train stats ({len(train_per_fold)}/{len(paths)} folds) --")
        for stat in TRAIN_STATS:
            ms = _mean_std([t.get(stat) for t in train_per_fold])
            if ms is None:
                continue
            train_agg[stat] = ms
            print(f"  {stat:<32} {ms['mean']:.4f} +/- {ms['std']:.4f}  (n={ms['n']})")
    out = os.path.join(results_dir, f"{tag}_mean_scores.json")
    with open(out, "w") as f:
        json.dump({"tag": tag, "folds": paths, "mean": agg, "train": train_agg}, f, indent=2)
    print(f"written to {out}")
    return agg

#
# per-label metric deltas (+ training-stat deltas) of each run vs the baseline
#
def compare_runs(paths):
    if len(paths) < 2:
        raise SystemExit("--compare needs at least two JSON files")
    baseline, others = paths[0], paths[1:]

    def resolve(p):
        """Path as given, else the same name under results/<TASK>/ (or flat results/ legacy)."""
        if os.path.exists(p):
            return p
        alt = os.path.join(read_results_dir(), p)
        return alt if os.path.exists(alt) else p

    def load_mean(p):
        """Flat {label: {metric: float}}, accepting per-fold or fold-mean files."""
        with open(resolve(p)) as f:
            d = json.load(f)
        kind = "fold-mean" if "mean" in d else "per-fold"
        raw = d["mean"] if kind == "fold-mean" else d["results"]["mean"]
        # fold-mean files nest {"mean", "std", "n"}; flatten to the mean
        flat = {label: {m: (v["mean"] if isinstance(v, dict) else v) for m, v in metrics.items()}
                for label, metrics in raw.items()}
        return kind, flat

    def load_train(p):
        """Flat {stat: float} training summary, or {} if unavailable.
        fold-mean files carry an aggregated "train" block; per-fold files have a
        sibling <run>_train.json."""
        path = resolve(p)
        with open(path) as f:
            d = json.load(f)
        if "train" in d:   # fold-mean: {stat: {mean, std, n}}
            return {k: (v["mean"] if isinstance(v, dict) else v) for k, v in d["train"].items()}
        tpath = path.replace("_scores.json", "_train.json")   # per-fold sibling
        if tpath != path and os.path.exists(tpath):
            with open(tpath) as f:
                return json.load(f)
        return {}

    kinds = {p: load_mean(p)[0] for p in paths}
    if len(set(kinds.values())) > 1:
        print("WARNING: mixing per-fold and fold-mean files in one compare")
        for p, k in kinds.items():
            print(f"  {k:<10} {_run_name(p)}")

    base = load_mean(baseline)[1]
    base_train = load_train(baseline)
    bname = _run_name(baseline)
    print(f"baseline : {bname}")
    for other in others:
        o = load_mean(other)[1]
        oname = _run_name(other)
        print(f"\n{oname} vs {bname} (mean over test set)")
        for label in base:
            for metric in SCORE_METRICS:
                bv, ov = base[label].get(metric), o[label].get(metric)
                if bv is None or ov is None:
                    continue
                print(f"  label {label:>10} | {metric:<32} "
                      f"{bv:.4f} -> {ov:.4f}   delta={ov - bv:+.4f}")
        # training cost / convergence deltas (needs a train summary on both sides)
        o_train = load_train(other)
        if base_train and o_train:
            print("  -- train --")
            for stat in TRAIN_STATS:
                bv, ov = base_train.get(stat), o_train.get(stat)
                if bv is None or ov is None:
                    continue
                print(f"  {stat:>32}   {bv:>12.4f} -> {ov:>12.4f}   delta={ov - bv:+.4f}")

#
# main
#
def main():
    p = argparse.ArgumentParser()
    # id
    p.add_argument("--tag")
    # modes
    p.add_argument("--tophat", action="store_true")
    p.add_argument("--bottomhat", action="store_true")
    p.add_argument("--morph-block", action="store_true")
    p.add_argument("--morph-loss", action="store_true")
    p.add_argument("--morph-bank", metavar="SPEC")
    p.add_argument("--conv-control", action="store_true")
    p.add_argument("--morph-unet", choices=["heavy", "balanced", "bottleneck"])
    # morph parameters
    p.add_argument("--morph-k", type=int, default=5)
    p.add_argument("--morph-k-max", type=int, default=11)
    p.add_argument("--morph-beta", type=float, default=10.0)
    p.add_argument("--morph-beta-final", type=float, default=30.0)
    p.add_argument("--survey-top-k", type=int, default=5)
    p.add_argument("--survey-workers", type=int, default=min(os.cpu_count() or 1, 8))
    p.add_argument("--survey-n", type=int, default=25,
                   help="cases the auto survey scores (0 = all); all-slices, train split")
    p.add_argument("--freeze-se", action="store_true")
    # training parameters
    p.add_argument("--epochs", type=int, default=config.HP["epochs"])
    p.add_argument("--patience", type=int, default=config.HP["patience"])
    p.add_argument("--batch-size", type=int, default=config.HP["batch_size"])
    p.add_argument("--patch-size", type=int, default=config.HP["patch_size"])
    p.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 6))
    p.add_argument("--lr", type=float, default=config.HP["lr"])
    p.add_argument("--iters-per-epoch", type=int, default=config.HP["iters_per_epoch"])
    p.add_argument("--val-cases", type=int, default=15)
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--val-batch", type=int, default=12)
    p.add_argument("--fg-fraction", type=float, default=config.HP["fg_fraction"])
    p.add_argument("--se-lr-mult", type=float, default=10.0)
    # static channel augmentation (from augment_channels.py)
    p.add_argument("--static-dir", default=None,
                   help="preprocessed dir with augmented npys (default: config.PREPROCESSED_DIR)")
    p.add_argument("--static-channels", type=int, default=0,
                   help="number of static filter channels between image and label in the npy")
    p.add_argument("--static-auto", action="store_true",
                   help="survey -> top-k STATIC filters -> precompute -> add as input channels (no "
                        "trainable blocks). Combine with --morph-bank auto for trainable blocks + static.")
    # other
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--test-only", action="store_true")
    p.add_argument("--fold-mean", metavar="TAG")
    p.add_argument("--compare", nargs="+", metavar="JSON")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.fold_mean:
        fold_mean(args.fold_mean)
        return
    if args.compare:
        compare_runs(args.compare)
        return
    if not args.tag:
        p.error("--tag is required")

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        # let Ada use TF32 matmuls and autotune convs for the fixed patch size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Resolve auto specs BEFORE build_loaders (the loader needs static_dir / static_channels).
    # The two auto modes are INDEPENDENT and compose:
    #   --morph-bank auto : survey -> top-k TRAINABLE -> morphological blocks (this line only)
    #   --static-auto     : survey -> top-k STATIC   -> extra input channels
    #   both flags        : trainable blocks + static channels (combined)
    if args.morph_bank == "auto":
        args.morph_bank = ensure_bank_spec(config.TASK, args.fold, config.DATA_DIR,
                                           args.survey_top_k, args.survey_workers, n=args.survey_n)
    if args.static_auto and not args.static_dir and args.static_channels == 0:
        sdir, n = ensure_static_channels(config.TASK, args.fold, config.DATA_DIR,
                                         args.survey_top_k, args.survey_workers, n=args.survey_n)
        if n > 0:
            args.static_dir = sdir
            args.static_channels = n

    train_loader, val_loader, test_loader, in_channels = build_loaders(args)

    if args.morph_unet:
        # morphological-separable U-Net: conv stages replaced by depthwise soft morphology
        # + 1x1 projection, per the chosen config. Its SEs end in ".se", so they pick up the
        # boosted SE lr and --freeze-se just like the bank/residual variants.
        model = MorphUNet(num_classes=config.NUM_CLASSES, in_channels=in_channels,
                          k=args.morph_k, beta=args.morph_beta, config=args.morph_unet).to(device)
    elif args.morph_bank:
        specs = parse_bank(args.morph_bank)
        n_extra = len(specs)
        base = UNet(num_classes=config.NUM_CLASSES, in_channels=in_channels + n_extra)
        if args.conv_control:
            model = ConvBankUNet(base, n_extra=n_extra, k=args.morph_k_max, in_channels=in_channels).to(device)
        else:
            model = MorphBankUNet(base, specs, k_max=args.morph_k_max, beta=args.morph_beta).to(device)
    elif args.morph_block or args.tophat or args.bottomhat:
        # trainable (--morph-block, learnable SE) or static (--tophat/--bottomhat, frozen SE)
        # residuals, both computed on the fly. under --morph-block default to top-hat.
        use_th = args.tophat or (args.morph_block and not args.bottomhat)
        use_bh = args.bottomhat
        base = UNet(num_classes=config.NUM_CLASSES, in_channels=in_channels + use_th + use_bh)
        model = MorphResidualUNet(base, k=args.morph_k, beta=args.morph_beta,
                                  use_tophat=use_th, use_bottomhat=use_bh).to(device)
        if not args.morph_block:            # static residual -> fixed (frozen) SE
            args.freeze_se = True
    else:
        model = UNet(num_classes=config.NUM_CLASSES, in_channels=in_channels).to(device)

    if args.freeze_se:                      # static: SE is a fixed structuring element
        for n, p in model.named_parameters():
            if n.endswith(".se"):
                p.requires_grad_(False)

    # (<tag>_f<fold>_{best.pth,last.pth,scores.json}
    stem = f"{args.tag}_f{args.fold}"
    if args.morph_unet:
        mode = f"morph-unet({args.morph_unet},k={args.morph_k},beta={args.morph_beta})"
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
    static_info = f" static_ch={args.static_channels} static_dir={args.static_dir}" if args.static_channels else ""
    print(f"[{stem}] device={device} mode={mode} loss={loss_desc} seed={args.seed} "
          f"fold={args.fold} loader_in_ch={in_channels} patch={args.patch_size} "
          f"iters/epoch={epoch_len} max-epochs={args.epochs} (<= {epoch_len * args.epochs} updates) "
          f"fg={args.fg_fraction}{static_info}")

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

    results_dir = task_results_dir()
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
        # geometric beta anneal (soft -> sharper morphology): on by default (--morph-beta-final),
        # only for models exposing set_beta (bank / residual -- baseline & conv control have none),
        # and skipped for a frozen SE (static residual stays a fully fixed operation) or when the
        # target equals the start. schedule is a pure function of epoch, so it survives --resume.
        anneal_beta = (args.morph_beta_final is not None
                       and args.morph_beta_final != args.morph_beta
                       and hasattr(model, "set_beta") and not args.freeze_se)
        cur_beta = args.morph_beta
        if anneal_beta:
            print(f"  beta anneal: {args.morph_beta:g} -> {args.morph_beta_final:g} "
                  f"(geometric over {args.epochs} epochs)")
        for epoch in range(start_epoch, args.epochs + 1):
            if anneal_beta:
                frac = (epoch - 1) / max(args.epochs - 1, 1)
                cur_beta = args.morph_beta * (args.morph_beta_final / args.morph_beta) ** frac
                model.set_beta(cur_beta)
            tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer, morph_loss)
            do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            if do_val:
                vd, per_class = run_val_dice(model, val_loader, device, config.NUM_CLASSES)
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
                      + "   |SE|: " + " ".join(f"{v:.3f}" for v in norms)
                      + (f"   beta={cur_beta:.1f}" if anneal_beta else ""))
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
