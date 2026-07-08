#
# Train and Test for U-Net
#

import argparse
import contextlib
import glob
import hashlib
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
from networks.morph_block import MorphBankUNet, ConvBankUNet
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

#
# pick gpu architecture
#
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#
# read morphbank spec
#
def _read_bank_spec(task, fold, root="results/explore"):
    path = os.path.join(root, f"{task}_bank.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        bank = json.load(f)
    return bank.get(str(fold))

#
# read static input spec
#
def _read_static_spec(task, fold, root="results/explore"):
    path = os.path.join(root, f"{task}_static.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        bank = json.load(f)
    return bank.get(str(fold))

#
# run morphology survey once (train split, all slices, n cases (n=0 is all cases))
#
def _run_survey(task, fold, dataset_dir, top_k, workers, root, n=25):
    print(f"no cached spec for {task} fold {fold}, running survey"
          f"(n={'all' if n == 0 else n} cases, all slices, train split)...", flush=True)
    subprocess.run([sys.executable, os.path.join("utilities", "morph_explore.py"), "survey",
                    str(dataset_dir), "--fold", str(fold), "--top-k", str(top_k),
                    "--n", str(n), "--split", "train", "--all-slices",
                    "--workers", str(workers), "--out-dir", root], check=True)

#
# resolve --morph-bank auto, reuse cached spec if present else run the survey once
#
def ensure_bank_spec(task, fold, dataset_dir, top_k, workers, root="results/explore", n=0):
    spec = _read_bank_spec(task, fold, root)
    if spec is None:
        _run_survey(task, fold, dataset_dir, top_k, workers, root, n)
        spec = _read_bank_spec(task, fold, root)
        if spec is None:
            raise RuntimeError(f"survey produced no spec for fold {fold}")
    print(f"trainable -> \"{spec}\"", flush=True)
    return spec

#
# build (or reuse) the augmented preprocessed dir holding `specs` as static channels, running
# augment_channels.py once; reuse only if the cached dir's manifest matches specs exactly (not
# just the count). `suffix` disambiguates the dir (per-fold for survey, spec-hash for manual)
#
def _ensure_static_dir(specs, suffix, perturb="none", strength=0.0, seed=0, fold=0):
    prep_dir = str(config.PREPROCESSED_DIR)
    # a perturbed dir gets its own name+cache: keyed by perturb+strength AND fold, because it only
    # holds that fold's TEST split (augment restricts to it), so it must not be shared across folds
    psuf = f"_{perturb}{strength:g}_f{fold}" if perturb != "none" else ""
    static_dir = prep_dir.rstrip("/") + f"_static_{suffix}{psuf}"
    manifest = os.path.join(static_dir, "filters.json")
    if os.path.exists(manifest):
        with open(manifest) as f:
            cached = json.load(f)
    else:
        cached = None
    if cached == specs:
        print(f"static dir exists (cached): {static_dir} ({len(specs)} filters)", flush=True)
    else:
        tag = f" perturb={perturb}{strength:g} fold={fold}(test)" if perturb != "none" else ""
        print(f"precomputing static channels: {specs}{tag} -> {static_dir}", flush=True)
        augment = os.path.join(config.PROJECT_ROOT, "datasets", "augment_channels.py")
        cmd = [sys.executable, augment, "--filters"] + specs + ["--src", prep_dir, "--out", static_dir]
        if perturb != "none":
            cmd += ["--perturb", perturb, "--perturb-strength", str(strength),
                    "--perturb-seed", str(seed), "--fold", str(fold)]
        subprocess.run(cmd, check=True)
    print(f"static -> {specs}  ({len(specs)} channels)", flush=True)
    return static_dir, len(specs)

#
# augmented dirs are keyed by a hash of the spec list, the filter channels are deterministic
# so any fold runs with the same specs, shares one dir and rebuilds are skipped
#
def _spec_suffix(specs):
    return "h" + hashlib.md5(",".join(specs).encode()).hexdigest()[:8]

#
# resolve static channels, read cached spec or run the full survey (per fold, since selection is
# fold dependent), then build/reuse the spec-hashed augmented dir, n_ch is the channel count
#
def ensure_static_channels(task, fold, dataset_dir, top_k, workers, root="results/explore", n=0):
    specs = _read_static_spec(task, fold, root)
    if specs is None:
        _run_survey(task, fold, dataset_dir, top_k, workers, root, n)
        specs = _read_static_spec(task, fold, root)
    if not specs:
        print("no static filters selected by survey", flush=True)
        return None, 0
    return _ensure_static_dir(specs, _spec_suffix(specs))

#
# resolve explicit --static-filters (bypass the survey): same spec-hashed dir, so it dedupes with
# the survey path whenever the specs coincide
#
def ensure_static_filters(specs):
    return _ensure_static_dir(specs, _spec_suffix(specs))

#
# build loaders
#
def build_loaders(args):
    with open(config.SPLITS_FILE, "rb") as f:
        splits = pickle.load(f)
    tr, vl, ts = (splits[args.fold]["train"], splits[args.fold]["val"], splits[args.fold]["test"])
    # data-efficiency sweep: keep only N training cases (val/test untouched). NESTED subsets across N
    # (shuffle once with a fixed seed, take the first N) so N=5 ⊂ N=10 ⊂ ...; the only variable that
    # moves between sweep points is training-set size, not which cases were drawn.
    n_train = getattr(args, "train_cases", 0) or 0
    if 0 < n_train < len(tr):
        order = list(tr)
        random.Random(args.seed).shuffle(order)
        tr = sorted(order[:n_train])
        print(f"data-efficiency: training on {n_train}/{len(splits[args.fold]['train'])} cases -> {tr}", flush=True)
    # --static-dir overrides to the augmented preprocessed dir, --static-channels sets the layout
    n_static = getattr(args, "static_channels", 0) or 0
    data_dir = getattr(args, "static_dir", None) or str(config.PREPROCESSED_DIR)
    # [image, filt_1, ..., filt_N, label], N = static_channels (0 for baseline)
    input_slice = tuple(range(1 + n_static))
    label_slice = 1 + n_static
    common = dict(target_size=args.patch_size, batch_size=args.batch_size, input_slice=input_slice,
                  label_slice=label_slice, num_processes=args.num_workers, fg_fraction=args.fg_fraction)
    cap = dict(num_batches=args.iters_per_epoch) if args.iters_per_epoch > 0 else {}
    train = None if args.test_only else NumpyDataSet(data_dir, keys=tr, **common, **cap)
    # val: full-slice like test, --val-cases caps to a fixed subset of volumes,
    # --val-batch packs full slices per forward pass to use the GPU
    val_keys = list(vl)
    if args.val_cases and 0 < args.val_cases < len(val_keys):
        val_keys = sorted(random.Random(args.seed).sample(val_keys, args.val_cases))
    val = None if args.test_only else NumpyDataSet(data_dir, keys=val_keys, mode="test", do_reshuffle=False,
                                                   **{**common, "batch_size": args.val_batch})
    # test: full-slice inference one at a time, all slices
    test = NumpyDataSet(data_dir, keys=ts, mode="test", do_reshuffle=False, **{**common, "batch_size": 1})
    in_channels = len(input_slice)
    return train, val, test, in_channels

#
# run epoch
#
def run_epoch(model, loader, device, dice_loss, ce_loss, optimizer=None, morph_loss=None, grad_clip=5.0):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    losses = []
    amp = device.type == "cuda"   # bf16 autocast on CUDA
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
                # clip grad norm so a foreground-sparse batch with a sharp Dice gradient can't
                # spike/diverge the epoch, tighter clip for sparse multi-class tasks
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")

#
# validation on full slices with global foreground Dice
#
def run_val_dice(model, loader, device, num_classes):
    model.eval()
    n_fg = max(num_classes - 1, 1)
    tp = torch.zeros(n_fg)
    fp = torch.zeros(n_fg)
    fn = torch.zeros(n_fg)
    amp = device.type == "cuda"
    chunk = [0]         # GPU sub batch cap, 0 = whole loader-batch, drops after a CUDA OOM
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
            while i < b:                                                    # forward in OOM safe chunks
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
                    print(f"CUDA OOM at chunk {step} -> retry at {chunk[0]}", flush=True)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)         # per-class global Dice
    present = (tp + fn) > 0                             # classes actually present in the val set
    mean = dice[present].mean().item() if present.any() else 0.0
    return mean, dice.tolist()

#
# test-time intensity perturbation, applied to the raw-image channel (0) only so any morph/static
# front-end channels are recomputed from (or kept consistent with) the shifted image. eval-time
# stress test for robustness: gamma (strictly monotonic -> rank-based morphology near-equivariant),
# contrast (affine about the mean, monotonic), noise (additive gaussian, non-monotonic -> tests the
# opening/closing denoising path). strength s: gamma/contrast are multiplicative factors (1 = id),
# noise is a sigma in normalized (z-scored) units. seed keeps the noise draw reproducible.
#
def _perturb_image(data, kind, s, gen):
    if kind == "none" or s == 0:
        return data
    x = data.clone()
    img = x[:, :1]
    if kind == "gamma":
        mn = img.amin((2, 3), keepdim=True)
        mx = img.amax((2, 3), keepdim=True)
        u = ((img - mn) / (mx - mn + 1e-6)).clamp(0, 1)
        img = u.pow(s) * (mx - mn) + mn
    elif kind == "contrast":
        m = img.mean((2, 3), keepdim=True)
        img = (img - m) * s + m
    elif kind == "noise":
        img = img + torch.randn(img.shape, generator=gen).to(img.device) * s
    x[:, :1] = img
    return x

#
# test phase
#
def evaluate_test(model, loader, device, json_path, num_workers=1, perturb="none", strength=0.0,
                  seed=0, advanced=True):
    model.eval()
    gen = torch.Generator().manual_seed(seed)   # cpu generator, device-agnostic reproducible noise
    # accumulate per-case pred/GT as uint8 (labels are small ints). Storing int64 preds + float
    # targets for the whole full-slice test set blew up CPU RAM and got the process OOM-killed.
    pred_dict, gt_dict = defaultdict(list), defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            data = batch["data"][0].float().to(device)
            data = _perturb_image(data, perturb, strength, gen)              # eval-time robustness stress
            target = batch["seg"][0][:, 0].to(torch.uint8).numpy()           # [b, H, W]
            pred = model(data).argmax(1).to(torch.uint8).cpu().numpy()       # [b, H, W]
            for i, fname in enumerate(batch["fnames"]):
                pred_dict[fname[0]].append(pred[i])
                gt_dict[fname[0]].append(target[i])
    pairs = [(np.stack(pred_dict[k]), np.stack(gt_dict[k])) for k in pred_dict]   # each [Z, H, W]
    # advanced=False drops the surface-distance metrics (HD95/ASSD) — the slow, scipy-bound part —
    # leaving Dice/precision/recall, which is all the robustness/data-efficiency curves need
    scores = aggregate_scores(pairs, evaluator=Evaluator, labels=LABELS,
        json_output_file=json_path, json_author="cv-project",
        json_task=config.TASK, num_workers=num_workers, advanced=advanced,
    )
    return scores

#
# find files
#
def _run_name(path):
    base = os.path.basename(path)
    for suf in ("_scores.json", ".json"):
        if base.endswith(suf):
            return base[:-len(suf)]
    return base

#
# all per run artifacts live under results/<TASK>/ so runs for different MSD tasks
# don't collide in one flat folder, task_results_dir() is the write target
#
def task_results_dir():
    d = os.path.join(config.PROJECT_ROOT, "results", config.TASK)
    os.makedirs(d, exist_ok=True)
    return d

#
# read_results_dir() is the READ source and falls back to results/
#
def read_results_dir():
    d = os.path.join(config.PROJECT_ROOT, "results", config.TASK)
    return d if os.path.isdir(d) else os.path.join(config.PROJECT_ROOT, "results")

#
# segmentation metrics reported by fold_mean, compare
# Dice + ASSD are the MSD primaries, HD95 is a robust boundary metric
# Precision/Recall explain why Dice moves, Jaccard (monotone with Dice)
# and the background dominated rates (Accuracy/FPR/TNR/NPV) are omitted
#
SCORE_METRICS = ("Dice", "Avg. Symmetric Surface Distance", "Hausdorff Distance 95",
                 "Precision", "Recall")
#
# per run training-cost, convergence stats across folds
#
TRAIN_STATS = ("best_fg_dice", "best_epoch", "epochs_to_90pct_best", "sec_per_epoch",
               "params_total", "params_frontend")

#
# {mean, std, n} over the non-None values
#
def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

#
# average each metric (and training stats) across all folds (<tag>_f<fold>_scores.json)
#
def fold_mean(tag):
    results_dir = read_results_dir()   # task subdir, or flat results/ as a fallback
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
    # training cost, convergence summary, averaged over the same folds
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
# robustness curve: mean foreground <metric> vs perturbation strength, one line per model.
# reads the fold-mean files fold_mean() writes for each (model, kind, strength):
#   <model>_<kind><strength>_mean_scores.json   (+ the clean <model>_mean_scores.json as the
#   identity anchor: strength 1.0 for gamma/contrast, 0.0 for noise). error bars are the
#   across-fold std already stored in those files, averaged over the foreground labels.
#
def plot_perturb(kind, models, metric="Dice"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results_dir = read_results_dir()

    def fg_metric(path):   # mean (and mean-of-per-label-std) of <metric> over foreground labels
        with open(path) as f:
            mean = json.load(f)["mean"]
        vals, stds = [], []
        for label, mets in mean.items():
            if str(label) == "0":            # skip background if present
                continue
            m = mets.get(metric)
            if isinstance(m, dict):
                vals.append(m["mean"]); stds.append(m.get("std", 0.0))
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.mean(stds))

    anchor = 1.0 if kind in ("gamma", "contrast") else 0.0   # identity strength for this kind
    plt.figure(figsize=(6, 4))
    for model in models:
        pts = []
        for p in glob.glob(os.path.join(results_dir, f"{model}_{kind}*_mean_scores.json")):
            tail = os.path.basename(p)[len(f"{model}_{kind}"):-len("_mean_scores.json")]
            try:
                s = float(tail)
            except ValueError:
                continue
            mv, sv = fg_metric(p)
            if mv is not None:
                pts.append((s, mv, sv))
        clean = os.path.join(results_dir, f"{model}_mean_scores.json")   # identity anchor
        if os.path.exists(clean) and not any(abs(s - anchor) < 1e-9 for s, _, _ in pts):
            mv, sv = fg_metric(clean)
            if mv is not None:
                pts.append((anchor, mv, sv))
        if not pts:
            print(f"no files match {model}_{kind}*_mean_scores.json in {results_dir}")
            continue
        pts.sort()
        xs, ys, es = zip(*pts)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=model)
        print(f"{model}: " + "  ".join(f"{s:g}->{y:.4f}" for s, y, _ in pts))
    xlab = {"gamma": "gamma  (1 = identity)", "contrast": "contrast factor  (1 = identity)",
            "noise": "gaussian sigma  (0 = clean)"}[kind]
    plt.axvline(anchor, color="k", ls=":", lw=0.8, alpha=0.5)
    plt.xlabel(xlab); plt.ylabel(f"mean foreground {metric}")
    plt.title(f"{config.TASK}: {metric} vs {kind} perturbation")
    plt.legend(); plt.grid(alpha=0.3)
    out = os.path.join(results_dir, f"{kind}_robustness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"written {out}")

#
# mean over foreground labels of a metric in a fold-mean file (+ mean of the per-label across-fold std)
#
def _fg_mean(path, metric):
    with open(path) as f:
        mean = json.load(f)["mean"]
    vals, stds = [], []
    for label, mets in mean.items():
        if str(label) == "0":            # skip background if present
            continue
        m = mets.get(metric)
        if isinstance(m, dict):
            vals.append(m["mean"]); stds.append(m.get("std", 0.0))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.mean(stds))

#
# data-efficiency curve: mean foreground <metric> vs #training cases, one line per model. reads the
# fold-mean files written for each (model, N): <model>_n<N>_mean_scores.json, plus the full-data run
# <model>_mean_scores.json as the top anchor (placed at the full per-fold train size). log-x.
#
def plot_data_efficiency(models, metric="Dice"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results_dir = read_results_dir()
    full_n = None                        # full per-fold train size (for the n0/full anchor + x pos)
    try:
        with open(config.SPLITS_FILE, "rb") as f:
            full_n = len(pickle.load(f)[0]["train"])
    except Exception:
        pass
    plt.figure(figsize=(6, 4))
    for model in models:
        pts = []
        for p in glob.glob(os.path.join(results_dir, f"{model}_n*_mean_scores.json")):
            tail = os.path.basename(p)[len(f"{model}_n"):-len("_mean_scores.json")]
            try:
                n = int(tail)
            except ValueError:
                continue                 # skip e.g. _n5_gamma0.5 (perturbed) files
            if n == 0:
                n = full_n or 0
            mv, sv = _fg_mean(p, metric)
            if mv is not None and n:
                pts.append((n, mv, sv))
        full = os.path.join(results_dir, f"{model}_mean_scores.json")   # full-data anchor
        if full_n and os.path.exists(full) and not any(n == full_n for n, _, _ in pts):
            mv, sv = _fg_mean(full, metric)
            if mv is not None:
                pts.append((full_n, mv, sv))
        if not pts:
            print(f"no files match {model}_n*_mean_scores.json in {results_dir}")
            continue
        pts.sort()
        xs, ys, es = zip(*pts)
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=model)
        print(f"{model}: " + "  ".join(f"N={n}->{y:.4f}" for n, y, _ in pts))
    plt.xscale("log")
    plt.xlabel("training cases (log scale)"); plt.ylabel(f"mean foreground {metric}")
    plt.title(f"{config.TASK}: data efficiency ({metric} vs training-set size)")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    out = os.path.join(results_dir, "data_efficiency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"written {out}")

#
# per label metric deltas (and training stat deltas) of each run vs the baseline
#
def compare_runs(paths):
    if len(paths) < 2:
        raise SystemExit("--compare needs at least two JSON files")
    baseline, others = paths[0], paths[1:]

    # path as given, else results/<TASK>/ (or results/)
    def resolve(p):
        if os.path.exists(p):
            return p
        alt = os.path.join(read_results_dir(), p)
        return alt if os.path.exists(alt) else p

    # {label: {metric: float}} accepting per-fold or fold-mean files
    def load_mean(p):
        with open(resolve(p)) as f:
            d = json.load(f)
        kind = "fold-mean" if "mean" in d else "per-fold"
        raw = d["mean"] if kind == "fold-mean" else d["results"]["mean"]
        # fold-mean files nest {"mean", "std", "n"}; flatten to the mean
        flat = {label: {m: (v["mean"] if isinstance(v, dict) else v) for m, v in metrics.items()}
                for label, metrics in raw.items()}
        return kind, flat

    # {stat: float} training summary, fold-mean files carry a train block,
    # per-fold files have a sibling <run>_train.json
    def load_train(p):
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
        print("Warning, mixing per-fold and fold-mean files in one compare")
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
        # training cost, convergence deltas (needs a train summary on both sides)
        o_train = load_train(other)
        if base_train and o_train:
            print("-- train --")
            for stat in TRAIN_STATS:
                bv, ov = base_train.get(stat), o_train.get(stat)
                if bv is None or ov is None:
                    continue
                print(f"  {stat:<32} {bv:>15.4f} -> {ov:>15.4f}   delta={ov - bv:+.4f}")

#
# main
#
def main():
    p = argparse.ArgumentParser()
    # id
    p.add_argument("--tag")
    # modes
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
    p.add_argument("--survey-n", type=int, default=25)
    p.add_argument("--survey-workers", type=int, default=min(os.cpu_count() or 1, 16))
    p.add_argument("--freeze-se", action="store_true")
    # training parameters
    p.add_argument("--epochs", type=int, default=config.HP["epochs"])
    p.add_argument("--patience", type=int, default=config.HP["patience"])
    p.add_argument("--batch-size", type=int, default=config.HP["batch_size"])
    p.add_argument("--patch-size", type=int, default=config.HP["patch_size"])
    p.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 16))
    p.add_argument("--lr", type=float, default=config.HP["lr"])
    p.add_argument("--iters-per-epoch", type=int, default=config.HP["iters_per_epoch"])
    p.add_argument("--val-cases", type=int, default=config.HP["val_cases"])
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--val-batch", type=int, default=config.HP["val_batch"])
    p.add_argument("--fg-fraction", type=float, default=config.HP["fg_fraction"])
    p.add_argument("--grad-clip", type=float, default=config.HP["grad_clip"])
    p.add_argument("--se-lr-mult", type=float, default=10.0)
    # static input channels
    p.add_argument("--static-dir", default=None)
    p.add_argument("--static-channels", type=int, default=0)
    p.add_argument("--static-auto", action="store_true")
    p.add_argument("--static-filters", nargs="+", metavar="SPEC")
    # other
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--train-cases", type=int, default=0)   # 0 = full split; N>0 = data-efficiency point
    p.add_argument("--resume", action="store_true")
    p.add_argument("--test-only", action="store_true")
    # eval-time robustness stress (reuses trained weights, no retraining): perturb the input image
    # then re-score. scores are written under a perturb-suffixed tag so the clean run isn't clobbered
    p.add_argument("--test-perturb", choices=["none", "gamma", "contrast", "noise"], default="none")
    p.add_argument("--perturb-strength", type=float, default=0.0)
    p.add_argument("--fast-eval", action="store_true")   # Dice-only test (skip slow HD95/ASSD)
    p.add_argument("--skip-existing", action="store_true")   # skip a point whose scores json already exists
    p.add_argument("--keep-static", action="store_true")   # keep the perturbed static dir (default: delete after eval)
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

    # --skip-existing: if this exact point's scores json already exists, skip the whole run (no model
    # build / augment / inference) and just refresh the plot so the curve stays complete. lets an
    # interrupted sweep resume without recomputing done points.
    if args.skip_existing:
        rt = f"{args.tag}_n{args.train_cases}" if args.train_cases and args.train_cases > 0 else args.tag
        st = rt if args.test_perturb == "none" else f"{rt}_{args.test_perturb}{args.perturb_strength:g}"
        jp = os.path.join(config.PROJECT_ROOT, "results", config.TASK, f"{st}_f{args.fold}_scores.json")
        if os.path.exists(jp):
            print(f"[{st}_f{args.fold}] scores exist, skipping (--skip-existing): {jp}", flush=True)
            if args.test_perturb != "none":
                try:
                    fold_mean(st)
                    plot_perturb(args.test_perturb,
                                 sorted({"baseline", "convctrl", "morphbank", "staticbank", args.tag}), "Dice")
                except Exception as e:
                    print(f"[viz] plot skipped: {e}", flush=True)
            return

    set_seed(args.seed)
    device = pick_device()
    if device.type == "cuda":
        # let Ada use TF32 matmuls and autotune convs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # resolve auto specs before build_loaders
    # the two auto modes are independent and compose:
    # --morph-bank auto : survey -> top-k trainable -> morphological blocks
    # --static-auto     : survey -> top-k static    -> extra input channels
    #   both flags      : trainable blocks and static channels combined
    if args.morph_bank == "auto":
        args.morph_bank = ensure_bank_spec(config.TASK, args.fold, config.DATA_DIR,
                                           args.survey_top_k, args.survey_workers, n=args.survey_n)
    # --static-filters : explicit specs -> extra input channels (takes priority over --static-auto)
    if args.static_filters and not args.static_dir and args.static_channels == 0:
        args.static_dir, args.static_channels = ensure_static_filters(args.static_filters)
    if args.static_auto and not args.static_dir and args.static_channels == 0:
        if args.test_perturb != "none":
            # test-only perturbed eval: the clean full augmented dir is never used, so DON'T build it
            # (484 cases of slow dome filters) -- just resolve the survey specs; the perturb block
            # below builds only this fold's perturbed TEST split.
            specs = _read_static_spec(config.TASK, args.fold)
            if specs is None:
                _run_survey(config.TASK, args.fold, config.DATA_DIR,
                            args.survey_top_k, args.survey_workers, "results/explore", args.survey_n)
                specs = _read_static_spec(config.TASK, args.fold)
            if specs:
                args.static_channels = len(specs)   # static_dir stays unset; set by perturb block
        else:
            sdir, n = ensure_static_channels(config.TASK, args.fold, config.DATA_DIR,
                                             args.survey_top_k, args.survey_workers, n=args.survey_n)
            if n > 0:
                args.static_dir = sdir
                args.static_channels = n

    # static channels + --test-perturb (staticbank robustness): rebuild channels on the PERTURBED
    # image (recompute -> no clean-channel leak), restricted to this fold's test split. specs come
    # from an explicit --static-dir manifest if present, else the survey (--static-auto/--static-filters).
    perturb_baked = False
    if args.test_perturb != "none" and (getattr(args, "static_channels", 0) or 0) > 0:
        manifest = os.path.join(args.static_dir, "filters.json") if args.static_dir else None
        if manifest and os.path.exists(manifest):
            specs = json.load(open(manifest))
        else:
            specs = _read_static_spec(config.TASK, args.fold) or args.static_filters
        if not specs:
            p.error("--test-perturb with static channels: could not resolve filter specs")
        args.static_dir, args.static_channels = _ensure_static_dir(
            specs, _spec_suffix(specs), perturb=args.test_perturb, strength=args.perturb_strength,
            seed=args.seed, fold=args.fold)
        perturb_baked = True
        print("static robustness: perturbation baked into channels (no in-loop perturb)", flush=True)

    train_loader, val_loader, test_loader, in_channels = build_loaders(args)

    if args.morph_unet:
        # morphological-separable U-Net, conv stages replaced by depthwise soft morphology
        # and 1x1 projection, per the chosen config, its SEs end in ".se", so they pick up the
        # boosted SE lr and --freeze-se just like the bank, residual variants
        model = MorphUNet(num_classes=config.NUM_CLASSES, in_channels=in_channels,
                          k=args.morph_k, beta=args.morph_beta, config=args.morph_unet).to(device)
    elif args.morph_bank:
        # trainable morph bank spanning the full library (exact / reconstruction / vdome-
        # surrogate ops); one extra input channel per spec token. --conv-control swaps the
        # bank for a parameter-matched learned conv front-end.
        specs = [s for s in args.morph_bank.split(",") if s]
        n_extra = len(specs)
        base = UNet(num_classes=config.NUM_CLASSES, in_channels=in_channels + n_extra)
        if args.conv_control:
            model = ConvBankUNet(base, n_extra=n_extra, k=args.morph_k_max, in_channels=in_channels).to(device)
        else:
            model = MorphBankUNet(base, specs, k=args.morph_k_max, beta=args.morph_beta).to(device)
    else:
        model = UNet(num_classes=config.NUM_CLASSES, in_channels=in_channels).to(device)

    if args.freeze_se:                      # static, SE is fixed
        for n, p in model.named_parameters():
            if n.endswith(".se"):
                p.requires_grad_(False)

    # <tag>_f<fold>_{best.pth,last.pth,scores.json}; a data-efficiency point trains its own model,
    # so fold N into the tag (run_tag) to keep its checkpoint/scores separate from the full-data run
    run_tag = f"{args.tag}_n{args.train_cases}" if args.train_cases and args.train_cases > 0 else args.tag
    stem = f"{run_tag}_f{args.fold}"
    if args.morph_unet:
        mode = f"morph-unet({args.morph_unet},k={args.morph_k},beta={args.morph_beta})"
    elif args.morph_bank:
        kind = "conv-control" if args.conv_control else "morph-bank"
        mode = f"{kind}([{args.morph_bank}],beta={args.morph_beta})"
    else:
        mode = "baseline"
    loss_desc = "dice+ce" + ("+morph" if args.morph_loss else "")
    epoch_len = args.iters_per_epoch if args.iters_per_epoch > 0 else len(train_loader)
    static_info = f" static_ch={args.static_channels} static_dir={args.static_dir}" if args.static_channels else ""
    print(f"[{stem}] device={device} mode={mode} loss={loss_desc} seed={args.seed} "
          f"fold={args.fold} loader_in_ch={in_channels} patch={args.patch_size} "
          f"iters/epoch={epoch_len} max-epochs={args.epochs} (<= {epoch_len * args.epochs} updates) "
          f"fg={args.fg_fraction} grad_clip={args.grad_clip:g}{static_info}")

    # foreground only Dice, background is most of pixels and its Dice is almost constant
    # so including it dilutes the gradient on the sparse target, CE still sees all classes
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()
    morph_loss = MorphConsistencyLoss().to(device) if args.morph_loss else None
    # give the morphological SE weights their own (higher) lr,
    # few, geometrically important params with sparse gradients, so a larger step helps them move
    # frozen SEs are requires_grad=False, excluded from the optimiser
    # SE tensors across all block types get the boosted lr (bank .se, ASF .ses.N,
    # recon .se_erode, vdome .gran_ses.N); scalar raw_h keeps the base lr
    def _is_morph_se(name):
        return (name.endswith(".se") or ".ses." in name
                or ".se_erode" in name or ".gran_ses." in name)
    se_named = [(n, p) for n, p in model.named_parameters() if _is_morph_se(n) and p.requires_grad]
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
        # selection metric is foreground Dice, best_dice tracks the best so far
        start_epoch, best_dice, best_epoch = 1, -1.0, 0
        t0, val_curve = time.time(), []   # wall-clock + (epoch, fg_dice) learning curve for summary
        # resume from the full-state last.pth (model + optim + scheduler + counters)
        if args.resume and os.path.exists(last_path):
            ck = torch.load(last_path, map_location=device)
            if isinstance(ck, dict) and "model" in ck:
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                scheduler.load_state_dict(ck["scheduler"])
                start_epoch = ck["epoch"] + 1
                best_dice, best_epoch = ck["best_val"], ck.get("best_epoch", 0)
                val_curve = ck.get("val_curve", [])   # restore learning curve
                print(f"resumed from epoch {ck['epoch']} (best fg-Dice={best_dice:.4f} @ ep {best_epoch}, "
                      f"{start_epoch - 1 - best_epoch}/{args.patience} epochs stale)")
        se_prev = {n: p.detach().clone() for n, p in se_named}   # to log how much the SEs move
        # geometric beta anneal (soft -> sharper morphology), on by default (--morph-beta-final),
        # only for models exposing set_beta (bank and residual, baseline and conv control have none),
        # and skipped for a frozen SE or when the target equals the start
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
            tr = run_epoch(model, train_loader, device, dice_loss, ce_loss, optimizer, morph_loss,
                           grad_clip=args.grad_clip)
            do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
            if do_val:
                vd, per_class = run_val_dice(model, val_loader, device, config.NUM_CLASSES)
                scheduler.step(-vd)    # scheduler minimises, we maximise Dice, so feed -Dice
                print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  val_fgDice={vd:.4f}  "
                      f"[per-class {' '.join(f'{d:.3f}' for d in per_class)}]")
                val_curve.append([epoch, round(vd, 5)])
            else:
                print(f"epoch {epoch:3d}/{args.epochs}  train={tr:.4f}  (val every {args.val_every})")
            if se_named:   # to see if the SE is actually learning, |Δ| per block + |SE| magnitude
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
                        "best_epoch": best_epoch, "val_curve": val_curve}, last_path)
            # early stop on epochs since last improvement (independent of --val-every)
            if args.patience and (epoch - best_epoch) >= args.patience:
                print(f"early stop: no val fg-Dice improvement for {epoch - best_epoch} epochs "
                      f"(best fg-Dice={best_dice:.4f} @ ep {best_epoch})")
                break

        # per run training summary, convergence and cost (--compare)
        elapsed = time.time() - t0
        front = sum(p.numel() for n, p in model.named_parameters()
                    if n.startswith(("blocks.", "front.", "tophat.", "bottomhat.")))
        total = sum(p.numel() for p in model.parameters())
        # epochs to first reach 90% of the best fg-Dice, a threshold-based convergence-speed metric
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
    # checkpoint loads from the clean stem; scores get a perturb-suffixed tag so a run like
    # `--tag baseline --test-perturb gamma --perturb-strength 0.5` writes baseline_gamma0.5_f<fold>
    # and `--fold-mean baseline_gamma0.5` / `--compare` pick it up unchanged
    score_tag = run_tag if args.test_perturb == "none" else f"{run_tag}_{args.test_perturb}{args.perturb_strength:g}"
    json_path = os.path.join(results_dir, f"{score_tag}_f{args.fold}_scores.json")
    # if the perturbation is already baked into the input dir (staticbank), don't apply it again in-loop
    loop_perturb = "none" if perturb_baked else args.test_perturb
    scores = evaluate_test(model, test_loader, device, json_path, num_workers=args.num_workers,
                           perturb=loop_perturb, strength=args.perturb_strength, seed=args.seed,
                           advanced=not args.fast_eval)
    print(f"[{stem}] mean scores written to {json_path}")
    for label, md in scores["mean"].items():
        print(f"  label {label}: Dice={md.get('Dice')} "
              f"ASSD={md.get('Avg. Symmetric Surface Distance')}")
    # incremental robustness plot: after each perturbed run, refresh this point's across-fold mean
    # and redraw <kind>_robustness.png from every model/strength scored so far. the curve fills in
    # as runs complete; failures here never fail the eval.
    if args.test_perturb != "none":
        print(f"[{stem}] perturb={args.test_perturb} strength={args.perturb_strength:g}", flush=True)
        try:
            fold_mean(score_tag)
            models = sorted({"baseline", "convctrl", "morphbank", "staticbank", args.tag})
            plot_perturb(args.test_perturb, models, "Dice")
        except Exception as e:
            print(f"[viz] perturb plot skipped: {e}", flush=True)
    # incremental data-efficiency plot: after each subsample run, refresh this N's across-fold mean
    # and redraw data_efficiency.png from every model/N trained so far (full-data anchor picked up
    # from the plain <model>_mean_scores.json). the curve fills in as runs complete.
    elif args.train_cases and args.train_cases > 0:
        try:
            fold_mean(score_tag)
            models = sorted({"baseline", "convctrl", "morphbank", "staticbank", args.tag})
            plot_data_efficiency(models, "Dice")
        except Exception as e:
            print(f"[viz] data-efficiency plot skipped: {e}", flush=True)

    # a perturbed static dir is a throwaway (test-split only, fully rebuildable) and each is ~GBs of
    # uncropped volumes, so delete it here rather than relying on a fragile shell glob. --keep-static
    # opts out. only fires for perturb_baked so a clean/training dir is never touched.
    if perturb_baked and args.static_dir and not args.keep_static:
        import shutil
        shutil.rmtree(args.static_dir, ignore_errors=True)
        print(f"[cleanup] removed perturbed static dir {args.static_dir}", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
