"""Generic per-method runner: run ONE traditional method over one CV fold and score it
per-volume, in the SAME space/conventions as the baseline U-Net (directly comparable).
Used by the per-tier drivers run_tier1.sh / run_tier2.sh / run_tier3.sh.

Examples:
  python3 run_experiment.py --method multiotsu --fold 0 --regime auto  --preprocessed-dir data/Task09_Spleen/preprocessed_A
  python3 run_experiment.py --method graphcut  --fold 0 --regime oracle --advanced
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import config
import io_utils
import metrics as M
import postprocess as P
import seeding
from methods import REGISTRY


def peak_rss_mb():
    """Best-effort peak resident memory (MB). Accurate on Linux/Mac; NaN on Windows
    without psutil -- report the paper numbers from the (Linux) VM."""
    try:
        import resource
        import sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / (1024.0 ** 2)
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024.0 ** 2)
        except Exception:
            return float("nan")


def build_method(name, args):
    cls = REGISTRY[name]
    if name == "otsu":
        return cls(per_slice=args.per_slice)
    if name == "multiotsu":
        return cls(classes=args.k, per_slice=args.per_slice)
    if name in ("kmeans", "gmm"):
        return cls(k=args.k)
    if name == "region_growing":
        return cls(tolerance=args.tolerance)
    if name == "watershed":
        return cls(gradient_sigma=args.gradient_sigma)
    if name == "chanvese":
        return cls(iterations=(args.iterations or None), smoothing=args.smoothing)
    if name == "morphgac":
        return cls(iterations=(args.iterations or None), smoothing=args.smoothing,
                   balloon=args.balloon, alpha=args.gac_alpha, sigma=args.gac_sigma)
    if name == "graphcut":
        return cls(lam=args.gc_lambda, sigma=args.gc_sigma, gmm_k=args.k)
    if name == "random_walker":
        return cls(beta=args.rw_beta)
    if name in ("gabor", "amfm", "granulometry"):
        return cls(k=args.k, andiff=not args.no_andiff)
    return cls()


def make_seeds(method, image, label, body, regime, prior, args):
    if not method.requires_seeds:
        return None
    if regime == "auto":
        if method.seed_type == "markers":
            fg, bg = prior.auto_markers(image, body)
            return {"fg": fg, "bg": bg}
        return {"points": prior.auto_seed_points(image, body, k=args.seed_k)}
    if method.seed_type == "markers":                       # oracle
        fg, bg = seeding.oracle_markers(label, body=body)
        return {"fg": fg, "bg": bg}
    return {"points": seeding.oracle_seed_points(label, k=args.seed_k)}


def localise(method, raw, image, label, regime, prior, args):
    """Turn a raw candidate mask into the final spleen mask (component pick + cleanup)."""
    if method.requires_seeds:                               # seeds already localise it
        return P.clean(raw, min_size=args.min_size, largest_cc=True, fill=True)
    raw = P.remove_small_objects(raw, args.min_size)
    if regime == "auto":
        sel = prior.select(raw, image)
    else:
        sel = P.select_component_by_overlap(raw, label == config.FOREGROUND_LABEL)
    return P.clean(sel, min_size=0, largest_cc=False, fill=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=sorted(REGISTRY))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--regime", choices=["auto", "oracle"], default="auto")
    p.add_argument("--track", default="A", help="bookkeeping label written into the output")
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--advanced", action="store_true", help="also HD95 / ASSD (needs medpy)")
    p.add_argument("--limit", type=int, default=0, help="only first N test cases (debug)")
    # method knobs
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--per-slice", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.06)
    p.add_argument("--gradient-sigma", type=float, default=1.0)
    p.add_argument("--seed-k", type=int, default=1)
    p.add_argument("--min-size", type=int, default=50)
    # Tier-2 knobs
    p.add_argument("--iterations", type=int, default=0, help="level-set iterations (0 = method default)")
    p.add_argument("--smoothing", type=int, default=1, help="level-set smoothing")
    p.add_argument("--balloon", type=float, default=1.0, help="morphgac balloon force")
    p.add_argument("--gac-alpha", type=float, default=100.0, help="morphgac inverse-gradient alpha")
    p.add_argument("--gac-sigma", type=float, default=2.0, help="morphgac inverse-gradient sigma")
    p.add_argument("--gc-lambda", type=float, default=2.0, help="graphcut Potts weight")
    p.add_argument("--gc-sigma", type=float, default=0.08, help="graphcut contrast sigma")
    p.add_argument("--rw-beta", type=float, default=130.0, help="random walker beta")
    # Tier-3 knobs
    p.add_argument("--no-andiff", action="store_true",
                   help="disable Perona-Malik denoise in the texture methods")
    args = p.parse_args()

    prep = Path(args.preprocessed_dir) if args.preprocessed_dir else config.PREPROCESSED_DIR
    splits = io_utils.load_splits(args.splits)
    test_names = io_utils.fold_cases(splits, args.fold, "test")
    train_names = io_utils.fold_cases(splits, args.fold, "train")
    if args.limit:
        test_names = test_names[:args.limit]

    method = build_method(args.method, args)
    prior = None
    if args.regime == "auto":
        prior = seeding.SpatialPrior.from_training(train_names, preprocessed_dir=prep)

    per_case, times = {}, {}
    for name, image, label in io_utils.iter_cases(test_names, preprocessed_dir=prep):
        body = image > 0
        seeds = make_seeds(method, image, label, body, args.regime, prior, args)
        t0 = time.perf_counter()
        raw = method.segment_volume(image, seeds=seeds)
        mask = localise(method, raw, image, label, args.regime, prior, args)
        times[name] = time.perf_counter() - t0
        per_case[name] = M.evaluate_case(mask, label, foreground_label=config.FOREGROUND_LABEL,
                                         spacing=None, advanced=args.advanced)
        print(f"  {name:16s} Dice={per_case[name]['Dice']:.4f}  t={times[name]:.2f}s", flush=True)

    agg = M.aggregate(per_case)
    tarr = np.array(list(times.values()), float)
    meta = {
        "method": args.method, "tier": getattr(method, "tier", 1),
        "fold": args.fold, "regime": args.regime, "track": args.track,
        "n_cases": len(per_case),
        "sec_per_case_mean": float(tarr.mean()) if tarr.size else float("nan"),
        "sec_per_case_std": float(tarr.std()) if tarr.size else float("nan"),
        "peak_rss_mb": peak_rss_mb(), "platform": platform.platform(),
    }
    out_dir = Path(args.out_dir) if args.out_dir else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.regime}_track{args.track}_f{args.fold}"
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"meta": meta, "mean": agg["mean"], "per_case": per_case, "times": times},
                  f, indent=2)

    print(f"\n== {stem} ==")
    for m, md in agg["mean"].items():
        print(f"  {m:10s} mean={md['mean']:.4f}  std={md['std']:.4f}  n={md['n']}")
    print(f"  time/case={meta['sec_per_case_mean']:.2f}s  peakRSS={meta['peak_rss_mb']:.0f}MB")
    print(f"  -> {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
