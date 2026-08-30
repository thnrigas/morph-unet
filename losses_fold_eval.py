#
# Fold aggregation + comparison for the loss-function experiment
#
# Renamed from part2_losses/train_eval.py so it does not clash with the morphological
# train_eval.py at the repo root (which has its own --fold-mean/--compare reading
# results/<TASK>/). This one reads the flat results/ that the loss track writes.
#
# Slimmed down to ONLY the two post-processing entry points that run_losses.sh
# uses after train_losses.py / test_losses.py have produced the per-fold scores:
#
#   python3 losses_fold_eval.py --fold-mean <tag>            # mean +/- std over folds
#   python3 losses_fold_eval.py --compare a.json b.json ...  # per-label Dice/ASSD deltas
#
# The training itself lives in train_losses.py (via pipeline_loss.py); nothing here
# imports a network or a loss, so the morphological baseline modules are gone.
#

import argparse
import glob
import json
import os

import numpy as np

import config


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
    p.add_argument("--fold-mean", metavar="TAG")
    p.add_argument("--compare", nargs="+", metavar="JSON")
    args = p.parse_args()

    if args.fold_mean:
        fold_mean(args.fold_mean)
        return
    if args.compare:
        compare_runs(args.compare)
        return
    p.error("nothing to do: pass --fold-mean TAG or --compare JSON [JSON ...]")


if __name__ == "__main__":
    main()
