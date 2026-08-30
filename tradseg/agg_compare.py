"""Collect every U-Net / hybrid result JSON across folds into ONE comparison table.

Pools per-case scores across folds for each experiment (test folds are disjoint, so case
names are unique) and re-aggregates with the shared metrics conventions -- the same numbers
as a proper 5-fold cross-validation. For hybrids it also reports the paired Dice delta vs
the same net WITHOUT refinement (carried in each hybrid JSON as `Dice_base`).

  python3 agg_compare.py                       # scans results/unets + results/hybrid
  python3 agg_compare.py --unet-dir ... --hybrid-dir ... --out results/comparison
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import config
import metrics as M

_FOLD_RE = re.compile(r"_f(\d+)\.json$")


def _experiment_key(path):
    """Strip the _f<fold> suffix -> the experiment identity shared across folds."""
    base = os.path.basename(path)
    return _FOLD_RE.sub("", base)


def _collect(paths):
    """path glob -> {experiment: {case: scores}} plus {experiment: base per_case} for hybrids."""
    exps, bases = {}, {}
    for path in sorted(paths):
        if not _FOLD_RE.search(path):
            continue
        with open(path) as f:
            data = json.load(f)
        key = _experiment_key(path)
        pc = data.get("per_case", {})
        exps.setdefault(key, {})
        bases.setdefault(key, {})
        _extra = ("Dice_base", "Dice_refined", "Dice_delta", "Dice_delta_refine",
                  "band_frac", "score", "selected", "sec")
        for case, sc in pc.items():
            # per_case["Dice"] holds the METHOD's score (for hybrids: the selective variant);
            # strip the bookkeeping keys, keep the metric dict.
            exps[key][case] = {k: v for k, v in sc.items() if k not in _extra}
            if "Dice_base" in sc:
                bases[key][case] = {"Dice": sc["Dice_base"]}
    return exps, bases


def _fmt(md):
    return f"{md['mean']:.4f}+-{md['std']:.4f} (n={md['n']})" if md["n"] else "  --  "


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet-dir", default=str(config.RESULTS_DIR / "unets"))
    p.add_argument("--hybrid-dir", default=str(config.RESULTS_DIR / "hybrid"))
    p.add_argument("--out", default=str(config.RESULTS_DIR / "comparison"))
    args = p.parse_args()

    unet_paths = glob.glob(os.path.join(args.unet_dir, "unet_*.json"))
    hyb_paths = glob.glob(os.path.join(args.hybrid_dir, "hybrid_*.json"))
    unet_exps, _ = _collect(unet_paths)
    hyb_exps, hyb_bases = _collect(hyb_paths)

    rows = {}
    for key, cases in {**unet_exps, **hyb_exps}.items():
        agg = M.aggregate(cases)["mean"]
        row = {"experiment": key, "n_folds_cases": len(cases), "metrics": agg}
        if key in hyb_bases and hyb_bases[key]:
            base_agg = M.aggregate(hyb_bases[key])["mean"]
            row["base_Dice"] = base_agg["Dice"]
            row["Dice_delta_vs_base"] = (agg["Dice"]["mean"] - base_agg["Dice"]["mean"]
                                         if agg["Dice"]["n"] and base_agg["Dice"]["n"] else float("nan"))
        rows[key] = row

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # CSV + console table
    metric_cols = ["Dice", "Jaccard", "HD95", "ASSD"]
    lines = ["experiment,n_cases," + ",".join(f"{m}_mean,{m}_std" for m in metric_cols)
             + ",Dice_delta_vs_base"]
    header = f"{'experiment':42s} {'cases':>6s}  " + "  ".join(f"{m:>16s}" for m in metric_cols) + "   dDice"
    print(header)
    print("-" * len(header))
    for key in sorted(rows):
        r = rows[key]
        md = r["metrics"]
        cells = "  ".join(_fmt(md.get(m, {"mean": float('nan'), "std": float('nan'), "n": 0}))
                          for m in metric_cols)
        delta = r.get("Dice_delta_vs_base")
        dtxt = f"{delta:+.4f}" if delta is not None and np.isfinite(delta) else "   -  "
        print(f"{key:42s} {r['n_folds_cases']:6d}  {cells}   {dtxt}")
        csv_cells = ",".join(f"{md.get(m, {}).get('mean', float('nan'))},"
                             f"{md.get(m, {}).get('std', float('nan'))}" for m in metric_cols)
        lines.append(f"{key},{r['n_folds_cases']},{csv_cells},"
                     f"{delta if delta is not None else float('nan')}")
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {os.path.join(args.out, 'summary.csv')}  |  {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
