#
# Collate the prune+finetune sweep into a Dice-vs-sparsity table.
# --------------------------------------------------------------
# Scans results/*_prune.json (written by prune.py) and prints, per model+method, a table of
# keep-ratio -> params-off% / unpruned / pruned / finetuned fg-Dice. Also writes prune_summary.csv.
#
# Usage:
#   python collate_prune.py                       # all *_prune.json in results/
#   python collate_prune.py --tag mpm_full_l2     # filter by base tag substring
#   python collate_prune.py --csv out.csv
#

import argparse
import csv
import glob
import json
import os
import re

import config

FIELDS = ["tag", "method", "alloc", "keep_ratio", "params_before", "params_after", "params_off_pct",
          "val_dice_unpruned", "val_dice_pruned", "val_dice_finetuned", "ft_skipped", "best_epoch",
          "finetune_epochs"]

# out_stem looks like:  <base>_prune-<method>-k<NN>_f<fold>
STEM_RE = re.compile(r"^(?P<base>.+)_prune-(?P<method>[a-z0-9]+)-k(?P<kk>\d+)_f(?P<fold>\d+)$")


def load_rows(results_dir, tag_filter):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_prune.json"))):
        with open(path) as f:
            d = json.load(f)
        stem = d.get("tag", os.path.basename(path)[:-len("_prune.json")])
        m = STEM_RE.match(stem)
        base = m.group("base") if m else stem
        fold = int(m.group("fold")) if m else -1
        if tag_filter and tag_filter not in base:
            continue
        d["base"], d["fold"] = base, fold
        rows.append(d)
    return rows


def fmt(v, nd=4):
    return "  -  " if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="", help="only rows whose base tag contains this substring")
    p.add_argument("--results-dir", default=os.path.join(config.PROJECT_ROOT, "results"))
    p.add_argument("--csv", default=os.path.join(config.PROJECT_ROOT, "results", "prune_summary.csv"))
    args = p.parse_args()

    rows = load_rows(args.results_dir, args.tag)
    if not rows:
        raise SystemExit(f"no *_prune.json found in {args.results_dir}"
                         + (f" matching '{args.tag}'" if args.tag else ""))

    # group by (base model+fold, method); sort within by keep_ratio
    groups = {}
    for r in rows:
        groups.setdefault((r["base"], r["fold"], r["method"]), []).append(r)

    for (base, fold, method) in sorted(groups):
        g = sorted(groups[(base, fold, method)], key=lambda r: r["keep_ratio"])
        print(f"\n=== {base}  fold {fold}  |  method={method} "
              f"===============================================")
        print(f"  {'keep':>5} {'params(M)':>10} {'off%':>6} "
              f"{'unpruned':>9} {'pruned':>8} {'finetuned':>10} {'final':>8} "
              f"{'recover':>8} {'best@ep':>9}")
        for r in g:
            pa = r.get("params_after")
            base_d = r.get("val_dice_unpruned")
            pr = r.get("val_dice_pruned")
            ft = r.get("val_dice_finetuned")
            # "final" = the model we actually keep: fine-tuned if it ran, else the pruned net as-is.
            final = ft if ft is not None else pr
            recover = (final - base_d) if (final is not None and base_d is not None) else None
            # finetuned column: value if it ran, "skip" if we deliberately skipped it, else "-"
            ft_cell = fmt(ft) if ft is not None else ("skip" if r.get("ft_skipped") else "  -  ")
            be, fe = r.get("best_epoch"), r.get("finetune_epochs")
            best_at = f"{be}/{fe}" if be is not None else ("skip" if r.get("ft_skipped") else "  -  ")
            print(f"  {r['keep_ratio']:>5.2f} "
                  f"{(pa/1e6 if pa else 0):>10.3f} "
                  f"{r.get('params_off_pct', 0):>6.1f} "
                  f"{fmt(base_d):>9} {fmt(pr):>8} "
                  f"{ft_cell:>10} {fmt(final):>8} "
                  f"{(fmt(recover) if recover is not None else '   -  '):>8} "
                  f"{best_at:>9}")

    # flat CSV for plotting
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base", "fold"] + FIELDS)
        for r in sorted(rows, key=lambda r: (r["base"], r["fold"], r["method"], r["keep_ratio"])):
            w.writerow([r["base"], r["fold"]] + [r.get(k) for k in FIELDS])
    print(f"\nwrote {args.csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
