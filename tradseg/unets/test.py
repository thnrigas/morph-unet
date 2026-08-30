"""Deterministic test of a trained U-Net on one fold -> per-volume Dice/HD95/ASSD using the
tradseg metrics (SAME conventions as the traditional methods, so numbers are comparable).
Raw argmax, no post-processing (matches the baseline's own eval). Reads the Track-A npy.

  python3 unets/test.py --tag mcdropout --fold 0 \
      --preprocessed-dir data/Task09_Spleen/preprocessed_A --advanced
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import argparse
import json
import time

import numpy as np

import config
import io_utils
import metrics as M
from predictor import UNetPredictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="mcdropout")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--weights-dir", default=None, help="dir with <tag>_f<fold>_best.pth (default RESULTS/unets)")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--advanced", action="store_true", help="also HD95 / ASSD")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    prep = args.preprocessed_dir or str(config.PREPROCESSED_DIR)
    splits = io_utils.load_splits(args.splits)
    weights_dir = args.weights_dir or str(config.RESULTS_DIR / "unets")
    out_dir = args.out_dir or str(config.RESULTS_DIR / "unets")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(weights_dir, f"{args.tag}_f{args.fold}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} (train first)")

    predictor = UNetPredictor(ckpt, num_classes=args.num_classes, dropout_p=args.dropout_p)
    names = io_utils.fold_cases(splits, args.fold, "test")
    if args.limit:
        names = names[:args.limit]

    per_case, times = {}, {}
    for name, image, label in io_utils.iter_cases(names, preprocessed_dir=prep):
        t0 = time.perf_counter()
        seg = predictor.predict(image, batch=args.batch)["seg"]
        times[name] = time.perf_counter() - t0
        per_case[name] = M.evaluate_case(seg, label, foreground_label=config.FOREGROUND_LABEL,
                                         spacing=None, advanced=args.advanced)
        print(f"  {name:16s} Dice={per_case[name]['Dice']:.4f}  t={times[name]:.2f}s", flush=True)

    agg = M.aggregate(per_case)
    tarr = np.array(list(times.values()), float)
    meta = {"model": f"unet_{args.tag}", "fold": args.fold, "track": "A", "device": str(predictor.device),
            "n_cases": len(per_case),
            "sec_per_case_mean": float(tarr.mean()) if tarr.size else float("nan")}
    stem = f"unet_{args.tag}_f{args.fold}"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump({"meta": meta, "mean": agg["mean"], "per_case": per_case, "times": times}, f, indent=2)
    print(f"\n== {stem} ==")
    for m, md in agg["mean"].items():
        print(f"  {m:10s} mean={md['mean']:.4f}  std={md['std']:.4f}  n={md['n']}")
    print(f"  -> {os.path.join(out_dir, stem + '.json')}")


if __name__ == "__main__":
    main()
