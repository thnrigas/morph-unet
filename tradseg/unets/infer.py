"""MC-Dropout inference over a fold -> per-case U-Net prediction + uncertainty maps, saved
for the hybrid to consume. This is the bridge from the U-Net to the traditional refiner.

For each TEST case it saves <out-dir>/preds/<tag>_f<fold>_<case>.npz with:
  seg        (Z,H,W) int16   argmax of the mean MC prediction
  fg_prob    (Z,H,W) float32  P(spleen) = mean_prob[:,1]      (U-Net soft output)
  entropy    (Z,H,W) float32  predictive entropy  (total uncertainty)
  mutual_info(Z,H,W) float32  epistemic uncertainty
  fg_var     (Z,H,W) float32  fg-prob variance    (epistemic proxy)
and a per-case Dice summary JSON (mean MC prediction vs GT).

  python3 unets/infer.py --tag mcdropout --fold 0 --mc-samples 30 \
      --preprocessed-dir data/Task09_Spleen/preprocessed_A
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import argparse
import json

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
    p.add_argument("--weights-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--mc-samples", type=int, default=30, help="stochastic passes T")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    prep = args.preprocessed_dir or str(config.PREPROCESSED_DIR)
    splits = io_utils.load_splits(args.splits)
    weights_dir = args.weights_dir or str(config.RESULTS_DIR / "unets")
    out_dir = args.out_dir or str(config.RESULTS_DIR / "unets")
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(weights_dir, f"{args.tag}_f{args.fold}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} (train first)")

    predictor = UNetPredictor(ckpt, num_classes=args.num_classes, dropout_p=args.dropout_p)
    names = io_utils.fold_cases(splits, args.fold, "test")
    if args.limit:
        names = names[:args.limit]

    per_case = {}
    for name, image, label in io_utils.iter_cases(names, preprocessed_dir=prep):
        m = predictor.predict_mc(image, T=args.mc_samples, temperature=args.temperature, batch=args.batch)
        fg_prob = m["mean_prob"][:, 1].astype(np.float32) if m["mean_prob"].shape[1] > 1 \
            else (1.0 - m["mean_prob"][:, 0]).astype(np.float32)
        np.savez_compressed(os.path.join(preds_dir, f"{args.tag}_f{args.fold}_{name}.npz"),
                            seg=m["seg"], fg_prob=fg_prob, entropy=m["entropy"],
                            mutual_info=m["mutual_info"], fg_var=m["fg_var"])
        per_case[name] = M.evaluate_case(m["seg"], label, foreground_label=config.FOREGROUND_LABEL,
                                         spacing=None, advanced=False)
        print(f"  {name:16s} Dice={per_case[name]['Dice']:.4f}  "
              f"mean_entropy={float(m['entropy'].mean()):.4f}", flush=True)

    agg = M.aggregate(per_case)
    stem = f"unet_{args.tag}_f{args.fold}_mcinfer"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump({"meta": {"tag": args.tag, "fold": args.fold, "mc_samples": args.mc_samples,
                            "temperature": args.temperature, "n_cases": len(per_case)},
                   "mean": agg["mean"], "per_case": per_case}, f, indent=2)
    print(f"\n[{stem}] preds -> {preds_dir}/  |  summary -> {os.path.join(out_dir, stem + '.json')}")


if __name__ == "__main__":
    main()
