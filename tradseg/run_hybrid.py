"""Run a U-Net + classical hybrid on one fold and score it against the U-Net baseline.

Two modes (see hybrid.py):
  --mode uncertainty : Hybrid #1, band = high-MC-uncertainty ring (needs the dropout net).
  --mode morph       : Hybrid #2, band = morphological ring (works on the dropout-free net).

Design (informed by the fold-0 diagnosis + the tier-2 oracle results):
  * random walker uses a SEED-AND-SOLVE setup (erode/dilate markers), not a thin band.
  * GAC level set snaps to edges (balloon=0), no ballooning.
  * the refiner sees a TRACK-B re-windowed image (stronger spleen edges), while seeds/band stay
    in the Track-A prediction space (exact closed-form remap, pixel-aligned).
  * CASE-SELECTIVE safety net: a per-case boundary uncertainty score decides whether to refine.
    The threshold is tuned on the fold's VALIDATION cases (GT allowed there), then applied to
    test -> fair. We report three numbers per refiner:
      always     : refine every case (raw refiner)
      selective  : refine only cases with score > tau (the fair method)
      oracle     : refine only where it actually helps (uses test GT -> UPPER BOUND only)

  python3 run_hybrid.py --tag mcdropout --fold 0 --mode uncertainty --refiner both --dropout-p 0.4
  python3 run_hybrid.py --tag weak02 --fold 0 --mode morph --refiner both --dropout-p 0.0
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "unets"))          # predictor / arch / mc_dropout

import argparse
import json
import math
import time

import numpy as np

import config
import io_utils
import metrics as M
import hybrid as H
from predictor import UNetPredictor

_UNC_KEYS = {"mutual_info": "mutual_info", "entropy": "entropy", "fg_var": "fg_var"}
_DICE = "Dice"


def _refiner_list(arg):
    return ["rw", "ls"] if arg == "both" else [arg]


def _fit_threshold(triples):
    """triples: list of (score, base_dice, refined_dice). Return tau maximizing the total Dice of
    'refine iff score > tau'. Ties -> highest tau (refine fewer cases). +-inf allow never/always."""
    scores = sorted({t[0] for t in triples})
    best_t, best_sum = math.inf, -1.0
    for t in [-math.inf] + scores + [math.inf]:
        s = sum((tr[2] if tr[0] > t else tr[1]) for tr in triples)
        if s >= best_sum:                     # >= keeps the largest tau among equals
            best_sum, best_t = s, t
    return best_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="base net tag (e.g. mcdropout | weak02)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--mode", choices=["uncertainty", "morph"], required=True)
    p.add_argument("--refiner", choices=["rw", "ls", "both"], default="both")
    # band params
    p.add_argument("--uncertainty", choices=list(_UNC_KEYS), default="entropy")
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--band-quantile", type=float, default=0.80)
    p.add_argument("--ring-radius", type=int, default=12)
    p.add_argument("--band-radius", type=int, default=6, help="morph-band half width")
    # refiner params
    p.add_argument("--rw-beta", type=float, default=130.0)
    p.add_argument("--rw-margin", type=int, default=8, help="erode/dilate margin for RW seeds")
    p.add_argument("--ls-iters", type=int, default=10)
    p.add_argument("--ls-smoothing", type=int, default=2)
    p.add_argument("--ls-balloon", type=float, default=0.0)
    # Track-B image for the refiner
    p.add_argument("--no-track-b", action="store_true", help="feed the refiner the Track-A image")
    # model / io
    p.add_argument("--dropout-p", type=float, default=0.4)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--preprocessed-dir", default=None)
    p.add_argument("--splits", default=None)
    p.add_argument("--weights-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--advanced", action="store_true", help="also HD95 / ASSD")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.mode == "uncertainty" and args.dropout_p <= 0:
        raise SystemExit("--mode uncertainty needs a dropout net (--dropout-p > 0).")

    prep = args.preprocessed_dir or str(config.PREPROCESSED_DIR)
    splits = io_utils.load_splits(args.splits)
    weights_dir = args.weights_dir or str(config.RESULTS_DIR / "unets")
    out_dir = args.out_dir or str(config.RESULTS_DIR / "hybrid")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(weights_dir, f"{args.tag}_f{args.fold}_best.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt} (train the '{args.tag}' net first)")

    refiners = _refiner_list(args.refiner)
    use_track_b = not args.no_track_b
    predictor = UNetPredictor(ckpt, num_classes=args.num_classes, dropout_p=args.dropout_p)
    fg = config.FOREGROUND_LABEL

    def process(names):
        """-> {name: {'score': float, 'base': metrics, 'refined': {r: metrics}, 'times': {r: s}}}"""
        res = {}
        for name, image, label in io_utils.iter_cases(names, preprocessed_dir=prep):
            img_ref = H.track_b_image(image) if use_track_b else image      # refiner's image
            if args.mode == "uncertainty":
                m = predictor.predict_mc(image, T=args.mc_samples, temperature=args.temperature,
                                         batch=args.batch)
                seg = m["seg"].astype(bool)
                band = H.uncertainty_band(seg, m[_UNC_KEYS[args.uncertainty]],
                                          quantile=args.band_quantile, ring_radius=args.ring_radius)
                score = H.boundary_score(seg, m["fg_var"])                  # MC boundary fg-variance
            else:
                d = predictor.predict(image, batch=args.batch)
                seg = d["seg"].astype(bool)
                band = H.morph_band(seg, radius=args.band_radius)
                score = H.boundary_score(seg, H.binary_entropy_map(d["prob"][:, 1]))  # det. analogue
            entry = {"score": score,
                     "base": M.evaluate_case(seg, label, foreground_label=fg, spacing=None,
                                             advanced=args.advanced),
                     "refined": {}, "times": {}}
            for r in refiners:
                t0 = time.perf_counter()
                refined = H.refine_volume(img_ref, seg, band, refiner=r, beta=args.rw_beta,
                                          rw_margin=args.rw_margin, iters=args.ls_iters,
                                          smoothing=args.ls_smoothing, balloon=args.ls_balloon)
                entry["times"][r] = time.perf_counter() - t0
                entry["refined"][r] = M.evaluate_case(refined, label, foreground_label=fg,
                                                      spacing=None, advanced=args.advanced)
            res[name] = entry
            deltas = " ".join(f"{r}:{entry['refined'][r][_DICE]-entry['base'][_DICE]:+.4f}"
                              for r in refiners)
            print(f"  {name:16s} base={entry['base'][_DICE]:.4f} score={score:.4f} d[{deltas}]",
                  flush=True)
        return res

    # 1) validation pass -> per-refiner threshold tau
    val_names = io_utils.fold_cases(splits, args.fold, "val")
    if args.limit:
        val_names = val_names[:args.limit]
    print(f"[val] fitting selection threshold on {len(val_names)} cases ...")
    val_res = process(val_names)
    tau = {}
    for r in refiners:
        tau[r] = _fit_threshold([(v["score"], v["base"][_DICE], v["refined"][r][_DICE])
                                 for v in val_res.values()])
        print(f"  tau[{r}] = {tau[r]:.5f}")

    # 2) test pass
    test_names = io_utils.fold_cases(splits, args.fold, "test")
    if args.limit:
        test_names = test_names[:args.limit]
    print(f"[test] {len(test_names)} cases ...")
    test_res = process(test_names)

    # 3) per refiner: three variants (always / selective / oracle) -> JSON
    for r in refiners:
        per_case, always, selective, oracle, base = {}, {}, {}, {}, {}
        for name, e in test_res.items():
            b, rf, sc = e["base"], e["refined"][r], e["score"]
            sel_by_unc = sc > tau[r]
            sel_by_gt = rf[_DICE] > b[_DICE]           # oracle: would refining help?
            base[name] = b
            always[name] = rf
            selective[name] = rf if sel_by_unc else b
            oracle[name] = rf if sel_by_gt else b
            per_case[name] = {**selective[name],
                              "Dice_base": b[_DICE], "Dice_refined": rf[_DICE],
                              "Dice_delta_refine": rf[_DICE] - b[_DICE],
                              "score": sc, "selected": bool(sel_by_unc),
                              "sec": e["times"][r]}
        A = lambda d: M.aggregate(d)["mean"]
        base_m, always_m, sel_m, oracle_m = A(base), A(always), A(selective), A(oracle)
        meta = {"hybrid": args.mode, "refiner": r, "base_tag": args.tag, "fold": args.fold,
                "track": "A", "refiner_image": "B" if use_track_b else "A",
                "uncertainty": args.uncertainty if args.mode == "uncertainty" else "det_entropy",
                "score_signal": "boundary_fg_var" if args.mode == "uncertainty" else "boundary_entropy",
                "tau": tau[r], "n_selected": int(sum(v["selected"] for v in per_case.values())),
                "band": ({"quantile": args.band_quantile, "ring_radius": args.ring_radius}
                         if args.mode == "uncertainty" else {"radius": args.band_radius}),
                "rw_margin": args.rw_margin, "ls_balloon": args.ls_balloon,
                "n_cases": len(per_case)}
        stem = f"hybrid_{args.mode}_{r}_{args.tag}_f{args.fold}"
        payload = {"meta": meta, "mean": sel_m, "base_mean": base_m,
                   "always_mean": always_m, "oracle_mean": oracle_m, "per_case": per_case}
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        db, da, ds, do = (base_m[_DICE]["mean"], always_m[_DICE]["mean"],
                          sel_m[_DICE]["mean"], oracle_m[_DICE]["mean"])
        print(f"== {stem} ==  base {db:.4f} | always {da:+.4f}->{da:.4f} | "
              f"selective {ds:.4f} ({meta['n_selected']}/{len(per_case)} refined) | oracle {do:.4f}")
        print(f"   -> {os.path.join(out_dir, stem + '.json')}")


if __name__ == "__main__":
    main()
