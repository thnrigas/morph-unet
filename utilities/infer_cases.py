#!/usr/bin/env python3
"""
Predictions of baseline and morph self (one seed / fold of the attention grid) on a few test cases,
saved for visualisation (slice overlays, MIP, 3D) on another machine.

The models are built and loaded by train_eval.main() itself (with --test-only), so the architecture
and the checkpoint (<tag>_f<fold>_best.pth) are exactly those of the run; the loader and the
test-set evaluation are swapped for a capture, so no score file is written or overwritten.
Inference repeats evaluate_test(): every slice along array axis 0 of the preprocessed volume,
padded (centred, image minimum) to a multiple of 16, fp32, argmax.

Case selection (unless --cases is given), from the per-case scores of the run, Vessel label:
  good          highest mean Dice of the two arms
  typical       mean Dice closest to the median
  bad           lowest mean Dice
  morph_better  largest HD95 reduction (baseline - morph)
  morph_worse   largest Dice loss (morph - baseline)
The JSON stores cases by position only, so each entry is matched to its case by the reference
voxel counts of both labels.

Output: <out>/<case>.npz (image uint8, gt, pred_baseline, pred_morph_self, spacing in the array's
axis order) and <out>/cases.json (selection and per-case Dice / HD95).

Usage, from the repo root in the project env:
  CUDA_VISIBLE_DEVICES=3 python3 utilities/infer_cases.py --suffix _s1_fv --fold 3
  CUDA_VISIBLE_DEVICES=""  python3 utilities/infer_cases.py ...        # CPU
"""
import argparse, glob, json, os, pickle, re, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config            # noqa: E402
import torch             # noqa: E402
import train_eval as te  # noqa: E402

ARMS = {"baseline":   ("grid_baseline",   ["--morph-unet", "none"]),
        "morph_self": ("grid_morph_self", ["--morph-unet", "none", "--morph-attn", "--morph-attn-src", "self"])}
LABEL_ID = {str(v): int(k) for k, v in config.LABELS.items() if int(k) > 0}   # {"Vessel": 1, "Tumour": 2}


def num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def load_model(arm, suffix, fold, seed):
    """Run train_eval.main() in --test-only mode and capture the loaded model."""
    tag, flags = ARMS[arm]
    got = {}
    te.build_loaders = lambda args: (None, None, None, 1)          # image channel only, as in the grid
    def capture(model, loader, device, json_path, **kw):
        got["model"], got["device"] = model, device
        return {"mean": {}}
    te.evaluate_test = capture
    sys.argv = ["train_eval.py", "--tag", tag + suffix, "--fold", str(fold), "--seed", str(seed),
                "--test-only"] + flags
    te.main()
    print(f"  (nothing written: evaluation replaced by a capture)", flush=True)
    model = got["model"]
    model.eval()
    return model, got["device"]


@torch.no_grad()
def predict(model, device, vol, batch=8, f=16):
    """vol: (D0, D1, D2) image in [0, 1]; 2D slices along axis 0, as the loader / evaluate_test."""
    out = np.zeros(vol.shape, np.uint8)
    h, w = vol.shape[1:]
    ph, pw = ((h + f - 1) // f) * f, ((w + f - 1) // f) * f
    t, l = (ph - h) // 2, (pw - w) // 2
    for s in range(0, vol.shape[0], batch):
        xs = []
        for j in range(s, min(s + batch, vol.shape[0])):
            sl = vol[j].astype(np.float32)
            xs.append(np.pad(sl, ((t, ph - h - t), (l, pw - w - l)), constant_values=sl.min())[None])
        x = torch.from_numpy(np.stack(xs)).to(device)
        p = model(x).argmax(1).cpu().numpy().astype(np.uint8)
        out[s:s + len(xs)] = p[:, t:t + h, l:l + w]
    return out


def dice(p, g):
    s = p.sum() + g.sum()
    return float(2 * (p & g).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_s1_fv", help="tag suffix of the seed ('' = seed 42, _s1_fv, ...)")
    ap.add_argument("--fold", type=int, default=3)
    ap.add_argument("--cases", default="", help="comma-separated case names; skips the automatic selection")
    ap.add_argument("--out", default="/gpu-data2/mis_mnn_team/vis_cases")
    a = ap.parse_args()
    m = re.search(r"_s(\d+)", a.suffix)
    seed = int(m.group(1)) if m else 42
    res = os.path.join(ROOT, "results", config.TASK)
    pre = str(config.PREPROCESSED_DIR)

    # ---- per-case scores of both arms, matched to case names by reference voxel counts ----
    with open(config.SPLITS_FILE, "rb") as fh:
        test_keys = sorted(pickle.load(fh)[a.fold]["test"])
    counts = {}
    for k in test_keys:
        lab = np.load(os.path.join(pre, k + ".npy"), mmap_mode="r")[1]
        counts[k] = tuple(int((lab == c).sum()) for c in LABEL_ID.values())
    by_count = {}
    for k, c in counts.items():
        by_count.setdefault(c, []).append(k)
    sc = {arm: json.load(open(os.path.join(res, f"{tag}{a.suffix}_f{a.fold}_scores.json")))["results"]["all"]
          for arm, (tag, _) in ARMS.items()}
    names = []
    for e in sc["baseline"]:
        c = tuple(int(num(e[n]["Total Positives Reference"])) for n in LABEL_ID)
        if len(by_count.get(c, [])) != 1:
            raise SystemExit(f"cannot match a score entry to a unique case (reference counts {c})")
        names.append(by_count[c][0])
    assert len(set(names)) == len(names) == len(sc["morph_self"])
    for eb, em in zip(sc["baseline"], sc["morph_self"]):           # same case order in both files
        assert all(num(eb[n]["Total Positives Reference"]) == num(em[n]["Total Positives Reference"]) for n in LABEL_ID)

    def arr(arm, lab, key): return np.array([num(e[lab][key]) for e in sc[arm]])
    dv_b, dv_m = arr("baseline", "Vessel", "Dice"), arr("morph_self", "Vessel", "Dice")
    hd_b, hd_m = arr("baseline", "Vessel", "Hausdorff Distance 95"), arr("morph_self", "Vessel", "Hausdorff Distance 95")
    avg = (dv_b + dv_m) / 2

    if a.cases:
        chosen = [(c.strip(), "given") for c in a.cases.split(",") if c.strip()]
    else:
        crit = [("good", -avg), ("typical", np.abs(avg - np.nanmedian(avg))), ("bad", avg),
                ("morph_better", -(hd_b - hd_m)), ("morph_worse", dv_m - dv_b)]
        chosen, used = [], set()
        for why, key in crit:                       # smallest key first, skip cases already chosen
            for i in np.argsort(np.where(np.isfinite(key), key, np.inf)):
                if names[i] not in used:
                    chosen.append((names[i], why)); used.add(names[i]); break
    print("\nselected cases (seed %d, fold %d):" % (seed, a.fold))
    info = []
    for case, why in chosen:
        i = names.index(case)
        d = {"case": case, "why": why}
        for arm in ARMS:
            for lab in LABEL_ID:
                d[f"{arm}_{lab}_dice"] = num(sc[arm][i][lab]["Dice"])
                d[f"{arm}_{lab}_hd95"] = num(sc[arm][i][lab]["Hausdorff Distance 95"])
        info.append(d)
        print(f"  {case:22s} {why:13s} Vessel Dice {d['baseline_Vessel_dice']:.3f} -> {d['morph_self_Vessel_dice']:.3f}"
              f"  HD95 {d['baseline_Vessel_hd95']:.1f} -> {d['morph_self_Vessel_hd95']:.1f}"
              f" | Tumour Dice {d['baseline_Tumour_dice']:.3f} -> {d['morph_self_Tumour_dice']:.3f}")

    # ---- volumes, spacing (from the raw NIfTI header, same axis order as the preprocessed array) ----
    from medpy.io import load as medpy_load
    vols = {}
    for d in info:
        case = d["case"]
        v = np.load(os.path.join(pre, case + ".npy"))
        raw = [p for p in glob.glob(os.path.join(str(config.IMAGES_DIR), case + "*.nii*"))
               if not os.path.basename(p).startswith("._")]
        _, hdr = medpy_load(raw[0])
        sp = tuple(float(s) for s in hdr.get_voxel_spacing())
        vols[case] = (v[0].astype(np.float32), v[1].astype(np.uint8), sp)
        d["array_shape"], d["spacing_mm"] = list(v.shape[1:]), list(sp)
    v0, _, sp0 = vols[info[0]["case"]]
    print(f"\norientation check ({info[0]['case']}): array {v0.shape}, spacing {sp0} mm."
          f"\n  2D slices = axes 1 x 2 = {v0.shape[1]} x {v0.shape[2]} px of {sp0[1]:.2f} x {sp0[2]:.2f} mm;"
          f" slicing axis 0 has spacing {sp0[0]:.2f} mm")

    # ---- inference with both arms ----
    preds = {d["case"]: {} for d in info}
    for arm in ARMS:
        print(f"\n== {arm} ==", flush=True)
        model, device = load_model(arm, a.suffix, a.fold, seed)
        for d in info:
            img, gt, _ = vols[d["case"]]
            p = predict(model, device, img)
            preds[d["case"]][arm] = p
            for lab, cid in LABEL_ID.items():
                d[f"{arm}_{lab}_dice_recomputed"] = dice(p == cid, gt == cid)
            print(f"  {d['case']:22s} Vessel Dice {d[f'{arm}_Vessel_dice_recomputed']:.3f}"
                  f" (score file {d[f'{arm}_Vessel_dice']:.3f})", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- save ----
    os.makedirs(a.out, exist_ok=True)
    for d in info:
        img, gt, sp = vols[d["case"]]
        np.savez_compressed(os.path.join(a.out, d["case"] + ".npz"),
                            image=np.round(np.clip(img, 0, 1) * 255).astype(np.uint8), gt=gt,
                            pred_baseline=preds[d["case"]]["baseline"], pred_morph_self=preds[d["case"]]["morph_self"],
                            spacing=np.array(sp), labels=json.dumps(LABEL_ID))
    meta = {"seed": seed, "fold": a.fold, "suffix": a.suffix, "slice_axis": 0, "cases": info}
    with open(os.path.join(a.out, "cases.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\nwrote {len(info)} cases to {a.out}")


if __name__ == "__main__":
    main()
