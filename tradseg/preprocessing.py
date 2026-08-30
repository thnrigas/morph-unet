"""Standalone dual-track preprocessing: raw MSD NIfTI -> (2, Z, S, S) npy + splits.pkl.

Track A ('fair'): reproduces the baseline U-Net data contract exactly -- fixed soft-tissue
    window (center 40, width 400), axial-first, label-free body crop, square-pad, resize
    256 (image bilinear, label nearest). Apples-to-apples input for the headline compare.

Track B ('trad'): traditional-optimised -- narrower soft-tissue window (center 50, width
    150 -> more fat/organ contrast), optional edge-preserving denoise, and an optional
    NO-resize variant that keeps native spacing (sidecar .json) for Tier-3 texture /
    mm-accurate surface metrics.

Nothing here is imported from my_unet-uncertainty. Case names match that pipeline's
(`<name>.npy` from `<name>.nii.gz`), so create_splits reproduces identical folds.

Usage (on the VM, DATA_DIR pointing at a writable copy of the raw task):
    DATA_DIR=$PWD/data/Task09_Spleen python -m tradseg.preprocessing --track A
    DATA_DIR=$PWD/data/Task09_Spleen python -m tradseg.preprocessing --track B
"""

import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import config


# --------------------------- array transforms --------------------------------
def normalize_ct(vol, center, width):
    lo, hi = center - width / 2.0, center + width / 2.0
    v = np.clip(vol.astype(np.float32), lo, hi)
    return (v - lo) / (hi - lo)


def axial_axis(spacing, shape):
    if spacing is not None and len(spacing) == len(shape):
        return int(np.argmax(spacing))
    return int(np.argmin(shape))


def body_bbox(vol01, thr=0.1, margin=8):
    from scipy.ndimage import label as cc_label
    inplane = (vol01 > thr).any(axis=0)
    if not inplane.any():
        return None
    lab, n = cc_label(inplane)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        inplane = lab == int(sizes.argmax())
    rows = np.where(inplane.any(axis=1))[0]
    cols = np.where(inplane.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    r0 = max(0, r0 - margin); c0 = max(0, c0 - margin)
    r1 = min(vol01.shape[1], r1 + margin); c1 = min(vol01.shape[2], c1 + margin)
    return r0, r1, c0, c1


def square_resize(vol, size, order):
    from scipy.ndimage import zoom
    _, h, w = vol.shape
    side = max(h, w)
    if h != side or w != side:
        vol = np.pad(vol, ((0, 0), (0, side - h), (0, side - w)), constant_values=0.0)
    if side != size:
        vol = zoom(vol, (1.0, size / side, size / side), order=order)
    if vol.shape[1] != size or vol.shape[2] != size:      # guard zoom round-off
        fixed = np.zeros((vol.shape[0], size, size), dtype=vol.dtype)
        h2, w2 = min(size, vol.shape[1]), min(size, vol.shape[2])
        fixed[:, :h2, :w2] = vol[:, :h2, :w2]
        vol = fixed
    return vol


def denoise(vol, method):
    if method in (None, "none"):
        return vol
    if method == "median":
        from scipy.ndimage import median_filter
        return median_filter(vol, size=(1, 3, 3))          # in-plane 3x3, per slice
    if method == "bilateral":
        from skimage.restoration import denoise_bilateral
        out = np.empty_like(vol)
        for z in range(vol.shape[0]):
            out[z] = denoise_bilateral(vol[z], sigma_color=0.05, sigma_spatial=2)
        return out
    raise ValueError(f"unknown denoise method: {method}")


# ------------------------------- config --------------------------------------
@dataclass
class PreprocConfig:
    ct_center: float = 40.0
    ct_width: float = 400.0
    size: Optional[int] = 256          # None -> keep native in-plane size (no resize)
    denoise: str = "none"
    body_thr: float = 0.1
    body_margin: int = 8
    keep_spacing: bool = False


TRACKS = {
    "A": PreprocConfig(ct_center=40, ct_width=400, size=256, denoise="none"),
    "B": PreprocConfig(ct_center=50, ct_width=150, size=256, denoise="median"),
    "Bnr": PreprocConfig(ct_center=50, ct_width=150, size=None, denoise="median",
                         keep_spacing=True),
}


# ------------------------------ per-case -------------------------------------
def _load_nifti(path):
    import nibabel as nib
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms())
    if data.ndim == 4:                                     # (H,W,D,1) or channel-last
        data = data[..., 0]
        zooms = zooms[:3]
    return data, zooms[:data.ndim]


def process_case(name, image_path, label_path, out_dir, cfg: PreprocConfig):
    image, spacing = _load_nifti(image_path)
    label, _ = _load_nifti(label_path)

    image = normalize_ct(image, cfg.ct_center, cfg.ct_width)
    ax = axial_axis(spacing, image.shape)
    image = np.moveaxis(image, ax, 0)
    label = np.moveaxis(label, ax, 0)
    new_spacing = [spacing[ax]] + [spacing[i] for i in range(len(spacing)) if i != ax]

    bbox = body_bbox(image, cfg.body_thr, cfg.body_margin)
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        image = image[:, r0:r1, c0:c1]
        label = label[:, r0:r1, c0:c1]

    if cfg.denoise != "none":
        image = denoise(image, cfg.denoise)

    if cfg.size is not None:
        image = square_resize(image, cfg.size, order=1)
        label = np.rint(square_resize(label, cfg.size, order=0))

    result = np.stack((image, label)).astype(np.float32)
    np.save(os.path.join(out_dir, name + ".npy"), result)

    if cfg.keep_spacing:
        with open(os.path.join(out_dir, name + ".spacing.json"), "w") as f:
            json.dump({"axial": new_spacing[0], "inplane": new_spacing[1:]}, f)
    return name


def create_splits(preprocessed_dir, out_file, seed=42, k=5):
    """Seeded k-fold CV, reproducing the baseline's algorithm exactly (same folds)."""
    names = sorted(p.name[:-4] for p in Path(preprocessed_dir).glob("*.npy"))
    if len(names) < k:
        raise ValueError(f"{k}-fold CV needs >= {k} samples, have {len(names)}")
    rng = random.Random(seed)
    rng.shuffle(names)
    chunks = [names[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        test = sorted(chunks[i])
        val = sorted(chunks[(i + 1) % k])
        used = set(test) | set(val)
        train = sorted(n for n in names if n not in used)
        splits.append({"train": train, "val": val, "test": test})
    all_test = sorted(s for sp in splits for s in sp["test"])
    assert all_test == sorted(names), "k-fold test coverage broken"
    with open(out_file, "wb") as f:
        pickle.dump(splits, f)
    return splits


# ------------------------------- driver --------------------------------------
def preprocess_all(track="A", data_dir=None, out_dir=None):
    cfg = TRACKS[track]
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"
    out_dir = Path(out_dir) if out_dir else data_dir / f"preprocessed_{track}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(fn for fn in os.listdir(image_dir)
                   if fn.endswith((".nii", ".nii.gz")) and not fn.startswith("._"))
    if not files:
        raise FileNotFoundError(f"no NIfTI images in {image_dir}")

    for fn in files:
        name = fn.split(".")[0]
        label_fn = fn.replace("_0000", "")          # no-op for MSD naming
        process_case(name, image_dir / fn, label_dir / label_fn, str(out_dir), cfg)
        print(f"  {name}", flush=True)

    splits_file = data_dir / "splits.pkl"           # names are track-independent
    create_splits(str(out_dir), str(splits_file))
    print(f"track {track}: {len(files)} cases -> {out_dir}\nsplits -> {splits_file}")


def main():
    p = argparse.ArgumentParser(description="Dual-track MSD preprocessing")
    p.add_argument("--track", choices=sorted(TRACKS), default="A")
    p.add_argument("--data-dir", default=None, help="defaults to config.DATA_DIR")
    p.add_argument("--out-dir", default=None, help="defaults to <data-dir>/preprocessed_<track>")
    args = p.parse_args()
    preprocess_all(track=args.track, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
