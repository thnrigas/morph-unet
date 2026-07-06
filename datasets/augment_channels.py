#!/usr/bin/env python3
#
# Augment preprocessed (image, label) npys with STATIC filter channels for training.
#
# Reads config.PREPROCESSED_DIR/*.npy (channel 0 = image, last channel = label), computes the
# requested static filters slice-wise along axis 0 (the plane the 2D loader trains on), and writes
# [image, filt_1, ..., filt_N, label] to --out. No raw re-preprocessing needed -- it reuses the
# already-preprocessed image. Filters are computed on norm01(image) so h/area thresholds match
# what the survey ranked on; channel 0 stays the original image (identical to the baseline input).
#
# Static-capable filters (all non-differentiable connected/oriented ops + the disk ops):
#   tophat:R  bottomhat:R  gradient:R  recontophat:R  asftophat:R  leveltophat:R
#   hdome:H   vdome:H:AREA  areaopen:AREA  areaclose:AREA
#
# Examples:
#   python datasets/augment_channels.py --filters recontophat:3 hdome:0.1 vdome:0.1:100
#   TASK=Task08_HepaticVessel python datasets/augment_channels.py --filters vdome:0.1:100 --workers 8
#
# Then train on the result by pointing the loader at --out with input_slice=(0..N), label_slice=N+1.
#

import argparse
import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np

# run from anywhere: put the project root (parent of this folder) on the import path so the
# shared project modules (config, utilities.morph_explore) resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utilities.morph_explore import (norm01, tophat, bottomhat, gradient, recon_tophat, hdome,
                                      volume_dome, asf_tophat, leveling_tophat, area_open, area_close)


def build_filter(spec):
    """'recontophat:3' -> fn(img2d). radius int, h float, area int."""
    p = spec.split(":")
    m = p[0]
    table = {
        "tophat":      lambda im: tophat(im, int(p[1])),
        "bottomhat":   lambda im: bottomhat(im, int(p[1])),
        "gradient":    lambda im: gradient(im, int(p[1])),
        "recontophat": lambda im: recon_tophat(im, int(p[1])),
        "asftophat":   lambda im: asf_tophat(im, int(p[1])),
        "leveltophat": lambda im: leveling_tophat(im, int(p[1])),
        "hdome":       lambda im: hdome(im, float(p[1])),
        "vdome":       lambda im: volume_dome(im, float(p[1]), int(p[2])),
        "areaopen":    lambda im: area_open(im, int(p[1])),
        "areaclose":   lambda im: area_close(im, int(p[1])),
    }
    if m not in table:
        raise SystemExit(f"unknown filter '{m}' in spec '{spec}'. known: {sorted(table)}")
    return table[m]


def _augment_case(fname, src, out, specs):
    arr = np.load(os.path.join(src, fname))          # (C>=2, Z, H, W): image ... label
    if arr.shape[0] < 2:
        raise SystemExit(f"{fname}: expected >=2 channels (image,label), got {arr.shape}")
    image = arr[0].astype(np.float32)
    label = arr[-1].astype(np.float32)
    inorm = norm01(image)                            # filter input matches the survey's calibration
    filters = [build_filter(s) for s in specs]
    chans = [image]                                  # ch0 stays the original image (baseline input)
    for f in filters:
        vol = np.empty_like(image)
        for z in range(image.shape[0]):
            vol[z] = f(inorm[z])
        chans.append(vol)
    chans.append(label)
    stacked = np.stack(chans).astype(np.float16)     # (1 + N + 1, Z, H, W)
    np.save(os.path.join(out, fname), stacked)
    return fname, stacked.shape


def main():
    ap = argparse.ArgumentParser(description="append static filter channels to preprocessed npys")
    ap.add_argument("--filters", nargs="+", required=True,
                    help="filter specs, e.g. recontophat:3 hdome:0.1 vdome:0.1:100 areaopen:150")
    ap.add_argument("--src", default=str(config.PREPROCESSED_DIR),
                    help="dir of (image,label) npys (default: config.PREPROCESSED_DIR)")
    ap.add_argument("--out", default=None, help="output dir (default: <src>_static)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    args = ap.parse_args()

    src = args.src
    out = args.out or (src.rstrip("/") + "_static")
    os.makedirs(out, exist_ok=True)
    for s in args.filters:                            # validate specs up front (fail fast)
        build_filter(s)
    cases = sorted(f for f in os.listdir(src) if f.endswith(".npy"))
    if not cases:
        raise SystemExit(f"no .npy files in {src}")

    n_ch = 1 + len(args.filters) + 1
    print(f"augmenting {len(cases)} cases from {src}")
    print(f"  filters: {args.filters}")
    print(f"  -> {out}   ({n_ch} channels: image + {len(args.filters)} filters + label)")
    print(f"  train with input_slice=(0..{len(args.filters)}), label_slice={n_ch - 1}")

    worker = partial(_augment_case, src=src, out=out, specs=args.filters)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for fname, shape in pool.imap_unordered(worker, cases):
                print(f"  {fname}  {shape}", flush=True)
    else:
        for fname in cases:
            _, shape = worker(fname)
            print(f"  {fname}  {shape}", flush=True)
    # manifest of the exact specs this dir was built from, written last (after every case
    # succeeded) so a crashed/partial run leaves no manifest and the dir is treated as stale.
    # consumers reuse the dir only if this list matches their specs exactly (not just the count).
    with open(os.path.join(out, "filters.json"), "w") as f:
        json.dump(args.filters, f)
    print("done.")


if __name__ == "__main__":
    main()
