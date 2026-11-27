#!/usr/bin/env python3
"""
Check the axial preprocessed data against the original folder. Run once on the cluster, after
`LAYOUT=axial python3 run_preprocessing.py`, from the repo root:

  python3 utilities/check_axial.py              # every case, slice counts, figure
  python3 utilities/check_axial.py --cases 10   # quick look: the first 10 cases only

1. Same data: the axial array of each case (2, Z, Y, X) must equal the original one (2, X, Y, Z')
   transposed, once the original's centred padding (z up to Z' = max(Z, 64)) is cut away. Image and
   label are compared, so the voxel count of every label is identical too.
2. No padding: the in-plane size of every axial slice (Task08: always 512 x 512).
3. What the 2D loader will see: slices per split of one fold, and how many contain each label (the
   class-balanced sampler draws its foreground slices from these).
4. A figure with one axial slice of four test cases (image + label contours), to check the
   orientation by eye: <data dir>/check_axial.png.
"""
import argparse, os, pickle, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402

COLOURS = ("#1baf7a", "#eda100")      # label contours in the figure: Vessel, Tumour


def compare(new, old):
    """Axial array vs the original one transposed, with its centred padding removed: (labels equal,
    images equal, max |image difference|); (False, False, nan) if the shapes do not fit."""
    _, Z, Y, X = new.shape
    if old.shape[1:] != (X, max(Y, 64), max(Z, 64)):
        return False, False, float("nan")
    cy, cz = (old.shape[2] - Y) // 2, (old.shape[3] - Z) // 2        # pad_nd_image pads centred
    ref = np.asarray(old[:, :, cy:cy + Y, cz:cz + Z]).transpose(0, 3, 2, 1)
    cur = np.asarray(new)
    diff = float(np.abs(cur[0].astype(np.float32) - ref[0].astype(np.float32)).max())
    return bool(np.array_equal(cur[1], ref[1])), diff == 0.0, diff


def figure(new_dir, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.4))
    for ax, n in zip(np.atleast_1d(axes), names):
        arr = np.load(os.path.join(new_dir, n + ".npy"), mmap_mode="r")
        z = int(np.argmax((np.asarray(arr[1]) > 0).sum(axis=(1, 2))))   # slice with the most labelled voxels
        img, lab = np.asarray(arr[0, z], np.float32), np.asarray(arr[1, z])
        ax.imshow(img, cmap="gray", interpolation="nearest")
        for (c, _), col in zip(sorted(config.LABELS.items()), COLOURS):
            if (lab == c).any():
                ax.contour(lab == c, levels=[0.5], colors=[col], linewidths=0.7)
        ax.set_title(f"{n}\naxial slice {z} of {arr.shape[1]}, {img.shape[0]} x {img.shape[1]}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([], [], color=col, label=name)
               for (_, name), col in zip(sorted(config.LABELS.items()), COLOURS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=0, help="check only the first N cases (0 = all)")
    ap.add_argument("--fold", type=int, default=0, help="fold whose splits are counted (default 0)")
    ap.add_argument("--fig", default="", help="figure path (default: <data dir>/check_axial.png)")
    a = ap.parse_args()

    old_dir = str(config.DATA_DIR / "preprocessed")
    new_dir = str(config.DATA_DIR / "preprocessed_axial")
    names = sorted(f[:-4] for f in os.listdir(new_dir) if f.endswith(".npy"))
    if a.cases:
        names = names[:a.cases]
    labels = dict(sorted(config.LABELS.items()))             # {1: "Vessel", 2: "Tumour"}
    print(f"original: {old_dir}\naxial:    {new_dir}\nchecking {len(names)} cases", flush=True)

    bad, img_diff, sizes, info = [], [], {}, {}
    for i, n in enumerate(names, 1):
        new = np.load(os.path.join(new_dir, n + ".npy"), mmap_mode="r")      # (2, Z, Y, X)
        old = np.load(os.path.join(old_dir, n + ".npy"), mmap_mode="r")      # (2, X, Y, Z')
        _, Z, Y, X = new.shape
        sizes[(Y, X)] = sizes.get((Y, X), 0) + 1
        lab_ok, img_ok, diff = compare(new, old)
        if not lab_ok:
            bad.append(n)
            print(f"  !! {n}: labels differ (axial {tuple(new.shape)}, original {tuple(old.shape)})", flush=True)
        elif not img_ok:
            img_diff.append(diff)
        lab = np.asarray(new[1])
        info[n] = (Z, {c: int((lab == c).any(axis=(1, 2)).sum()) for c in labels})
        if i % 25 == 0 or i == len(names):
            print(f"  {i}/{len(names)} checked", flush=True)

    ok = len(names) - len(bad)
    print(f"\nlabels identical to the original (transposed, padding removed): {ok}/{len(names)}"
          + (f"; DIFFERENT: {', '.join(bad)}" if bad else ""))
    print(f"images identical: {ok - len(img_diff)}/{ok}"
          + (f"; {len(img_diff)} differ, max |diff| {max(img_diff):.4f} (CT window changed since the "
             f"original folder was made?)" if img_diff else ""))
    print("in-plane size of the axial slices (cases): "
          + ", ".join(f"{y} x {x}: {c}" for (y, x), c in sorted(sizes.items())))
    zs = np.array([z for z, _ in info.values()])
    print(f"slices per volume: min {zs.min()}, median {int(np.median(zs))}, max {zs.max()}, total {zs.sum()}")

    with open(config.SPLITS_FILE, "rb") as f:
        split = pickle.load(f)[a.fold]
    print(f"\nfold {a.fold}: axial slices seen by the 2D loader"
          + (" (only the checked cases)" if a.cases else ""))
    for part in ("train", "val", "test"):
        ks = [k for k in split[part] if k in info]
        tot = sum(info[k][0] for k in ks)
        per = ", ".join(f"with {name} {sum(info[k][1][c] for k in ks)} ({100 * sum(info[k][1][c] for k in ks) / max(tot, 1):.0f}%)"
                        for c, name in labels.items())
        print(f"  {part:5s}: {len(ks):3d} cases, {tot:5d} slices; {per}")

    show = [k for k in split["test"] if k in info][:4] or names[:4]
    out = a.fig or str(config.DATA_DIR / "check_axial.png")
    figure(new_dir, show, out)
    print(f"\nfigure: {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
