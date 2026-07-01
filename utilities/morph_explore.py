#
# Morphology Explorer
#
# extract images, apply morphological transformations (top-hat, bottom-hat)
# visualise and quantify where the residual lands relative to ground truth
# use to experiment and prune possible dead ends
#
# two input modes
# i. precomputed : 4-channel .npy (image, tophat, bottomhat, label), shows exactly what the network receives
# ii. fresh : any single-channel image (.npy / .nii / .nii.gz), morphology computed on the fly with a configurable
# structuring element, so we can sweep se_radius / mode / dimensionality
#
# works for 2D and 3D, and for any number of label classes
#
# examples
#
# preprocessed volume
# python utilities/morph_explore.py data/Task04_Hippocampus/preprocessed/hippocampus_001.npy
#
# a different SE size, computed fresh from the image channel
# python utilities/morph_explore.py <img.npy> --fresh --se-radius 4
#
# a raw 2D image and its label, no precomputed channels
# python utilities/morph_explore.py retina_01.png --label retina_01_mask.png --fresh --se-radius 3
#
# pick a slice explicitly instead of the max-foreground one, save without showing
# python utilities/morph_explore.py vol.npy --slice 20 --no-show --out panel.png
#

import argparse
import os
import numpy as np


#
# IO
#
def _load_any(path):
    """Load .npy / .nii(.gz) / common image formats -> float ndarray."""
    ext = path.lower()
    if ext.endswith(".npy"):
        return np.load(path)
    if ext.endswith((".nii", ".nii.gz")):
        from medpy.io import load                       # already a project dep
        data, _ = load(path)
        return data
    from PIL import Image                                # png/jpg/tif ...
    arr = np.asarray(Image.open(path))
    return arr.mean(-1) if arr.ndim == 3 else arr        # collapse RGB to gray


def load_inputs(args):
    """
    Return (image, tophat, bottomhat, label) as float arrays of identical shape.
    label may be None. tophat/bottomhat are computed fresh when not available
    (or when --fresh is given).
    """
    raw = _load_any(args.path).astype(np.float64)

    # this project's stacked layout: (4, ...) with channels img/top/bot/label.
    # --fresh still takes image+label from the stack, it only recomputes residuals.
    stacked = raw.ndim >= 3 and raw.shape[0] == 4
    if stacked:
        image, label = raw[0], raw[3]
        tophat, bottomhat = (None, None) if args.fresh else (raw[1], raw[2])
    else:
        # single image channel; drop a leading singleton axis if present
        image = raw[0] if (raw.ndim >= 3 and raw.shape[0] == 1) else raw
        label = _load_any(args.label).astype(np.float64) if args.label else None
        tophat = bottomhat = None

    # normalise image to [0,1] for display/computation stability
    rng = image.max() - image.min()
    image = (image - image.min()) / rng if rng > 0 else image

    if tophat is None or bottomhat is None:
        tophat, bottomhat = compute_residuals(image, args.se_radius)
    return image, tophat, bottomhat, label


#
# morphology
#
def compute_residuals(image, se_radius):
    """White top-hat (x-opening) and black bottom-hat (closing-x). 2D or 3D."""
    from scipy.ndimage import grey_opening, grey_closing
    from skimage.morphology import ball, disk
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}; "
                         "check the input layout (expected a single image channel)")
    fp = ball(se_radius) if image.ndim == 3 else disk(se_radius)
    tophat = np.clip(image - grey_opening(image, footprint=fp), 0, None)
    bottomhat = np.clip(grey_closing(image, footprint=fp) - image, 0, None)
    return tophat, bottomhat


#
# slice selection
#
def pick_slice(label, image, forced):
    """Index of a representative 2D slice along axis 0 (3D), or None for 2D."""
    if image.ndim == 2:
        return None
    if forced is not None:
        return forced
    if label is not None and label.any():
        return int((label > 0).sum(axis=tuple(range(1, label.ndim))).argmax())
    return image.shape[0] // 2


def _slice(a, z):
    return a if (a is None or z is None) else a[z]


#
# quantitative
#
def region_stats(residual, label):
    """
    Enrichment of residual energy at the label boundary vs interior vs background.
    Returns a dict; boundary is a 1-voxel ring around any foreground class.
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    fg = label > 0
    if not fg.any():
        return None
    inside = binary_erosion(fg)
    boundary = binary_dilation(fg) & ~inside
    background = ~binary_dilation(fg)
    tot = residual.sum() + 1e-9
    return dict(
        mean_boundary=residual[boundary].mean(),
        mean_inside=residual[inside].mean() if inside.any() else 0.0,
        mean_background=residual[background].mean(),
        pct_energy_boundary=100 * residual[boundary].sum() / tot,
        pct_energy_inside=100 * residual[inside].sum() / tot,
        pct_energy_background=100 * residual[background].sum() / tot,
        boundary_voxel_frac=100 * boundary.mean(),
        enrichment=(residual[boundary].mean() / (residual[background].mean() + 1e-9)),
    )


def print_stats(name, s):
    if s is None:
        print(f"  {name}: no label -> skipped")
        return
    print(f"  {name}: enrichment(boundary/background)={s['enrichment']:.1f}x  "
          f"mean[bnd/in/bg]={s['mean_boundary']:.4f}/{s['mean_inside']:.4f}/{s['mean_background']:.4f}")
    print(f"      %energy bnd/in/bg = {s['pct_energy_boundary']:.1f}/"
          f"{s['pct_energy_inside']:.1f}/{s['pct_energy_background']:.1f}  "
          f"(boundary = {s['boundary_voxel_frac']:.1f}% of voxels)")


#
# rendering
#
def render(image, tophat, bottomhat, label, z, out, show):
    import matplotlib.pyplot as plt
    im = _slice(image, z)
    tp = _slice(tophat, z)
    bt = _slice(bottomhat, z)
    lb = _slice(label, z)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(im, cmap="gray")
    ax[0].set_title("image")
    ax[1].imshow(tp, cmap="magma")
    ax[1].set_title("top-hat")
    ax[2].imshow(bt, cmap="magma")
    ax[2].set_title("bottom-hat")
    ax[3].imshow(im, cmap="gray")
    ax[3].set_title("image + label contour")
    if lb is not None:
        # overlay each foreground class contour on image AND on the residuals
        for a in (ax[1], ax[2], ax[3]):
            a.contour(lb, levels=np.arange(0.5, lb.max() + 1), colors="cyan", linewidths=0.7)
    for a in ax:
        a.axis("off")
    title = "morphology explorer" + (f"  (z={z})" if z is not None else "  (2D)")
    fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"saved figure -> {out}")
    if show:
        plt.show()
    plt.close(fig)


#
# main
#
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="4-channel .npy, or a single image (.npy/.nii/.png/...)")
    p.add_argument("--label", help="separate label file (when not in a 4-channel stack)")
    p.add_argument("--fresh", action="store_true", help="ignore any precomputed channels; recompute morphology")
    p.add_argument("--se-radius", type=int, default=2, help="structuring-element radius")
    p.add_argument("--slice", type=int, default=None, help="axis-0 slice (3D); default=max foreground")
    p.add_argument("--out", default=None, help="path to save the figure (PNG)")
    p.add_argument("--no-show", action="store_true", help="do not open an interactive window")
    args = p.parse_args()

    image, tophat, bottomhat, label = load_inputs(args)
    z = pick_slice(label, image, args.slice)

    print(f"input: {os.path.basename(args.path)}  shape={image.shape}  "
          f"se_radius={args.se_radius}  slice={z}")
    if label is not None:
        print("TOP-HAT")
        print_stats("tophat", region_stats(tophat, label))
        print("BOTTOM-HAT")
        print_stats("bottomhat", region_stats(bottomhat, label))

    render(image, tophat, bottomhat, label, z,
           args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
