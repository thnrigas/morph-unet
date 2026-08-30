"""Anchored, band-limited refinement of a U-Net segmentation by a classical method.

The two hybrid ideas share ONE engine and differ only in how the refinement band is drawn:

  * Hybrid #1 - uncertainty-gated (on the MC-Dropout net): the band is the set of
    high-uncertainty voxels near the predicted boundary (uncertainty = MC entropy /
    mutual_info / fg-prob variance).  This is the "gate": the classical method is invited
    in exactly where the network is unsure.

  * Hybrid #2 - weakened baseline (on the dropout-FREE weak net): that net has no
    uncertainty, so the band is a purely MORPHOLOGICAL ring straddling the predicted
    boundary (dilation XOR erosion).

In BOTH cases the refiner may only RELABEL voxels inside the band; every voxel outside the
band keeps the U-Net's label.  This ANCHORS the hybrid to the network -- the result equals
the U-Net except inside a thin band -- so a classical refiner can fix boundary/leakage
errors without dragging a strong prediction toward its own (worse) fixed point.  The whole
premise (discussed with the user) is that unrestricted "refine to convergence" degrades a
good CNN; band-anchoring is what makes a gain possible.

Everything is 2D per axial slice, matching the rest of the project.  No post-processing is
added on top (the U-Net baseline is scored raw), so any Dice delta is attributable to the
refiner alone.
"""

import numpy as np


# --------------------------------------------------------------------------- morphology
def _disk(radius):
    from skimage.morphology import disk
    return disk(int(radius))


def _dilate2d(mask2d, radius):
    if radius <= 0:
        return np.asarray(mask2d, bool)
    from skimage.morphology import binary_dilation
    return binary_dilation(np.asarray(mask2d, bool), _disk(radius))


def _erode2d(mask2d, radius):
    if radius <= 0:
        return np.asarray(mask2d, bool)
    from skimage.morphology import binary_erosion
    return binary_erosion(np.asarray(mask2d, bool), _disk(radius))


def _boundary2d(mask2d, radius=1):
    """Symmetric boundary ring of a 2D mask (dilate XOR erode)."""
    return _dilate2d(mask2d, radius) & ~_erode2d(mask2d, radius)


# --------------------------------------------------------------------------- bands
def morph_band(seg, radius=6):
    """Hybrid #2 band: a ring of width ~2*radius straddling the predicted boundary,
    per slice. Voxels here get relabelled; everything else keeps the U-Net label."""
    seg = np.asarray(seg, bool)
    band = np.zeros(seg.shape, bool)
    for z in range(seg.shape[0]):
        if seg[z].any():
            band[z] = _dilate2d(seg[z], radius) & ~_erode2d(seg[z], radius)
    return band


def uncertainty_band(seg, uncertainty, quantile=0.80, ring_radius=12, boundary_radius=1):
    """Hybrid #1 band: near-boundary voxels whose uncertainty is in the top (1-quantile)
    fraction *within that near-boundary region*.

    Kept deliberately conservative (anchored to the predicted boundary): the refiner does
    not hallucinate spleen on slices where the net predicted nothing. `uncertainty` is any
    per-voxel map (higher = less certain), e.g. MC mutual information."""
    seg = np.asarray(seg, bool)
    unc = np.asarray(uncertainty, np.float32)
    band = np.zeros(seg.shape, bool)
    for z in range(seg.shape[0]):
        b = _boundary2d(seg[z], boundary_radius)
        if not b.any():
            continue
        near = _dilate2d(b, ring_radius)
        u = unc[z]
        vals = u[near]
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        thr = float(np.quantile(vals[finite], quantile))
        band[z] = near & np.isfinite(u) & (u >= thr)
    return band


# --------------------------------------------------------------------------- Track-B image
def track_b_image(image, a_center=40.0, a_width=400.0, b_center=50.0, b_width=150.0, median=True):
    """Exact re-windowing of a Track-A [0,1] image to Track-B contrast, ON THE SAME grid.

    Track A/B differ only in the CT window (A: c40/w400 -> HU[-160,240]; B: c50/w150 ->
    HU[-25,125]) + a median denoise on B. Soft tissue (spleen ~50 HU, liver, muscle) lies
    inside A's window, so its HU is recoverable from the A image losslessly; re-windowing to
    B is then closed-form and PIXEL-ALIGNED with the U-Net prediction (no resampling, no
    separate preprocessing). Values A clipped (bone>240 / air<-160) are outside B's window too,
    so the clip maps them correctly. This lets the refiner see B's stronger spleen edges while
    seeds/band stay in A-space.  x_B = clip((x_A*a_width + (a_lo - b_lo)) / b_width, 0, 1)."""
    a_lo = a_center - a_width / 2.0
    b_lo = b_center - b_width / 2.0
    xb = (np.asarray(image, np.float32) * a_width + (a_lo - b_lo)) / b_width
    xb = np.clip(xb, 0.0, 1.0)
    if median:
        from scipy.ndimage import median_filter
        xb = median_filter(xb, size=(1, 3, 3))          # match Track B's in-plane 3x3 median
    return xb.astype(np.float32)


# --------------------------------------------------------------------------- per-case score
def boundary_score(seg, value_map, boundary_radius=1):
    """Pooled mean of a per-voxel map over the predicted-boundary ring -> one scalar per case.

    A GT-free case-quality signal: with value_map = MC fg-prob variance it is the boundary
    epistemic disagreement (best rank-correlation with per-case Dice in our data, Spearman
    ~-0.92); with value_map = single-forward predictive entropy it is the deterministic
    analogue (for the dropout-free weak net). Higher -> less certain -> lower expected Dice."""
    seg = np.asarray(seg, bool)
    tot, cnt = 0.0, 0
    for z in range(seg.shape[0]):
        b = _boundary2d(seg[z], boundary_radius)
        if b.any():
            v = np.asarray(value_map[z])[b]
            v = v[np.isfinite(v)]
            tot += float(v.sum()); cnt += int(v.size)
    return tot / cnt if cnt else 0.0


def binary_entropy_map(fg_prob, eps=1e-6):
    """Per-voxel predictive entropy of a single-forward fg probability (deterministic nets)."""
    p = np.clip(np.asarray(fg_prob, np.float64), eps, 1.0 - eps)
    return (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))).astype(np.float32)


# --------------------------------------------------------------------------- refiners
def refine_random_walker(image, seg, band=None, beta=130.0, tol=1e-3, rw_margin=8,
                         restrict_to_band=False):
    """Grady random walker with a SEED-AND-SOLVE setup (the tier-2 oracle configuration that
    scored best on boundary metrics): confident interior = erode(pred, rw_margin) -> fg marker,
    far exterior = ~dilate(pred, rw_margin) -> bg marker, the ring in between is unlabelled and
    RW places the boundary there from image evidence. This gives RW the WIDE gap it needs (the
    old thin-band markers starved it). Anchoring holds: seeded pixels return as their marker.
    restrict_to_band=True additionally commits changes only inside `band` (uncertainty gate)."""
    from skimage.segmentation import random_walker
    seg = np.asarray(seg, bool)
    out = seg.copy()
    for z in range(seg.shape[0]):
        s = seg[z]
        if not s.any():
            continue
        fg = _erode2d(s, rw_margin)
        if not fg.any():
            fg = s                            # tiny structure: seed with the mask itself
        bg = ~_dilate2d(s, rw_margin)
        if not bg.any():
            continue
        markers = np.zeros(s.shape, np.int32)
        markers[bg] = 1                       # confident background
        markers[fg] = 2                       # confident foreground
        if not (markers == 0).any():          # no unlabelled gap -> nothing to solve
            continue
        try:
            lab = random_walker(image[z].astype(np.float64), markers, beta=beta, tol=tol, mode="bf")
            res = (lab == 2)
        except Exception:
            continue                          # numerical failure -> keep the U-Net slice
        if restrict_to_band and band is not None:
            b = np.asarray(band[z], bool)
            out[z] = np.where(b, res, s)      # commit changes only inside the band
        else:
            out[z] = res
    return out


def refine_level_set(image, seg, band, kind="gac", iters=10, smoothing=2, balloon=0.0,
                     alpha=100.0, sigma=2.0, lambda1=1.0, lambda2=1.0):
    """Morphological level set (GAC edge-based, or Chan-Vese region-based) initialised at the
    U-Net contour and evolved a few iterations, then CLAMPED to the band: only changes inside
    the band are accepted. Bridges to Ch.17 (active contours / morphological curvature flows).

    balloon=0 (default): the contour only SNAPS to the nearest strong edge (no inflation), which
    avoids the over-growth/precision-collapse seen with balloon=+1. Paired with a Track-B image
    (stronger spleen edges) this recovers the boundary without ballooning into neighbours."""
    from skimage.segmentation import (morphological_geodesic_active_contour,
                                      inverse_gaussian_gradient, morphological_chan_vese)
    seg = np.asarray(seg, bool)
    band = np.asarray(band, bool)
    out = seg.copy()
    for z in range(seg.shape[0]):
        if not band[z].any() or not seg[z].any():
            continue
        init = seg[z].astype(np.uint8)
        img = image[z].astype(np.float32)
        if kind == "gac":
            g = inverse_gaussian_gradient(img, alpha=alpha, sigma=sigma)
            ls = morphological_geodesic_active_contour(
                g, iters, init_level_set=init, smoothing=smoothing,
                threshold="auto", balloon=balloon)
        else:                                 # region-based Chan-Vese (no edge stop)
            ls = morphological_chan_vese(
                img, iters, init_level_set=init, smoothing=smoothing,
                lambda1=lambda1, lambda2=lambda2)
        ls = ls.astype(bool)
        res = seg[z].copy()
        res[band[z]] = ls[band[z]]            # ANCHOR: accept changes only inside the band
        out[z] = res
    return out


# --------------------------------------------------------------------------- dispatch
def refine_volume(image, seg, band, refiner="rw", **params):
    """Dispatch to a refiner by name: 'rw' (seed-and-solve random walker) | 'ls' (=GAC)."""
    if refiner == "rw":
        return refine_random_walker(image, seg, band,
                                    beta=params.get("beta", 130.0),
                                    tol=params.get("tol", 1e-3),
                                    rw_margin=params.get("rw_margin", 8),
                                    restrict_to_band=params.get("restrict_to_band", False))
    if refiner in ("ls", "ls_gac", "ls_cv"):
        kind = "cv" if refiner == "ls_cv" or params.get("ls_kind") == "cv" else "gac"
        return refine_level_set(
            image, seg, band, kind=kind,
            iters=params.get("iters", 10), smoothing=params.get("smoothing", 2),
            balloon=params.get("balloon", 0.0), alpha=params.get("alpha", 100.0),
            sigma=params.get("sigma", 2.0), lambda1=params.get("lambda1", 1.0),
            lambda2=params.get("lambda2", 1.0))
    raise ValueError(f"unknown refiner {refiner!r} (use 'rw' or 'ls')")
