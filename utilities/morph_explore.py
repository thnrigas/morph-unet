#
# Morphology Explorer
#
# two modes
#
#   explore  render every filter in the vocabulary on the densest foreground slice of one
#            case, for a specific dataset or all of them, visual only, no ranking
#   survey   batch ranking to pick filters before training, modality-aware preprocessing,
#            an SE-size sweep and bands scored over all foreground slices
#            by a target-concentration ranking, computes only, no figures
#
# examples
#   # visualise the full filter bank on one dataset (or every data/Task*)
#   python3 utilities/morph_explore.py explore data/Task08_HepaticVessel
#   python3 utilities/morph_explore.py explore all
#   # survey a dataset, rank residuals over 25 cases, all foreground slices, train split
#   python3 utilities/morph_explore.py survey data/Task08_HepaticVessel
#   python3 utilities/morph_explore.py survey data/Task01_BrainTumour --channel 0
#
# (`morph_explore.py <dataset>` still works, it defaults to the explore subcommand)
#

import argparse
import glob
import hashlib
import json
import os
import pickle
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
import numpy as np

#
# IO, load image formats to float arrays
#
def load_any(path):
    ext = path.lower()
    if ext.endswith(".npy"):
        return np.load(path).astype(np.float64)
    if ext.endswith((".nii", ".nii.gz")):
        from medpy.io import load
        data, _ = load(path)
        return data.astype(np.float64)
    from PIL import Image
    arr = np.asarray(Image.open(path)).astype(np.float64)
    return arr.mean(-1) if arr.ndim == 3 else arr

def load_meta(ds_dir):
    with open(os.path.join(ds_dir, "dataset.json")) as f:
        d = json.load(f)
    return d.get("modality", {"0": "MRI"}), d.get("labels", {})

def norm01(v):
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 0 else v * 0.0

#
# preprocessing (windowing / normalisation), modality-aware
#
def preprocess(vol, modality_str, channel=0, n_mod=1, ct_center=40.0, ct_width=400.0, p_lo=0.5, p_hi=99.5):
    if vol.ndim == 4:       # multi-modal, sequences stacked on one axis
        # the modality axis is the one whose length equals the modality count
        # leaving 3 spatial axes to match the label
        cand = [ax for ax in range(4) if vol.shape[ax] == n_mod]
        vol = np.take(vol, channel, axis=cand[-1] if cand else 3)
    v = vol.astype(np.float64)
    if modality_str.upper().startswith("CT"):
        lo, hi = ct_center - ct_width / 2.0, ct_center + ct_width / 2.0
        v = np.clip(v, lo, hi)
    else:
        fg = v[v > v.min()]
        lo, hi = np.percentile(fg, [p_lo, p_hi]) if fg.size else (v.min(), v.max())
        v = np.clip(v, lo, hi)
    return norm01(v)

#
# morphology (2D or 3D)
#
def _footprint(ndim, r):
    from skimage.morphology import ball, disk
    return ball(r) if ndim == 3 else disk(r)

def opening_of(img, r):
    from scipy.ndimage import grey_opening
    return grey_opening(img, footprint=_footprint(img.ndim, r))

def closing_of(img, r):
    from scipy.ndimage import grey_closing
    return grey_closing(img, footprint=_footprint(img.ndim, r))

def tophat(img, r):
    return np.clip(img - opening_of(img, r), 0, None)

def bottomhat(img, r):
    return np.clip(closing_of(img, r) - img, 0, None)

def gradient(img, r):
    from scipy.ndimage import grey_dilation, grey_erosion
    fp = _footprint(img.ndim, r)
    return np.clip(grey_dilation(img, footprint=fp) - grey_erosion(img, footprint=fp), 0, None)

#
# reconstruction top-hat = img - opening-by-reconstruction, marker = erosion(img, disk_r)
# kills small bright structures, geodesic reconstruction regrows the survivors to their exact
# original shape, residual is a boundary-faithful map of the removed thin structures
# non differentiable, static channel only, not a trainable SE block
#
def recon_tophat(img, r):
    from scipy.ndimage import grey_erosion
    from skimage.morphology import reconstruction
    marker = grey_erosion(img, footprint=_footprint(img.ndim, r))
    opened = reconstruction(marker, img, method="dilation")
    return np.clip(img - opened, 0, None)

#
# h-dome = img - reconstruction(img - h, img), extracts every bright peak/ridge rising more
# than height h above its surroundings, size and shape agnostic (no radius), robust to h
# non differentiable (iterative), static channel only
#
def hdome(img, h):
    from skimage.morphology import reconstruction
    rec = reconstruction(img - float(h), img, method="dilation")
    return np.clip(img - rec, 0, None)

#
# area opening, drop bright connected components smaller than area px, keep the rest at their
# exact shape, a denoiser (thin long vessels survive, small bright blobs don't), static
#
def area_open(img, area):
    from skimage.morphology import area_opening
    return area_opening(img.astype(np.float32), area_threshold=int(area))

#
# area closing (dual of area_open), fill dark components smaller than area, bridges small dark
# gaps in the bright vessel tree (recall), static
#
def area_close(img, area):
    from skimage.morphology import area_closing
    return area_closing(img.astype(np.float32), area_threshold=int(area))

#
# area-gated h-dome (cv18 volume marker), keep only bright domes that are both high-contrast
# (rise > h) and large enough (>= area px), isolates vessels better than contrast or size alone
# static (non differentiable)
#
def volume_dome(img, h, area):
    return area_open(hdome(img, h), area)

#
# alternating sequential filter, cascade opening-then-closing at radii 1..r
# (multiscale, edge-respecting simplification / denoise), static
#
def asf(img, r):
    from scipy.ndimage import grey_opening, grey_closing
    out = img
    for rr in range(1, int(r) + 1):
        fp = _footprint(out.ndim, rr)
        out = grey_closing(grey_opening(out, footprint=fp), footprint=fp)
    return out

#
# img - ASF, the bright detail the ASF removed (multiscale top-hat), cleaner thin-structure
# highlighter than a single-scale opening, static
#
def asf_tophat(img, r):
    return np.clip(img - asf(img, r), 0, None)

#
# morphological leveling toward a Gaussian marker (sigma=r), edge-preserving simplification,
# flattens small fluctuations without blurring or shifting strong contours, iterate the marker
# g <- (f wedge dilate(g)) vee erode(g) to near idempotence, static
#
def leveling(img, r, iters=30):
    from scipy.ndimage import gaussian_filter, grey_dilation, grey_erosion
    f = img.astype(np.float32)
    g = gaussian_filter(f, sigma=float(r))
    fp = _footprint(f.ndim, 1)
    for _ in range(int(iters)):
        g_new = np.maximum(np.minimum(f, grey_dilation(g, footprint=fp)), grey_erosion(g, footprint=fp))
        if np.abs(g_new - g).max() < 1e-5:
            g = g_new
            break
        g = g_new
    return g

#
# img - leveling, fine bright detail the leveling simplified away, static
#
def leveling_tophat(img, r):
    return np.clip(img - leveling(img, r), 0, None)

#
# (L, L) binary line SE through the centre at angle radians (2D only)
#
def _line_footprint(length, angle):
    r = length // 2
    fp = np.zeros((length, length), dtype=bool)
    for t in range(-r, r + 1):
        y = int(round(r - t * np.sin(angle)))
        x = int(round(r + t * np.cos(angle)))
        if 0 <= y < length and 0 <= x < length:
            fp[y, x] = True
    return fp

#
# orientation-invariant line top-hat, max over oriented line SEs of (img - opening), so a
# tubular structure survives at whatever angle it runs, 2D only, falls back to isotropic on 3D
#
def line_tophat(img, r, n_angles=4):
    from scipy.ndimage import grey_opening
    if img.ndim != 2:
        return tophat(img, r)
    length, best = 2 * r + 1, None
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        th = np.clip(img - grey_opening(img, footprint=_line_footprint(length, a)), 0, None)
        best = th if best is None else np.maximum(best, th)
    return best

#
# orientation-invariant line bottom-hat, max over oriented line SEs of (closing - img), for
# dark tubular structures, 2D only, falls back to the isotropic bottom-hat on 3D
#
def line_bottomhat(img, r, n_angles=4):
    from scipy.ndimage import grey_closing
    if img.ndim != 2:
        return bottomhat(img, r)
    length, best = 2 * r + 1, None
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        bh = np.clip(grey_closing(img, footprint=_line_footprint(length, a)) - img, 0, None)
        best = bh if best is None else np.maximum(best, bh)
    return best

#
# granulometric band (opening γ), bright structure with scale in (r_lo, r_hi]
#
def band(img, r_lo, r_hi):
    return np.clip(opening_of(img, r_lo) - opening_of(img, r_hi), 0, None)

#
# anti-granulometric band (closing φ), dark structure with scale in (r_lo, r_hi]
#
def band_dark(img, r_lo, r_hi):
    return np.clip(closing_of(img, r_hi) - closing_of(img, r_lo), 0, None)

#
# slice selection, (axis, index) of the largest target slice over all axes
#
def pick_slice(label, forced=None):
    if label is None or label.ndim == 2 or not (label > 0).any():
        return None
    if forced is not None:
        return 0, forced
    best = None
    for ax in range(label.ndim):
        counts = (label > 0).sum(axis=tuple(i for i in range(label.ndim) if i != ax))
        idx, area = int(counts.argmax()), int(counts.max())
        if best is None or area > best[2]:
            best = (ax, idx, area)
    return best[0], best[1]


def take(a, sel):
    if a is None or sel is None:
        return a
    ax, idx = sel
    return np.take(a, idx, axis=ax)

#
# reorder img axes to match the label (medpy permutes the spatial axes of some 4D volumes),
# when several permutations match the shape disambiguate by the one whose label foreground sits
# on image tissue (highest mean intensity), since a wrong swap lands the label on background
#
def align_axes(img, label):
    from itertools import permutations
    if img.shape == label.shape:
        return img
    fg = label > 0
    best, best_score = img, -1.0
    for perm in permutations(range(img.ndim)):
        if tuple(img.shape[p] for p in perm) == label.shape:
            cand = img.transpose(perm)
            score = float(cand[fg].mean()) if fg.any() else 0.0
            if score > best_score:
                best, best_score = cand, score
    return best

#
# stats: where residual energy lands relative to the target
#
# AUC = P(residual on a random fg pixel > a random bg pixel), via Mann-Whitney U, threshold-free
# discriminability (0.5 = chance, 1.0 = perfectly separable), background subsampled for speed
#
def _auc(pos, neg, max_n=20000):
    from scipy.stats import rankdata
    pos = np.asarray(pos, np.float64).ravel()
    neg = np.asarray(neg, np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return 0.5
    if neg.size > max_n:
        neg = np.random.default_rng(0).choice(neg, max_n, replace=False)
    ranks = rankdata(np.concatenate([pos, neg]))
    n1 = pos.size
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * neg.size))

def region_stats(residual, label):
    from scipy.ndimage import binary_erosion, binary_dilation
    fg = label > 0
    if not fg.any():
        return None
    inside = binary_erosion(fg)
    boundary = binary_dilation(fg) & ~inside
    background = ~binary_dilation(fg)
    tot = residual.sum() + 1e-9
    pct_on_target = 100 * residual[fg].sum() / tot
    target_area_pct = 100 * fg.mean()
    res_fg, res_bg = residual[fg], residual[background]
    return dict(
        enrichment=(residual[boundary].mean() / (residual[background].mean() + 1e-9)),
        mean_boundary=residual[boundary].mean(),
        mean_inside=residual[inside].mean() if inside.any() else 0.0,
        mean_background=residual[background].mean(),
        pct_on_target=pct_on_target,
        target_area_pct=target_area_pct,
        concentration=pct_on_target / (target_area_pct + 1e-9),
        # discriminability of the residual as an fg-vs-bg pixel score (complements concentration)
        auc=_auc(res_fg, res_bg),
        fisher=float((res_fg.mean() - res_bg.mean()) ** 2 / (res_fg.var() + res_bg.var() + 1e-9)),
    )

#
# rendering
#
# grid of every filter in the library at the given radius / h / area, with the label contour
# overlaid, so a single case shows the full vocabulary (trainable + static) side by side
#
def render_all(img2d, label2d, r, h, area, out, show, save_individual=True):
    import matplotlib.pyplot as plt
    lo = max(r - 1, 1)
    panels = [
        ("image", img2d, "gray", False),
        ("image + label", img2d, "gray", True),
        (f"top-hat r={r}", tophat(img2d, r), "magma", True),
        (f"bottom-hat r={r}", bottomhat(img2d, r), "magma", True),
        (f"gradient r={r}", gradient(img2d, r), "magma", True),
        (f"recon-tophat r={r}", recon_tophat(img2d, r), "magma", True),
        (f"line-tophat r={r}", line_tophat(img2d, r), "magma", True),
        (f"line-bottomhat r={r}", line_bottomhat(img2d, r), "magma", True),
        (f"asf-tophat r={r}", asf_tophat(img2d, r), "magma", True),
        (f"leveling r={r}", leveling(img2d, r), "gray", True),
        (f"leveling-tophat r={r}", leveling_tophat(img2d, r), "magma", True),
        (f"h-dome h={h}", hdome(img2d, h), "magma", True),
        (f"volume-dome h={h} a={area}", volume_dome(img2d, h, area), "magma", True),
        (f"area-open a={area} (denoise)", area_open(img2d, area), "gray", True),
        (f"area-close a={area} (bridge)", area_close(img2d, area), "gray", True),
        (f"gband {lo}-{r}", band(img2d, lo, r), "magma", True),
        (f"fband {lo}-{r}", band_dark(img2d, lo, r), "magma", True),
    ]
    ncol = 5
    nrow = (len(panels) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow))
    axf = ax.ravel()
    has_lbl = label2d is not None and label2d.max() > 0
    for a, (title, m, cmap, cont) in zip(axf, panels):
        a.imshow(m, cmap=cmap); a.set_title(title, fontsize=8)
        if cont and has_lbl:
            a.contour(label2d, levels=np.arange(0.5, label2d.max() + 1), colors="cyan", linewidths=0.5)
    for a in axf:
        a.axis("off")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved grid -> {out}")
        if save_individual:
            # also save one PNG per filter next to the grid: <stem>_<filter>.png
            stem, ext = os.path.splitext(out)
            ext = ext or ".png"
            for title, m, cmap, cont in panels:
                f2, a2 = plt.subplots(figsize=(4, 4))
                a2.imshow(m, cmap=cmap); a2.set_title(title, fontsize=10); a2.axis("off")
                if cont and has_lbl:
                    a2.contour(label2d, levels=np.arange(0.5, label2d.max() + 1), colors="cyan", linewidths=0.7)
                safe = title.replace(" ", "_").replace("=", "").replace("/", "-")
                f2.savefig(f"{stem}_{safe}{ext}", dpi=120, bbox_inches="tight"); plt.close(f2)
            print(f"saved {len(panels)} per-filter PNGs -> {stem}_<filter>{ext}")
    if show:
        plt.show()
    plt.close(fig)

#
# subcommand: explore — render every filter in the vocabulary on the densest-foreground
# slice of one (densest) case per dataset, and print a single-slice concentration/AUC
# ranking of that same bank (a quick preview of what `survey` aggregates over the dataset).
# Pass one or more MSD task dirs, or "all" for every data/Task* dir.
#
#
# expand the 'all' shortcut to every data/Task* dir that has an imagesTr/
#
def _resolve_datasets(datasets):
    if datasets == ["all"]:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        return sorted(d for d in glob.glob(os.path.join(root, "Task*"))
                      if os.path.isdir(os.path.join(d, "imagesTr")))
    return datasets

#
# subcommand: explore — render every filter on one representative case per dataset (visual only,
# no scoring; use `survey` for train-split filter selection over all slices / n cases)
#
def cmd_explore(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for ddir in _resolve_datasets(args.datasets):
        try:
            task_name = os.path.basename(ddir.rstrip('/'))
            modality, _ = load_meta(ddir)
            mod = modality.get("0", "MRI")
            ig, lg = os.path.join(ddir, "imagesTr"), os.path.join(ddir, "labelsTr")
            cases = sorted(f for f in os.listdir(ig) if f.endswith(".nii.gz") and not f.startswith("."))
            if not cases:
                print(f"{ddir}: no NIfTI cases found, skipping", flush=True)
                continue
            best = None
            for fn in cases[:args.scan]:                 # pick the densest-foreground case
                lbl = load_any(os.path.join(lg, fn)); lbl = lbl[..., 0] if lbl.ndim == 4 else lbl
                fgv = (lbl > 0).sum()
                if best is None or fgv > best[1]:
                    best = (fn, fgv)
            fn = best[0]
            img = preprocess(load_any(os.path.join(ig, fn)), mod, 0, len(modality))
            lbl = load_any(os.path.join(lg, fn)); lbl = lbl[..., 0] if lbl.ndim == 4 else lbl
            img = align_axes(img, lbl); sel = pick_slice(lbl)
            i2, l2 = take(img, sel), take(lbl, sel)
            out_path = os.path.join(args.out_dir, f"{task_name}_explore.png")
            print(f"{task_name}: selected case {fn}, slice {sel} -> {out_path}", flush=True)
            render_all(i2, l2, args.se_radius, args.h, args.area, out_path,
                       show=not args.no_show, save_individual=False)
        except Exception as e:
            print(f"{ddir}: FAILED {type(e).__name__}: {e}", flush=True)

#
# subcommand: survey (batch ranking) — parallel, optional train-split + all-slices,
# with an auto-selector that emits the --morph-bank spec from the ranking
#
METRICS = ("concentration", "enrichment", "auc", "fisher")   # accumulated per residual

#
# accumulate the per-(residual, class) metric sums over a case's foreground slices, slices axis 0
# to match the training loader (the exact plane the network trains on), scores each labelled class
# separately (plus the pooled 'all') so e.g. Vessel and Tumour rank independently, picklable
#
def _survey_case(fn, image_dir, label_dir, mod, channel, n_mod, radii, bands, all_slices,
                 h_values=(), areas=(), classes=()):
    img = preprocess(load_any(os.path.join(image_dir, fn)), mod, channel, n_mod)
    lbl = load_any(os.path.join(label_dir, fn))
    if lbl.ndim == 4:
        lbl = lbl[..., 0]
    img = align_axes(img, lbl)
    if lbl.ndim == 2 or not (lbl > 0).any():
        axis, idxs = None, [None]
    else:
        axis = 0                                   # same plane the loader slices
        counts = (lbl > 0).sum(axis=tuple(i for i in range(lbl.ndim) if i != axis))
        idxs = [int(i) for i in np.where(counts > 0)[0]] if all_slices else [int(counts.argmax())]

    acc = {}   # (key, class) -> [conc_sum, enrich_sum, auc_sum, fisher_sum, count]
    # score against the pooled foreground ("all") and each labelled class independently
    targets = [(None, "all")] + list(classes)

    def note(key, res, lbl2d):
        for cid, cname in targets:
            mask = (lbl2d > 0) if cid is None else (lbl2d == cid)
            if not mask.any():
                continue
            s = region_stats(res, mask)
            if s:
                a = acc.setdefault((key, cname), [0.0, 0.0, 0.0, 0.0, 0])
                for i, m in enumerate(METRICS):
                    a[i] += s[m]
                a[4] += 1

    for idx in idxs:
        img2d = img if idx is None else np.take(img, idx, axis=axis)
        lbl2d = lbl if idx is None else np.take(lbl, idx, axis=axis)
        for r in radii:
            note(f"tophat r={r}", tophat(img2d, r), lbl2d)
            note(f"bottomhat r={r}", bottomhat(img2d, r), lbl2d)
            note(f"gradient r={r}", gradient(img2d, r), lbl2d)
            note(f"ltophat r={r}", line_tophat(img2d, r), lbl2d)          # diagnostic (oriented, bright)
            note(f"lbottomhat r={r}", line_bottomhat(img2d, r), lbl2d)    # diagnostic (oriented, dark)
            note(f"recontophat r={r}", recon_tophat(img2d, r), lbl2d)     # static (connected, boundary-faithful)
            note(f"asftophat r={r}", asf_tophat(img2d, r), lbl2d)         # static (multiscale simplification residual)
            note(f"leveltophat r={r}", leveling_tophat(img2d, r), lbl2d)  # static (leveling residual, edge-preserving)
        for h in h_values:
            note(f"hdome h={h}", hdome(img2d, h), lbl2d)                  # static (connected, contrast, radius-free)
            for area in areas:
                note(f"vdome h={h} a={area}", volume_dome(img2d, h, area), lbl2d)   # static (cv18 volume: contrast+area)
        for lo, hi in bands:
            note(f"gband {lo}-{hi}", band(img2d, lo, hi), lbl2d)
            note(f"fband {lo}-{hi}", band_dark(img2d, lo, hi), lbl2d)
    return fn, acc, len(idxs)

#
# --morph-bank tokens from a survey key (0, 1, or 2). full grammar (networks/morph_block.py):
# exact (tophat/bottomhat/gradient/asftophat) AND reconstruction/dome/leveling
# (recontophat/hdome/vdome/leveltophat). granulometric bands have no dedicated op -- they are a
# linear difference of two same-family residuals (gband = tophat(hi)-tophat(lo);
# fband = bottomhat(hi)-bottomhat(lo)), which the U-Net's linear first conv forms from the two
# appended channels, so a band EXPANDS to its two tophats/bottomhats. vdome -> surrogate (h only).
# only connected-only operators (area_open/area_close) have no soft op and remain static-only.
#
def _trainable_specs_from_key(key):
    for mode in ("tophat", "bottomhat", "gradient"):
        if key.startswith(f"{mode} r="):
            return [f"{mode}:{int(key.split('r=')[1])}"]
    if key.startswith("asftophat r="):
        return [f"asftophat:{int(key.split('r=')[1])}"]
    if key.startswith("recontophat r="):
        return [f"recontophat:{int(key.split('r=')[1])}"]
    if key.startswith("leveltophat r="):
        return [f"leveltophat:{int(key.split('r=')[1])}"]
    if key.startswith("hdome h="):
        return [f"hdome:{key.split('h=')[1].split()[0]}"]
    if key.startswith("vdome h="):
        return [f"vdome:{key.split('h=')[1].split()[0]}"]
    if key.startswith("gband "):
        lo, hi = key.split()[1].split("-")
        return [f"tophat:{int(lo)}", f"tophat:{int(hi)}"]
    if key.startswith("fband "):
        lo, hi = key.split()[1].split("-")
        return [f"bottomhat:{int(lo)}", f"bottomhat:{int(hi)}"]
    return []

def _select_spec(rows, k):
    # k caps the number of TOKENS (channels); a band consumes two. dedup keeps a tophat shared
    # between a band and a standalone pick from being emitted twice.
    picked = []
    for row in rows:
        for spec in _trainable_specs_from_key(row["key"]):
            if spec not in picked and len(picked) < k:
                picked.append(spec)
    return ",".join(picked)

# mapping from survey key prefix → augment_channels.py filter spec
_STATIC_KEY_MAP = {
    "recontophat r=": lambda key: f"recontophat:{key.split('r=')[1]}",
    "asftophat r=":   lambda key: f"asftophat:{key.split('r=')[1]}",
    "leveltophat r=": lambda key: f"leveltophat:{key.split('r=')[1]}",
    "hdome h=":       lambda key: f"hdome:{key.split('h=')[1]}",
    "vdome h=":       lambda key: _vdome_spec(key),
}

def _vdome_spec(key):
    # "vdome h=0.1 a=50" → "vdome:0.1:50"
    parts = key.split()
    h = parts[1].split("=")[1]
    a = parts[2].split("=")[1]
    return f"vdome:{h}:{a}"

#
# top-k static (non differentiable) filters from the ranking, formatted as augment_channels.py
# --filters specs, the connected/oriented operators that can't be trainable SE blocks and must
# be pre-computed as static input channels
#
def _select_static(rows, k):
    picked = []
    for row in rows:
        key = row["key"]
        for prefix, to_spec in _STATIC_KEY_MAP.items():
            if key.startswith(prefix) and len(picked) < k:
                spec = to_spec(key)
                if spec not in picked:
                    picked.append(spec)
                break
    return picked


def cmd_survey(args):
    modality, labels = load_meta(args.dataset_dir)
    classes = [(int(k), v) for k, v in labels.items() if int(k) != 0]   # foreground classes for per-class scoring
    mod = modality[str(args.channel)] if str(args.channel) in modality else modality.get("0", "MRI")
    task = os.path.basename(args.dataset_dir.rstrip("/"))
    os.makedirs(args.out_dir, exist_ok=True)
    bands = list(zip(args.bands[0::2], args.bands[1::2]))
    img_dir = os.path.join(args.dataset_dir, "imagesTr")
    lbl_dir = os.path.join(args.dataset_dir, "labelsTr")

    cases = sorted(f for f in os.listdir(img_dir) if f.endswith(".nii.gz") and not f.startswith("."))
    tag = "all"
    if args.split == "train":                          # default: selection must use training data only
        splits_path = os.path.join(args.dataset_dir, "splits.pkl")
        if os.path.exists(splits_path):
            with open(splits_path, "rb") as f:
                keys = set(pickle.load(f)[args.fold]["train"])
            cases = [fn for fn in cases if fn.replace(".nii.gz", "") in keys]
            tag = f"train-f{args.fold}"
        else:
            print(f"no splits.pkl in {args.dataset_dir} (run_preprocessing first for "
                  f"train selection), falling back to all cases", flush=True)
    if args.n > 0:
        cases = cases[:args.n]

    # parallel per-case accumulation over all foreground slices (or one, per --all-slices)
    worker = partial(_survey_case, image_dir=img_dir, label_dir=lbl_dir, mod=mod, channel=args.channel,
                     n_mod=len(modality), radii=args.radii, bands=bands, all_slices=args.all_slices,
                     h_values=args.h_values, areas=args.areas, classes=classes)
    total, n_slices = {}, 0
    print(f"surveying {len(cases)} cases ({tag}, all_slices={args.all_slices}, workers={args.workers})...", flush=True)

    def consume(res):
        nonlocal n_slices
        fn, acc, nsl = res
        n_slices += nsl
        for key, vals in acc.items():
            t = total.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
            for i in range(5):
                t[i] += vals[i]
        print(f"    {fn} ({nsl} slices)", flush=True)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(worker, cases):
                consume(res)
    else:
        for fn in cases:
            consume(worker(fn))

    # per-(residual, class) metric means, plus a combined selectivity x discriminability score
    rows = []
    for (key, cls), t in total.items():
        if t[4] > 0:
            row = {"key": key, "class": cls, **{m: t[i] / t[4] for i, m in enumerate(METRICS)}}
            row["conc_auc"] = row["concentration"] * row["auc"]
            rows.append(row)
    rows.sort(key=lambda r: r[args.rank_by], reverse=True)
    # the trainable-bank / static specs are selected from the pooled 'all' ranking (train pipeline
    # consumes one spec per fold); the per-class tables below are diagnostic
    pooled = [r for r in rows if r["class"] == "all"]
    spec = _select_spec(pooled, args.top_k)
    static_specs = _select_static(pooled, args.top_k)

    cols = ["concentration", "enrichment", "auc", "fisher", "conc_auc"]
    order = ["all"] + [cn for _, cn in classes]
    classes_present = [c for c in order if any(r["class"] == c for r in rows)]
    head = [f"SUMMARY  {task}  ({tag}, modality={mod}, {len(cases)} cases, {n_slices} slices, "
            f"rank-by={args.rank_by})"]
    for cls in classes_present:
        crows = sorted((r for r in rows if r["class"] == cls), key=lambda r: r[args.rank_by], reverse=True)
        head.append(f"\n--- class: {cls} ---")
        head.append(f"{'residual':13s} " + " ".join(f"{c:>13s}" for c in cols))
        for row in crows:
            head.append(f"{row['key']:13s} " + " ".join(f"{row[c]:13.3f}" for c in cols))
        if crows:
            head.append(f">>> best {cls} ({args.rank_by}) = {crows[0][args.rank_by]:.3f}  ({crows[0]['key']})")
    head.append("")
    if pooled:
        head.append(f"  selected --morph-bank \"{spec}\"  (trainable, from pooled 'all')")
        if static_specs:
            head.append(f"  selected --filters {' '.join(static_specs)}")
            head.append(f"      datasets/augment_channels.py --filters {' '.join(static_specs)}")
            head.append(f"      train_eval.py --static-channels {len(static_specs)}")
    print("\n".join(head))
    out_path = os.path.join(args.out_dir, f"{task}_{tag}_stats.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(head) + "\n")

    # whole-dataset surveys (tag "all", not a fold's train split) write to separate *_all_* files,
    # so they never clobber the leakage-clean per-fold specs that train_eval's --*-auto consumes
    fname_tag = "all_" if tag == "all" else ""
    # machine-readable handoff: fold -> spec, consumed by `train_eval.py --morph-bank auto`
    bank_path = os.path.join(args.out_dir, f"{task}_{fname_tag}bank.json")
    bank = {}
    if os.path.exists(bank_path):
        with open(bank_path) as f:
            bank = json.load(f)
    bank[str(args.fold)] = spec
    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)

    # machine-readable static handoff: fold -> list of filter specs for augment_channels.py
    static_path = os.path.join(args.out_dir, f"{task}_{fname_tag}static.json")
    static_bank = {}
    if os.path.exists(static_path):
        with open(static_path) as f:
            static_bank = json.load(f)
    static_bank[str(args.fold)] = static_specs
    with open(static_path, "w") as f:
        json.dump(static_bank, f, indent=2)

    print(f"\n[written] {out_path}")
    print(f"[written] {bank_path}  (fold {args.fold} -> \"{spec}\")")
    if static_specs:
        print(f"[written] {static_path}  (fold {args.fold} -> {static_specs})")

    # --augment: precompute the selected static filters into a per-fold augmented dir, so the whole
    # static pipeline (survey -> select top-k -> precompute) is one command. train_eval just consumes
    # the printed --static-dir / --static-channels; no survey/augment logic lives in the trainer.
    if args.augment and static_specs:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
        aug = os.path.join(root_dir, "datasets", "augment_channels.py")
        src = os.path.join(args.dataset_dir, "preprocessed")
        suffix = "h" + hashlib.md5(",".join(static_specs).encode()).hexdigest()[:8]
        out = os.path.join(args.dataset_dir, f"preprocessed_static_{suffix}")
        if not os.path.isdir(src):
            print(f"[augment] SKIPPED: preprocessed dir not found ({src}). Run run_preprocessing.py first.")
        else:
            print(f"\n[augment] precomputing {len(static_specs)} static channels -> {out}", flush=True)
            subprocess.run([sys.executable, aug, "--filters", *static_specs,
                            "--src", src, "--out", out, "--workers", str(args.workers)], check=True)
            print(f">>> ready. train with:\n>>>   python train_eval.py --tag static_cv18 "
                  f"--static-dir {out} --static-channels {len(static_specs)} --fold {args.fold}")


#
# main
#
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("explore")
    pe.add_argument("datasets", nargs="+")
    pe.add_argument("--se-radius", type=int, default=2)
    pe.add_argument("--h", type=float, default=0.1)
    pe.add_argument("--area", type=int, default=100)
    pe.add_argument("--scan", type=int, default=5)
    pe.add_argument("--out-dir", default="results/explore")
    pe.add_argument("--no-show", action="store_true")
    pe.set_defaults(func=cmd_explore)

    ps = sub.add_parser("survey")
    ps.add_argument("dataset_dir")
    ps.add_argument("--n", type=int, default=25)
    ps.add_argument("--channel", type=int, default=0)
    ps.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3, 5])
    ps.add_argument("--bands", type=int, nargs="+", default=[1, 2, 2, 3, 3, 5])
    ps.add_argument("--h-values", dest="h_values", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ps.add_argument("--areas", type=int, nargs="+", default=[50, 150])
    ps.add_argument("--split", choices=["all", "train"], default="train")
    ps.add_argument("--fold", type=int, default=0)
    ps.add_argument("--all-slices", dest="all_slices", action="store_true", default=True)
    ps.add_argument("--one-slice", dest="all_slices", action="store_false")
    ps.add_argument("--rank-by", choices=["concentration", "auc", "fisher", "conc_auc"], default="conc_auc")
    ps.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    ps.add_argument("--top-k", type=int, default=5)
    ps.add_argument("--out-dir", default="results/explore")
    ps.add_argument("--augment", action="store_true")
    ps.set_defaults(func=cmd_survey)
    return p


def main():
    import sys
    argv = sys.argv[1:]
    if argv and argv[0] not in ("explore", "survey", "-h", "--help"):
        argv = ["explore"] + argv
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
