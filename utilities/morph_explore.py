#
# Morphology Explorer
#
# one tool, two subcommands:
#
#   explore  - single-case deep dive. 4-panel (image, top-hat, bottom-hat, image+label),
#              2D or 3D, precomputed 4-channel .npy OR fresh morphology, with region stats.
#   survey   - batch, per-dataset look to pick a task before training: modality-aware
#              preprocessing (CT windowing / MRI normalisation), an SE-size sweep and
#              granulometric bands, scored by a target-concentration ranking.
#
# examples
#   # single preprocessed volume (what the network receives)
#   python utilities/morph_explore.py explore data/Task04_Hippocampus/preprocessed/hippocampus_001.npy
#   # a raw image + its label, fresh morphology at a chosen SE size
#   python utilities/morph_explore.py explore img.nii.gz --label mask.nii.gz --fresh --se-radius 3
#   # survey a dataset: sweep radii + bands over 8 cases, render the first 2, rank residuals
#   python utilities/morph_explore.py survey data/Task08_HepaticVessel --n 8 --viz 2
#   python utilities/morph_explore.py survey data/Task01_BrainTumour --channel 0   # FLAIR
#
# (bare `morph_explore.py <path>` still works: it defaults to the explore subcommand.)
#

import argparse
import json
import os
import pickle
from functools import partial
from multiprocessing import Pool
import numpy as np


#
# IO
#
def load_any(path):
    """Load .npy / .nii(.gz) / common image formats -> float ndarray."""
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
def preprocess(vol, modality_str, channel=0, n_mod=1, ct_center=40.0, ct_width=400.0,
               p_lo=0.5, p_hi=99.5):
    """Normalise to [0,1] matched to the modality.

    CT  : clip to a Hounsfield window so soft tissue keeps contrast (vs air/bone).
    MRI : robust percentile clip of the foreground, then min-max (intensity is relative).
    """
    if vol.ndim == 4:                       # multi-modal: sequences stacked on one axis
        # the modality axis is the one whose length equals the modality count
        # (medpy may place it first or last), leaving 3 spatial axes to match the label
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
# morphology (2D or 3D, chosen by ndim)
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
    """morphological gradient = dilation - erosion (boundary-selective)."""
    from scipy.ndimage import grey_dilation, grey_erosion
    fp = _footprint(img.ndim, r)
    return np.clip(grey_dilation(img, footprint=fp) - grey_erosion(img, footprint=fp), 0, None)


def _line_footprint(length, angle):
    """(L, L) binary line SE through the centre at `angle` radians (2-D only)."""
    r = length // 2
    fp = np.zeros((length, length), dtype=bool)
    for t in range(-r, r + 1):
        y = int(round(r - t * np.sin(angle)))
        x = int(round(r + t * np.cos(angle)))
        if 0 <= y < length and 0 <= x < length:
            fp[y, x] = True
    return fp


def line_tophat(img, r, n_angles=4):
    """orientation-invariant line top-hat: max over oriented line SEs of (img - opening),
    so a tubular structure survives at whatever angle it runs. 2-D slices only; falls back
    to the isotropic top-hat on 3-D input."""
    from scipy.ndimage import grey_opening
    if img.ndim != 2:
        return tophat(img, r)
    length, best = 2 * r + 1, None
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        th = np.clip(img - grey_opening(img, footprint=_line_footprint(length, a)), 0, None)
        best = th if best is None else np.maximum(best, th)
    return best


def line_bottomhat(img, r, n_angles=4):
    """orientation-invariant line bottom-hat: max over oriented line SEs of (closing - img),
    for dark tubular structures. 2-D slices only; falls back to the isotropic bottom-hat on 3-D."""
    from scipy.ndimage import grey_closing
    if img.ndim != 2:
        return bottomhat(img, r)
    length, best = 2 * r + 1, None
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        bh = np.clip(grey_closing(img, footprint=_line_footprint(length, a)) - img, 0, None)
        best = bh if best is None else np.maximum(best, bh)
    return best


def band(img, r_lo, r_hi):
    """granulometric band (opening γ): bright structure with scale in (r_lo, r_hi]."""
    return np.clip(opening_of(img, r_lo) - opening_of(img, r_hi), 0, None)


def band_dark(img, r_lo, r_hi):
    """anti-granulometric band (closing φ): dark structure with scale in (r_lo, r_hi]."""
    return np.clip(closing_of(img, r_hi) - closing_of(img, r_lo), 0, None)


#
# slice selection: (axis, index) of the largest-target slice over all axes
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


def align_axes(img, label):
    """Reorder img axes to match the label (medpy permutes the spatial axes of some
    4-D volumes). When several permutations match the shape — e.g. two equal-length
    axes — disambiguate by choosing the one whose label foreground sits on image
    tissue (highest mean intensity), since a wrong swap lands the label on background."""
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
def _auc(pos, neg, max_n=20000):
    """AUC = P(residual on a random fg pixel > a random bg pixel), via Mann-Whitney U.
    Threshold-free discriminability: 0.5 = chance, 1.0 = perfectly separable. Background is
    subsampled for speed."""
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
def render_single(image, th, bh, label, out, show):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    ax[0].imshow(image, cmap="gray"); ax[0].set_title("image")
    ax[1].imshow(th, cmap="magma"); ax[1].set_title("top-hat")
    ax[2].imshow(bh, cmap="magma"); ax[2].set_title("bottom-hat")
    ax[3].imshow(image, cmap="gray"); ax[3].set_title("image + label contour")
    if label is not None and label.max() > 0:
        for a in (ax[1], ax[2], ax[3]):
            a.contour(label, levels=np.arange(0.5, label.max() + 1), colors="cyan", linewidths=0.7)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130, bbox_inches="tight"); print(f"saved figure -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def render_grid(img2d, label2d, radii, bands, out, dark_bands=()):
    import matplotlib.pyplot as plt
    ncol = max(len(radii) + 1, len(bands) + len(dark_bands) + 1)
    fig, ax = plt.subplots(3, ncol, figsize=(3.0 * ncol, 9.2))
    for a in ax.ravel():
        a.axis("off")

    def contour(a):
        if label2d is not None and label2d.max() > 0:
            a.contour(label2d, levels=np.arange(0.5, label2d.max() + 1), colors="cyan", linewidths=0.6)

    ax[0, 0].imshow(img2d, cmap="gray"); ax[0, 0].set_title("image (preprocessed)")
    ax[1, 0].imshow(img2d, cmap="gray"); ax[1, 0].set_title("image + label"); contour(ax[1, 0])
    ax[2, 0].imshow(img2d, cmap="gray"); ax[2, 0].set_title("image")
    for j, r in enumerate(radii, start=1):
        ax[0, j].imshow(tophat(img2d, r), cmap="magma"); ax[0, j].set_title(f"top-hat r={r}"); contour(ax[0, j])
        ax[1, j].imshow(bottomhat(img2d, r), cmap="magma"); ax[1, j].set_title(f"bottom-hat r={r}"); contour(ax[1, j])
    # row 2: bright bands (opening γ) then dark bands (closing φ)
    row2 = [(f"γ{lo}-γ{hi}", band(img2d, lo, hi)) for lo, hi in bands] \
        + [(f"φ{lo}-φ{hi}", band_dark(img2d, lo, hi)) for lo, hi in dark_bands]
    for j, (name, m) in enumerate(row2, start=1):
        if j < ncol:
            ax[2, j].imshow(m, cmap="magma"); ax[2, j].set_title(name); contour(ax[2, j])
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


#
# subcommand: explore (single case)
#
def cmd_explore(args):
    raw = load_any(args.path)
    stacked = raw.ndim >= 3 and raw.shape[0] == 4      # (img, top, bot, label)
    if stacked:
        image, label = raw[0], raw[3]
        th, bh = (None, None) if args.fresh else (raw[1], raw[2])
    else:
        image = raw[0] if (raw.ndim >= 3 and raw.shape[0] == 1) else raw
        label = load_any(args.label) if args.label else None
        th = bh = None

    image = norm01(image)
    if th is None or bh is None:
        th, bh = tophat(image, args.se_radius), bottomhat(image, args.se_radius)

    sel = pick_slice(label if label is not None else image, args.slice)
    print(f"input: {os.path.basename(args.path)}  shape={image.shape}  "
          f"se_radius={args.se_radius}  slice={sel}")
    if label is not None:
        for name, res in (("TOP-HAT", th), ("BOTTOM-HAT", bh)):
            s = region_stats(res, label)
            if s:
                print(f"{name}: enrichment(bnd/bg)={s['enrichment']:.1f}x  "
                      f"concentration(on-target/area)={s['concentration']:.2f}x  "
                      f"%energy on target={s['pct_on_target']:.1f} (area {s['target_area_pct']:.1f}%)")

    render_single(take(image, sel), take(th, sel), take(bh, sel),
                  take(label, sel), args.out, show=not args.no_show)


#
# subcommand: survey (batch ranking) — parallel, optional train-split + all-slices,
# with an auto-selector that emits the --morph-bank spec from the ranking
#
METRICS = ("concentration", "enrichment", "auc", "fisher")   # accumulated per residual


def _survey_case(fn, image_dir, label_dir, mod, channel, n_mod, radii, bands, all_slices):
    """Accumulate the per-residual metric sums over a case's foreground slices.
    Slices spatial axis 0 to match the training loader (NumpyDataLoader always slices axis 0),
    so selection is measured on the exact plane the network trains on. Module-level (picklable)."""
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

    acc = {}   # key -> [conc_sum, enrich_sum, auc_sum, fisher_sum, count]

    def note(key, res, lbl2d):
        s = region_stats(res, lbl2d)
        if s:
            a = acc.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
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
            note(f"ltophat r={r}", line_tophat(img2d, r), lbl2d)         # diagnostic (oriented, bright)
            note(f"lbottomhat r={r}", line_bottomhat(img2d, r), lbl2d)   # diagnostic (oriented, dark)
        for lo, hi in bands:
            note(f"gband {lo}-{hi}", band(img2d, lo, hi), lbl2d)
            note(f"fband {lo}-{hi}", band_dark(img2d, lo, hi), lbl2d)
    return fn, acc, len(idxs)


def _select_spec(rows, k):
    """--morph-bank spec from the ranking: top-k distinct (mode, radius) among the modes that
    have a trainable SoftMorph2D block (tophat/bottomhat/gradient). Bands and line top-hats are
    diagnostic only — bands are redundant (the U-Net synthesises them) and oriented line SEs
    aren't representable by the disk-initialised trainable block."""
    picked = []
    for row in rows:
        key = row["key"]
        for mode in ("tophat", "bottomhat", "gradient"):
            if key.startswith(f"{mode} r=") and len(picked) < k:
                r = int(key.split("r=")[1])
                if (mode, r) not in picked:
                    picked.append((mode, r))
    return ",".join(f"{m}:{r}" for m, r in picked)


def cmd_survey(args):
    modality, _ = load_meta(args.dataset_dir)
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
            print(f"[warn] no splits.pkl in {args.dataset_dir} (run run_preprocessing first for "
                  f"train-only selection); falling back to ALL cases", flush=True)
    if args.n > 0:
        cases = cases[:args.n]

    # visual sanity: render the first --viz cases (one representative slice each)
    for fn in cases[:args.viz]:
        lbl = load_any(os.path.join(lbl_dir, fn)); lbl = lbl[..., 0] if lbl.ndim == 4 else lbl
        img = align_axes(preprocess(load_any(os.path.join(img_dir, fn)), mod, args.channel, len(modality)), lbl)
        sel = pick_slice(lbl)
        render_grid(take(img, sel), take(lbl, sel), args.radii, bands,
                    os.path.join(args.out_dir, f"{task}_{fn.replace('.nii.gz','')}.png"), dark_bands=bands)

    # parallel per-case accumulation over all foreground slices (or one, per --all-slices)
    worker = partial(_survey_case, image_dir=img_dir, label_dir=lbl_dir, mod=mod, channel=args.channel,
                     n_mod=len(modality), radii=args.radii, bands=bands, all_slices=args.all_slices)
    total, n_slices = {}, 0
    print(f"surveying {len(cases)} cases ({tag}, all_slices={args.all_slices}, workers={args.workers}) ...", flush=True)

    def consume(res):
        nonlocal n_slices
        fn, acc, nsl = res
        n_slices += nsl
        for key, vals in acc.items():
            t = total.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0])
            for i in range(5):
                t[i] += vals[i]
        print(f"  {fn}  ({nsl} slices)", flush=True)

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(worker, cases):
                consume(res)
    else:
        for fn in cases:
            consume(worker(fn))

    # per-residual metric means, plus a combined selectivity x discriminability score
    rows = []
    for key, t in total.items():
        if t[4] > 0:
            row = {"key": key, **{m: t[i] / t[4] for i, m in enumerate(METRICS)}}
            row["conc_auc"] = row["concentration"] * row["auc"]
            rows.append(row)
    rows.sort(key=lambda r: r[args.rank_by], reverse=True)
    spec = _select_spec(rows, args.top_k)

    cols = ["concentration", "enrichment", "auc", "fisher", "conc_auc"]
    head = [f"===== SUMMARY  {task}  ({tag}, modality={mod}, {len(cases)} cases, {n_slices} slices, "
            f"rank-by={args.rank_by}) =====",
            f"{'residual':13s} " + " ".join(f"{c:>13s}" for c in cols)]
    for row in rows:
        head.append(f"{row['key']:13s} " + " ".join(f"{row[c]:13.3f}" for c in cols))
    if rows:
        head.append(f">>> best ({args.rank_by}) = {rows[0][args.rank_by]:.3f}  ({rows[0]['key']})")
        head.append(f">>> selected --morph-bank \"{spec}\"")
    print("\n".join(head))
    out_path = os.path.join(args.out_dir, f"{task}_{tag}_stats.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(head) + "\n")

    # machine-readable handoff: fold -> spec, consumed by `train_eval.py --morph-bank auto`
    bank_path = os.path.join(args.out_dir, f"{task}_bank.json")
    bank = {}
    if os.path.exists(bank_path):
        with open(bank_path) as f:
            bank = json.load(f)
    bank[str(args.fold)] = spec
    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"\n[written] {out_path}\n[written] {bank_path}  (fold {args.fold} -> \"{spec}\")")


#
# main
#
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("explore", help="single-case 4-panel + region stats")
    pe.add_argument("path", help="4-channel .npy, or a single image (.npy/.nii/.png/...)")
    pe.add_argument("--label", help="separate label file (when not a 4-channel stack)")
    pe.add_argument("--fresh", action="store_true", help="ignore precomputed channels; recompute morphology")
    pe.add_argument("--se-radius", type=int, default=2)
    pe.add_argument("--slice", type=int, default=None, help="force axis-0 slice (3D)")
    pe.add_argument("--out", default=None, help="save the figure (PNG)")
    pe.add_argument("--no-show", action="store_true")
    pe.set_defaults(func=cmd_explore)

    ps = sub.add_parser("survey", help="batch SE sweep + bands + concentration ranking")
    ps.add_argument("dataset_dir", help="MSD task dir with imagesTr/ labelsTr/ dataset.json")
    ps.add_argument("--n", type=int, default=8, help="cases to score (0 = all)")
    ps.add_argument("--viz", type=int, default=3, help="render panels for the first N cases")
    ps.add_argument("--channel", type=int, default=0, help="modality channel for multi-modal images")
    ps.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3, 5])
    ps.add_argument("--bands", type=int, nargs="+", default=[1, 2, 2, 3, 3, 5], help="flat lo hi lo hi ...")
    ps.add_argument("--split", choices=["all", "train"], default="train",
                    help="'train' (default) = only --fold's training keys (needs splits.pkl), avoids "
                         "test leakage in selection; 'all' = every case (exploration/cross-task ranking)")
    ps.add_argument("--fold", type=int, default=0)
    ps.add_argument("--all-slices", dest="all_slices", action="store_true", default=True,
                    help="score every foreground slice (default)")
    ps.add_argument("--one-slice", dest="all_slices", action="store_false",
                    help="quick: score only the densest foreground slice per case")
    ps.add_argument("--rank-by", choices=["concentration", "auc", "fisher", "conc_auc"],
                    default="conc_auc", help="metric to rank/select by (default: concentration x auc)")
    ps.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    ps.add_argument("--top-k", type=int, default=4, help="how many (mode,radius) to auto-select for --morph-bank")
    ps.add_argument("--out-dir", default="results/explore")
    ps.set_defaults(func=cmd_survey)
    return p


def main():
    import sys
    argv = sys.argv[1:]
    # backward-compat: `morph_explore.py <path> ...` -> `explore <path> ...`
    if argv and argv[0] not in ("explore", "survey", "-h", "--help"):
        argv = ["explore"] + argv
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
