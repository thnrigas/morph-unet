"""Per-volume segmentation metrics, reproducing the baseline U-Net's conventions.

The baseline (my_unet-uncertainty/evaluation) scores each case as a whole volume,
in the preprocessed 256x256 space, WITHOUT voxel spacing (so surface distances are
in resized-voxel units), for the foreground label only, and averages across cases
with nanmean. We reproduce those conventions here (standalone) so our numbers are
directly comparable to the U-Net's.

Reproduced NaN handling:
  * Dice/Jaccard: NaN iff BOTH prediction and reference are empty (dropped by
    nanmean). Empty prediction vs non-empty reference -> 0 (penalised).
  * HD/HD95/ASD/ASSD: NaN if prediction OR reference is empty or full.
"""

import numpy as np


def _counts(pred, ref):
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    return tp, fp, fn


def dice(pred, ref):
    tp, fp, fn = _counts(pred, ref)
    if tp + fp + fn == 0:            # both empty
        return float("nan")
    return 2.0 * tp / (2.0 * tp + fp + fn)


def jaccard(pred, ref):
    tp, fp, fn = _counts(pred, ref)
    if tp + fp + fn == 0:
        return float("nan")
    return tp / (tp + fp + fn)


def precision(pred, ref):
    tp, fp, fn = _counts(pred, ref)
    if tp + fp == 0:
        return float("nan")
    return tp / (tp + fp)


def recall(pred, ref):
    tp, fp, fn = _counts(pred, ref)
    if tp + fn == 0:
        return float("nan")
    return tp / (tp + fn)


def _surface(name, pred, ref, spacing):
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if (not pred.any()) or pred.all() or (not ref.any()) or ref.all():
        return float("nan")
    try:
        from medpy.metric.binary import hd, hd95, asd, assd
    except Exception as exc:  # pragma: no cover - depends on env
        raise ImportError("medpy is required for surface metrics: pip install medpy") from exc
    fn = {"HD": hd, "HD95": hd95, "ASD": asd, "ASSD": assd}[name]
    return float(fn(pred, ref, voxelspacing=spacing))


def evaluate_case(pred_vol, gt_vol, foreground_label=1, spacing=None, advanced=True):
    """Metrics for one case. gt_vol is the integer label volume; the foreground
    label is compared, everything else is background."""
    ref = np.asarray(gt_vol) == foreground_label
    pred = np.asarray(pred_vol).astype(bool)
    res = {
        "Dice": dice(pred, ref),
        "Jaccard": jaccard(pred, ref),
        "Precision": precision(pred, ref),
        "Recall": recall(pred, ref),
    }
    if advanced:
        res["HD95"] = _surface("HD95", pred, ref, spacing)
        res["ASSD"] = _surface("ASSD", pred, ref, spacing)
    return res


def aggregate(case_scores):
    """case_scores: dict[name] -> dict[metric] -> value.
    Returns {'per_case': ..., 'mean': {metric: {'mean':, 'std':, 'n':}}}."""
    metrics = sorted({m for d in case_scores.values() for m in d})
    mean = {}
    for m in metrics:
        vals = np.array([case_scores[c].get(m, np.nan) for c in case_scores], dtype=float)
        finite = np.isfinite(vals)
        if finite.any():
            mean[m] = {"mean": float(np.nanmean(vals)),
                       "std": float(np.nanstd(vals)),
                       "n": int(finite.sum())}
        else:
            mean[m] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"per_case": case_scores, "mean": mean}
