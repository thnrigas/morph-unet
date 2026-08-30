"""Tier-1 intensity-threshold baselines: Otsu and multi-Otsu.

Both emit a RAW soft-tissue candidate mask on the body voxels; picking the spleen
among the candidate blobs (largest CC / spatial prior / seed) is the runner's job.
These are deliberately weak baselines: on CT the spleen is nearly isointense with
liver/kidney/muscle, so intensity alone cannot separate them -- that is the point.
"""

import numpy as np

from methods.base import Segmenter, apply_per_slice, body_mask
import config


class OtsuSegmenter(Segmenter):
    name = "otsu"

    def __init__(self, per_slice=False):
        self.per_slice = per_slice

    def _seg(self, arr):
        from skimage.filters import threshold_otsu
        body = body_mask(arr)
        vals = arr[body]
        if vals.size < 2 or vals.min() == vals.max():
            return np.zeros(arr.shape, bool)
        t = threshold_otsu(vals)
        return (arr > t) & body

    def segment_volume(self, image, seeds=None):
        return apply_per_slice(self._seg, image) if self.per_slice else self._seg(image)


class MultiOtsuSegmenter(Segmenter):
    """Multi-Otsu into `classes` intensity bands; keep the band that contains the
    spleen target intensity (~0.52 in the fixed [0,1] window)."""

    name = "multiotsu"

    def __init__(self, classes=3, target_intensity=None, per_slice=False):
        self.classes = classes
        self.target = (config.spleen_target_intensity()
                       if target_intensity is None else float(target_intensity))
        self.per_slice = per_slice

    def _seg(self, arr):
        from skimage.filters import threshold_multiotsu
        body = body_mask(arr)
        vals = arr[body]
        if vals.size < self.classes or np.unique(vals).size < self.classes:
            return np.zeros(arr.shape, bool)
        try:
            thr = threshold_multiotsu(vals, classes=self.classes)
        except ValueError:
            return np.zeros(arr.shape, bool)
        band = int(np.digitize(self.target, thr))       # 0..len(thr)
        regions = np.digitize(arr, thr)
        return (regions == band) & body

    def segment_volume(self, image, seeds=None):
        return apply_per_slice(self._seg, image) if self.per_slice else self._seg(image)
