"""Tier-1 seeded region growing (intensity flood fill from seed points).

Grows connected regions whose intensity is within `tolerance` of the seed value.
Seeds come from the runner: AUTO -> spatial-prior peak, ORACLE -> deepest GT voxel.
Leakage into isointense neighbours (liver/kidney) is the expected failure mode.
"""

import numpy as np

from .base import Segmenter, body_mask


class RegionGrowingSegmenter(Segmenter):
    name = "region_growing"
    requires_seeds = True
    seed_type = "points"

    def __init__(self, tolerance=0.06, connectivity=1):
        self.tolerance = float(tolerance)
        self.connectivity = int(connectivity)

    def segment_volume(self, image, seeds=None):
        from skimage.segmentation import flood
        points = (seeds or {}).get("points") or []
        if not points:
            return np.zeros(image.shape, bool)
        body = body_mask(image)
        out = np.zeros(image.shape, bool)
        for pt in points:
            out |= flood(image, seed_point=tuple(pt),
                         connectivity=self.connectivity, tolerance=self.tolerance)
        return out & body
