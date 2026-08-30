"""Tier-1 marker-controlled watershed on the gradient-magnitude relief.

Foreground/background markers come from the runner (AUTO -> spatial prior,
ORACLE -> GT). The watershed floods the gradient image from the markers; the basin
grown from the fg marker is the segmentation. This is the morphology/eikonal-flavour
baseline (connects to the Ch.17 watershed <-> eikonal discussion).
"""

import numpy as np
from scipy import ndimage as ndi

from .base import Segmenter, body_mask


class WatershedSegmenter(Segmenter):
    name = "watershed"
    requires_seeds = True
    seed_type = "markers"

    def __init__(self, gradient_sigma=1.0):
        self.gradient_sigma = float(gradient_sigma)

    def segment_volume(self, image, seeds=None):
        from skimage.segmentation import watershed
        s = seeds or {}
        fg = s.get("fg")
        bg = s.get("bg")
        if fg is None or not np.any(fg):
            return np.zeros(image.shape, bool)
        relief = ndi.gaussian_gradient_magnitude(image.astype(np.float32), self.gradient_sigma)
        markers = np.zeros(image.shape, np.int32)
        if bg is not None:
            markers[np.asarray(bg, bool)] = 1
        markers[np.asarray(fg, bool)] = 2
        labels = watershed(relief, markers)
        return (labels == 2) & body_mask(image)
