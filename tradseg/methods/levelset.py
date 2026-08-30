"""Tier-2 morphological level sets: Chan-Vese (region) and Geodesic AC (edge).

Both are seeded via the fg marker as the INITIAL level set (auto: spatial-prior blob;
oracle: eroded GT), evolved **2D per slice**, then the component overlapping the init is
kept. Region-based ACWE has no edge stop → it may leak into isointense neighbours;
edge-based GAC balloons out to gradient edges → stops at the spleen boundary. Connects to
Ch.17 (active contours / level sets / morphological curvature flows).

Uses skimage's morphological approximations (no PDE numerics), which is exactly the
morphology↔level-set bridge from the course.
"""

import numpy as np

from .base import Segmenter, body_mask
import postprocess as P


def _keep_init_component(region2d, init2d):
    """Keep the connected component of the evolved region that overlaps the init."""
    region2d = np.asarray(region2d, bool)
    if not region2d.any() or not init2d.any():
        return np.zeros_like(region2d, bool)
    return P.select_component_by_overlap(region2d, init2d)


class ChanVeseSegmenter(Segmenter):
    """Morphological Chan-Vese (Active Contours Without Edges), region-based."""

    name = "chanvese"
    tier = 2
    requires_seeds = True
    seed_type = "markers"

    def __init__(self, iterations=None, smoothing=2, lambda1=1.0, lambda2=1.0):
        self.iterations = 40 if iterations is None else int(iterations)
        self.smoothing = int(smoothing)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def segment_volume(self, image, seeds=None):
        from skimage.segmentation import morphological_chan_vese
        fg = (seeds or {}).get("fg")
        if fg is None or not np.any(fg):
            return np.zeros(image.shape, bool)
        fg = np.asarray(fg, bool)
        out = np.zeros(image.shape, bool)
        for z in range(image.shape[0]):
            init = fg[z]
            if not init.any():
                continue
            ls = morphological_chan_vese(image[z].astype(np.float32), self.iterations,
                                         init_level_set=init.astype(np.uint8),
                                         smoothing=self.smoothing,
                                         lambda1=self.lambda1, lambda2=self.lambda2)
            out[z] = _keep_init_component(ls.astype(bool), init)
        return out & body_mask(image)


class MorphGACSegmenter(Segmenter):
    """Morphological Geodesic Active Contour, edge-based (balloon toward edges)."""

    name = "morphgac"
    tier = 2
    requires_seeds = True
    seed_type = "markers"

    def __init__(self, iterations=None, smoothing=1, balloon=1.0, threshold="auto",
                 alpha=100.0, sigma=2.0):
        self.iterations = 60 if iterations is None else int(iterations)
        self.smoothing = int(smoothing)
        self.balloon = float(balloon)
        self.threshold = threshold
        self.alpha = float(alpha)
        self.sigma = float(sigma)

    def segment_volume(self, image, seeds=None):
        from skimage.segmentation import (morphological_geodesic_active_contour,
                                          inverse_gaussian_gradient)
        fg = (seeds or {}).get("fg")
        if fg is None or not np.any(fg):
            return np.zeros(image.shape, bool)
        fg = np.asarray(fg, bool)
        out = np.zeros(image.shape, bool)
        for z in range(image.shape[0]):
            init = fg[z]
            if not init.any():
                continue
            gimg = inverse_gaussian_gradient(image[z].astype(np.float32),
                                             alpha=self.alpha, sigma=self.sigma)
            ls = morphological_geodesic_active_contour(
                gimg, self.iterations, init_level_set=init.astype(np.uint8),
                smoothing=self.smoothing, threshold=self.threshold, balloon=self.balloon)
            out[z] = _keep_init_component(ls.astype(bool), init)
        return out & body_mask(image)
