"""Tier-2 random walker (Grady), 2D per slice, seeded with fg/bg markers.

Solves the combinatorial Dirichlet problem: each unlabeled pixel is assigned to the
marker (fg/bg) its random walk is most likely to reach first, with transition weights
governed by intensity gradients (beta). Robust seeded method, no new dependency.
"""

import numpy as np

from .base import Segmenter, body_mask


class RandomWalkerSegmenter(Segmenter):
    name = "random_walker"
    tier = 2
    requires_seeds = True
    seed_type = "markers"

    def __init__(self, beta=130.0, tol=1e-3):
        self.beta = float(beta)
        self.tol = float(tol)

    def segment_volume(self, image, seeds=None):
        from skimage.segmentation import random_walker
        s = seeds or {}
        fg = s.get("fg")
        if fg is None or not np.any(fg):
            return np.zeros(image.shape, bool)
        fg = np.asarray(fg, bool)
        bg = np.asarray(s.get("bg"), bool) if s.get("bg") is not None else np.zeros_like(fg)
        body = body_mask(image)
        out = np.zeros(image.shape, bool)
        for z in range(image.shape[0]):
            if not fg[z].any():
                continue
            markers = np.zeros(image[z].shape, np.int32)
            markers[bg[z]] = 1
            markers[fg[z]] = 2
            if not (markers == 1).any():           # need a bg seed -> use the image border
                markers[0, :] = 1; markers[-1, :] = 1; markers[:, 0] = 1; markers[:, -1] = 1
                markers[fg[z]] = 2
            if (markers == 0).sum() == 0:           # nothing left to label
                out[z] = fg[z] & body[z]
                continue
            try:
                lab = random_walker(image[z].astype(np.float64), markers,
                                    beta=self.beta, tol=self.tol, mode="bf")
                out[z] = (lab == 2) & body[z]
            except Exception:
                out[z] = fg[z] & body[z]
        return out
