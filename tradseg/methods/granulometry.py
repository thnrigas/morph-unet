"""Tier-3 granulometry segmenter (Ch.13 Maragos): local morphological pattern-spectrum
features (multiscale white/black top-hats) + intensity -> K-means, keep the cluster nearest
the spleen intensity. Reuses the feature-clustering machinery in methods/texture.py.
"""

import numpy as np

from methods.texture import _FeatureCluster
import texture_utils as T


class GranulometrySegmenter(_FeatureCluster):
    name = "granulometry"

    def __init__(self, radii=(1, 2, 4, 8), **kw):
        super().__init__(**kw)
        self.radii = tuple(int(r) for r in radii)

    def features_slice(self, img2d):
        s = self._prep_slice(img2d)
        g = T.granulometry_features(s, self.radii)          # (2*len(radii), H, W)
        return np.concatenate([s[None], g], 0)
