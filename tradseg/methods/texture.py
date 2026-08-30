"""Tier-3 texture segmenters (Ch.13 Maragos): Gabor filterbank and AM-FM / Teager-energy
Dominant Component Analysis.

Both extract per-pixel texture features (2D per slice, optional Perona-Malik denoise),
cluster the body-voxel feature vectors with K-means, and keep the cluster whose mean CT
intensity is closest to the spleen target. The result is a RAW candidate mask; the runner
then localises the spleen component via the spatial prior (auto) / GT overlap (oracle) --
exactly the intensity-method path (requires_seeds=False).

Expectation: the spleen is barely textured on CT and iso-texture with liver/kidney, so
these are weak on their own -- but they are the 'texture school' representatives and the
natural feature source for later hybrids.
"""

import numpy as np

from methods.base import Segmenter, body_mask
import config
import texture_utils as T


class _FeatureCluster(Segmenter):
    """Cluster per-pixel feature vectors, keep the cluster nearest the spleen intensity."""

    tier = 3
    requires_seeds = False

    def __init__(self, k=3, target_intensity=None, andiff=True,
                 max_fit_voxels=100_000, random_state=0):
        self.k = int(k)
        self.target = (config.spleen_target_intensity()
                       if target_intensity is None else float(target_intensity))
        self.andiff = bool(andiff)
        self.max_fit_voxels = int(max_fit_voxels)
        self.random_state = int(random_state)

    def _prep_slice(self, img2d):
        return T.perona_malik(img2d) if self.andiff else img2d.astype(np.float32)

    def features_slice(self, img2d):
        """Return a (F, H, W) feature stack for one slice. OVERRIDE."""
        raise NotImplementedError

    def segment_volume(self, image, seeds=None):
        from sklearn.cluster import KMeans
        body = body_mask(image)
        rows, inten_rows, where = [], [], []
        for z in range(image.shape[0]):
            bz = body[z]
            if not bz.any():
                continue
            fz = self.features_slice(image[z])                     # (F, H, W)
            rows.append(fz.reshape(fz.shape[0], -1)[:, bz.ravel()].T)   # (nz, F)
            inten_rows.append(image[z][bz])
            where.append((z, np.flatnonzero(bz.ravel())))
        if not rows:
            return np.zeros(image.shape, bool)
        X = np.concatenate(rows, 0).astype(np.float64)
        inten = np.concatenate(inten_rows, 0)
        if X.shape[0] < self.k:
            return np.zeros(image.shape, bool)

        # standardize; floor the std so near-constant features contribute ~0 instead of
        # amplified noise (robustness where a feature is flat, e.g. after strong denoise)
        mu = X.mean(0); sd = np.maximum(X.std(0), 1e-3)
        Xs = (X - mu) / sd
        rng = np.random.default_rng(self.random_state)
        fit = Xs if Xs.shape[0] <= self.max_fit_voxels else \
            Xs[rng.choice(Xs.shape[0], self.max_fit_voxels, replace=False)]
        km = KMeans(n_clusters=self.k, n_init=4, random_state=self.random_state).fit(fit)
        labels = km.predict(Xs)

        means = np.array([inten[labels == c].mean() if np.any(labels == c) else np.inf
                          for c in range(self.k)])
        target_c = int(np.argmin(np.abs(means - self.target)))
        sel = labels == target_c

        out = np.zeros(image.shape, bool)
        off = 0
        for (z, flat_idx) in where:
            n = flat_idx.size
            plane = out[z].ravel()
            plane[flat_idx] = sel[off:off + n]
            out[z] = plane.reshape(image.shape[1:])
            off += n
        return out & body


class GaborSegmenter(_FeatureCluster):
    """Gabor filterbank energy features + intensity -> K-means."""

    name = "gabor"

    def __init__(self, frequencies=(0.15, 0.3), n_orient=4, **kw):
        super().__init__(**kw)
        self.bank = T.gabor_bank(frequencies, n_orient)

    def features_slice(self, img2d):
        s = self._prep_slice(img2d)
        gf = T.gabor_features(s, self.bank)
        return np.concatenate([s[None], gf], 0)


class AmFmSegmenter(_FeatureCluster):
    """AM-FM Dominant Component Analysis: per Gabor channel compute the Teager energy,
    keep the dominant (max-energy) channel's amplitude + energy, plus intensity."""

    name = "amfm"

    def __init__(self, frequencies=(0.15, 0.3), n_orient=4, **kw):
        super().__init__(**kw)
        self.bank = T.gabor_bank(frequencies, n_orient)

    def features_slice(self, img2d):
        from skimage.filters import gabor
        s = self._prep_slice(img2d)
        amps, engs = [], []
        for f, t in self.bank:
            re, im = gabor(s, frequency=f, theta=t)
            amp = np.sqrt(re.astype(np.float32) ** 2 + im.astype(np.float32) ** 2)
            amps.append(amp)
            engs.append(T.teager_energy_2d(amp))
        amps = np.stack(amps); engs = np.stack(engs)
        dom = np.argmax(engs, axis=0)
        dom_amp = np.take_along_axis(amps, dom[None], 0)[0]
        dom_eng = np.take_along_axis(engs, dom[None], 0)[0]
        return np.stack([s.astype(np.float32), dom_amp, dom_eng])
