"""Tier-1 intensity-clustering baselines: K-means and GMM on the [0,1] CT values.

Cluster the body-voxel intensities into k groups, keep the cluster whose mean is
closest to the spleen target intensity -> a soft-tissue candidate band. Like the
Otsu baselines these cannot separate isointense organs; the runner localises the
spleen among the candidate blobs.
"""

import numpy as np

from methods.base import Segmenter, body_mask
import config


class _IntensityCluster(Segmenter):
    def __init__(self, k=3, target_intensity=None, max_fit_voxels=200_000, random_state=0):
        self.k = int(k)
        self.target = (config.spleen_target_intensity()
                       if target_intensity is None else float(target_intensity))
        self.max_fit_voxels = int(max_fit_voxels)
        self.random_state = int(random_state)

    def _fit(self, fit_vals):
        """Return (centers (k,), predict_fn(values (n,1)) -> labels (n,))."""
        raise NotImplementedError

    def segment_volume(self, image, seeds=None):
        body = body_mask(image)
        coords = np.flatnonzero(body.ravel())
        if coords.size < self.k:
            return np.zeros(image.shape, bool)
        vals = image.ravel()[coords].reshape(-1, 1).astype(np.float64)
        rng = np.random.default_rng(self.random_state)
        fit_vals = vals
        if vals.shape[0] > self.max_fit_voxels:
            sub = rng.choice(vals.shape[0], self.max_fit_voxels, replace=False)
            fit_vals = vals[sub]
        centers, predict = self._fit(fit_vals)
        target_cluster = int(np.argmin(np.abs(centers.ravel() - self.target)))
        labels = predict(vals)
        out = np.zeros(image.size, bool)
        out[coords] = labels == target_cluster
        return out.reshape(image.shape) & body


class KMeansSegmenter(_IntensityCluster):
    name = "kmeans"

    def _fit(self, fit_vals):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.k, n_init=4, random_state=self.random_state).fit(fit_vals)
        return km.cluster_centers_.ravel(), km.predict


class GMMSegmenter(_IntensityCluster):
    name = "gmm"

    def _fit(self, fit_vals):
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=self.k, random_state=self.random_state).fit(fit_vals)
        return gm.means_.ravel(), gm.predict
