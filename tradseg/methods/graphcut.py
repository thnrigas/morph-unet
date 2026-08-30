"""Tier-2 graph cut = GMM data term + contrast-sensitive Potts smoothness, 2D per slice.

Binary s-t min-cut via PyMaxflow (Boykov-Kolmogorov). This IS the GMM + MRF/Potts method:
- data (unary): a GMM fitted on the case's body-voxel intensities; the component nearest
  the spleen target intensity gives P(fg|I), the rest P(bg|I).
- smoothness (pairwise): contrast-sensitive Potts, lam*exp(-(Ii-Ij)^2/(2 sigma^2)).
- hard seeds: fg/bg markers from the runner become infinite-capacity t-links.

Convention: source side = FG, sink side = BG. `get_grid_segments` returns True on the
sink (BG) side, so FG = ~segments. Unary caps: sourcecap = Dbg, sinkcap = Dfg.
"""

import numpy as np

from .base import Segmenter, body_mask
import config

INF = 1e9
EPS = 1e-6


class GraphCutSegmenter(Segmenter):
    name = "graphcut"
    tier = 2
    requires_seeds = True
    seed_type = "markers"

    def __init__(self, lam=2.0, sigma=0.08, gmm_k=3, target_intensity=None,
                 max_fit_voxels=200_000):
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.gmm_k = int(gmm_k)
        self.target = (config.spleen_target_intensity()
                       if target_intensity is None else float(target_intensity))
        self.max_fit_voxels = int(max_fit_voxels)

    def _fit_gmm(self, image, body):
        from sklearn.mixture import GaussianMixture
        vals = image[body].reshape(-1, 1).astype(np.float64)
        if vals.shape[0] < self.gmm_k:
            return None, None
        if vals.shape[0] > self.max_fit_voxels:
            sub = np.random.default_rng(0).choice(vals.shape[0], self.max_fit_voxels, replace=False)
            fit_vals = vals[sub]
        else:
            fit_vals = vals
        gm = GaussianMixture(n_components=self.gmm_k, random_state=0).fit(fit_vals)
        fg_comp = int(np.argmin(np.abs(gm.means_.ravel() - self.target)))
        return gm, fg_comp

    def _unary(self, gm, fg_comp, img2d):
        proba = gm.predict_proba(img2d.reshape(-1, 1).astype(np.float64))
        pfg = np.clip(proba[:, fg_comp].reshape(img2d.shape), EPS, 1.0 - EPS)
        dfg = -np.log(pfg)            # cost of labeling FG
        dbg = -np.log(1.0 - pfg)      # cost of labeling BG
        return dfg, dbg

    def _cut_slice(self, img2d, fg2d, bg2d, body2d, gm, fg_comp):
        import maxflow
        h, w = img2d.shape
        dfg, dbg = self._unary(gm, fg_comp, img2d)
        g = maxflow.GraphFloat()
        nodeids = g.add_grid_nodes((h, w))

        sig2 = 2.0 * self.sigma * self.sigma
        # right neighbours
        wr = self.lam * np.exp(-((img2d - np.roll(img2d, -1, axis=1)) ** 2) / sig2)
        wr[:, -1] = 0.0
        g.add_grid_edges(nodeids, weights=wr,
                         structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
        # down neighbours
        wd = self.lam * np.exp(-((img2d - np.roll(img2d, -1, axis=0)) ** 2) / sig2)
        wd[-1, :] = 0.0
        g.add_grid_edges(nodeids, weights=wd,
                         structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)

        src = dbg.astype(np.float64).copy()      # sourcecap paid if node is BG (sink side)
        snk = dfg.astype(np.float64).copy()      # sinkcap paid if node is FG (source side)
        src[fg2d] = INF; snk[fg2d] = 0.0         # hard FG -> stays on source side
        snk[bg2d] = INF; src[bg2d] = 0.0         # hard BG -> stays on sink side
        outside = ~body2d
        snk[outside] = INF; src[outside] = 0.0   # out-of-body -> BG
        g.add_grid_tedges(nodeids, src, snk)

        g.maxflow()
        seg_sink = g.get_grid_segments(nodeids)  # True = sink = BG
        return (~seg_sink) & body2d

    def segment_volume(self, image, seeds=None):
        s = seeds or {}
        fg = s.get("fg")
        if fg is None or not np.any(fg):
            return np.zeros(image.shape, bool)
        fg = np.asarray(fg, bool)
        bg = np.asarray(s.get("bg"), bool) if s.get("bg") is not None else np.zeros_like(fg)
        body = body_mask(image)
        gm, fg_comp = self._fit_gmm(image, body)
        if gm is None:
            return np.zeros(image.shape, bool)
        out = np.zeros(image.shape, bool)
        for z in range(image.shape[0]):
            if not fg[z].any():
                continue
            out[z] = self._cut_slice(image[z].astype(np.float64), fg[z], bg[z], body[z], gm, fg_comp)
        return out
