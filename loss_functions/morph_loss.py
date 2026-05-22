#!/usr/bin/env python
"""
Morphological consistency loss : penalises how much soft opening, closing changes the
predicted foreground probabilities. A clean segmentation is near-idempotent under open/close,
so a non-zero value flags speckle (removed by opening) and pinholes (filled by closing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphConsistencyLoss(nn.Module):
    """|p - opening(p)| + |closing(p) - p| over the non-background classes."""

    def __init__(self, weight=0.1, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.weight = weight
        self.k = k
        self.pad = k // 2
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):                       # x: (N,1,H,W) -> (N,1,k*k,H,W)
        N, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(N, C, self.k * self.k, H, W)

    def _dilation(self, x):
        return torch.logsumexp(self.beta * self._neigh(x), dim=2) / self.beta

    def _erosion(self, x):
        return -torch.logsumexp(-self.beta * self._neigh(x), dim=2) / self.beta

    def forward(self, pred_softmax):    # (B, C, H, W), classes 1... are foreground
        fg = pred_softmax[:, 1:]
        B, C, H, W = fg.shape
        x = fg.reshape(B * C, 1, H, W)
        opening = self._dilation(self._erosion(x))
        closing = self._erosion(self._dilation(x))
        consistency = (x - opening).abs().mean() + (x - closing).abs().mean()
        return self.weight * consistency
