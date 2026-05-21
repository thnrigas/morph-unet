#!/usr/bin/env python
"""
Morphological block : top-hat, bottom-hat with a trainable structuring element (SE),
erosion and dilation approximated via LogSumExp so backpropagation works (dead gradient problem).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMorph2D(nn.Module):
    """Differentiable top-hat or bottom-hat with a learnable structuring element"""

    def __init__(self, k=5, beta=10.0, mode="tophat"):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        assert mode in ("tophat", "bottomhat"), "mode must be 'tophat' or 'bottomhat'"
        self.k = k
        self.pad = k // 2
        self.mode = mode
        # one weight per offset in the k x k window; zero init -> flat SE
        self.se = nn.Parameter(torch.zeros(1, 1, k * k, 1, 1))
        # fixed LSE temperature (buffer -> moves with .to(device))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):                    # x: (B,1,H,W)
        # (B,1,k*k,H,W) local neighbourhoods
        B, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(B, C, self.k * self.k, H, W)

    def soft_dilation(self, x):
        # soft dilation = (1/b) * logsumexp(b * (neigh + SE))
        n = self._neigh(x) + self.se
        return torch.logsumexp(self.beta * n, dim=2) / self.beta

    def soft_erosion(self, x):
        # soft erosion = -(1/b) * logsumexp(-b * (neigh - SE))
        n = self._neigh(x) - self.se
        return -torch.logsumexp(-self.beta * n, dim=2) / self.beta

    def forward(self, x):
        if self.mode == "tophat":
            # white top-hat: x - opening(x), opening = dilation(erosion(x))
            opened = self.soft_dilation(self.soft_erosion(x))
            return torch.clamp(x - opened, min=0.0)
        # bottom-hat: closing(x) - x, closing = erosion(dilation(x))
        closed = self.soft_erosion(self.soft_dilation(x))
        return torch.clamp(closed - x, min=0.0)

    @torch.no_grad()
    def learned_se(self):
        """Learned structuring element as a (k, k) numpy array"""
        return self.se.detach().cpu().view(self.k, self.k).numpy()


class MorphResidualUNet(nn.Module):
    """
    Morphological front-end + U-Net. Prepends a learnable top-hat and/or bottom-hat channel to the image and
    feeds it/them to unet (in_channels = 1 + #residuals).
    """

    def __init__(self, base_unet, k=5, beta=10.0, use_tophat=True, use_bottomhat=False):
        super().__init__()
        assert use_tophat or use_bottomhat, "enable at least one residual"
        self.tophat = SoftMorph2D(k=k, beta=beta, mode="tophat") if use_tophat else None
        self.bottomhat = SoftMorph2D(k=k, beta=beta, mode="bottomhat") if use_bottomhat else None
        self.unet = base_unet

    def forward(self, x):
        chans = [x]
        if self.tophat is not None:
            chans.append(self.tophat(x))
        if self.bottomhat is not None:
            chans.append(self.bottomhat(x))
        return self.unet(torch.cat(chans, dim=1))
