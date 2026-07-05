#
# Morphological Block
#
# x - opening(x) (top-hat), closing(x) - x (bottom-hat) with a trainable structuring element (SE)
# we add these operations as extra input channel in the vanilla U-Net for it to learn additional information
# we approximate dilation and erosion via logsumexp as hard min/max operations cause gradients
# to become zero and backpropagation not to work (the dead gradient problem)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Differentiable top-hat or bottom-hat with a learnable structuring element
#
class SoftMorph2D(nn.Module):

    def __init__(self, k=5, beta=10.0, mode="tophat", init_radius=None, init_inside=0.5, init_outside=0.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        assert mode in ("tophat", "bottomhat", "gradient"), "mode must be tophat/bottomhat/gradient"
        self.k = k
        self.pad = k // 2
        self.mode = mode
        # SE = one weight per offset in the k x k window.
        #   init_radius=None -> flat SE (zeros);
        #   init_radius=r    -> `init_inside` in a radius-r disk, `init_outside` elsewhere.
        # the block starts as a radius-r disk in a larger window; only the inside-outside gap
        # matters (top-hat is shift-invariant), and a ~0.5 gap keeps the outside able to grow.
        se0 = (torch.zeros(k * k) if init_radius is None
               else self._disk_se(k, init_radius, init_inside, init_outside))
        self.se = nn.Parameter(se0.view(1, 1, k * k, 1, 1))
        self.register_buffer("beta", torch.tensor(float(beta)))

    @staticmethod
    def _disk_se(k, radius, inside=0.5, outside=0.0):
        """(k*k,) SE: `inside` in a centred radius-`radius` disk, `outside` elsewhere."""
        r = k // 2
        radius = min(radius, r)                       # disk must fit the window
        offs = torch.arange(k) - r
        dr, dc = torch.meshgrid(offs, offs, indexing="ij")
        disk = (dr ** 2 + dc ** 2) <= radius ** 2
        se = torch.full((k, k), float(outside))
        se[disk] = float(inside)
        return se.reshape(-1)

    def _neigh(self, x):
        # x: (B,1,H,W) -> (B,1,k*k,H,W) (local neighbourhoods)
        B, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(B, C, self.k * self.k, H, W)

    # soft dilation = (1/b) * logsumexp(b * (neigh + SE))
    def soft_dilation(self, x):
        n = self._neigh(x) + self.se
        return torch.logsumexp(self.beta * n, dim=2) / self.beta

    # soft erosion = -(1/b) * logsumexp(-b * (neigh - SE))
    def soft_erosion(self, x):
        n = self._neigh(x) - self.se
        return -torch.logsumexp(-self.beta * n, dim=2) / self.beta

    def set_beta(self, beta):
        # anneal the log-sum-exp temperature: higher beta -> sharper (more faithful) morphology
        self.beta.fill_(float(beta))

    def forward(self, x):
        if self.mode == "tophat":
            # top-hat : x - opening(x), opening = dilation(erosion(x))
            opened = self.soft_dilation(self.soft_erosion(x))
            return torch.clamp(x - opened, min=0.0)
        if self.mode == "gradient":
            # morphological gradient : dilation(x) - erosion(x) (boundary-selective, >= 0)
            return torch.clamp(self.soft_dilation(x) - self.soft_erosion(x), min=0.0)
        # bottom-hat : closing(x) - x, closing = erosion(dilation(x))
        closed = self.soft_erosion(self.soft_dilation(x))
        return torch.clamp(closed - x, min=0.0)

    @torch.no_grad()
    def learned_se(self):
        # learned structuring element as a (k, k) numpy array
        return self.se.detach().cpu().view(self.k, self.k).numpy()

#
# Morphological U-Net
# prepends a learnable top-hat and/or bottom-hat channel to the image
# and feeds it/them to unet (in_channels = 1 + #residuals)
#
class MorphResidualUNet(nn.Module):

    def __init__(self, base_unet, k=5, beta=10.0, use_tophat=True, use_bottomhat=False):
        super().__init__()
        assert use_tophat or use_bottomhat, "enable at least one residual"
        self.tophat = SoftMorph2D(k=k, beta=beta, mode="tophat") if use_tophat else None
        self.bottomhat = SoftMorph2D(k=k, beta=beta, mode="bottomhat") if use_bottomhat else None
        self.unet = base_unet

    def set_beta(self, beta):
        for blk in (self.tophat, self.bottomhat):
            if blk is not None:
                blk.set_beta(beta)

    def forward(self, x):
        # morph residuals are computed from the image channel only (ch 0); static filter
        # channels pass through to the UNet unchanged alongside the morph outputs
        img = x[:, :1]                    # (B, 1, H, W) — the image
        chans = [x]                       # all input channels (image + any static filters)
        if self.tophat is not None:
            chans.append(self.tophat(img))
        if self.bottomhat is not None:
            chans.append(self.bottomhat(img))
        return self.unet(torch.cat(chans, dim=1))


#
# Morphological Bank U-Net
# prepends a bank of trainable-SE residual channels, seeded from survey-picked
# (polarity, radius) specs, then feeds image + bank to the U-Net.
# every block shares a fixed k_max window; each SE is disk-initialised at its survey
# radius (0 inside / init_outside elsewhere), so blocks start diverse and can still grow.
# specs: list of (mode, radius), e.g. [("tophat", 3), ("bottomhat", 1)]
#
class MorphBankUNet(nn.Module):

    def __init__(self, base_unet, specs, k_max=11, beta=10.0, init_inside=0.5, init_outside=0.0):
        super().__init__()
        assert specs, "need at least one (mode, radius) spec"
        self.blocks = nn.ModuleList([
            SoftMorph2D(k=k_max, beta=beta, mode=mode, init_radius=r,
                        init_inside=init_inside, init_outside=init_outside)
            for mode, r in specs])
        self.unet = base_unet

    def set_beta(self, beta):
        for blk in self.blocks:
            blk.set_beta(beta)

    def forward(self, x):
        # morph bank residuals from image channel only (ch 0); static channels pass through
        img = x[:, :1]
        chans = [x] + [blk(img) for blk in self.blocks]
        return self.unet(torch.cat(chans, dim=1))


#
# Matched control: a learned conv front-end producing the SAME number of extra
# channels at matched parameter count (conv k*k weights ~ SE k*k weights), so a
# comparison isolates morphology's rank/shape selectivity from "extra channels".
#
class ConvBankUNet(nn.Module):

    def __init__(self, base_unet, n_extra, k=5, in_channels=1):
        super().__init__()
        self.front = nn.Conv2d(in_channels, n_extra, kernel_size=k, padding=k // 2)
        self.act = nn.ReLU(inplace=True)
        self.unet = base_unet

    def forward(self, x):
        # conv front-end reads all input channels; output is cat'd alongside x
        extra = self.act(self.front(x))
        return self.unet(torch.cat([x, extra], dim=1))
