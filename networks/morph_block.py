#
# Morphological filter bank (unified, trainable soft-morphology)
#
# One trainable bank spanning the full survey library, built from two soft primitives --
# soft dilation and soft erosion (logsumexp at temperature beta, annealed during training to
# avoid the dead-gradient problem). SEs are learnable; reconstruction / dome ops also learn h.
# Three tiers of operators:
#
#   exact          tophat / bottomhat / gradient / opening, ASF, oriented (line) tophat.
#                  fixed-depth compositions of dilate/erode; no approximation.
#   reconstruction hdome, reconstruction top-hat. geodesic reconstruction iterates to
#                  stability; we TRUNCATE to a few soft iterations (partial reconstruction).
#   surrogate      vdome. area-opening (a connected-component op) has no differentiable
#                  relaxation, so its size gate is approximated by a learnable granulometric
#                  opening -- the union of openings by a disk + oriented lines.
#
# SE initialisation is by ROLE (the crucial detail from the SoftMorph2D -> soft-bank merge):
#   * growable grey residual SEs  (tophat/bottomhat/gradient/asf, recon-tophat's erosion)
#       -> disk, 0.5 inside / 0.0 outside. the whole window participates so the SE can grow
#          spatially, and erode/dilate cancel the 0.5 inflation. identical to the original
#          MorphBankUNet, so existing flat-op (tophat/bottomhat/gradient) runs reproduce.
#   * faithful flat support SEs, where the support must EXCLUDE the rest of the window for a
#     correct max/min (reconstruction geodesic connectivity; vdome's granulometric size gate;
#     oriented lines, whose orientation must be preserved)
#       -> 0.0 on support / -1e4 off it. a soft/inflated SE here floods the reconstruction to
#          the mask and outputs zeros, and washes out the shape selectivity of the size gate.
#
# Every op consumes a single-channel image (B,1,H,W) and returns one channel; MorphBankUNet
# concatenates them as extra input channels. ConvBankUNet is the parameter-matched control.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


#
# structuring-element initialisers  (a (k*k,) additive grey structuring function)
#
_NEG = 1e4

def _offsets(k):
    r = k // 2
    o = torch.arange(k) - r
    dy, dx = torch.meshgrid(o, o, indexing="ij")
    return dy.reshape(-1).float(), dx.reshape(-1).float()

def disk_se(k, radius, inside=0.5, outside=0.0):
    # growable grey SE (exact ops): `inside` on a centred radius-`radius` disk, `outside`
    # elsewhere. with outside=0 the whole window participates and the SE can grow spatially.
    dy, dx = _offsets(k)
    se = torch.full_like(dy, float(outside))
    se[(dy ** 2 + dx ** 2) <= float(radius) ** 2] = float(inside)
    return se

def hard_flat_se(k, radius):
    # faithful flat SE: 0 on the radius-`radius` disk, -1e4 off it (excluded from both the max
    # of dilation and the min of erosion). used where the support must be exact.
    return disk_se(k, radius, inside=0.0, outside=-_NEG)

def line_se(k, angle):
    # faithful flat line SE through the centre at `angle` (radians): 0 within 0.5 of the line,
    # -1e4 elsewhere -- a hard support so the orientation is preserved.
    dy, dx = _offsets(k)
    perp = (dx * math.sin(angle) - dy * math.cos(angle)).abs()
    se = torch.full_like(dy, -_NEG)
    se[perp <= 0.5] = 0.0
    return se

def parabolic_se(k, sigma=None):
    # b(d) = -||d||^2/(2 sigma^2): the morphological analogue of a Gaussian, dense gradient.
    if sigma is None:
        sigma = max(k // 2, 1) / 2.0
    dy, dx = _offsets(k)
    return -((dy ** 2 + dx ** 2) / (2.0 * sigma ** 2))

def elementary_se(k):
    # radius-1 hard flat SE: the fixed geodesic connectivity used by reconstruction.
    return hard_flat_se(k, 1)

def _inv_softplus(y):
    # so softplus(raw) == y at init (keeps a learnable non-negative h)
    return math.log(math.expm1(float(y)))

def _make_se(k, init="disk", radius=None, angle=0.0):
    if init == "line":
        se = line_se(k, angle)
    elif init == "parabolic":
        se = parabolic_se(k)
    elif init == "hard":
        se = hard_flat_se(k, radius if radius is not None else k // 2)
    else:                                        # "disk" -> growable 0.5/0 (exact ops)
        se = disk_se(k, radius if radius is not None else k // 2)
    return se.view(1, 1, k * k, 1, 1)

#
# base: the soft dilation / erosion primitives + opening / closing / reconstruction
#
class _SoftMorph(nn.Module):

    def __init__(self, k, beta):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.pad = k // 2
        self.register_buffer("beta", torch.tensor(float(beta)))

    def set_beta(self, beta):
        # anneal the log-sum-exp temperature: higher beta -> sharper (more faithful) morphology
        self.beta.fill_(float(beta))

    def _neigh(self, x):
        # (B,1,H,W) -> (B,1,k*k,H,W)
        B, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(B, C, self.k * self.k, H, W)

    def dilate(self, x, se, debias=False):
        n = self._neigh(x) + se
        out = torch.logsumexp(self.beta * n, dim=2) / self.beta
        if debias:
            # logsumexp overshoots max by up to log(n_support)/beta; subtract it so the soft
            # dilation approximates the true max (needed inside reconstruction, where an
            # inflated dilation floods the marker to the mask and kills the residual).
            n_on = (se > -1e3).float().sum(dim=2).clamp(min=1.0)
            out = out - torch.log(n_on) / self.beta
        return out

    def erode(self, x, se):
        n = self._neigh(x) - se
        return -torch.logsumexp(-self.beta * n, dim=2) / self.beta

    def opening(self, x, se):
        return self.dilate(self.erode(x, se), se)

    def closing(self, x, se):
        return self.erode(self.dilate(x, se), se)

    def reconstruct(self, marker, mask, se, iters):
        # geodesic reconstruction by dilation, TRUNCATED to `iters` soft steps (approx.)
        g = marker
        for _ in range(iters):
            g = torch.minimum(self.dilate(g, se, debias=True), mask)   # pointwise min differentiable a.e.
        return g

#
# tier 1: exact ops (fixed-depth compositions, no approximation)
#
class Tophat(_SoftMorph):
    def __init__(self, k, beta, init="disk", radius=None, angle=0.0):
        super().__init__(k, beta)
        self.se = nn.Parameter(_make_se(k, init, radius, angle))

    def forward(self, x):
        return (x - self.opening(x, self.se)).clamp(min=0.0)

    @torch.no_grad()
    def learned_se(self):
        return self.se.detach().cpu().view(self.k, self.k).numpy()

class Bottomhat(_SoftMorph):
    def __init__(self, k, beta, init="disk", radius=None):
        super().__init__(k, beta)
        self.se = nn.Parameter(_make_se(k, init, radius))

    def forward(self, x):
        return (self.closing(x, self.se) - x).clamp(min=0.0)

class Gradient(_SoftMorph):
    def __init__(self, k, beta, init="disk", radius=None):
        super().__init__(k, beta)
        self.se = nn.Parameter(_make_se(k, init, radius))

    def forward(self, x):
        return (self.dilate(x, self.se) - self.erode(x, self.se)).clamp(min=0.0)

class ASF(_SoftMorph):
    # alternating sequential filter: open-then-close at increasing scales, one learnable SE per
    # scale (radii 1..n_scales). residual=True -> ASF top-hat (x - ASF).
    def __init__(self, k, beta, n_scales=3, residual=False):
        super().__init__(k, beta)
        self.ses = nn.ParameterList(
            [nn.Parameter(_make_se(k, "disk", radius=r)) for r in range(1, n_scales + 1)])
        self.residual = residual

    def forward(self, x):
        out = x
        for se in self.ses:
            out = self.closing(self.opening(out, se), se)
        return (x - out).clamp(min=0.0) if self.residual else out

#
# tier 2: truncated soft reconstruction (approximate)
#
class HDome(_SoftMorph):
    # h-dome = x - reconstruct(x - h, x). learnable h (>=0 via softplus); the geodesic
    # connectivity is a FIXED elementary SE (standard reconstruction).
    def __init__(self, k, beta, h_init=0.1, iters=5):
        super().__init__(k, beta)
        self.iters = iters
        self.raw_h = nn.Parameter(torch.tensor(_inv_softplus(h_init)))
        self.register_buffer("se_recon", elementary_se(k).view(1, 1, k * k, 1, 1))

    def forward(self, x):
        h = F.softplus(self.raw_h)
        rec = self.reconstruct(x - h, x, self.se_recon, self.iters)
        return (x - rec).clamp(min=0.0)

class ReconTophat(_SoftMorph):
    # reconstruction top-hat = x - reconstruct(erode(x), x). learnable (growable) erosion SE
    # sets the scale removed; fixed elementary geodesic SE regrows the survivors.
    def __init__(self, k, beta, radius=3, iters=5):
        super().__init__(k, beta)
        self.iters = iters
        self.se_erode = nn.Parameter(_make_se(k, "disk", radius=radius))   # growable
        self.register_buffer("se_recon", elementary_se(k).view(1, 1, k * k, 1, 1))

    def forward(self, x):
        marker = self.erode(x, self.se_erode)
        rec = self.reconstruct(marker, x, self.se_recon, self.iters)
        return (x - rec).clamp(min=0.0)

class Leveling(_SoftMorph):
    # leveling top-hat = x - level(x), level = iterate g <- max(min(x, dilate(g)), erode(g))
    # toward a Gaussian marker (learnable sigma) to near-idempotence. TRUNCATED to a few soft
    # iterations (approx., same truncation caveat as hdome/recontophat). geodesic connectivity
    # is the fixed elementary SE (standard leveling); sigma is the learnable marker scale.
    def __init__(self, k, beta, sigma_init=3.0, iters=8):
        super().__init__(k, beta)
        self.iters = iters
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(sigma_init)))
        self.register_buffer("se", elementary_se(k).view(1, 1, k * k, 1, 1))

    @staticmethod
    def _gaussian_blur(x, sigma):
        # separable Gaussian marker, kernel support +/- 3 sigma (differentiable w.r.t. sigma
        # via the sample grid), same padding convention as the morphology ops.
        radius = max(int(3.0 * float(sigma.detach())), 1)
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        x = F.conv2d(x, kernel.unsqueeze(2), padding=(0, radius))    # horizontal
        x = F.conv2d(x, kernel.unsqueeze(3), padding=(radius, 0))    # vertical
        return x

    def forward(self, x):
        sigma = F.softplus(self.raw_sigma)
        g = self._gaussian_blur(x, sigma)
        for _ in range(self.iters):
            g = torch.maximum(torch.minimum(x, self.dilate(g, self.se, debias=True)),
                               self.erode(g, self.se))
        return (x - g).clamp(min=0.0)

#
# tier 3: vdome surrogate (contrast gate + learnable granulometric size gate)
#
class VDomeSurrogate(_SoftMorph):
    # vdome = area_open(hdome(.)): keep bright domes that are both high-contrast (rise > h) AND
    # large. area-opening is combinatorial, so the size gate is a granulometric opening -- the
    # union (max) of openings by a disk + oriented lines. those size-gate SEs use HARD support
    # (0/-1e4) so the shape selectivity that keeps large / thin structures is preserved.
    def __init__(self, k, beta, h_init=0.1, iters=5, n_orient=2):
        super().__init__(k, beta)
        self.iters = iters
        self.raw_h = nn.Parameter(torch.tensor(_inv_softplus(h_init)))
        self.register_buffer("se_recon", elementary_se(k).view(1, 1, k * k, 1, 1))
        gran = [_make_se(k, "hard", radius=k // 2)]
        gran += [_make_se(k, "line", angle=math.pi * i / n_orient) for i in range(n_orient)]
        self.gran_ses = nn.ParameterList([nn.Parameter(s) for s in gran])

    def forward(self, x):
        h = F.softplus(self.raw_h)
        dome = (x - self.reconstruct(x - h, x, self.se_recon, self.iters)).clamp(min=0.0)
        opens = [self.opening(dome, se) for se in self.gran_ses]
        return torch.stack(opens, dim=0).amax(dim=0)     # union of openings

#
# spec grammar -> op.  one token per channel, examples:
#   tophat:3   bottomhat:2   gradient:1   line:3:0   asf:3   asftophat:3
#   hdome:0.1  recontophat:3  vdome:0.1  leveltophat:3
#
def build_op(spec, k, beta):
    parts = spec.split(":")
    name = parts[0]
    if name == "tophat":
        return Tophat(k, beta, radius=int(parts[1]))
    if name == "bottomhat":
        return Bottomhat(k, beta, radius=int(parts[1]))
    if name == "gradient":
        return Gradient(k, beta, radius=int(parts[1]))
    if name == "line":
        idx = int(parts[2]) if len(parts) > 2 else 0
        return Tophat(k, beta, init="line", angle=math.pi * idx / 4)
    if name == "asf":
        return ASF(k, beta, n_scales=int(parts[1]), residual=False)
    if name == "asftophat":
        return ASF(k, beta, n_scales=int(parts[1]), residual=True)
    if name == "hdome":
        return HDome(k, beta, h_init=float(parts[1]))
    if name == "recontophat":
        return ReconTophat(k, beta, radius=int(parts[1]))
    if name == "leveltophat":
        return Leveling(k, beta, sigma_init=float(parts[1]))
    if name == "vdome":
        return VDomeSurrogate(k, beta, h_init=float(parts[1]))
    raise ValueError(f"unknown morph-bank spec: {spec!r}")

#
# Morphological Bank U-Net: prepend the trainable morph channels to the image (+ any static
# channels), then feed to the U-Net. specs are full-grammar strings (one channel each).
#
class MorphBankUNet(nn.Module):

    def __init__(self, base_unet, specs, k=11, beta=10.0):
        super().__init__()
        assert specs, "need at least one spec"
        self.blocks = nn.ModuleList([build_op(s, k, beta) for s in specs])
        self.unet = base_unet

    def set_beta(self, beta):
        for blk in self.blocks:
            blk.set_beta(beta)

    def forward(self, x):
        # morph residuals from the image channel only (ch 0); static channels pass through in x
        img = x[:, :1]
        chans = [x] + [blk(img) for blk in self.blocks]
        return self.unet(torch.cat(chans, dim=1))

#
# Matched control: a learned conv front-end producing the SAME number of extra channels, so a
# comparison isolates morphology's rank/shape selectivity from "extra learned channels".
# (channel-matched; parameter-matched exactly for the flat ops, approximately once the bank
# holds reconstruction/dome ops that carry their own h / granulometric params.)
#
class ConvBankUNet(nn.Module):

    def __init__(self, base_unet, n_extra, k=5, in_channels=1):
        super().__init__()
        self.front = nn.Conv2d(in_channels, n_extra, kernel_size=k, padding=k // 2)
        self.act = nn.ReLU(inplace=True)
        self.unet = base_unet

    def forward(self, x):
        extra = self.act(self.front(x))
        return self.unet(torch.cat([x, extra], dim=1))
