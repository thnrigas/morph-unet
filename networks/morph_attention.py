#
# Morphological Attention U-Net
#
# An example architecture that adds *morphological attention* on top of the MPM MorphUNet
# (networks/morph_unet.py). Instead of the usual convolutional attention gate (Attention
# U-Net, Oktay et al. 2018), the gating signal is produced by grey-scale morphology:
#
#   top-hat(x)    = x - opening(x)      -> bright structures thinner than the SE (e.g. vessels)
#   bottom-hat(x) = closing(x) - x      -> dark  structures thinner than the SE
#
# where opening = dilation(erosion(.)) and closing = erosion(dilation(.)) with a shared
# learnable structuring element. The two hat responses are the classic morphological
# thin-/tubular-structure detectors, so this attention is a natural fit for targets like
# hepatic vessels: the gate learns to keep exactly the geometry morphology exposes.
#
# Each encoder skip is passed through a MorphAttentionGate before being concatenated into
# the decoder, so the network attends to morphological saliency at every resolution while
# reusing the exact MPM neuron, its checkpointing, and every MorphUNet flag (config,
# half_morph, tie_mirror, beta annealing via set_beta).
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if __package__ in (None, ""):          # allow `python networks/morph_attention.py` self-test
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.morph_unet import MorphUNet, SoftMorph2d


#
# shared-SE grey morphology exposing pure dilation/erosion (and openings/closings).
# distinct from SoftMorph2d, whose forward already fuses dilation+erosion into the MPM
# neuron; here we need the individual operators to build top-hat / bottom-hat. SE is flat-
# initialised (zeros) so the gate starts as a plain square-SE morphology (near neutral top-hat).
#
# pad : how the border is extended before the k x k neighbourhoods are taken.
#   "neutral" - the identity of each operator: -inf for dilation (max), +inf for erosion (min), so a
#               border pixel only sees its real neighbours (= the SE cropped at the border). The soft
#               max/min is a log-MEAN-exp, i.e. the logsumexp minus log(#valid neighbours): with fewer
#               terms at the border a plain logsumexp would shift by log(k*k / #valid)/beta and leave a
#               rim in the hats. In the interior the only difference to the plain logsumexp is a
#               constant -/+ log(k*k)/beta, which cancels exactly inside opening/closing, so the
#               interior hats are unchanged.
#   "zero"    - the original behaviour (unfold's zero padding, plain logsumexp). Erosion at the border
#               is pulled towards 0, giving a top-hat rim around every patch. Kept to reproduce the
#               earlier unet_morphattn* runs.
#
# reflect : use the adjoint pair, so opening/closing are true openings/closings for ANY learned SE g.
# With the neighbourhood read at x+o (unfold), the max-plus convolution / min-plus correlation are
#   dilation  (f (+) g)(x) = max_o f(x+o) + g(-o)     -> the SE reflected: se.flip(2)
#   erosion   (f (-) g)(x) = min_o f(x+o) - g(o)
# (unfold lists the k*k offsets row-major from (-r,-r) to (+r,+r), so reversing that axis maps o -> -o).
# reflect=False is the original code, whose dilation used g(+o): the opening then stamps back the
# MIRRORED SE, can exceed the input by up to max_o[g(o) - g(-o)] and the top-hat can go negative. Both
# agree while the SE is symmetric, e.g. at the zero init. Kept to reproduce the earlier runs.
#
class GreyMorph2d(nn.Module):

    def __init__(self, channels, k=3, beta=10.0, pad="neutral", reflect=True):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        assert pad in ("neutral", "zero"), "pad must be 'neutral' or 'zero'"
        self.k = k
        self.pad = k // 2
        self.pad_mode = pad
        self.reflect = reflect
        self.se = nn.Parameter(torch.zeros(1, channels, k * k, 1, 1))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x, fill):
        # (B,C,H,W) -> (B,C,k*k,H,W); the border is extended with `fill` (the operator's identity)
        B, C, H, W = x.shape
        if self.pad_mode == "zero":
            return F.unfold(x, self.k, padding=self.pad).view(B, C, self.k * self.k, H, W)
        xp = F.pad(x, (self.pad,) * 4, value=fill)
        return F.unfold(xp, self.k).view(B, C, self.k * self.k, H, W)

    def _log_count(self, H, W, ref):
        # log(#in-image neighbours) per pixel, (H,W): the separable product of the valid row and
        # column counts of the k x k window, which is k*k in the interior and smaller at the border
        def count(L):
            i = torch.arange(L, device=ref.device)
            return (i + self.pad).clamp(max=L - 1) - (i - self.pad).clamp(min=0) + 1
        return torch.log((count(H)[:, None] * count(W)[None, :]).to(ref.dtype))

    def dilate(self, x):
        b = self.beta
        se = self.se.flip(2) if self.reflect else self.se   # g(-o): the adjoint of erode (see above)
        out = torch.logsumexp(b * (self._neigh(x, float("-inf")) + se), dim=2)
        if self.pad_mode == "neutral":
            out = out - self._log_count(x.shape[2], x.shape[3], out)
        return out / b

    def erode(self, x):
        b = self.beta                              # classical erosion: min over (neigh - SE)
        out = torch.logsumexp(-b * (self._neigh(x, float("inf")) - self.se), dim=2)
        if self.pad_mode == "neutral":
            out = out - self._log_count(x.shape[2], x.shape[3], out)
        return -out / b

    def opening(self, x):
        return self.dilate(self.erode(x))          # remove small BRIGHT structures

    def closing(self, x):
        return self.erode(self.dilate(x))          # fill  small DARK   structures

    def set_beta(self, beta):
        self.beta.fill_(float(beta))


#
# morphological spatial attention gate: gate = sigmoid(conv1x1([top-hat, bottom-hat])),
# out = x * gate. per-channel [0,1] re-weighting driven purely by morphological saliency.
# source : which feature the hats are computed on; the gate always multiplies the skip `x`.
#   "self"  - the skip `x` itself
#   "cross" - the decoder context `g` (the upsampled decoder feature, same C/H/W as the skip). Same
#             parameters as self (only the hats' input changes); the 1x1 conv mixes all of g's hat
#             channels, so g's and x's channels need not correspond.
# The warm init is refused for "cross": it pairs hat channel c with skip channel c, which is only
# meaningful when both come from the same tensor (self); for cross it is an arbitrary pairing.
#
class MorphAttentionGate(nn.Module):

    def __init__(self, channels, k=3, beta=10.0, warm=0.0, pad="neutral", reflect=True, source="self"):
        super().__init__()
        if source not in ("self", "cross"):
            raise ValueError("source must be 'self' or 'cross'")
        if source == "cross" and warm > 0:
            raise ValueError("the warm init pairs hat channel c with skip channel c, which has no meaning "
                             "when the hats come from the decoder; use warm=0 (2*sigmoid, identity init) "
                             "with source='cross'")
        self.use_ckpt = True                       # toggled by MorphUNet.set_checkpointing
        self.source = source
        self.morph = GreyMorph2d(channels, k=k, beta=beta, pad=pad, reflect=reflect)
        self.gate = nn.Conv2d(2 * channels, channels, 1)
        self.warm = float(warm)
        if self.warm > 0:
            # WARM-START (--morph-attn-warm): the gate reads the top-hat MINUS bottom-hat morphological
            # contrast from step 0 (identity-mapped conv), applied at a learnable strength g (init = warm).
            # gate = 1 + g*tanh(contrast) in [1-g, 1+g] -> morphology is HALF-ACTIVE at init instead of the
            # plain identity start; the morphological analogue of the linear gate's gamma=0.5 warm init.
            with torch.no_grad():
                self.gate.weight.zero_(); self.gate.bias.zero_()
                idx = torch.arange(channels)
                self.gate.weight[idx, idx, 0, 0] = 1.0              # + top-hat channel c
                self.gate.weight[idx, channels + idx, 0, 0] = -1.0  # - bottom-hat channel c
            self.g = nn.Parameter(torch.tensor(self.warm))
        else:
            # IDENTITY init: zero-init conv -> gate = 2*sigmoid(0) = 1 -> skip passes through unchanged
            # (the ReZero gamma=0 analogue, a plain-U-Net start).
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _forward(self, x, g):
        s = g if self.source == "cross" else x     # the feature whose morphology drives the gate
        tophat = s - self.morph.opening(s)         # bright thin structures
        bothat = self.morph.closing(s) - s         # dark  thin structures
        z = self.gate(torch.cat([tophat, bothat], dim=1))
        if self.warm > 0:
            gate = 1.0 + self.g * torch.tanh(z)    # half-active morphology at init (see __init__)
        else:
            gate = 2.0 * torch.sigmoid(z)          # identity at init
        return x * gate

    def forward(self, x, g=None):
        # x: the encoder skip (gated); g: the upsampled decoder feature (needed for source="cross")
        if self.source == "cross" and g is None:
            raise ValueError("source='cross' needs the decoder feature g")
        # 4 unfolds (open=2, close=2) on the full skip -> checkpoint in training to keep memory flat
        if self.use_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, g, use_reentrant=False)
        return self._forward(x, g)


#
# Morphological Attention U-Net: MorphUNet + a MorphAttentionGate on every encoder skip.
# inherits all MorphUNet construction flags (config, half_morph, tie_mirror, k, beta).
# attn_k : SE size of the top-hat/bottom-hat gates, either one odd int for all four skips or a
#          sequence of four (att1..att4, i.e. enc1..enc4). Independent of the backbone's `k`.
# attn_pad : border handling of the gates' morphology, "neutral" or "zero" (see GreyMorph2d).
#            Only the gates; the backbone's morphological layers keep their own padding.
# attn_reflect : adjoint dilation (reflected SE), so the hats come from true openings/closings
#                (see GreyMorph2d). False reproduces the earlier runs.
# attn_src : "self" (hats of the skip) or "cross" (hats of the upsampled decoder feature), see
#            MorphAttentionGate. The gate always multiplies the skip.
#
class MorphAttentionUNet(MorphUNet):

    def __init__(self, num_classes, in_channels=1, fs=64, k=3, beta=10.0, config="heavy",
                 half_morph=False, tie_mirror=False, conv_stem=False, checkpoint=True, impl="fast",
                 act="leaky", attn_k=3, attn_beta=10.0, dropout=0.0, attn_warm=0.0,
                 attn_pad="neutral", attn_reflect=True, attn_src="self"):
        super().__init__(num_classes, in_channels=in_channels, fs=fs, k=k, beta=beta,
                         config=config, half_morph=half_morph, tie_mirror=tie_mirror,
                         conv_stem=conv_stem, checkpoint=checkpoint, impl=impl, act=act,
                         dropout=dropout)
        ks = [attn_k] * 4 if isinstance(attn_k, int) else list(attn_k)
        assert len(ks) == 4, "attn_k must be one int or four (att1..att4)"
        def gate(ch, kk):
            return MorphAttentionGate(ch, k=kk, beta=attn_beta, warm=attn_warm, pad=attn_pad,
                                      reflect=attn_reflect, source=attn_src)
        # one gate per skip, sized to that encoder stage's channel count
        self.att1 = gate(fs,     ks[0])
        self.att2 = gate(fs * 2, ks[1])
        self.att3 = gate(fs * 4, ks[2])
        self.att4 = gate(fs * 8, ks[3])
        self.set_checkpointing(self.ckpt_enabled)  # re-apply so the new gates pick up the flag

    def set_beta(self, beta):
        # anneal the neuron SEs (base class) and the attention SEs together
        super().set_beta(beta)
        for mod in self.modules():
            if isinstance(mod, GreyMorph2d):
                mod.set_beta(beta)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        # gate each skip by morphological saliency (of the skip, or of the upsampled decoder feature
        # for cross) before concatenation; each upsampled feature is computed once and used for both
        u4 = self.up4(c);  d4 = self.dec4(torch.cat([u4, self.att4(e4, u4)], 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, self.att3(e3, u3)], 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, self.att2(e2, u2)], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, self.att1(e1, u1)], 1))
        return self.final(d1)


#
# self-test:  python networks/morph_attention.py
#
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 64, 64, device=dev)
    for cfg in ("balanced", "heavy"):
        net = MorphAttentionUNet(num_classes=3, in_channels=1, config=cfg).to(dev).train()
        out = net(x)
        out.mean().backward()
        total = sum(p.numel() for p in net.parameters())
        attn = sum(p.numel() for m in net.modules() if isinstance(m, MorphAttentionGate)
                   for p in m.parameters())
        print(f"{cfg:9s} out={tuple(out.shape)} params={total/1e6:.2f}M attn={attn/1e3:.1f}k  fwd+bwd OK")
