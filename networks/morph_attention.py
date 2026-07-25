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
class GreyMorph2d(nn.Module):

    def __init__(self, channels, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.pad = k // 2
        self.se = nn.Parameter(torch.zeros(1, channels, k * k, 1, 1))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):
        B, C, H, W = x.shape
        return F.unfold(x, self.k, padding=self.pad).view(B, C, self.k * self.k, H, W)

    def dilate(self, x):
        b = self.beta
        return torch.logsumexp(b * (self._neigh(x) + self.se), dim=2) / b

    def erode(self, x):
        b = self.beta                              # classical erosion: min over (neigh - SE)
        return -torch.logsumexp(-b * (self._neigh(x) - self.se), dim=2) / b

    def opening(self, x):
        return self.dilate(self.erode(x))          # remove small BRIGHT structures

    def closing(self, x):
        return self.erode(self.dilate(x))          # fill  small DARK   structures

    def set_beta(self, beta):
        self.beta.fill_(float(beta))


#
# morphological spatial attention gate: gate = sigmoid(conv1x1([top-hat, bottom-hat])),
# out = x * gate. per-channel [0,1] re-weighting driven purely by morphological saliency.
#
class MorphAttentionGate(nn.Module):

    def __init__(self, channels, k=3, beta=10.0, warm=0.0):
        super().__init__()
        self.use_ckpt = True                       # toggled by MorphUNet.set_checkpointing
        self.morph = GreyMorph2d(channels, k=k, beta=beta)
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

    def _forward(self, x):
        tophat = x - self.morph.opening(x)         # bright thin structures
        bothat = self.morph.closing(x) - x         # dark  thin structures
        z = self.gate(torch.cat([tophat, bothat], dim=1))
        if self.warm > 0:
            gate = 1.0 + self.g * torch.tanh(z)    # half-active morphology at init (see __init__)
        else:
            gate = 2.0 * torch.sigmoid(z)          # identity at init
        return x * gate

    def forward(self, x):
        # 4 unfolds (open=2, close=2) on the full skip -> checkpoint in training to keep memory flat
        if self.use_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


#
# Morphological Attention U-Net: MorphUNet + a MorphAttentionGate on every encoder skip.
# inherits all MorphUNet construction flags (config, half_morph, tie_mirror, k, beta).
#
class MorphAttentionUNet(MorphUNet):

    def __init__(self, num_classes, in_channels=1, fs=64, k=3, beta=10.0, config="heavy",
                 half_morph=False, tie_mirror=False, conv_stem=False, checkpoint=True, impl="fast",
                 act="leaky", attn_k=3, attn_beta=10.0, dropout=0.0, attn_warm=0.0):
        super().__init__(num_classes, in_channels=in_channels, fs=fs, k=k, beta=beta,
                         config=config, half_morph=half_morph, tie_mirror=tie_mirror,
                         conv_stem=conv_stem, checkpoint=checkpoint, impl=impl, act=act,
                         dropout=dropout)
        # one gate per skip, sized to that encoder stage's channel count
        self.att1 = MorphAttentionGate(fs,     k=attn_k, beta=attn_beta, warm=attn_warm)
        self.att2 = MorphAttentionGate(fs * 2, k=attn_k, beta=attn_beta, warm=attn_warm)
        self.att3 = MorphAttentionGate(fs * 4, k=attn_k, beta=attn_beta, warm=attn_warm)
        self.att4 = MorphAttentionGate(fs * 8, k=attn_k, beta=attn_beta, warm=attn_warm)
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
        # gate each skip by its morphological saliency before concatenation
        d4 = self.dec4(torch.cat([self.up4(c), self.att4(e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(e1)], 1))
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
