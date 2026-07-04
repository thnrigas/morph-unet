#
# Morphological U-Net (Setting-1 style morphological CNN)
#
# Replaces selected U-Net conv stages with morphological-separable blocks:
#   depthwise grey-morphology (a shared learnable SE, soft dilation + erosion) for the
#   spatial / geometric non-linearity, followed by a 1x1 conv that projects/mixes the
#   channels. The 1x1 is the "linear activation" that makes a morphological stack a
#   universal approximator (Fotopoulos & Maragos, "Training Deep Morphological Neural
#   Networks as Universal Approximators"): pure morphological layers collapse and have
#   sparse gradients, so a linear step between them is required, not optional.
#
# Why depthwise + 1x1 instead of a faithful channel-mixing max-plus conv: the latter
# needs a (B, C_out, C_in*k*k, H, W) intermediate, which is infeasible at U-Net channel
# counts (512-1024). Depthwise morphology keeps the morphological character cheaply
# while the 1x1 does the C_in->C_out projection -- which also *reduces* parameters vs
# the baseline double 3x3 conv (one 1x1 instead of two 3x3), and the 1x1s become the
# structured-pruning (TropNNC) targets.
#
# soft dilation/erosion via logsumexp (temperature beta) to avoid the dead-gradient
# problem of hard min/max. SE is zero-initialised (flat); per-channel scale/bias start
# neutral. beta can be annealed up during training (soft gradient -> sharper morphology).
#
# three configurations, selected by `config`:
#   heavy       - every encoder/decoder stage + bottleneck is morphological
#   balanced    - only the high-resolution stages (fine tubular geometry, e.g. vessels)
#   bottleneck  - only the (parameter-heavy) center stage
#

import torch
import torch.nn as nn
import torch.nn.functional as F


#
# depthwise grey morphology: alpha * (soft_dilation(x) + soft_erosion(x)) + bias
# one shared structuring element per channel, with a per-channel scale/bias
# (the MPM layer of Fotopoulos & Maragos with lambda = 0.5, shared weights, biased)
#
class SoftMorph2d(nn.Module):

    def __init__(self, channels, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.pad = k // 2
        # shared SE per channel (flat init -> starts near identity), per-channel affine
        self.se = nn.Parameter(torch.zeros(1, channels, k * k, 1, 1))
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):
        # x: (B,C,H,W) -> (B,C,k*k,H,W) (local neighbourhoods)
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
        self.beta.fill_(float(beta))

    def forward(self, x):
        return self.alpha * (self.soft_dilation(x) + self.soft_erosion(x)) + self.bias


#
# sublayers: a plain 3x3 conv unit, or a morphological-separable unit
#
class ConvUnit(nn.Module):

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MorphUnit(nn.Module):

    # depthwise soft morphology on in_ch, then a 1x1 projection to out_ch (+ norm + act)
    def __init__(self, in_ch, out_ch, k=3, beta=10.0):
        super().__init__()
        self.morph = SoftMorph2d(in_ch, k=k, beta=beta)
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.proj(self.morph(x))))


#
# a U-Net stage = two sublayers (in->out, out->out) with a residual on the second
# (residual connections improve generalisation of morphological nets -- the RMPM trick)
#
class Stage(nn.Module):

    def __init__(self, in_ch, out_ch, morph=False, k=3, beta=10.0):
        super().__init__()
        if morph:
            self.sub1 = MorphUnit(in_ch, out_ch, k=k, beta=beta)
            self.sub2 = MorphUnit(out_ch, out_ch, k=k, beta=beta)
        else:
            self.sub1 = ConvUnit(in_ch, out_ch)
            self.sub2 = ConvUnit(out_ch, out_ch)

    def forward(self, x):
        x = self.sub1(x)
        return x + self.sub2(x)     # sub2 is out->out, so the residual shapes match


#
# which stages are morphological, per configuration
#
STAGE_CONFIGS = {
    "heavy":      {"enc1", "enc2", "enc3", "enc4", "center", "dec4", "dec3", "dec2", "dec1"},
    "balanced":   {"enc1", "enc2", "dec2", "dec1"},   # high-resolution stages only
    "bottleneck": {"center"},
    "none":       set(),                              # all-conv reference
}


class MorphUNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, fs=64, k=3, beta=10.0, config="heavy"):
        super().__init__()
        morph = STAGE_CONFIGS[config] if isinstance(config, str) else set(config)
        self.config = config

        def stage(name, i, o):
            return Stage(i, o, morph=(name in morph), k=k, beta=beta)

        # encoder
        self.enc1 = stage("enc1", in_channels, fs)
        self.enc2 = stage("enc2", fs, fs * 2)
        self.enc3 = stage("enc3", fs * 2, fs * 4)
        self.enc4 = stage("enc4", fs * 4, fs * 8)
        self.pool = nn.MaxPool2d(2, stride=2)
        # bottleneck
        self.center = stage("center", fs * 8, fs * 16)
        # decoder (transpose-conv upsampling, skip-concat)
        self.up4 = nn.ConvTranspose2d(fs * 16, fs * 8, 2, stride=2)
        self.dec4 = stage("dec4", fs * 16, fs * 8)
        self.up3 = nn.ConvTranspose2d(fs * 8, fs * 4, 2, stride=2)
        self.dec3 = stage("dec3", fs * 8, fs * 4)
        self.up2 = nn.ConvTranspose2d(fs * 4, fs * 2, 2, stride=2)
        self.dec2 = stage("dec2", fs * 4, fs * 2)
        self.up1 = nn.ConvTranspose2d(fs * 2, fs, 2, stride=2)
        self.dec1 = stage("dec1", fs * 2, fs)
        # 1x1 output head stays linear
        self.final = nn.Conv2d(fs, num_classes, 1)

    def set_beta(self, beta):
        for mod in self.modules():
            if isinstance(mod, SoftMorph2d):
                mod.set_beta(beta)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(c), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


#
# self-test: build the reference + the three configs, report parameter counts
# (total and the morphological share) and a forward+backward timing on this machine.
# run:  python networks/morph_unet.py
#
def _counts(model):
    total = sum(p.numel() for p in model.parameters())
    morph = sum(p.numel()
                for mod in model.modules() if isinstance(mod, SoftMorph2d)
                for p in mod.parameters())
    return total, morph


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    import time

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    x = torch.randn(2, 1, 64, 64, device=dev)
    print(f"device={dev}\n")
    print(f"{'config':11s} {'out shape':18s} {'params':>9s} {'morph':>9s} {'fwd+bwd':>10s}")
    for cfg in ("none", "bottleneck", "balanced", "heavy"):
        net = MorphUNet(num_classes=3, in_channels=1, config=cfg).to(dev)
        total, morph = _counts(net)
        net.train()
        out = net(x)
        out.mean().backward()          # warmup (builds graph / caches kernels)
        _sync(dev)
        t0 = time.time()
        for _ in range(5):
            net.zero_grad(set_to_none=True)
            net(x).mean().backward()
        _sync(dev)
        dt = (time.time() - t0) / 5 * 1000
        print(f"{cfg:11s} {str(tuple(out.shape)):18s} "
              f"{total/1e6:7.2f}M {morph/1e3:7.1f}k {dt:8.1f} ms")
