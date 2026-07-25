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
from torch.utils.checkpoint import checkpoint


#
# activation factory. "leaky" (nnU-Net default) is used for training; "relu" gives the plain
# ReLU that the tropical-geometry framework (and TropNNC) is derived for, so a network meant
# for TropNNC pruning can be built/fine-tuned with exact ReLU non-linearities.
#
def make_act(act="leaky"):
    return nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(inplace=True)


#
# depthwise grey morphology -- the Max-Plus-Min (MPM) neuron of Fotopoulos & Maragos,
# "Training Deep Morphological Neural Networks as Universal Approximators" (Setting 1):
#
#   out = alpha * ( [ b_dil  v  max_j(x_j + se_dil) ]     # biased soft dilation (max-plus)
#                 + [ b_ero  ^  min_j(x_j + se_ero) ] )   # biased soft erosion  (min-plus)
#
# Per the paper the dilation (max-plus) and erosion (min-plus) paths SHARE the structuring
# element and carry DIFFERENT biases -- and the bias enters through the join (v) / meet (^),
# NOT as an addition, so the two biases are genuinely distinct and cannot be folded into one
# additive constant (which is what a trailing "+ bias" would have been). Both paths use
# x_j + se (min-plus erosion is min_j(x_j + se), not the classical min_j(x_j - se)).
#
# Within a neuron the dilation (max-plus) and erosion (min-plus) paths SHARE ONE structuring
# element -- a single parameter `se`, tied at init AND throughout training (the paper's hard
# weight-tie) -- while carrying DIFFERENT biases. (The separate-but-init-tied behaviour lives
# one level up, across paired neurons in different layers; see MorphUNet's `tie_pairs`.)
#
# Init follows Appendix C for MPM/lambda=1/2 nets: "all morphological layers are initialised
# to follow a standard distribution" (SE ~ N(0,1)). The sum of max and min is naturally
# zero-mean, so no negative-mean shift (needed only for pure max-plus) is required.
#
# max/min are softened with a LogSumExp of temperature beta to avoid the dead gradients of
# hard min/max; each bias is folded into the LSE via logaddexp so no extra
# (B,C,k*k+1,H,W) tensor is materialised.
#
class SoftMorph2d(nn.Module):

    def __init__(self, channels, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.pad = k // 2
        # ONE structuring element per channel, shared by both paths (hard weight-tie), N(0,1)
        self.se = nn.Parameter(torch.randn(1, channels, k * k, 1, 1))
        # distinct join/meet biases + per-channel linear scaling of the sum
        self.b_dil = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.b_ero = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def _neigh(self, x):
        # x: (B,C,H,W) -> (B,C,k*k,H,W) (local neighbourhoods)
        B, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.pad)
        return cols.view(B, C, self.k * self.k, H, W)

    def set_beta(self, beta):
        self.beta.fill_(float(beta))

    def forward(self, x):
        # unfold ONCE and share the neighbourhood tensor between the two paths: the
        # (B,C,k*k,H,W) cols are this layer's dominant activation.
        b = self.beta
        n = self._neigh(x) + self.se                          # (B,C,k*k,H,W), shared SE
        # biased soft dilation: (1/b) logsumexp over the neighbourhood joined with b_dil
        dil = torch.logaddexp(torch.logsumexp(b * n, dim=2), b * self.b_dil) / b
        # biased soft erosion: -(1/b) logsumexp over -(neighbourhood) met with b_ero
        ero = -torch.logaddexp(torch.logsumexp(-b * n, dim=2), -b * self.b_ero) / b
        return self.alpha * (dil + ero)


#
# sublayers: a plain 3x3 conv unit, or a morphological-separable unit
#
class ConvUnit(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, act="leaky"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MorphUnit(nn.Module):

    # depthwise soft morphology on in_ch, then a 1x1 projection to out_ch (+ norm + act)
    def __init__(self, in_ch, out_ch, k=3, beta=10.0, act="leaky"):
        super().__init__()
        self.morph = SoftMorph2d(in_ch, k=k, beta=beta)
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = make_act(act)
        # when structurally pruned, `_in_keep` holds the surviving input-channel indices and the
        # forward selects them before the (shrunk) morph/proj. A buffer (not a patched method) so
        # it follows .to(device) and the whole pruned module pickles/reloads cleanly.
        self.register_buffer("_in_keep", None)

    def forward(self, x):
        if self._in_keep is not None:
            x = x[:, self._in_keep]
        return self.act(self.norm(self.proj(self.morph(x))))


class ConvSepUnit(nn.Module):

    # the plain-conv TWIN of MorphUnit: a DEPTHWISE 3x3 conv (one k*k linear filter per input
    # channel -- the ordinary linear analogue of the depthwise soft morphology) followed by the
    # SAME 1x1 projection to out_ch (+ norm + act). Identical separable factorisation and, to
    # within a couple of params per channel, the SAME parameter budget as MorphUnit -- only the
    # per-channel spatial operator differs (a learned linear 3x3 here vs. soft max-plus/min-plus
    # morphology there). Running this at the same config isolates exactly what the morphology
    # buys over an equally-sized depthwise conv, and it trains much faster (no unfold / logsumexp,
    # a single cuDNN depthwise kernel, and no gradient-checkpointing needed).
    #
    # Param count per unit (k=3): depthwise = in*(k*k) + in bias = 10*in; MorphUnit's SoftMorph2d
    # = in*(k*k) + 3*in (se + b_dil + b_ero + alpha) = 12*in. The 1x1 proj (in*out + out) is
    # identical and dominates, so the two units match to well under 0.1% of total params.
    def __init__(self, in_ch, out_ch, k=3, beta=10.0, act="leaky"):
        super().__init__()
        # `beta` is accepted for signature parity with the Unit factory in Stage; a conv has no
        # temperature, so it is ignored (set_beta / beta-warmup become no-ops on this unit).
        self.dw = nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch)   # depthwise spatial
        self.proj = nn.Conv2d(in_ch, out_ch, 1)                              # 1x1 channel mix
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = make_act(act)
        # same pruning hook as MorphUnit: surviving input-channel indices (buffer -> .to()/pickle
        # safe). Pruning targets the depthwise filters dw.weight[keep] + proj.weight[:, keep].
        self.register_buffer("_in_keep", None)

    def forward(self, x):
        if self._in_keep is not None:
            x = x[:, self._in_keep]
        return self.act(self.norm(self.proj(self.dw(x))))


class HybridUnit(nn.Module):

    # half the *input* channels go through soft morphology, the other half through a
    # plain 3x3 conv; the two outputs are concatenated to out_ch. Splitting on the input
    # is what buys the speed-up -- the morphological unfold (this layer's cost) then runs
    # on only in_ch//2 channels. Falls back to a full ConvUnit when the input is too thin
    # to split (e.g. the 1-channel image stage), where morphology would be meaningless anyway.
    def __init__(self, in_ch, out_ch, k=3, beta=10.0, act="leaky"):
        super().__init__()
        self.in_m = in_ch // 2                         # channels routed to morphology
        self.in_c = in_ch - self.in_m                  # channels routed to conv
        self.out_m = out_ch // 2 if self.in_m > 0 else 0
        self.out_c = out_ch - self.out_m
        self.morph = MorphUnit(self.in_m, self.out_m, k=k, beta=beta, act=act) if self.out_m else None
        self.conv = ConvUnit(self.in_c, self.out_c, act=act) if self.out_c else None

    def forward(self, x):
        if self.morph is None:                         # thin input -> all conv
            return self.conv(x)
        xm, xc = x[:, :self.in_m], x[:, self.in_m:]
        return torch.cat([self.morph(xm), self.conv(xc)], dim=1)


#
# STRICT (paper "Setting 1" conv) morphology: a full channel-mixing max-plus + min-plus
# convolution. Each output channel takes a soft max/min over ALL input channels and the k*k
# structuring shifts (unlike the depthwise SoftMorph2d) -- the faithful morphological conv, so
# the max-plus weights carry the bulk of the parameters (which is what makes it prune the
# paper's way). Shared weight W for both paths, distinct join/meet biases (the MPM neuron).
#
# Implemented as a numerically-stable LOG-DOMAIN MATMUL:
#   soft-dilation = (1/b) logsumexp_{ik}( b(U + W) ) = (1/b) log( exp(bW) @ exp(bU) )
# with a separate max subtracted from U (per location) and W (per output channel) so the
# exponentials never overflow -- this keeps it a single (im2col) matmul instead of the
# infeasible (B, C_out, C_in*k*k, H, W) tensor.
#
class StrictMorph2d(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, beta=10.0):
        super().__init__()
        assert k % 2 == 1, "kernel size k must be odd"
        self.k = k
        self.pad = k // 2
        self.out_ch = out_ch
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch * k * k))   # structuring functions, N(0,1)
        self.b_dil = nn.Parameter(torch.randn(out_ch))                   # join bias
        self.b_ero = nn.Parameter(torch.randn(out_ch))                   # meet bias
        self.register_buffer("beta", torch.tensor(float(beta)))

    def set_beta(self, beta):
        self.beta.fill_(float(beta))

    def _lse(self, U, sign):
        # logsumexp_{ik}( sign*beta*(U + W) ), stable, as a log-matmul -> (B, out, L)
        b = self.beta
        sU = (sign * b) * U                              # (B, ik, L)
        sW = (sign * b) * self.weight                    # (out, ik)
        uM = sU.amax(dim=1, keepdim=True)                # (B, 1, L)
        wM = sW.amax(dim=1, keepdim=True)                # (out, 1)
        prod = torch.einsum('oi,bil->bol',
                            torch.exp(sW - wM), torch.exp(sU - uM))   # (B, out, L)
        return uM + wM.unsqueeze(0) + torch.log(prod.clamp_min(1e-30))

    def forward(self, x):
        B, C, H, W = x.shape
        b = self.beta
        U = F.unfold(x, self.k, padding=self.pad)        # (B, in*k*k, H*W)
        bd = self.b_dil.view(1, -1, 1)
        be = self.b_ero.view(1, -1, 1)
        dil = torch.logaddexp(self._lse(U, 1.0), b * bd) / b            # biased soft dilation
        ero = -torch.logaddexp(self._lse(U, -1.0), -b * be) / b        # biased soft erosion
        return (dil + ero).view(B, self.out_ch, H, W)


class StrictMorphUnit(nn.Module):

    # paper Setting-1 conv block: full max-plus/min-plus morphological conv (does the channel
    # mixing itself), then a DEPTHWISE 3x3 linear activation (per-channel, 9*out_ch params),
    # then norm + act. Contrast MorphUnit, which is depthwise-morph then a 1x1 channel mix.
    def __init__(self, in_ch, out_ch, k=3, beta=10.0, act="leaky"):
        super().__init__()
        self.morph = StrictMorph2d(in_ch, out_ch, k=k, beta=beta)
        self.act_conv = nn.Conv2d(out_ch, out_ch, k, padding=k // 2, groups=out_ch)  # depthwise linear
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = make_act(act)

    def forward(self, x):
        return self.act(self.norm(self.act_conv(self.morph(x))))


class ConvMPMUnit(nn.Module):

    # ConvSep's twin with a MORPHOLOGICAL channel mixer. The spatial op is the SAME depthwise 3x3
    # LINEAR conv as ConvSepUnit, but the 1x1 LINEAR channel projection is replaced by a 1x1 MPM
    # neuron -- a soft max-plus (join) + min-plus (meet) mix over the input channels, i.e.
    # StrictMorph2d with k=1 (no spatial extent, pure channel mixing). Motivation: in ConvSep /
    # MorphUnit the channel mixing is a dense linear sum -- every input channel feeds every output
    # a little -- so channel usage is inherently NON-sparse. A max-plus mixer's output is instead
    # dominated by the single winning input channel per neuron (soft-argmax), so a channel that
    # never wins contributes nothing and is prunable: the morphology now lives exactly where
    # channel selection / structured sparsity happens. Same param budget class as ConvSep (the 1x1
    # mixer has in*out weights either way; the MPM adds only 2*out bias params). beta is the
    # LogSumExp temperature (soft -> hard as it grows; follows set_beta / beta-warmup like the
    # other soft-morph units). No (B,C,k*k,H,W) neighbourhood tensor (k=1), so like ConvSep it does
    # NOT need gradient checkpointing.
    def __init__(self, in_ch, out_ch, k=3, beta=10.0, act="leaky"):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch)   # depthwise spatial (linear)
        self.mix = StrictMorph2d(in_ch, out_ch, k=1, beta=beta)              # 1x1 max-plus/min-plus channel MPM
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = make_act(act)
        # pruning hook parity with MorphUnit/ConvSepUnit: surviving input-channel indices. Pruning
        # this unit drops dw.weight[keep] AND the mix's input columns mix.weight[:, keep].
        self.register_buffer("_in_keep", None)

    def forward(self, x):
        if self._in_keep is not None:
            x = x[:, self._in_keep]
        return self.act(self.norm(self.mix(self.dw(x))))


#
# a U-Net stage = two sublayers (in->out, out->out) with a residual on the second
# (residual connections improve generalisation of morphological nets -- the RMPM trick).
# mode: "conv" (plain), "morph" (MPM neurons), or "half" (HybridUnit: half morph, half conv).
# impl: "fast" (depthwise-morph + 1x1) or "paper" (StrictMorphUnit: full max-plus + depthwise 3x3).
#
class Stage(nn.Module):

    def __init__(self, in_ch, out_ch, mode="conv", k=3, beta=10.0, impl="fast", act="leaky",
                 dropout=0.0):
        super().__init__()
        self.mode = mode
        self.impl = impl
        self.use_ckpt = True            # toggled by MorphUNet.set_checkpointing
        # spatial (channel-wise) dropout on the stage OUTPUT, applied AFTER any checkpointed
        # compute so it never interacts with the checkpoint RNG. Identity when dropout==0 (the
        # default) -> existing models are bit-for-bit unchanged.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if mode == "conv":
            self.sub1, self.sub2 = ConvUnit(in_ch, out_ch, act=act), ConvUnit(out_ch, out_ch, act=act)
        elif mode == "half":
            self.sub1 = HybridUnit(in_ch, out_ch, k=k, beta=beta, act=act)
            self.sub2 = HybridUnit(out_ch, out_ch, k=k, beta=beta, act=act)
        else:                            # "morph"
            if impl == "paper":
                Unit = StrictMorphUnit
            elif impl == "convsep":      # depthwise-conv twin (matched params, no morphology)
                Unit = ConvSepUnit
            elif impl == "convmpm":      # depthwise 3x3 conv + MORPHOLOGICAL 1x1 (MPM) channel mixer
                Unit = ConvMPMUnit
            else:
                Unit = MorphUnit
            self.sub1 = Unit(in_ch, out_ch, k=k, beta=beta, act=act)
            self.sub2 = Unit(out_ch, out_ch, k=k, beta=beta, act=act)

    def _forward(self, x):
        x = self.sub1(x)
        return x + self.sub2(x)     # sub2 is out->out, so the residual shapes match

    def forward(self, x):
        # gradient-checkpoint the morphological stages: their (B,C,k*k,H,W) neighbourhood
        # tensors are the dominant activation and holding them for all the balanced stages
        # overflows the GPU, so recompute them in backward instead of storing. plain conv
        # stages are cheap and left un-checkpointed. use_reentrant=False so a non-grad input
        # (e.g. enc1's raw image) and bf16 autocast both work. Checkpointing is a pure
        # compute<->memory trade with NO effect on the maths, so it can be turned off
        # (set_checkpointing(False)) when there is enough VRAM to go faster.
        # convsep stages are cheap (a single depthwise cuDNN kernel, no (B,C,k*k,H,W) tensor),
        # so they are NOT checkpointed -- checkpointing would only add a redundant recompute.
        # convmpm's log-domain channel matmul stores large (B,out,L) intermediates, so like the
        # soft-morph units it is checkpointed; only the cheap depthwise-conv twin (convsep) is not.
        heavy = self.mode in ("morph", "half") and self.impl != "convsep"
        if self.use_ckpt and heavy and self.training and torch.is_grad_enabled():
            return self.drop(checkpoint(self._forward, x, use_reentrant=False))
        return self.drop(self._forward(x))


#
# which stages are morphological, per configuration
#
STAGE_CONFIGS = {
    "heavy":      {"enc1", "enc2", "enc3", "enc4", "center", "dec4", "dec3", "dec2", "dec1"},
    "balanced":   {"enc1", "enc2", "dec2", "dec1"},   # high-resolution stages only
    "bottleneck": {"center"},
    "deep":       {"enc4", "center", "dec4"},         # bottleneck + its two skip-adjacent neighbours
    # morphology everywhere, progressively freeing the high-res (expensive) levels back to conv:
    "full":       {"enc1", "enc2", "enc3", "enc4", "center", "dec4", "dec3", "dec2", "dec1"},  # all -> 18 morph layers (== heavy)
    "full_l1":    {"enc2", "enc3", "enc4", "center", "dec4", "dec3", "dec2"},                   # linear level 1 -> 14 morph layers
    "full_l2":    {"enc3", "enc4", "center", "dec4", "dec3"},                                   # linear levels 1-2 -> 10 morph layers
    "none":       set(),                              # all-conv reference
}


def _tie_se(src_stage, dst_stage):
    # copy structuring elements from src_stage's morph neurons into dst_stage's matching
    # ones (same sub-module path AND same shape), so the pair starts identical. They remain
    # separate Parameters, so training lets them diverge. Shape-mismatched neurons (e.g. the
    # in->out sub1 of an encoder vs. its decoder mirror) are skipped. Biases stay independent.
    src = dict(src_stage.named_modules())
    for name, dm in dst_stage.named_modules():
        sm = src.get(name)
        if (isinstance(dm, SoftMorph2d) and isinstance(sm, SoftMorph2d)
                and sm.se.shape == dm.se.shape):
            with torch.no_grad():
                dm.se.copy_(sm.se)


class MorphUNet(nn.Module):

    # half_morph : morphological stages use HybridUnit (half morph, half conv) instead of full
    #              MorphUnit -- ~halves the morphological cost for faster training.
    # tie_mirror : encoder<->decoder mirror stages (enc1<->dec1, enc2<->dec2) start with
    #              *identical* structuring elements, then diverge freely during training.
    # conv_stem  : a 3x3 conv lifts the raw input to fs channels BEFORE any morphology, so the
    #              first morphological neuron sees smooth conv features (denoising) rather than
    #              the raw 1-channel intensities -- consistent with the paper's linear/morph
    #              alternation (Setting 3). Without it, enc1 morphology runs on the raw input.
    # checkpoint : gradient-checkpoint the morph stages (memory<->compute trade, no maths change).
    # act        : "leaky" (nnU-Net default, used for training) or "relu" (plain ReLU, the
    #              non-linearity the tropical-geometry / TropNNC pruning theory is derived for).
    def __init__(self, num_classes, in_channels=1, fs=64, k=3, beta=10.0, config="heavy",
                 half_morph=False, tie_mirror=False, conv_stem=False, checkpoint=True, impl="fast",
                 act="leaky", dropout=0.0):
        super().__init__()
        morph = STAGE_CONFIGS[config] if isinstance(config, str) else set(config)
        self.config = config
        morph_mode = "half" if half_morph else "morph"

        def stage(name, i, o):
            return Stage(i, o, mode=(morph_mode if name in morph else "conv"), k=k, beta=beta,
                         impl=impl, act=act, dropout=dropout)

        # optional denoising conv stem: lift in_channels -> fs before enc1 (which then runs on fs)
        self.stem = ConvUnit(in_channels, fs, act=act) if conv_stem else nn.Identity()
        enc1_in = fs if conv_stem else in_channels
        # encoder
        self.enc1 = stage("enc1", enc1_in, fs)
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

        if tie_mirror:                       # start each enc<->dec mirror from the same SEs
            for enc, dec in [(self.enc1, self.dec1), (self.enc2, self.dec2),
                             (self.enc3, self.dec3), (self.enc4, self.dec4)]:
                _tie_se(enc, dec)            # a no-op where the pair isn't morphological
        self.set_checkpointing(checkpoint)

    def set_checkpointing(self, flag):
        # toggle gradient checkpointing on every stage/gate that supports it (has use_ckpt)
        self.ckpt_enabled = flag
        for mod in self.modules():
            if hasattr(mod, "use_ckpt"):
                mod.use_ckpt = flag

    def set_beta(self, beta):
        for mod in self.modules():
            if isinstance(mod, (SoftMorph2d, StrictMorph2d)):
                mod.set_beta(beta)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
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
