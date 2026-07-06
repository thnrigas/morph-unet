#
# Structural channel pruning for the FAST morphological U-Net (networks/morph_unet.py, impl="fast")
# ---------------------------------------------------------------------------------------------
# The fast MorphUnit is:  x -> SoftMorph2d (depthwise, per-channel SE) -> 1x1 proj (in->out)
#                           -> InstanceNorm -> act.
# The depthwise SE and the per-channel bias/scale (b_dil, b_ero, alpha) all live on the *input*
# channels; the 1x1 proj reads those same input channels (one weight column per input channel).
# So "a morphological channel" == one INPUT channel of a MorphUnit, and erasing it means dropping
#   se[:, i], b_dil[:, i], b_ero[:, i], alpha[:, i]  AND  proj.weight[:, i]   (its 1x1 column).
# The proj OUTPUT width is untouched, so the stage's output channel count is unchanged -> residual
# adds and skip-concats still line up. Pruning input channels is therefore LOCAL and structurally
# safe: no cascade. This module rebuilds each pruned MorphUnit with genuinely smaller tensors
# (real parameter/FLOP savings) and slices its input at forward time.
#
# Two importance criteria (both ranked PER UNIT over its input channels):
#   * "l1"    : ||SE_i||  -- the paper-faithful magnitude score (Fotopoulos & Maragos 2024,
#               Dimitriadis & Maragos 2021): prune channels whose structuring element is small.
#   * "l1x1"  : ||proj[:,i]|| * |alpha_i| * spread(SE_i)  -- the combined score. spread = max-min
#               of the SE (its actual morphological effect; a flat SE is inert even if ||SE|| is
#               large), |alpha| the channel scale, ||proj[:,i]|| how much the 1x1 actually uses it.
#   * "morph" : morphology-native saliency. Data-free part |alpha_i| * spread(SE_i); optional
#               data-driven multiplier = off-centre win-rate (how often the channel's max-plus
#               argmax picks a NEIGHBOUR, not the centre pixel = how much morphology it truly does;
#               a channel that always keeps the centre is an identity and is prunable). This is the
#               depthwise analog of the max-plus "winner" statistic (Zhang et al. ISMM 2019).
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.morph_unet import MorphUNet, MorphUnit, SoftMorph2d, Stage


# --------------------------------------------------------------------------------------
# discovery: every MorphUnit in the net (these are the prunable morph blocks)
# --------------------------------------------------------------------------------------
def morph_units(model):
    """name -> MorphUnit, in call order."""
    return {n: m for n, m in model.named_modules() if isinstance(m, MorphUnit)}


def _in_ch(unit):
    return unit.morph.se.shape[1]


# --------------------------------------------------------------------------------------
# per-input-channel importance scores (one vector of length in_ch per MorphUnit)
# --------------------------------------------------------------------------------------
def _se_l2(unit):
    # ||SE_i|| over the k*k offsets -> (in,)
    return unit.morph.se[0, :, :, 0, 0].norm(dim=1)


def _se_spread(unit):
    # max_j SE_ij - min_j SE_ij -> the channel's morphological "throw" (0 == inert/identity)
    se = unit.morph.se[0, :, :, 0, 0]                 # (in, kk)
    return se.max(dim=1).values - se.min(dim=1).values


def _alpha_abs(unit):
    return unit.morph.alpha[0, :, 0, 0].abs()         # (in,)


def _proj_in_norm(unit):
    # ||proj[:, i]|| : the 1x1 weight mass that reads input channel i -> (in,)
    W = unit.proj.weight[:, :, 0, 0]                  # (out, in)
    return W.norm(dim=0)


def score_unit(unit, criterion, act_rate=None):
    """(in_ch,) importance of each input channel under `criterion`.

    act_rate: optional (in_ch,) off-centre win-rate from calibration (used by "morph").
    """
    if criterion == "l1":
        return _se_l2(unit)
    if criterion == "l1x1":
        s = _proj_in_norm(unit) * _alpha_abs(unit) * _se_spread(unit)
        return s
    if criterion == "morph":
        s = _alpha_abs(unit) * _se_spread(unit)
        if act_rate is not None:
            s = s * (act_rate + 1e-6)                 # keep channels that actually dilate/erode
        return s
    raise ValueError(f"unknown criterion {criterion!r}")


# --------------------------------------------------------------------------------------
# data-driven off-centre win-rate (for the "morph" criterion)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_winner_rates(model, calib_batches, device):
    """{unit_name: (in_ch,) off-centre win-rate} over calibration inputs.

    For each SoftMorph2d input channel, the fraction of (batch, position) where the max-plus
    dilation argmax OR the min-plus erosion argmin falls on a NON-centre offset. High == the
    channel genuinely reshapes its neighbourhood; ~0 == it just passes the centre pixel through.
    """
    units = morph_units(model)
    acc = {n: None for n in units}
    cnt = {n: 0 for n in units}
    centre = None
    handles = []

    def mk_hook(name, unit):
        def hook(mod, inp):
            nonlocal centre
            x = inp[0]
            k, pad = mod.k, mod.pad
            kk = k * k
            c = kk // 2                                # centre offset index
            n = F.unfold(x, k, padding=pad)           # (B, in*kk, L)
            B = x.shape[0]
            ic = x.shape[1]
            n = n.view(B, ic, kk, -1)                 # (B, in, kk, L)
            se = mod.se[0, :, :, 0, 0].unsqueeze(0).unsqueeze(-1)   # (1, in, kk, 1)
            val = n + se
            dil_win = val.argmax(dim=2)               # (B, in, L)  dilation winner offset
            ero_win = val.argmin(dim=2)               # (B, in, L)  erosion  winner offset
            off = ((dil_win != c) | (ero_win != c)).float().mean(dim=(0, 2)).cpu()   # (in,)
            if acc[name] is None:
                acc[name] = off
            else:
                acc[name] += off
            cnt[name] += 1
        return hook

    for n, u in units.items():
        handles.append(u.morph.register_forward_pre_hook(mk_hook(n, u.morph)))
    model.eval()
    for xb in calib_batches:
        model(xb.to(device))
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(cnt[n], 1)) for n in units}


# --------------------------------------------------------------------------------------
# structural surgery: rebuild a MorphUnit keeping only `keep` input channels
# --------------------------------------------------------------------------------------
def prune_unit(unit, keep):
    """In-place shrink a MorphUnit to the boolean/index `keep` over its input channels."""
    if keep.dtype == torch.bool:
        keep = keep.nonzero(as_tuple=False).flatten()
    keep = keep.to(torch.long).sort().values
    dev = unit.morph.se.device
    old_morph = unit.morph
    new_in = keep.numel()
    out_ch = unit.proj.weight.shape[0]

    # shrunk depthwise morphology (copy the surviving per-channel params)
    m = SoftMorph2d(new_in, k=old_morph.k, beta=float(old_morph.beta)).to(dev)
    with torch.no_grad():
        m.se.copy_(old_morph.se[:, keep])
        m.b_dil.copy_(old_morph.b_dil[:, keep])
        m.b_ero.copy_(old_morph.b_ero[:, keep])
        m.alpha.copy_(old_morph.alpha[:, keep])
    unit.morph = m

    # shrunk 1x1 projection (drop the pruned input columns; outputs unchanged)
    new_proj = nn.Conv2d(new_in, out_ch, 1).to(dev)
    with torch.no_grad():
        new_proj.weight.copy_(unit.proj.weight[:, keep])
        new_proj.bias.copy_(unit.proj.bias)
    unit.proj = new_proj

    unit.register_buffer("_in_keep", keep.to(dev))
    return unit


# --------------------------------------------------------------------------------------
# whole-model pruning: score every morph unit, keep top-`keep_ratio` input channels each
# --------------------------------------------------------------------------------------
@torch.no_grad()
def prune_morph_channels(model, criterion="l1", keep_ratio=0.5, calib_batches=None, device="cpu",
                         min_keep=1, verbose=True):
    """Structurally prune input channels of every MorphUnit. Returns a per-unit report dict."""
    units = morph_units(model)
    rates = None
    if criterion == "morph" and calib_batches is not None:
        rates = collect_winner_rates(model, calib_batches, device)

    report = {}
    for name, u in units.items():
        ic = _in_ch(u)
        s = score_unit(u, criterion, act_rate=(rates.get(name) if rates else None))
        k = max(min_keep, int(round(keep_ratio * ic)))
        keep = torch.topk(s, k).indices
        report[name] = {"in_before": ic, "in_after": k}
        prune_unit(u, keep)
        if verbose:
            print(f"  {name:22s} in {ic:4d} -> {k:4d}  ({criterion})")
    return report


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------
# self-test:  python networks/prune_morph.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    net = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="fast",
                    conv_stem=True, checkpoint=False).to(dev).eval()
    x = torch.randn(2, 1, 64, 64, device=dev)
    y0 = net(x)
    p0 = count_params(net)
    calib = [torch.randn(1, 1, 64, 64, device=dev) for _ in range(3)]
    for crit in ("l1", "l1x1", "morph"):
        net2 = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="fast",
                         conv_stem=True, checkpoint=False).to(dev).eval()
        net2.load_state_dict(net.state_dict())
        rep = prune_morph_channels(net2, criterion=crit, keep_ratio=0.5, calib_batches=calib,
                                   device=dev, verbose=False)
        y1 = net2(x)
        p1 = count_params(net2)
        ok = tuple(y1.shape) == tuple(y0.shape) and torch.isfinite(y1).all().item()
        print(f"{crit:6s}: params {p0/1e6:.3f}M -> {p1/1e6:.3f}M "
              f"({100*(1-p1/p0):.1f}% off)  out={tuple(y1.shape)}  finite={ok}")
    print("OK")
