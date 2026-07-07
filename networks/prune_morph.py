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
# Importance criteria (all ranked PER UNIT over its input channels):
#   * "l1x1"  : ||proj[:,i]|| * |alpha_i| * spread(SE_i)  -- the combined score. spread = max-min
#               of the SE (its actual morphological effect; a flat SE is inert even if ||SE|| is
#               large), |alpha| the channel scale, ||proj[:,i]|| how much the 1x1 actually uses it.
#   * "morph" : morphology-native saliency. Data-free part |alpha_i| * spread(SE_i); optional
#               data-driven multiplier = off-centre win-rate (how often the channel's max-plus
#               argmax picks a NEIGHBOUR, not the centre pixel = how much morphology it truly does;
#               a channel that always keeps the centre is an identity and is prunable). This is the
#               depthwise analog of the max-plus "winner" statistic (Zhang et al. ISMM 2019).
#   * "lin"   : ||proj[:,i]|| * |alpha_i| -- morphology-AGNOSTIC output contribution (data-free);
#               keeps inert-but-informative linear channels that "morph" would discard.
#   * "act"   : ||proj[:,i]|| * E|morph_i(x)| -- DATA-DRIVEN output contribution. The measured
#               mean-abs morphed activation replaces the data-free spread(SE) proxy, so it reflects
#               the real signal a channel injects into the next layer (incl. pooling/norm/residual
#               upstream). The activation-times-weight channel-saliency of Molchanov et al. (2017),
#               adapted to the depthwise-morph + 1x1 unit.
#   * "fb"    : GLOBAL importance from an HMM forward-backward over the morph-unit chain. Unigram
#               prior = the "act" activation prob (L1-normalised); bigram transition = empirical
#               per-patch co-activation between successive units. The posterior gamma = alpha*beta
#               couples upstream reachability with downstream influence -> a global per-channel score
#               (importance propagation, cf. NISP, Yu et al. CVPR 2018). See
#               collect_forward_backward_importance.
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


def score_unit(unit, criterion, act_rate=None, act_mag=None):
    """(in_ch,) importance of each input channel under `criterion`.

    act_rate: optional (in_ch,) off-centre win-rate from calibration (used by "morph").
    act_mag : optional (in_ch,) mean |morph output| from calibration (used by "act").
    """
    if criterion == "l1x1":
        s = _proj_in_norm(unit) * _alpha_abs(unit) * _se_spread(unit)
        return s
    if criterion == "morph":
        s = _alpha_abs(unit) * _se_spread(unit)
        if act_rate is not None:
            s = s * (act_rate.to(s.device) + 1e-6)    # win-rate is gathered on CPU; match the score's device
        return s
    if criterion == "act":
        # DATA-DRIVEN output contribution: ||proj[:,i]|| * E|morph_i(x)|. The 1x1 column is the
        # fan-out weight; the measured mean-abs morphed activation is the REAL signal channel i
        # injects (already carries alpha, the SE effect, and everything upstream -- pooling, norm,
        # the residual -- so it captures the "input side" without needing ill-defined fan-in
        # weights). Falls back to the pure proj-norm if no calibration was supplied.
        s = _proj_in_norm(unit)
        if act_mag is not None:
            s = s * act_mag.to(s.device)
        return s
    if criterion == "lin":
        # pure output-contribution, morphology-AGNOSTIC: ||proj[:,i]|| * |alpha_i|. Keeps channels
        # the 1x1 actually reads hard even when their SE is an identity (spread~0). The deliberate
        # complement of "morph": it PRESERVES inert-but-informative linear pathways instead of
        # discarding them -- an identity morph followed by its proj column is still a useful linear
        # projection of the input channel.
        return _proj_in_norm(unit) * _alpha_abs(unit)
    if criterion == "random":
        # SANITY BASELINE: ignore every weight/activation and score channels at random, so top-k
        # keeps a uniformly random keep_ratio of channels. If the informed criteria don't beat this,
        # they aren't buying anything. Uses the global RNG (seeded in prune.py) for reproducibility.
        return torch.rand(_in_ch(unit), device=unit.morph.se.device)
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
# data-driven mean |morph output| per input channel (for the "act" criterion)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_act_mag(model, calib_batches, device):
    """{unit_name: (in_ch,) mean |morph output|} over the calibration inputs.

    A forward hook grabs each SoftMorph2d's OUTPUT (the morphed activation alpha*(dil+ero), one map
    per input channel) and averages its absolute value over batch and space. This is the actual
    signal each input channel feeds into the unit's 1x1 -- the honest, measured replacement for the
    data-free spread(SE) proxy, and it already reflects pooling / norm / the residual upstream.
    """
    units = morph_units(model)
    acc = {n: None for n in units}
    cnt = {n: 0 for n in units}
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            a = out.abs().mean(dim=(0, 2, 3)).cpu()    # (B,in,H,W) -> (in,)
            acc[name] = a if acc[name] is None else acc[name] + a
            cnt[name] += 1
        return hook

    for n, u in units.items():
        handles.append(u.morph.register_forward_hook(mk_hook(n)))
    model.eval()
    for xb in calib_batches:
        model(xb.to(device))
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(cnt[n], 1)) for n in units}


# --------------------------------------------------------------------------------------
# HMM forward-backward global channel importance (for the "fb" criterion)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_forward_backward_importance(model, calib_batches, device, eps=1e-8):
    """{unit_name: (in_ch,) gamma} -- GLOBAL per-channel posterior occupancy via HMM forward-backward.

    Morph units are treated as an HMM chain in forward order. For unit L, patch n, input channel i:
        s[L][n,i] = spatial mean |morph output|      (per-patch L1 activation prevalence; the Part-1
                                                       quantity, == the data-driven factor of "act")
    Prior      pi[L]_i        proportional to  E_n s[L][n,i]                       (unigram prob)
    Transition T[L]_{i->j}    proportional to  E_n s[L][n,i]*s[L+1][n,j]  (row-normalised co-activation;
                              a per-patch SCALAR statistic, so it is resolution- and graph-agnostic --
                              it survives the pool/concat/residual/skip routing between morph units).
    Forward alpha (upstream reachability), backward beta (downstream influence), and the posterior
    gamma = alpha*beta give one GLOBAL importance per channel -- then pruned as a plain unigram score
    through the usual local/global keep-ratio machinery. Seeding beta at the last layer with the LOSS
    gradient instead of uniform would turn this into the cross-layer Taylor propagation (NISP-style);
    kept uniform here to stay strictly no-grad.
    """
    units = morph_units(model)
    names = list(units.keys())                          # named_modules() == forward/registration order
    traces = {n: [] for n in names}
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            traces[name].append(out.abs().mean(dim=(2, 3)).cpu())    # (B,in,H,W) -> (B,in)
        return hook

    for n in names:
        handles.append(units[n].morph.register_forward_hook(mk_hook(n)))
    model.eval()
    for xb in calib_batches:
        model(xb.to(device))
    for h in handles:
        h.remove()

    S = [torch.cat(traces[n], dim=0) for n in names]    # each (N, C_L); rows aligned by patch across L
    L, N = len(S), S[0].shape[0]
    pi = [(s.mean(dim=0) + eps) for s in S]
    pi = [p / p.sum() for p in pi]
    T = []                                              # row-stochastic co-activation transitions
    for l in range(L - 1):
        M = (S[l].t() @ S[l + 1]) / N + eps             # (C_l, C_{l+1})
        T.append(M / M.sum(dim=1, keepdim=True))

    alpha = [pi[0].clone()]                             # forward messages (renormalised each step)
    for l in range(1, L):
        a = pi[l] * (alpha[l - 1] @ T[l - 1])
        alpha.append(a / (a.sum() + eps))
    beta = [None] * L                                   # backward messages
    beta[L - 1] = torch.ones_like(pi[L - 1]) / pi[L - 1].numel()
    for l in range(L - 2, -1, -1):
        b = T[l] @ beta[l + 1]
        beta[l] = b / (b.sum() + eps)

    dev = units[names[0]].morph.se.device
    gamma = {}
    for l, n in enumerate(names):
        g = alpha[l] * beta[l]
        gamma[n] = (g / (g.sum() + eps)).to(dev)        # per-layer posterior, on the model's device
    return gamma


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
# per-layer score normalisation (only for cross-layer GLOBAL ranking; raw magnitudes are not
# comparable across layers). "max" maps each layer's scores to [0,1] by its best channel, so a
# layer with a PEAKY importance profile (few channels do the work = redundant tail) sheds more
# channels than a layer with a FLAT profile (every channel useful) -- which is exactly the
# "some layers are more redundant" intuition.
# --------------------------------------------------------------------------------------
def _normalize(s, mode):
    if mode == "none":
        return s
    if mode == "max":
        return s / (s.max() + 1e-12)
    if mode == "mean":
        return s / (s.mean() + 1e-12)
    if mode == "l2":
        return s / (s.norm() + 1e-12)
    if mode == "zscore":
        return (s - s.mean()) / (s.std() + 1e-12)
    raise ValueError(f"unknown global-norm {mode!r}")


def _all_scores(model, criterion, calib_batches, device):
    """({unit_name: MorphUnit}, {unit_name: (in_ch,) score}) under `criterion` -- computed ONCE,
    before any surgery, so global allocation ranks all channels on the unpruned model."""
    units = morph_units(model)
    if criterion == "fb":                               # global posterior, computed for all units at once
        if calib_batches is None:
            raise ValueError("criterion 'fb' needs calibration batches")
        return units, collect_forward_backward_importance(model, calib_batches, device)
    rates = act_mag = None
    if calib_batches is not None:
        if criterion == "morph":
            rates = collect_winner_rates(model, calib_batches, device)
        elif criterion == "act":
            act_mag = collect_act_mag(model, calib_batches, device)
    scores = {n: score_unit(u, criterion,
                            act_rate=(rates.get(n) if rates else None),
                            act_mag=(act_mag.get(n) if act_mag else None))
              for n, u in units.items()}
    return units, scores


def _local_keep(units, scores, keep_ratio, min_keep):
    """UNIFORM allocation: each unit keeps its own top round(keep_ratio*in_ch) (>= min_keep)."""
    out = {}
    for name, u in units.items():
        ic = _in_ch(u)
        k = min(ic, max(min_keep, int(round(keep_ratio * ic))))
        out[name] = torch.topk(scores[name], k).indices
    return out


def _global_keep(units, scores, keep_ratio, min_keep, global_norm):
    """GLOBAL allocation: a single budget of round(keep_ratio*TOTAL) channels is shared across all
    units and handed to the globally highest-scored channels -- so a redundant layer can give up
    channels to a layer that needs them (non-uniform sparsity). Each unit keeps at least `min_keep`
    (a hard floor that can push the final total above the budget when keep_ratio is tiny). Scores
    are made cross-layer-comparable by per-layer `global_norm` first."""
    nscore = {n: _normalize(s, global_norm) for n, s in scores.items()}
    total = sum(_in_ch(u) for u in units.values())
    budget = int(round(keep_ratio * total))
    keep, pool = {}, []                                # pool: (norm_score, name, local_idx)
    for name, u in units.items():
        m = min(min_keep, _in_ch(u))
        order = torch.argsort(nscore[name], descending=True)
        keep[name] = set(order[:m].tolist())           # reserved floor per layer
        for j in order[m:].tolist():
            pool.append((float(nscore[name][j]), name, j))
    remaining = max(0, budget - sum(len(v) for v in keep.values()))
    pool.sort(key=lambda t: t[0], reverse=True)
    for _, name, j in pool[:remaining]:                # fill the rest by global rank
        keep[name].add(j)
    dev = next(iter(units.values())).morph.se.device
    return {name: torch.tensor(sorted(idx), dtype=torch.long, device=dev)
            for name, idx in keep.items()}


# --------------------------------------------------------------------------------------
# whole-model pruning: score every morph unit, keep top channels (local or global budget)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def prune_morph_channels(model, criterion="l1x1", keep_ratio=0.5, calib_batches=None, device="cpu",
                         min_keep=1, alloc="local", global_norm="max", verbose=True):
    """Structurally prune input channels of every MorphUnit. Returns a per-unit report dict.

    alloc="local"  : each unit keeps keep_ratio of ITS OWN channels (uniform sparsity).
    alloc="global" : one keep_ratio*TOTAL budget shared across units (non-uniform sparsity), with a
                     guaranteed `min_keep` floor per unit and per-layer `global_norm` for ranking.
    """
    units, scores = _all_scores(model, criterion, calib_batches, device)
    keep_idx = (_global_keep(units, scores, keep_ratio, min_keep, global_norm)
                if alloc == "global" else _local_keep(units, scores, keep_ratio, min_keep))
    report = {}
    for name, u in units.items():
        ic = _in_ch(u)
        keep = keep_idx[name]
        report[name] = {"in_before": ic, "in_after": int(keep.numel())}
        prune_unit(u, keep)
        if verbose:
            print(f"  {name:22s} in {ic:4d} -> {keep.numel():4d}  ({criterion}, {alloc})")
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

    def fresh():
        n2 = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="fast",
                       conv_stem=True, checkpoint=False).to(dev).eval()
        n2.load_state_dict(net.state_dict())
        return n2

    # local (uniform) allocation, every criterion
    for crit in ("l1x1", "morph", "lin", "act", "fb", "random"):
        net2 = fresh()
        prune_morph_channels(net2, criterion=crit, keep_ratio=0.5, calib_batches=calib,
                             device=dev, verbose=False)
        y1 = net2(x)
        p1 = count_params(net2)
        ok = tuple(y1.shape) == tuple(y0.shape) and torch.isfinite(y1).all().item()
        print(f"local  {crit:6s}: params {p0/1e6:.3f}M -> {p1/1e6:.3f}M "
              f"({100*(1-p1/p0):.1f}% off)  out={tuple(y1.shape)}  finite={ok}")

    # global (shared-budget) allocation, min_keep floor -> non-uniform per-layer sparsity
    for crit in ("lin", "morph", "fb"):
        net2 = fresh()
        rep = prune_morph_channels(net2, criterion=crit, keep_ratio=0.5, calib_batches=calib,
                                   device=dev, alloc="global", global_norm="max", min_keep=2,
                                   verbose=False)
        y1 = net2(x)
        p1 = count_params(net2)
        widths = ",".join(str(r["in_after"]) for r in rep.values())
        ok = (tuple(y1.shape) == tuple(y0.shape) and torch.isfinite(y1).all().item()
              and all(r["in_after"] >= 2 for r in rep.values()))
        print(f"global {crit:6s}: params {p0/1e6:.3f}M -> {p1/1e6:.3f}M "
              f"({100*(1-p1/p0):.1f}% off)  per-layer[{widths}]  finite/floor={ok}")
    print("OK")
