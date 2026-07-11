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

from networks.morph_unet import (MorphUNet, MorphUnit, SoftMorph2d, Stage, ConvSepUnit,
                                  ConvMPMUnit, StrictMorph2d)


# --------------------------------------------------------------------------------------
# discovery: every prunable unit in the net.
# Two unit types share the SAME shape contract -- a per-INPUT-channel spatial op feeding a
# 1x1 proj (out<-in) -- so the whole pipeline is generic once a handful of helpers dispatch
# on type: the fast MorphUnit (depthwise soft morphology + 1x1) and its plain-conv twin
# ConvSepUnit (depthwise 3x3 + 1x1). Pruning an input channel drops that channel's spatial
# filter AND the matching proj input column; the proj OUTPUT width is untouched -> LOCAL, no
# cascade -- identical for both. The morphology-native criteria (l1x1/morph) need the SE and
# are morph-only; the agnostic ones (lin=weight-norm, act, fb, random) transfer to convsep.
# --------------------------------------------------------------------------------------
# Three unit types share the per-INPUT-channel spatial-op + 1x1-mixer contract, so the whole
# pipeline is generic once a few helpers dispatch on type: MorphUnit (depthwise soft morphology +
# linear 1x1 `proj`), ConvSepUnit (depthwise 3x3 + linear 1x1 `proj`), and ConvMPMUnit (depthwise
# 3x3 + MORPHOLOGICAL 1x1 `mix`, a StrictMorph2d(k=1) max-plus/min-plus channel MPM). Pruning an
# input channel drops that channel's spatial filter AND the matching 1x1 input column -> LOCAL, no
# cascade, identical for all three. The morphology-native criteria (l1x1/morph) need the SoftMorph2d
# SE, so they are MorphUnit-only; the agnostic ones (lin/act/fb/fbnew/random) transfer to both twins.
_PRUNABLE = (MorphUnit, ConvSepUnit, ConvMPMUnit)


def morph_units(model):
    """name -> prunable unit (MorphUnit / ConvSepUnit / ConvMPMUnit), in call order."""
    return {n: m for n, m in model.named_modules() if isinstance(m, _PRUNABLE)}


def _is_conv(unit):
    return isinstance(unit, ConvSepUnit)


def _is_convmpm(unit):
    return isinstance(unit, ConvMPMUnit)


def _is_morph(unit):
    # the soft-morph unit (per-channel SE); the ONLY unit the morphology-native criteria apply to
    return isinstance(unit, MorphUnit)


def _spatial(unit):
    # the per-input-channel spatial sub-module (calibration hook target):
    #   convsep/convmpm -> the depthwise 3x3 conv `dw`;  morph -> the SoftMorph2d `morph`.
    # Each outputs one activation map per INPUT channel, so act/fb/fbnew read the same quantity.
    return unit.morph if _is_morph(unit) else unit.dw


def _dev(unit):
    # convmpm's 1x1 mixer is a StrictMorph2d (`mix`); the other two have a linear 1x1 `proj`
    return unit.mix.weight.device if _is_convmpm(unit) else unit.proj.weight.device


def _in_ch(unit):
    if _is_morph(unit):
        return unit.morph.se.shape[1]
    return unit.dw.weight.shape[0]                    # depthwise conv weight: (in, 1, k, k)


def _dw_norm(unit):
    # per-channel L2 norm of the depthwise 3x3 filters -> (in,); convsep analogue of SE spread
    w = unit.dw.weight                                # (in, 1, k, k)
    return w.view(w.shape[0], -1).norm(dim=1)


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
    # ||mixer[:, i]|| : the 1x1 weight mass that reads input channel i -> (in,). For convmpm the
    # mixer is a StrictMorph2d(k=1) whose structuring weight is (out, in*1*1) = (out, in), so the
    # per-input-channel fan-out is the column norm exactly as for the linear 1x1 -- a large column
    # means channel i is a strong max-plus/min-plus contributor (can win) for some output.
    if _is_convmpm(unit):
        W = unit.mix.weight                           # (out, in)
        return W.norm(dim=0)
    W = unit.proj.weight[:, :, 0, 0]                  # (out, in)
    return W.norm(dim=0)


def score_unit(unit, criterion, act_rate=None, act_mag=None):
    """(in_ch,) importance of each input channel under `criterion`.

    act_rate: optional (in_ch,) off-centre win-rate from calibration (used by "morph").
    act_mag : optional (in_ch,) mean |morph output| from calibration (used by "act").
    """
    if criterion == "l1x1":
        if not _is_morph(unit):
            # no SoftMorph SE/alpha on a conv/convmpm unit; use the depthwise-filter energy as the
            # spatial-effect proxy (the 1x1-mixer column norm x depthwise 3x3 filter norm)
            return _proj_in_norm(unit) * _dw_norm(unit)
        s = _proj_in_norm(unit) * _alpha_abs(unit) * _se_spread(unit)
        return s
    if criterion == "morph":
        if not _is_morph(unit):
            raise ValueError("criterion 'morph' is SoftMorph-specific; not valid for conv/convmpm units")
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
        if not _is_morph(unit):
            return _proj_in_norm(unit)                # no alpha on conv/convmpm -> pure 1x1 mixer norm
        return _proj_in_norm(unit) * _alpha_abs(unit)
    if criterion == "random":
        # SANITY BASELINE: ignore every weight/activation and score channels at random, so top-k
        # keeps a uniformly random keep_ratio of channels. If the informed criteria don't beat this,
        # they aren't buying anything. Uses the global RNG (seeded in prune.py) for reproducibility.
        return torch.rand(_in_ch(unit), device=_dev(unit))
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
        handles.append(_spatial(u).register_forward_hook(mk_hook(n)))
    model.eval()
    for xb in calib_batches:
        model(xb.to(device))
    for h in handles:
        h.remove()
    return {n: (acc[n] / max(cnt[n], 1)) for n in units}


# --------------------------------------------------------------------------------------
# foreground-restricted mean |morph output| per input channel (for the "fbnew" criterion)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_fg_act_mag(model, calib_batches, device, eps=1e-8):
    """{unit_name: (in_ch,) mean |morph output| over foreground positions} -- the "act" quantity
    but averaged ONLY over the feature-map positions whose receptive field covers a foreground
    (vessel/tumour) voxel, instead of over the whole map.

    The receptive field of a feature-map pixel is approximated by its footprint at the unit's own
    spatial resolution: the label's foreground mask (seg>0) is max-pooled down to the feature map
    size (H_L, W_L), so a position counts iff ANY foreground voxel falls in the input region that
    feeds it. Because the mask is derived purely from the resolution, an encoder unit and its
    symmetric decoder unit (same resolution in a U-Net) see the IDENTICAL foreground mask -- i.e.
    the decoder inherits the receptive field of its mirror encoder layer, as intended.

    calib_batches: list of (data, seg) pairs; seg is the integer label map [b,1,H0,W0].
    """
    units = morph_units(model)
    num = {n: None for n in units}                       # sum |act| over fg positions -> (in,)
    den = {n: 0.0 for n in units}                        # count of fg positions (channel-agnostic)
    cur = {"fg": None}                                   # current batch fg mask, set before forward
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            B, C, H, W = out.shape
            m = F.adaptive_max_pool2d(cur["fg"], (H, W))          # (B,1,H,W) in {0,1}, RF footprint
            a = (out.abs() * m).sum(dim=(0, 2, 3)).cpu()          # (in,) fg-weighted activation mass
            num[name] = a if num[name] is None else num[name] + a
            den[name] += float(m.sum().item())
        return hook

    for n, u in units.items():
        handles.append(_spatial(u).register_forward_hook(mk_hook(n)))
    model.eval()
    for data, seg in calib_batches:
        cur["fg"] = (seg > 0).float().to(device)                  # (B,1,H0,W0) foreground mask
        model(data.to(device))
    for h in handles:
        h.remove()
    return {n: (num[n] / (den[n] + eps)) for n in units}          # mean |act| over fg positions


# --------------------------------------------------------------------------------------
# HMM forward-backward global channel importance (for the "fb" criterion)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_forward_backward_importance(model, calib_batches, device, eps=1e-8,
                                        use_skips=True, emission_backward=True,
                                        residual_smooth=1.0, fg_restrict=False):
    """{unit_name: (in_ch,) gamma} -- GLOBAL per-channel posterior occupancy via HMM forward-backward,
    now run over the U-Net's REAL routing instead of a bare linear chain.

    States = each morph unit's input channels. For unit L, patch n, channel i:
        s[L][n,i] = spatial mean |morph output|.
    Emission / prior  pi[L]_i  is proportional to  E_n s[L][n,i]  ( = E[morph(i)] ).  It is applied in
    BOTH passes. This matters: with a row-stochastic transition T, the bare backward recursion
    beta = T @ beta keeps beta uniform (T @ const = const), so beta collapses to 1 and gamma degenerates
    to alpha. Carrying the DOWNSTREAM emission pi through the backward step, beta = T @ (pi * beta), is
    what makes "downstream influence" actually vary per channel (emission_backward=True).

    Graph (use_skips=True):
      * BACKBONE : consecutive morph units L -> L+1 (encoder down / bottleneck / decoder up).
      * SKIP     : enc_k.sub2 -> dec_k.sub1, the U-Net skip. The skip INJECTS the encoder stage's
                   channels into the decoder as new inputs, so the encoder's importance reaches the
                   decoder directly, not only diffused through the bottleneck.
    Transition  T_{i->j} propto E_n s[L][n,i]*s[L+1][n,j], row-normalised. RESIDUAL add-one (Laplace)
    smoothing (residual_smooth>0) is added ONLY on the stage-boundary edges X.sub2 -> Y.sub1, the exact
    edges the Stage residual out=sub1(x)+sub2(sub1(x)) feeds -- a guaranteed baseline flow there, and
    left alone on within-stage and skip edges.

    Messages: alpha[L] = pi[L] * sum_in(alpha_src @ T) ; beta[L] = sum_out(T @ (pi_tgt * beta_tgt)) ;
    gamma = alpha*beta renormalised per layer. Because every edge points forward in registration order
    (skips included), one topological sweep each way is exact. Set use_skips=False,
    emission_backward=False, residual_smooth=0 to recover the legacy linear-chain fb.

    fg_restrict=True ("fbfg" = foreground forward-backward): the per-patch state s[L][n,i] is the mean
    |morph output| over ONLY the foreground receptive-field positions (seg>0 max-pooled to the unit's
    resolution, exactly as in collect_fg_act_mag) instead of the whole map. Everything downstream --
    pi, co-activation transitions, skips, residual smoothing, alpha/beta -- is unchanged; it is the
    fixed fb driven by foreground-only statistics. Then calib_batches must be (data, seg) pairs.
    """
    units = morph_units(model)
    names = list(units.keys())                          # named_modules() == forward/registration order
    idx = {n: i for i, n in enumerate(names)}
    traces = {n: [] for n in names}
    cur = {"fg": None}                                  # current-batch fg mask (fg_restrict only)
    handles = []

    def mk_hook(name):
        def hook(mod, inp, out):
            if fg_restrict:                             # mean |morph| over foreground positions only
                _, _, H, W = out.shape
                m = F.adaptive_max_pool2d(cur["fg"], (H, W))         # (B,1,H,W) RF footprint in {0,1}
                s = (out.abs() * m).sum(dim=(2, 3)) / (m.sum(dim=(2, 3)) + eps)   # (B,in)
            else:
                s = out.abs().mean(dim=(2, 3))          # (B,in,H,W) -> (B,in) full-map mean
            traces[name].append(s.cpu())
        return hook

    for n in names:
        handles.append(_spatial(units[n]).register_forward_hook(mk_hook(n)))
    model.eval()
    if fg_restrict:
        for data, seg in calib_batches:
            cur["fg"] = (seg > 0).float().to(device)    # (B,1,H0,W0) foreground mask
            model(data.to(device))
    else:
        for xb in calib_batches:
            model(xb.to(device))
    for h in handles:
        h.remove()

    S = [torch.cat(traces[n], dim=0) for n in names]    # each (N, C_L); rows aligned by patch across L
    L, N = len(S), S[0].shape[0]
    pi = [(s.mean(dim=0) + eps) for s in S]             # emission / prior  pi_i propto E[morph(i)]
    pi = [p / p.sum() for p in pi]

    def transition(a, b, smooth):
        M = (S[a].t() @ S[b]) / N                       # (C_a, C_b) co-activation
        if smooth > 0:                                  # residual ~ add-one (Laplace) smoothing:
            M = M + smooth * M.mean()                   #   a guaranteed baseline flow on THIS edge
        M = M + eps
        return M / M.sum(dim=1, keepdim=True)           # row-stochastic

    edges_in = {l: [] for l in range(L)}                # target -> [(source, T)]
    edges_out = {l: [] for l in range(L)}               # source -> [(target, T)]
    def add_edge(a, b, smooth):
        T = transition(a, b, smooth)
        edges_out[a].append((b, T)); edges_in[b].append((a, T))

    # A Stage is  out = sub1(x) + sub2(sub1(x))  -- the ONLY residual in the net, an identity around
    # sub2 whose skip feeds the NEXT stage. So a residual lives on exactly one kind of backbone edge:
    # X.sub2 -> Y.sub1 (a stage boundary). Within-stage edges (sub1 -> sub2) carry no residual, and the
    # U-Net skips are a separate concat -- so add-one smoothing is applied ONLY where a residual exists.
    for l in range(L - 1):                              # backbone chain
        is_residual = names[l].endswith(".sub2") and names[l + 1].endswith(".sub1")
        add_edge(l, l + 1, residual_smooth if is_residual else 0.0)
    if use_skips:                                       # U-Net skips: enc_k.sub2 -> dec_k.sub1 (injects
        for n in names:                                 # the encoder's channels; a skip, NOT a residual)
            if n.startswith("dec") and n.endswith(".sub1"):
                enc = "enc" + n[3:n.index(".")] + ".sub2"
                if enc in idx:
                    add_edge(idx[enc], idx[n], 0.0)

    alpha = [None] * L                                  # forward: emission * incoming messages
    for l in range(L):
        if edges_in[l]:
            a = pi[l] * sum(alpha[s] @ T for (s, T) in edges_in[l])
        else:
            a = pi[l].clone()                           # source node -> just the prior
        alpha[l] = a / (a.sum() + eps)

    beta = [None] * L                                   # backward: outgoing messages carry downstream pi
    for l in range(L - 1, -1, -1):
        if edges_out[l]:
            b = sum(T @ ((pi[t] * beta[t]) if emission_backward else beta[t])
                    for (t, T) in edges_out[l])
        else:
            b = torch.ones_like(pi[l])                  # sink node -> uniform
        beta[l] = b / (b.sum() + eps)

    dev = _dev(units[names[0]])
    return {n: ((alpha[i] * beta[i]) / ((alpha[i] * beta[i]).sum() + eps)).to(dev)
            for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------
# fb-MORPH: foreground forward-backward whose TRANSITIONS come from the morphological 1x1
# neuron's actual channel selection, not co-activation ("fbmorph" criterion, ConvMPM only).
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect_morph_routing_importance(model, calib_batches, device, eps=1e-8, use_skips=True,
                                     emission_backward=True, residual_smooth=1.0,
                                     max_pos_per_batch=20000):
    """{unit_name: (in_ch,) gamma} -- same foreground forward-backward as fbfg, but the transition
    T_{i->j} between successive states is the ROUTING FREQUENCY of the morphological 1x1 mixer instead
    of channel co-activation. ConvMPM only (needs the StrictMorph2d `mix`).

    For a ConvMPM unit, mix = StrictMorph2d(k=1): output j is a soft max-plus JOIN and min-plus MEET
    over the input channels, both of the SAME score  U_i + W_ji  (U = the depthwise-conv output that
    feeds the mixer, W = mix.weight (out,in)). So per position, output j has two winners:
        max-winner(j) = argmax_i (U_i + W_ji)   (the join / dilation choice)
        min-winner(j) = argmin_i (U_i + W_ji)   (the meet / erosion choice)
    Counting, over FOREGROUND positions only, how often input channel i wins (either choice) for output
    j gives R[i,j] = how strongly channel i routes to output j through the morphological neuron. That is
    the transition, row-normalised. This is causal routing ("j actually picked i"), unlike fbfg's
    correlation ("i and j were co-active").

    Where routing is well defined vs. not:
      * CLEAN (use mix routing R): within-stage sub1->sub2, encoder backbone (pool keeps channels),
        enc4.sub2->center, and the U-Net SKIP enc_k.sub2->dec_k.sub1 (the encoder output is concatenated
        verbatim into the decoder's skip columns, so enc's own R maps straight in).
      * NON-CLEAN (fall back to fbfg co-activation): every backbone edge that lands on a decoder sub1,
        because that input is `cat[ConvTranspose(up), skip]` -- the up path is a LINEAR channel-mixing
        transpose-conv, not a morphological selection, so no single MPM neuron routes it.
    Residual add-one smoothing is applied only on the stage-boundary edges X.sub2->Y.sub1, and the
    downstream emission pi (foreground E[morph]) is carried through the backward pass, exactly as in fb.
    """
    units = morph_units(model)
    names = list(units.keys())
    for n in names:
        if not _is_convmpm(units[n]):
            raise ValueError(f"fbmorph needs ConvMPM units (morphological 1x1 mixer); {n} is "
                             f"{type(units[n]).__name__}. Use it on a --morph-impl convmpm model.")
    idx = {n: i for i, n in enumerate(names)}
    traces = {n: [] for n in names}                      # per-patch fg-mean |U| per input ch (co-act + pi)
    Rmax = {n: None for n in names}                      # (in,out) max-plus join win counts
    Rmin = {n: None for n in names}                      # (in,out) min-plus meet win counts
    cur = {"fg": None}
    handles = []

    def count_routing(name, U, W):                       # U:(B,in,H,W)  W:(out,in)
        B, C, H, W_ = U.shape
        m = F.adaptive_max_pool2d(cur["fg"], (H, W_)) > 0.5              # (B,1,H,W) fg footprint
        sel = U.permute(0, 2, 3, 1)[m[:, 0]]             # (P, in) foreground positions only
        traces[name].append(((U.abs() * m).sum(dim=(2, 3))              # (B,in) fg-mean |U| for co-act/pi
                             / (m.sum(dim=(2, 3)) + eps)).cpu())
        P, inC = sel.shape
        outC = W.shape[0]
        if P == 0:
            z = torch.zeros(inC, outC)
            Rmax[name] = z.clone() if Rmax[name] is None else Rmax[name]
            Rmin[name] = z.clone() if Rmin[name] is None else Rmin[name]
            return
        if P > max_pos_per_batch:                        # cap positions to bound the argmax cost
            sel = sel[torch.randperm(P, device=sel.device)[:max_pos_per_batch]]
            P = sel.shape[0]
        cmax = torch.zeros(inC * outC, device=sel.device)
        cmin = torch.zeros(inC * outC, device=sel.device)
        ar = torch.arange(outC, device=sel.device)
        step = max(1, int(1e8 // (outC * inC)))          # chunk rows so (chunk,out,in) stays ~1e8 floats
        for s in range(0, P, step):
            sc = sel[s:s + step].unsqueeze(1) + W.unsqueeze(0)          # (p,out,in) = U_i + W_ji
            wmax = sc.argmax(dim=2); wmin = sc.argmin(dim=2)            # (p,out) winning input channel
            cmax += torch.bincount((wmax * outC + ar).flatten(), minlength=inC * outC).float()
            cmin += torch.bincount((wmin * outC + ar).flatten(), minlength=inC * outC).float()
        cmax = cmax.view(inC, outC).cpu(); cmin = cmin.view(inC, outC).cpu()
        Rmax[name] = cmax if Rmax[name] is None else Rmax[name] + cmax
        Rmin[name] = cmin if Rmin[name] is None else Rmin[name] + cmin

    def mk_pre(name):
        def hook(mod, inp):
            count_routing(name, inp[0], mod.weight)      # inp[0] = dw output = mixer input; mod.weight (out,in)
        return hook

    for n in names:
        handles.append(units[n].mix.register_forward_pre_hook(mk_pre(n)))
    model.eval()
    for data, seg in calib_batches:
        cur["fg"] = (seg > 0).float().to(device)
        model(data.to(device))
    for h in handles:
        h.remove()

    S = [torch.cat(traces[n], dim=0) for n in names]     # (N, in) fg-mean per patch
    N = S[0].shape[0]
    L = len(names)
    pi = [(s.mean(dim=0) + eps) for s in S]
    pi = [p / p.sum() for p in pi]
    R = [(Rmax[n] + Rmin[n]) for n in names]             # (in,out) combined join+meet routing counts

    def norm_rows(M, smooth):
        if smooth > 0:
            M = M + smooth * M.mean()
        M = M + eps
        return M / M.sum(dim=1, keepdim=True)

    def coact(a, b, smooth):                             # fbfg-style co-activation transition
        return norm_rows((S[a].t() @ S[b]) / N, smooth)

    edges_in = {l: [] for l in range(L)}
    edges_out = {l: [] for l in range(L)}
    def add_edge(a, b, T):
        edges_out[a].append((b, T)); edges_in[b].append((a, T))

    for l in range(L - 1):
        dest = names[l + 1]
        is_boundary = names[l].endswith(".sub2") and dest.endswith(".sub1")
        smooth = residual_smooth if is_boundary else 0.0
        if dest.startswith("dec") and dest.endswith(".sub1"):           # up path: ConvTranspose+concat
            add_edge(l, l + 1, coact(l, l + 1, smooth))                 # -> co-activation fallback
        else:                                                          # clean: source mix routing
            assert R[l].shape[1] == _in_ch(units[dest]), \
                f"routing edge {names[l]}->{dest}: out {R[l].shape[1]} != in {_in_ch(units[dest])}"
            add_edge(l, l + 1, norm_rows(R[l].clone(), smooth))
    if use_skips:                                        # skip: enc_k.sub2 routing into dec skip columns
        for n in names:
            if n.startswith("dec") and n.endswith(".sub1"):
                enc = "enc" + n[3:n.index(".")] + ".sub2"
                if enc in idx:
                    in_dec = _in_ch(units[n]); out_enc = R[idx[enc]].shape[1]
                    up = in_dec - out_enc                                # concat = [up, skip]; skip is last
                    T = torch.zeros(_in_ch(units[enc]), in_dec)
                    T[:, up:] = R[idx[enc]]                              # enc output == dec skip columns
                    add_edge(idx[enc], idx[n], norm_rows(T, 0.0))

    alpha = [None] * L
    for l in range(L):
        if edges_in[l]:
            a = pi[l] * sum(alpha[s] @ T for (s, T) in edges_in[l])
        else:
            a = pi[l].clone()
        alpha[l] = a / (a.sum() + eps)
    beta = [None] * L
    for l in range(L - 1, -1, -1):
        if edges_out[l]:
            b = sum(T @ ((pi[t] * beta[t]) if emission_backward else beta[t])
                    for (t, T) in edges_out[l])
        else:
            b = torch.ones_like(pi[l])
        beta[l] = b / (b.sum() + eps)

    dev = _dev(units[names[0]])
    return {n: ((alpha[i] * beta[i]) / ((alpha[i] * beta[i]).sum() + eps)).to(dev)
            for i, n in enumerate(names)}


# --------------------------------------------------------------------------------------
# structural surgery: rebuild a MorphUnit keeping only `keep` input channels
# --------------------------------------------------------------------------------------
def prune_unit(unit, keep):
    """In-place shrink a prunable unit (MorphUnit or ConvSepUnit) to `keep` input channels."""
    if keep.dtype == torch.bool:
        keep = keep.nonzero(as_tuple=False).flatten()
    keep = keep.to(torch.long).sort().values
    dev = _dev(unit)
    new_in = keep.numel()
    out_ch = unit.mix.out_ch if _is_convmpm(unit) else unit.proj.weight.shape[0]

    if _is_convmpm(unit):
        # shrink the depthwise 3x3 (keep surviving per-channel filters) + drop the morph mixer's
        # input columns. mix is StrictMorph2d(k=1): weight (out, in*1*1)=(out, in), so column i is
        # input channel i; the per-output join/meet biases are unchanged (output width untouched).
        old_dw = unit.dw
        k, pad = old_dw.kernel_size[0], old_dw.padding[0]
        new_dw = nn.Conv2d(new_in, new_in, k, padding=pad, groups=new_in).to(dev)
        with torch.no_grad():
            new_dw.weight.copy_(old_dw.weight[keep])
            new_dw.bias.copy_(old_dw.bias[keep])
        unit.dw = new_dw
        old_mix = unit.mix
        new_mix = StrictMorph2d(new_in, out_ch, k=old_mix.k, beta=float(old_mix.beta)).to(dev)
        with torch.no_grad():
            new_mix.weight.copy_(old_mix.weight[:, keep])
            new_mix.b_dil.copy_(old_mix.b_dil)
            new_mix.b_ero.copy_(old_mix.b_ero)
        unit.mix = new_mix
        unit.register_buffer("_in_keep", keep.to(dev))
        return unit

    if _is_conv(unit):
        # shrink the depthwise 3x3 (keep the surviving per-channel filters) + drop proj columns
        old_dw = unit.dw
        k, pad = old_dw.kernel_size[0], old_dw.padding[0]
        new_dw = nn.Conv2d(new_in, new_in, k, padding=pad, groups=new_in).to(dev)
        with torch.no_grad():
            new_dw.weight.copy_(old_dw.weight[keep])       # (in,1,k,k) -> keep rows
            new_dw.bias.copy_(old_dw.bias[keep])
        unit.dw = new_dw
        new_proj = nn.Conv2d(new_in, out_ch, 1).to(dev)
        with torch.no_grad():
            new_proj.weight.copy_(unit.proj.weight[:, keep])
            new_proj.bias.copy_(unit.proj.bias)
        unit.proj = new_proj
        unit.register_buffer("_in_keep", keep.to(dev))
        return unit

    old_morph = unit.morph
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


def _all_scores(model, criterion, calib_batches, device, fb_opts=None):
    """({unit_name: MorphUnit}, {unit_name: (in_ch,) score}) under `criterion` -- computed ONCE,
    before any surgery, so global allocation ranks all channels on the unpruned model."""
    units = morph_units(model)
    if criterion == "fb":                               # global posterior, computed for all units at once
        if calib_batches is None:
            raise ValueError("criterion 'fb' needs calibration batches")
        return units, collect_forward_backward_importance(model, calib_batches, device, **(fb_opts or {}))
    if criterion == "fbfg":                             # the fixed fb, but foreground-restricted stats
        if calib_batches is None:
            raise ValueError("criterion 'fbfg' needs (data, seg) calibration batches")
        return units, collect_forward_backward_importance(model, calib_batches, device,
                                                          fg_restrict=True, **(fb_opts or {}))
    if criterion == "fbmorph":                          # fg fb with morphological-selection transitions
        if calib_batches is None:
            raise ValueError("criterion 'fbmorph' needs (data, seg) calibration batches")
        return units, collect_morph_routing_importance(model, calib_batches, device, **(fb_opts or {}))
    if criterion == "fbnew":                            # act, but restricted to foreground receptive fields
        if calib_batches is None:
            raise ValueError("criterion 'fbnew' needs (data, seg) calibration batches")
        fg_mag = collect_fg_act_mag(model, calib_batches, device)
        scores = {n: score_unit(u, "act", act_mag=fg_mag.get(n)) for n, u in units.items()}
        return units, scores
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
    dev = _dev(next(iter(units.values())))
    return {name: torch.tensor(sorted(idx), dtype=torch.long, device=dev)
            for name, idx in keep.items()}


# --------------------------------------------------------------------------------------
# whole-model pruning: score every morph unit, keep top channels (local or global budget)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def prune_morph_channels(model, criterion="l1x1", keep_ratio=0.5, calib_batches=None, device="cpu",
                         min_keep=1, alloc="local", global_norm="max", verbose=True, fb_opts=None):
    """Structurally prune input channels of every MorphUnit. Returns a per-unit report dict.

    alloc="local"  : each unit keeps keep_ratio of ITS OWN channels (uniform sparsity).
    alloc="global" : one keep_ratio*TOTAL budget shared across units (non-uniform sparsity), with a
                     guaranteed `min_keep` floor per unit and per-layer `global_norm` for ranking.
    fb_opts        : dict forwarded to the 'fb' forward-backward (use_skips/emission_backward/
                     residual_smooth); ignored by the other criteria.
    """
    units, scores = _all_scores(model, criterion, calib_batches, device, fb_opts=fb_opts)
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

    # --- CONVSEP twin: the agnostic criteria (lin/act/fb/random) must prune it identically ---
    print("--- convsep (depthwise 3x3 + 1x1) ---")
    cnet = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="convsep",
                     conv_stem=True, checkpoint=False).to(dev).eval()
    cy0 = cnet(x)
    cp0 = count_params(cnet)
    ccalib = [torch.randn(1, 1, 64, 64, device=dev) for _ in range(3)]
    for crit in ("l1x1", "lin", "act", "fb", "random"):     # l1x1 falls back to dw-norm on conv
        c2 = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="convsep",
                       conv_stem=True, checkpoint=False).to(dev).eval()
        c2.load_state_dict(cnet.state_dict())
        for alloc in ("local", "global"):
            c3 = MorphUNet(num_classes=3, in_channels=1, fs=16, config="full_l2", impl="convsep",
                           conv_stem=True, checkpoint=False).to(dev).eval()
            c3.load_state_dict(cnet.state_dict())
            prune_morph_channels(c3, criterion=crit, keep_ratio=0.5, calib_batches=ccalib,
                                 device=dev, alloc=alloc, global_norm="max", min_keep=2, verbose=False)
            cy1 = c3(x)
            cp1 = count_params(c3)
            ok = tuple(cy1.shape) == tuple(cy0.shape) and torch.isfinite(cy1).all().item()
            print(f"convsep {crit:6s} {alloc:6s}: {cp0/1e6:.3f}M -> {cp1/1e6:.3f}M "
                  f"({100*(1-cp1/cp0):.1f}% off)  out={tuple(cy1.shape)}  finite={ok}")
    print("OK")
