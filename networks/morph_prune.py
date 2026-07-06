#
# Winner-frequency (tropical routing) pruning for morphological layers
# ---------------------------------------------------------------------
# Honest, tractable version of the "pruning as path-finding" idea, restricted to the
# STRICT morphological conv (networks/morph_unet.py:StrictMorph2d), the only unit with a
# genuine channel-level max-plus routing (the fast/depthwise MorphUnit has NO max over
# channels -- its 1x1 is a plain sum -- so winner-routing is undefined there).
#
# We do NOT dress this as a generative HMM/EM (that is a category error: backprop != the
# Forward-Backward posterior, and softmax/abs of conv weights is not a stochastic operator).
# What is mathematically real: a max-plus forward pass IS Viterbi in the max-plus semiring,
# so "how often does input channel i provide the argmax winner" is a well-defined,
# MNN-specific routing statistic. We compare three channel-importance criteria:
#
#   l1     : ||W column-block of channel i||  (data-free magnitude, the baseline to beat)
#   act    : mean |input activation| of channel i over calibration data (generic data-driven)
#   winner : how often channel i wins the max-plus argmax over calibration data (the proposal)
#
# The point of the module is the FALSIFIABLE TEST: if `winner` beats both `l1` and `act` on
# the Dice-retention-vs-sparsity curve, the "MNN winner structure gives a better signal"
# hypothesis holds; if winner ~= act, the idea is just generic data-driven pruning (still
# beats l1, but no special sauce).
#
# Pruning is applied per morphological sub-network implicitly: scores are ranked PER LAYER
# (each morph layer sits in a skip-free linear->morph->linear->morph chain), so no cross-
# scale / cross-skip comparison is made. Removal is done by input-channel masking (correct
# MPM semantics via -inf/+inf per path), which measures criterion quality without the risky
# structural rebuild; turning a surviving mask into real FLOP/param savings is a later step.
#

import types
import torch
import torch.nn.functional as F

from networks.morph_unet import StrictMorph2d


# --------------------------------------------------------------------------------------
# masked forward for StrictMorph2d: prune input channels with per-path infinities.
# dilation is a soft-max over (U + W) -> pruned columns get -inf (never win the max).
# erosion  is a soft-min over (U + W) -> pruned columns get +inf (never win the min).
# a single masked activation value cannot do both, which is why we mask the WEIGHT here.
# --------------------------------------------------------------------------------------
def _lse_ext(U, W, beta, sign):
    # numerically-stable logsumexp_{ik}( sign*beta*(U + W) ) as a log-matmul -> (B, out, L)
    sU = (sign * beta) * U                    # (B, ik, L)
    sW = (sign * beta) * W                    # (out, ik)   (+-inf columns -> exp 0)
    uM = sU.amax(dim=1, keepdim=True)         # (B, 1, L)
    wM = sW.amax(dim=1, keepdim=True)         # (out, 1)
    prod = torch.einsum("oi,bil->bol", torch.exp(sW - wM), torch.exp(sU - uM))
    return uM + wM.unsqueeze(0) + torch.log(prod.clamp_min(1e-30))


def _masked_strict_forward(self, x):
    b = self.beta
    U = F.unfold(x, self.k, padding=self.pad)                 # (B, in*kk, L)
    W = self.weight                                           # (out, in*kk)
    keep = getattr(self, "_prune_keep", None)                 # (in,) bool or None
    if keep is not None and not bool(keep.all()):
        kk = self.k * self.k
        col_prune = (~keep).repeat_interleave(kk).view(1, -1)   # (1, in*kk)
        Wd = W.masked_fill(col_prune, float("-inf"))            # drop from the max
        We = W.masked_fill(col_prune, float("inf"))             # drop from the min
    else:
        Wd = We = W
    bd = self.b_dil.view(1, -1, 1)
    be = self.b_ero.view(1, -1, 1)
    dil = torch.logaddexp(_lse_ext(U, Wd, b, 1.0), b * bd) / b
    ero = -torch.logaddexp(_lse_ext(U, We, b, -1.0), -b * be) / b
    B, _, H, Wd_ = x.shape
    return (dil + ero).view(B, self.out_ch, H, Wd_)


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------
def strict_layers(model):
    """name -> StrictMorph2d, in call order. These are the prunable morph layers."""
    return {n: m for n, m in model.named_modules() if isinstance(m, StrictMorph2d)}


def _in_ch(m):
    return m.weight.shape[1] // (m.k * m.k)


# --------------------------------------------------------------------------------------
# per-input-channel importance scores (one vector of length in_ch per layer)
# --------------------------------------------------------------------------------------
def score_l1(m):
    """Data-free magnitude: L2 norm of each input channel's weight column-block over outputs."""
    kk = m.k * m.k
    W = m.weight.view(m.out_ch, _in_ch(m), kk)          # (out, in, kk)
    return W.norm(dim=(0, 2))                            # (in,)


@torch.no_grad()
def _acc_act(m, x, store):
    # mean |activation| per input channel, accumulated over calibration batches
    s = x.abs().mean(dim=(0, 2, 3))                     # (in,)
    store["act"] = store.get("act", 0) + s.cpu()
    store["n"] = store.get("n", 0) + 1


@torch.no_grad()
def _acc_winner(m, x, store, out_chunk=4):
    # hard max-plus winner histogram: for each (output channel o, position l) find
    # argmax_ik (U + W[o]); the input channel of that winner casts one vote. This is the
    # Case-A / Viterbi statistic (which channel provides the max-plus maximiser).
    kk = m.k * m.k
    ic = _in_ch(m)
    U = F.unfold(x, m.k, padding=m.pad)                 # (B, in*kk, L)
    W = m.weight                                        # (out, in*kk)
    votes = torch.zeros(ic)
    for o0 in range(0, m.out_ch, out_chunk):            # chunk outputs to bound memory
        Wc = W[o0:o0 + out_chunk]                       # (c, in*kk)
        s = U.unsqueeze(1) + Wc.unsqueeze(0).unsqueeze(-1)   # (B, c, in*kk, L)
        win = s.argmax(dim=2)                           # (B, c, L) winning ik index
        ch = (win // kk).reshape(-1)                    # -> input channel of the winner
        votes += torch.bincount(ch.cpu(), minlength=ic).float()
    store["winner"] = store.get("winner", 0) + votes


@torch.no_grad()
def _acc_winner_soft(m, x, store, out_chunk=4):
    # SOFT responsibility (Case B), faithful to the beta-10 trained net: for each output o
    # and position l, r_ik = softmax_ik(beta*(U+W[o])) for dilation and softmax_ik(-beta*(U+W))
    # for erosion. A channel's score sums BOTH paths (erosion-critical channels must survive),
    # then over its k*k offsets, outputs and data. Never exactly zero -> a smooth ranking.
    kk = m.k * m.k
    ic = _in_ch(m)
    b = m.beta
    U = F.unfold(x, m.k, padding=m.pad)                 # (B, in*kk, L)
    W = m.weight                                        # (out, in*kk)
    mass = torch.zeros(ic)
    for o0 in range(0, m.out_ch, out_chunk):
        Wc = W[o0:o0 + out_chunk]
        s = U.unsqueeze(1) + Wc.unsqueeze(0).unsqueeze(-1)     # (B, c, in*kk, L)
        r = torch.softmax(b * s, dim=2) + torch.softmax(-b * s, dim=2)   # dil + ero
        r = r.sum(dim=(0, 3))                            # (c, in*kk)  sum over batch, positions
        r = r.view(-1, ic, kk).sum(dim=2).sum(dim=0)     # (in,)       sum offsets + chunk outputs
        mass += r.cpu()
    store["winner_soft"] = store.get("winner_soft", 0) + mass


# --------------------------------------------------------------------------------------
# PATH-COMPOSED scores (the "global path" version of the routing idea).
#
# Per morph layer we build a routing matrix R[o, i] (out x in) = how much output channel o
# depends on input channel i, aggregated over data+positions (soft: dilation+erosion softmax
# responsibility; hard: max/min winner counts). Then, along each skip-free channel-identity
# CHAIN, we propagate importance BACKWARD:  imp_in = (R (+ I if residual))^T @ imp_out.
# A channel scores high iff it routes into important downstream channels -- coupling the
# per-layer signal across the chain, which the plain per-layer winner ignores.
#
# Scope (matches the U-Net's real wiring, see notes below):
#   * chains link consecutive morph layers with out_ch == in_ch AND either same stage or
#     both in the encoder/center block (pool preserves channels -> the encoder is one long
#     chain, which is where deep paths actually exist);
#   * center->decoder and decoder->decoder are NOT linked: the upconv mixes channels and the
#     skip-CONCAT branches the graph (a decoder morph's input is [deep ; encoder-skip]).
#     So each decoder stage is its own 2-hop chain, concat = boundary. No full-DAG BP.
#   * the within-stage residual (out = h + sub2(h)) adds an identity term, applied as R + I
#     on square (out==in) morph layers, which are exactly the residual sub2's.
# A length-1 chain reduces to winner_soft (R^T @ 1 = per-channel responsibility), so path_*
# strictly generalises the per-layer score.
#
# RECOMMENDATION: prefer `path_soft` in general. We train the network SOFT (LogSumExp, beta
# warmed up to 10), so its actual routing IS the softmax responsibility -- scoring with the
# hard argmax (`path_hard`, the beta->inf Viterbi limit) measures a regime the net never runs
# in, and is brittle to ties/noise. `path_hard` is kept only as a control column to confirm
# empirically that the beta-consistent soft variant wins. At beta=10 the soft responsibilities
# are already peaked, so soft composition stays discriminative (no diffusion collapse).
# --------------------------------------------------------------------------------------
def _stage(name):
    return name.split(".")[0]


_ENC_CENTER = {"enc1", "enc2", "enc3", "enc4", "center"}


@torch.no_grad()
def _acc_routing(m, x, store, do_soft, do_hard, out_chunk=4):
    # accumulate the (out x in) routing matrix R for this morph layer over calibration data
    kk = m.k * m.k
    ic = _in_ch(m)
    b = m.beta
    U = F.unfold(x, m.k, padding=m.pad)                 # (B, in*kk, L)
    W = m.weight
    if do_soft and "R_soft" not in store:
        store["R_soft"] = torch.zeros(m.out_ch, ic)
    if do_hard and "R_hard" not in store:
        store["R_hard"] = torch.zeros(m.out_ch, ic)
    for o0 in range(0, m.out_ch, out_chunk):
        Wc = W[o0:o0 + out_chunk]
        s = U.unsqueeze(1) + Wc.unsqueeze(0).unsqueeze(-1)     # (B, c, in*kk, L)
        if do_soft:
            r = torch.softmax(b * s, dim=2) + torch.softmax(-b * s, dim=2)   # dil + ero
            r = r.sum(dim=(0, 3)).view(-1, ic, kk).sum(dim=2)                 # (c, in)
            store["R_soft"][o0:o0 + r.shape[0]] += r.cpu()
        if do_hard:
            wd = s.argmax(dim=2)                          # (B, c, L) dilation winners
            we = (-s).argmax(dim=2)                       # (B, c, L) erosion winners
            for j in range(s.shape[1]):
                ch = (torch.cat([wd[:, j], we[:, j]]).reshape(-1) // kk).cpu()
                store["R_hard"][o0 + j] += torch.bincount(ch, minlength=ic).float()


def _find_chains(layers):
    """List of chains (lists of layer names) that are channel-identity + skip-free."""
    names = list(layers)
    chains, cur = [], []
    for n in names:
        if cur:
            prev = cur[-1]
            linked = (layers[prev].out_ch == _in_ch(layers[n])
                      and (_stage(prev) == _stage(n)
                           or (_stage(prev) in _ENC_CENTER and _stage(n) in _ENC_CENTER)))
            if linked:
                cur.append(n)
                continue
            chains.append(cur)
        cur = [n]
    if cur:
        chains.append(cur)
    return chains


def _compose_paths(layers, R_by_name):
    """Backward-compose importance along each chain. Returns {name: per-input-channel score}."""
    scores = {}
    for chain in _find_chains(layers):
        imp_out = None
        for n in reversed(chain):                        # deepest layer of the chain first
            R = R_by_name[n].clone()
            if R.shape[0] == R.shape[1]:                 # square -> residual sub2 -> add I
                R = R + torch.eye(R.shape[0])
            if imp_out is None:
                imp_out = torch.ones(R.shape[0])
            imp_in = R.t() @ imp_out                     # (in,) importance of this layer's inputs
            scores[n] = imp_in
            imp_out = imp_in / imp_in.mean().clamp_min(1e-9)   # normalise (scale-free ranking)
    return scores


@torch.no_grad()
def collect_scores(model, calib_batches, device, criteria=("l1", "act", "winner", "winner_soft")):
    """Run a forward calibration pass and return {layer_name: {criterion: score(in_ch)}}.

    calib_batches: iterable of input tensors (B, C, H, W). Use small B (1-2) -- the winner
    statistic materialises a (B, out_chunk, in*kk, L) tensor per chunk.
    """
    layers = strict_layers(model)
    stores = {n: {} for n in layers}
    p_soft, p_hard = "path_soft" in criteria, "path_hard" in criteria
    need_data = any(c in ("act", "winner", "winner_soft", "path_soft", "path_hard") for c in criteria)

    handles = []
    if need_data:
        for n, m in layers.items():
            def pre_hook(mod, inp, _store=stores[n]):
                x = inp[0]
                if "act" in criteria:
                    _acc_act(mod, x, _store)
                if "winner" in criteria:
                    _acc_winner(mod, x, _store)
                if "winner_soft" in criteria:
                    _acc_winner_soft(mod, x, _store)
                if p_soft or p_hard:
                    _acc_routing(mod, x, _store, p_soft, p_hard)
            handles.append(m.register_forward_pre_hook(pre_hook))

    model.eval()
    if need_data:
        for xb in calib_batches:
            model(xb.to(device))
    for h in handles:
        h.remove()

    out = {}
    for n, m in layers.items():
        sc = {}
        if "l1" in criteria:
            sc["l1"] = score_l1(m).detach().cpu()
        if "act" in criteria:
            sc["act"] = stores[n]["act"] / max(stores[n].get("n", 1), 1)
        if "winner" in criteria:
            sc["winner"] = stores[n]["winner"]
        if "winner_soft" in criteria:
            sc["winner_soft"] = stores[n]["winner_soft"]
        out[n] = sc

    # path-composed scores: build routing matrices, then backward-compose per chain
    for want, rkey, skey in ((p_soft, "R_soft", "path_soft"), (p_hard, "R_hard", "path_hard")):
        if not want:
            continue
        R_by_name = {n: stores[n][rkey] for n in layers}
        ps = _compose_paths(layers, R_by_name)
        for n in layers:
            out[n][skey] = ps[n]
    return out


# --------------------------------------------------------------------------------------
# masks + application (per-layer top-k keep, ranked within the layer / sub-network)
# --------------------------------------------------------------------------------------
def build_masks(scores, criterion, keep_ratio):
    """{layer: keep_bool(in_ch)} keeping the top `keep_ratio` input channels by `criterion`."""
    masks = {}
    for name, sc in scores.items():
        s = sc[criterion]
        ic = s.numel()
        k = max(1, int(round(keep_ratio * ic)))
        keep = torch.zeros(ic, dtype=torch.bool)
        keep[torch.topk(s, k).indices] = True
        masks[name] = keep
    return masks


def apply_masks(model, masks):
    """Install masks (patch StrictMorph2d.forward + set _prune_keep). Returns restore()."""
    layers = strict_layers(model)
    patched = []
    for name, keep in masks.items():
        m = layers[name]
        m._prune_keep = keep.to(m.weight.device)
        m.forward = types.MethodType(_masked_strict_forward, m)
        patched.append(m)

    def restore():
        for m in patched:
            if hasattr(m, "_prune_keep"):
                del m._prune_keep
            if "forward" in m.__dict__:          # drop the instance-level patch
                del m.__dict__["forward"]
    return restore


# --------------------------------------------------------------------------------------
# falsifiable comparison: Dice-retention vs keep_ratio, per criterion
# --------------------------------------------------------------------------------------
@torch.no_grad()
def compare_criteria(model, calib_batches, eval_fn, keep_ratios,
                     criteria=("l1", "act", "winner"), device="cpu"):
    """eval_fn(model)->float (e.g. foreground Dice on a val set). Returns {crit: {ratio: score}}."""
    scores = collect_scores(model, calib_batches, device, criteria)
    results = {c: {} for c in criteria}
    base = eval_fn(model)
    print(f"unpruned: {base:.4f}")
    for crit in criteria:
        for r in keep_ratios:
            restore = apply_masks(model, build_masks(scores, crit, r))
            try:
                results[crit][r] = eval_fn(model)
            finally:
                restore()
            print(f"  {crit:7s} keep={r:.2f} -> {results[crit][r]:.4f}")
    return base, results


# --------------------------------------------------------------------------------------
# self-test on random data: builds a small strict MorphUNet, checks the three scores
# compute and rank channels DIFFERENTLY, and that masked forward still runs + prunes.
#   run:  python networks/morph_prune.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from networks.morph_unet import MorphUNet

    dev = torch.device("cuda" if torch.cuda.is_available()
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(0)
    # tiny heavy strict net: morph everywhere -> a long encoder chain + separate decoder 2-hops
    net = MorphUNet(num_classes=3, in_channels=1, fs=8, k=3, config="heavy",
                    impl="paper", checkpoint=False).to(dev).eval()
    calib = [torch.randn(1, 1, 64, 64, device=dev) for _ in range(3)]

    crits = ("l1", "act", "winner", "winner_soft", "path_soft", "path_hard")
    sc = collect_scores(net, calib, dev, criteria=crits)
    layers = strict_layers(net)
    chains = _find_chains(layers)
    print(f"strict layers: {len(sc)}   chains: {[len(c) for c in chains]} "
          f"(expect one long encoder chain + short decoder ones)")

    import torch as _t
    def top(v, k=8): return set(_t.argsort(v, descending=True)[:k].tolist())
    def ov(a, b): return len(top(a) & top(b)) / 8.0
    name0 = max(sc, key=lambda n: sc[n]["l1"].numel())   # a layer with many input channels
    s = sc[name0]
    print(f"[{name0}] in_ch={s['l1'].numel()}  top-8 overlaps: "
          f"l1-winnerSoft={ov(s['l1'], s['winner_soft']):.2f}  "
          f"winnerSoft-pathSoft={ov(s['winner_soft'], s['path_soft']):.2f}  "
          f"pathSoft-pathHard={ov(s['path_soft'], s['path_hard']):.2f}  (low => differ)")

    # masked forward runs and actually changes the output (channels pruned)
    x = calib[0]
    y0 = net(x)
    restore = apply_masks(net, build_masks(sc, "path_soft", keep_ratio=0.5))
    y1 = net(x)
    restore()
    delta = (y0 - y1).abs().mean().item()
    print(f"masked forward OK: output changed by {delta:.4f} at keep=0.5 "
          f"(finite={_t.isfinite(y1).all().item()})")
    print("OK")