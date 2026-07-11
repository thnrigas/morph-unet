#
# Linear-Attention U-Net -- attention-GATED skip connections with LINEAR (kernel) attention.
# =========================================================================================
# Same U-Net skeleton as networks/morph_unet.py (conv encoder/decoder, transpose-conv
# upsampling, skip-concat), but each skip is REFINED by a linear cross-attention block before
# it is concatenated: the upsampled decoder feature is the QUERY, the encoder skip is the
# KEY/VALUE. This is the Attention-U-Net idea (Oktay et al. 2018, "gate the skip with the
# decoder context") but with LINEAR attention instead of the additive gate -- so it is the
# plain-conv, linear-attention counterpart to the morphological-attention net in
# networks/morph_attention.py (which this file deliberately does NOT depend on).
#
# Why LINEAR attention (Katharopoulos et al. 2020, "Transformers are RNNs"): softmax attention
# over pixels is O(N^2) in the number of pixels N=H*W, which is infeasible at the high-res skips
# (e.g. 128x128 = 16384 tokens). With a positive feature map phi = elu(.)+1 the softmax is
# replaced by phi(Q) ( phi(K)^T V ), and the associativity lets us contract K,V over tokens
# FIRST -> a (d x d) context matrix -> total cost O(N * d^2), linear in pixels. Every skip can
# therefore afford cross-attention, including the finest one.
#
# The refinement is residual with a ReZero gate (gamma init 0): at initialisation each skip is
# the vanilla U-Net skip, and training smoothly ramps in the attention -- stable, and a clean
# ablation (gamma measures how much attention the model actually chose to use).
#

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the proven conv building blocks + the two-sublayer residual Stage from morph_unet
from networks.morph_unet import ConvUnit, Stage, make_act


class LinearAttention2d(nn.Module):
    #
    # Linear (kernel) CROSS-attention over pixels: query source `g` attends over key/value
    # source `x` (same H,W, same channel count ch). Multi-head; feature map phi = elu+1.
    #
    #   phi(q_i)^T ( sum_j phi(k_j) v_j^T )        numerator  (per query pixel i)
    #   ---------------------------------------
    #   phi(q_i)^T ( sum_j phi(k_j) )              denominator (normaliser)
    #
    # Contract K,V over the N tokens FIRST (the (d_k x d_v) context) -> O(N d^2), not O(N^2).
    #
    def __init__(self, ch, heads=4, dim_head=None):
        super().__init__()
        dim_head = dim_head or max(ch // heads, 1)
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_k = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_v = nn.Conv2d(ch, inner, 1, bias=False)
        self.to_out = nn.Conv2d(inner, ch, 1)

    def forward(self, g, x):
        B, _, H, W = x.shape
        h, d, N = self.heads, self.dim_head, H * W
        q = self.to_q(g).view(B, h, d, N)
        k = self.to_k(x).view(B, h, d, N)
        v = self.to_v(x).view(B, h, d, N)
        q = F.elu(q) + 1.0                              # positive feature map phi(Q)
        k = F.elu(k) + 1.0                              # phi(K)
        kv = torch.einsum('bhkn,bhvn->bhkv', k, v)      # (B,h,d_k,d_v) context, contracted over tokens
        ksum = k.sum(dim=-1)                            # (B,h,d_k) = sum_j phi(k_j)
        num = torch.einsum('bhkn,bhkv->bhvn', q, kv)    # (B,h,d_v,N)
        den = torch.einsum('bhkn,bhk->bhn', q, ksum)    # (B,h,N) normaliser
        out = num / (den.unsqueeze(2) + 1e-6)           # (B,h,d_v,N)
        out = out.reshape(B, h * d, H, W)
        return self.to_out(out)


class LinAttnGate(nn.Module):
    #
    # Refine an encoder skip `x` with the decoder context `g` via linear cross-attention,
    # residually with a ReZero gate. Returns a skip of the SAME shape as `x` (so the usual
    # skip-concat in the decoder is unchanged). gamma init 0 -> starts as the plain skip.
    #
    def __init__(self, ch, heads=4, gamma_init=0.0):
        super().__init__()
        self.attn = LinearAttention2d(ch, heads=heads)
        self.gamma = nn.Parameter(torch.full((1,), float(gamma_init)))

    def forward(self, g, x):
        return x + self.gamma * self.attn(g, x)


class LinAttnUNet(nn.Module):
    #
    # Plain-conv U-Net (same channel schedule as MorphUNet) with a LinAttnGate on every skip.
    # conv_stem lifts the raw input to fs before enc1 (matches morph_unet's conv_stem option).
    # heads : number of linear-attention heads per skip gate.
    #
    def __init__(self, num_classes, in_channels=1, fs=64, conv_stem=True, heads=4, act="leaky",
                 gamma_init=0.0):
        super().__init__()
        def cstage(i, o):
            return Stage(i, o, mode="conv", act=act)
        def gate(ch):
            return LinAttnGate(ch, heads=heads, gamma_init=gamma_init)

        self.stem = ConvUnit(in_channels, fs, act=act) if conv_stem else nn.Identity()
        enc1_in = fs if conv_stem else in_channels
        # encoder
        self.enc1 = cstage(enc1_in, fs)
        self.enc2 = cstage(fs, fs * 2)
        self.enc3 = cstage(fs * 2, fs * 4)
        self.enc4 = cstage(fs * 4, fs * 8)
        self.pool = nn.MaxPool2d(2, stride=2)
        # bottleneck
        self.center = cstage(fs * 8, fs * 16)
        # decoder (transpose-conv up, linear-attention-gated skip, conv stage)
        self.up4 = nn.ConvTranspose2d(fs * 16, fs * 8, 2, stride=2)
        self.gate4 = gate(fs * 8)
        self.dec4 = cstage(fs * 16, fs * 8)
        self.up3 = nn.ConvTranspose2d(fs * 8, fs * 4, 2, stride=2)
        self.gate3 = gate(fs * 4)
        self.dec3 = cstage(fs * 8, fs * 4)
        self.up2 = nn.ConvTranspose2d(fs * 4, fs * 2, 2, stride=2)
        self.gate2 = gate(fs * 2)
        self.dec2 = cstage(fs * 4, fs * 2)
        self.up1 = nn.ConvTranspose2d(fs * 2, fs, 2, stride=2)
        self.gate1 = gate(fs)
        self.dec1 = cstage(fs * 2, fs)
        self.final = nn.Conv2d(fs, num_classes, 1)

    # accepted for API parity with MorphUNet (no morphology/checkpointing here) -> no-ops
    def set_checkpointing(self, flag):
        pass

    def set_beta(self, beta):
        pass

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        u4 = self.up4(c);  d4 = self.dec4(torch.cat([u4, self.gate4(u4, e4)], 1))
        u3 = self.up3(d4); d3 = self.dec3(torch.cat([u3, self.gate3(u3, e3)], 1))
        u2 = self.up2(d3); d2 = self.dec2(torch.cat([u2, self.gate2(u2, e2)], 1))
        u1 = self.up1(d2); d1 = self.dec1(torch.cat([u1, self.gate1(u1, e1)], 1))
        return self.final(d1)


#
# self-test:  python networks/linear_attention.py
#
if __name__ == "__main__":
    import time
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 128, 128, device=dev)
    net = LinAttnUNet(num_classes=3, in_channels=1, conv_stem=True, heads=4).to(dev)
    total = sum(p.numel() for p in net.parameters())
    attn = sum(p.numel() for m in net.modules() if isinstance(m, LinearAttention2d)
               for p in m.parameters())
    net.train()
    out = net(x); out.mean().backward()                 # warmup / graph
    gammas = [round(m.gamma.item(), 4) for m in net.modules() if isinstance(m, LinAttnGate)]
    print(f"device={dev}  params={total/1e6:.2f}M  attention={attn/1e3:.1f}k")
    print(f"out={tuple(out.shape)}  finite={torch.isfinite(out).all().item()}  gamma(init)={gammas}")
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        net.zero_grad(set_to_none=True); net(x).mean().backward()
    if dev.type == "cuda": torch.cuda.synchronize()
    print(f"fwd+bwd {(time.time()-t0)/5*1000:.1f} ms/iter (b=2, 128x128)")
