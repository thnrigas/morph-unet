#
# SE asymmetry of the morphological-attention gates in trained checkpoints
#
# Before the reflection fix (--morph-attn-reflect off, every earlier unet_morphattn* run), the gate's
# dilation used the SE g(+o) instead of g(-o). The "opening" then stamps back the MIRRORED SE: it can
# exceed the input by up to
#
#   asym = max_o [ g(o) - g(-o) ]      (>= 0, per channel; 0 iff the SE is symmetric)
#
# and the top-hat can go negative by the same amount (the closing/bottom-hat likewise). At the zero init
# asym = 0, so this measures how far training moved the SEs away from symmetry, i.e. how much the old
# hats could differ from true top-hats/bottom-hats.
#
# Columns, per gate (att1..att4 = enc1..enc4 skips):
#   asym mean/max : asym over the channels, in feature-value units (the hard-morphology bound)
#   |se| mean     : typical SE magnitude, for scale
#   rel mean      : mean over channels of asym / (max g - min g); 0 = symmetric, 1 = the asymmetry is as
#                   large as the SE's whole range
#   beta          : the temperature stored in the checkpoint (the end of the anneal)
#
# run:  python3 utilities/se_asymmetry.py results/Task08_HepaticVessel/unet_morphattn_f0_best.pth [...]
#

import argparse

import torch


def load_state(path):
    ck = torch.load(path, map_location="cpu")
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck   # _last.pth holds full state


def report(path):
    sd = load_state(path)
    keys = sorted(k for k in sd if k.startswith("att") and k.endswith(".morph.se"))
    print(f"\n{path}")
    if not keys:
        print("  no morphological-attention gates (att*.morph.se) in this checkpoint")
        return
    warm = any(k.startswith("att") and k.endswith(".g") for k in sd)
    print(f"  gate form: {'1 + g*tanh (warm)' if warm else '2*sigmoid (identity init)'}")
    print(f"  {'gate':5s} {'C':>5s} {'k':>3s} {'asym mean':>10s} {'asym max':>9s} "
          f"{'|se| mean':>10s} {'rel mean':>9s} {'beta':>6s}")
    for key in keys:
        se = sd[key].float()[0, :, :, 0, 0]                      # (C, k*k), offsets row-major
        k = int(round(se.shape[1] ** 0.5))
        asym = (se - se.flip(1)).amax(dim=1)                     # flip(1): offset o -> -o
        spread = (se.amax(dim=1) - se.amin(dim=1)).clamp_min(1e-12)
        gate = key.split(".")[0]
        beta = sd.get(f"{gate}.morph.beta")
        beta = f"{float(beta):6.1f}" if beta is not None else "     -"
        print(f"  {gate:5s} {se.shape[0]:5d} {k:3d} {asym.mean().item():10.4f} {asym.max().item():9.4f} "
              f"{se.abs().mean().item():10.4f} {(asym / spread).mean().item():9.3f} {beta}")


def main():
    p = argparse.ArgumentParser(description="SE asymmetry of the morphological-attention gates")
    p.add_argument("checkpoints", nargs="+", help="_best.pth (state dict) or _last.pth (full state)")
    for path in p.parse_args().checkpoints:
        report(path)


if __name__ == "__main__":
    main()
