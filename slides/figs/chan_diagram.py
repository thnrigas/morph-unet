#!/usr/bin/env python3
"""Slide 11 — per-layer surviving channels under GLOBAL allocation (keep 0.1), deep & bottleneck.
Shows how each criterion distributes a shared budget across depth; fb starves the center layers.
Data recovered in deepbot_widths.json (global, keep 0.1)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W=json.load(open("/tmp/claude-1001/-home-Kasimatis-Documents-kasimat/"
                 "f6662d64-bc71-456a-915b-ec84152c4c68/scratchpad/deepbot_widths.json"))
OUT="/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"
ORIG={"enc4.sub1":256,"enc4.sub2":512,"center.sub1":512,"center.sub2":1024,
      "dec4.sub1":1024,"dec4.sub2":512}
ORDER={"mpm_deep":["enc4.sub1","enc4.sub2","center.sub1","center.sub2","dec4.sub1","dec4.sub2"],
       "mpm_bottleneck":["center.sub1","center.sub2"]}
CRIT=[("l1x1","l1x1g","#9aa6b2"),("lin","ling","#7fb0d6"),("act","actg","#c9a24b"),
      ("morph","morphg","#b39ddb"),("fb","fbg","#0F766E")]  # (legend, json-key, colour); fb = bold accent
INK="#20222b"; MUT="#5b6672"

def draw(tag,out,title):
    layers=ORDER[tag]; x=np.arange(len(layers)); n=len(CRIT); bw=0.16
    fig,ax=plt.subplots(figsize=(13.5 if tag=="mpm_deep" else 9.0,6.4),dpi=200)
    for c,(lab,key,col) in enumerate(CRIT):
        pct=[100*W[tag][key][L]/ORIG[L] for L in layers]
        bold = key=="fbg"
        ax.bar(x+(c-(n-1)/2)*bw, pct, bw, label=lab, color=col,
               edgecolor=INK if bold else "white", linewidth=1.3 if bold else 0.6,
               zorder=3 if bold else 2)
    # shade the center layers
    for i,L in enumerate(layers):
        if L.startswith("center"):
            ax.axvspan(i-0.5,i+0.5,color="#00000008",zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([L.replace(".","\n") for L in layers],fontsize=13)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_ylabel("% of channels kept  (global, keep 0.1)",fontsize=14)
    ax.set_title(title,fontsize=16,fontweight="bold",color=INK,loc="left")
    ax.legend(frameon=False,ncol=5,fontsize=13,loc="upper center",
              bbox_to_anchor=(0.5,-0.13),columnspacing=1.4)
    ax.grid(True,axis="y",color="#e7e7ec",lw=0.8,zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.margins(x=0.02)
    fig.savefig(f"{OUT}/{out}.png",bbox_inches="tight",facecolor="white")
    plt.close(fig); print("saved",out)

draw("mpm_deep","chan_deep","In deep, fb concentrates its budget in enc4 and starves the center")
draw("mpm_bottleneck","chan_bottleneck","In bottleneck, fb keeps sub1 and prunes sub2 hardest")
