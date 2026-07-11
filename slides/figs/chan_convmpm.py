#!/usr/bin/env python3
"""Per-layer surviving channels for the ConvMPM models under GLOBAL allocation, comparing the
forward-backward family (fb / fbfg / fb-morph) with random. Reads layer_widths + layer_in_before
straight from results/*_prune.json, so it renders whatever prunes have finished (missing methods are
skipped). One panel per model, at a chosen keep ratio."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
OUT="/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"
INK="#20222b"; MUT="#5b6672"
# (legend, tag-suffix, colour); fb-morph is the bold accent -- the new morphological-routing scheme
CRIT=[("fb","fbg","#9aa6b2"),("fbfg","fbfgg","#7fb0d6"),
      ("fb-morph","fbmorphg","#0F766E"),("random","randomg","#c98a8a")]

def draw(tag, kk, out, title):
    order=None; series=[]
    for lab,suf,col in CRIT:
        p=f"{RES}/{tag}_prune-{suf}-{kk}_f0_prune.json"
        if not os.path.exists(p): continue
        d=json.load(open(p)); lw=d.get("layer_widths"); ib=d.get("layer_in_before")
        if not lw or not ib: continue
        if order is None: order=list(lw.keys())
        pct=[100*lw[L]/ib[L] for L in order]
        series.append((lab,col,pct,lab=="fb-morph"))
    if not series:
        print(f"skip {out}: no data yet"); return
    x=np.arange(len(order))
    fig,ax=plt.subplots(figsize=(15,6.2),dpi=200)
    for lab,col,pct,bold in series:
        ax.plot(x,pct,marker="o",ms=7 if bold else 5.5,lw=3.4 if bold else 2.0,
                color=col,label=lab,zorder=4 if bold else 3,
                markeredgecolor="white",markeredgewidth=0.7)
    cidx=[i for i,L in enumerate(order) if L.startswith("center")]
    if cidx: ax.axvspan(min(cidx)-0.5,max(cidx)+0.5,color="#0f766e10",zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([L.replace(".sub",".") for L in order],
                    rotation=55,ha="right",fontsize=10.5)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_ylabel(f"% channels kept  (global, keep {int(kk[1:])/100:g})",fontsize=13.5)
    ax.set_title(title,fontsize=15.5,fontweight="bold",color=INK,loc="left")
    ax.legend(frameon=False,ncol=4,fontsize=13,loc="upper right")
    ax.grid(True,axis="y",color="#e7e7ec",lw=0.8,zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
    fig.savefig(f"{OUT}/{out}.png",bbox_inches="tight",facecolor="white")
    plt.close(fig); print("saved",out,"(methods:",", ".join(s[0] for s in series)+")")

# keep 0.1 (k10) is the most aggressive shared budget -> where routing choices separate most
draw("convmpm_small","k10","chan_convmpm_small",
     "ConvMPM (fs=13): where each criterion spends a keep-0.1 budget across depth")
draw("convmpm_small2m","k10","chan_convmpm_small2m",
     "ConvMPM (fs=37, ~2M): per-layer channels kept at keep 0.1")
