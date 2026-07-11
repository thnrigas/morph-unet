#!/usr/bin/env python3
"""Slide 11 — per-layer channel survival (global keep 0.1) as LINE plots.
morphological heavy and linear (heavy)=convsep read their per-layer widths straight from the
_prune.json files. full_l2's summary jsons lack per-layer widths, so those were recovered from
the prune log into full_l2_widths.json (one entry per global scheme)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
OUT="/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"
INK="#20222b"; MUT="#5b6672"
CRIT=[("l1x1","l1x1g","#9aa6b2"),("lin","ling","#7fb0d6"),("act","actg","#c9a24b"),
      ("fb","fbg","#0F766E")]   # fb bold

def _plot_series(order,series,crits,out,title):
    fig,ax=plt.subplots(figsize=(14.5,6.6),dpi=200)
    x=np.arange(len(order))
    cidx=[i for i,L in enumerate(order) if L.startswith("center")]
    if cidx: ax.axvspan(min(cidx)-0.5,max(cidx)+0.5,color="#0f766e10",zorder=0)
    for lab,key,col in crits:
        if lab not in series: continue
        ys,c=series[lab]; bold=lab=="fb"
        ax.plot(x,ys,marker="o",ms=7 if bold else 5.5,lw=3.4 if bold else 2.1,
                color=c,label=lab,zorder=4 if bold else 3,
                markeredgecolor="white",markeredgewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels([L.replace(".sub",".") for L in order],
                    rotation=55,ha="right",fontsize=11.5)
    ax.tick_params(axis="y",labelsize=12.5)
    ax.set_ylabel("% channels kept (global, keep 0.1)",fontsize=14)
    ax.set_title(title,fontsize=16,fontweight="bold",color=INK,loc="left")
    if cidx: ax.text((min(cidx)+max(cidx))/2, ax.get_ylim()[1]*0.95,"center",
                     ha="center",va="top",fontsize=13,color="#0f766e",fontweight="bold")
    ax.legend(frameon=False,ncol=4,fontsize=13,loc="upper right")
    ax.grid(True,axis="y",color="#e7e7ec",lw=0.8,zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.margins(x=0.01)
    fig.savefig(f"{OUT}/{out}.png",bbox_inches="tight",facecolor="white")
    plt.close(fig); print("saved",out)

def lineplot(tag,fold,crits,out,title):
    """per-layer widths from each scheme's _prune.json."""
    order=None; series={}
    for lab,key,col in crits:
        f=f"{RES}/{tag}_prune-{key}-k10_f{fold}_prune.json"
        if not os.path.exists(f): continue
        d=json.load(open(f)); lw=d.get("layer_widths"); bf=d.get("layer_in_before")
        if not lw: continue
        if order is None: order=list(lw.keys())
        series[lab]=([100*lw[L]/bf[L] for L in order],col)
    _plot_series(order,series,crits,out,title)

def lineplot_recovered(widths_json,crits,out,title):
    """per-layer widths recovered from the prune log, keyed by global-scheme (ling/actg/...)."""
    W=json.load(open(widths_json)); order=None; series={}
    for lab,key,col in crits:
        if key not in W: continue
        lw=W[key]["layer_widths"]; bf=W[key]["layer_in_before"]
        if order is None: order=list(lw.keys())
        series[lab]=([100*lw[L]/bf[L] for L in order],col)
    _plot_series(order,series,crits,out,title)

lineplot("morphunet_heavy",0,CRIT,"chan_heavy",
         "In morphological (heavy), fb starves the center layers")
lineplot("convsep_heavy",1,[c for c in CRIT if c[0]!="l1x1"],"chan_linear",
         "Linear (heavy) shows the same center-pruning pattern")
lineplot_recovered(f"{HERE}/full_l2_widths.json",CRIT,"chan_full_l2",
         "full_l2 repeats it: every criterion collapses the center, keeps the shallow layers")
