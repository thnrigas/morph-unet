#!/usr/bin/env python3
"""Slide 11 — per-model pruning result tables as PNGs, LANDSCAPE (transposed):
rows = keep ratio 0.01..0.7, columns = scheme x {local, global} + random(local).
Cell = MACRO test Dice; background = Δ vs unpruned (good/warn/bad); 'NF' = kept w/o fine-tune;
'–' = not run. Wide aspect so it fits a 16:9 slide.
Usage:  python3 prune_tables.py [tag ...]   (default: all five)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
OUT="/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"
KEEPS=[0.01,0.03,0.05,0.1,0.3,0.5,0.7]
KK={0.01:"k01",0.03:"k03",0.05:"k05",0.1:"k10",0.3:"k30",0.5:"k50",0.7:"k70"}
MORPH=["l1x1","lin","morph","act","fb"]; LIN=["lin","act","fb"]
CFG={"mpm_deep":(2,MORPH,"prune_deep"),"mpm_bottleneck":(2,MORPH,"prune_bottleneck"),
     "mpm_full_l2":(0,MORPH,"prune_full_l2"),"morphunet_heavy":(0,MORPH,"prune_heavy"),
     "convsep_heavy":(1,LIN,"prune_convsep"),
     "convmpm_small":(0,[],"prune_convmpm_small"),
     "convmpm_small2m":(0,[],"prune_convmpm_small2m")}
# ConvMPM models were pruned only under GLOBAL allocation with the forward-backward family + random,
# so they get an explicit column set instead of the default local/global sweep.
CMPM_COLS=[("fb","global","g"),("fbfg","global","g"),("fbmorph","global","g"),("random","global","g")]
COLS_OVERRIDE={"convmpm_small":CMPM_COLS,"convmpm_small2m":CMPM_COLS}
BG={"good":"#e2f0ea","warn":"#f7edd8","bad":"#f7dedb","pend":"#f1f2f4","na":"#f1f2f4"}
FG={"good":"#157F52","warn":"#9A6B0E","bad":"#B23A2E","pend":"#9aa2ab"}
INK="#20222b"; MUT="#5b6672"; LINE="#d9dde2"

def load(stem):
    p=f"{RES}/{stem}_prune.json"
    if not os.path.exists(p): return None
    d=json.load(open(p)); sc=f"{RES}/{stem}_scores.json"; mac=None
    if os.path.exists(sc):
        r=json.load(open(sc)).get("results",{}).get("mean")
        if r: mac=(r["Vessel"]["Dice"]+r["Tumour"]["Dice"])/2
    return d,mac

def cls(dv): return "good" if dv>=-0.005 else ("warn" if dv>=-0.015 else "bad")
def base_macro(tag,fold):
    r=json.load(open(f"{RES}/{tag}_f{fold}_scores.json"))["results"]["mean"]
    return (r["Vessel"]["Dice"]+r["Tumour"]["Dice"])/2

def render(tag, fold=None, out=None):
    _fold,crits,_out=CFG[tag]
    fold = _fold if fold is None else fold
    out  = _out  if out  is None else out
    base=base_macro(tag,fold)
    if tag in COLS_OVERRIDE:
        cols=list(COLS_OVERRIDE[tag])
    else:
        cols=[]                          # (scheme, alloc, suffix)
        for m in crits: cols+=[(m,"local",""),(m,"global","g")]
        cols.append(("fbnew","global","g"))   # proposed scheme, global-only (extreme keep 0.01)
        cols.append(("fbfg","global","g"))    # foreground-restricted fixed fb (global)
        cols.append(("random","local",""))
        cols.append(("random","global","g"))  # random control under GLOBAL allocation (convsep extremes)
    nC=len(cols)

    def getcell(k,col):
        m,alloc,suf=col
        r=load(f"{tag}_prune-{m}{suf}-{KK[k]}_f{fold}")
        if not r: return ("–","pend",False)
        d,mac=r
        if mac is None: return ("·","pend",False)
        return (f"{mac:.3f}", cls(mac-base), d.get("ft_skipped",False))

    # rows are keep ratios. Drop a keep row only when it is entirely empty AND not one of the extreme
    # ratios 0.01/0.03/0.05 (those always stay, even if blank).
    ALWAYS={0.01,0.03,0.05}
    show=[k for k in KEEPS if k in ALWAYS or any(getcell(k,c)[0] not in ("–","·") for c in cols)]
    nR=len(show)

    labw=2.0; cw=2.05; rh=1.06        # wider columns + bigger rows for readability
    Wtab=labw+nC*cw
    top=2.4; bot=1.25                 # title band / legend band
    H=top+nR*rh+bot
    fig,ax=plt.subplots(figsize=(0.84*Wtab,0.84*H),dpi=200); ax.axis("off")
    ax.set_xlim(0,Wtab); ax.set_ylim(0,H)
    yhead=H-top+0.15
    # column headers (scheme over alloc)
    ax.text(labw/2, yhead, "keep", ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=INK)
    for j,(m,alloc,suf) in enumerate(cols):
        x=labw+j*cw+cw/2
        ax.text(x, yhead+0.32, m, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=INK, fontfamily="monospace")
        ax.text(x, yhead-0.22, alloc, ha="center", va="center", fontsize=10,
                color=(FG["good"] if alloc=="global" else MUT), fontfamily="monospace")
    ax.plot([0.1,Wtab-0.1],[yhead-0.58]*2,color=INK,lw=1.4)
    # rows
    for i,k in enumerate(show):
        y=yhead-1.2-i*rh
        ax.text(labw-0.18, y, f"{k:g}", ha="right", va="center", fontsize=13.5,
                fontweight="bold", color=INK, fontfamily="monospace")
        for j,col in enumerate(cols):
            txt,cl,nf=getcell(k,col); x=labw+j*cw
            ax.add_patch(Rectangle((x,y-0.49),cw,0.98,facecolor=BG[cl],
                         edgecolor="white",lw=1.1,zorder=1))
            ax.text(x+cw/2, y, txt, ha="center", va="center", fontsize=12.5,
                    fontfamily="monospace", color=FG[cl],
                    fontweight="bold" if cl!="pend" else "normal", zorder=2)
            if nf:
                ax.text(x+cw-0.08, y+0.34, "NF", ha="right", va="center", fontsize=8,
                        color=FG["good"], fontweight="bold", zorder=3)
    # title + subtitle + legend
    ax.text(0.05, H-0.62, tag, ha="left", va="center", fontsize=16.5,
            fontweight="bold", color=INK, fontfamily="monospace")
    ax.text(0.05, H-1.25, f"macro test Dice · unpruned {base:.3f} · Δ-coloured · NF = kept w/o fine-tune · – = not run",
            ha="left", va="center", fontsize=11, color=MUT)
    lx=labw
    for name,cl in [("Δ≥−0.005","good"),("−0.005…−0.015","warn"),("Δ<−0.015","bad"),("not run","pend")]:
        ax.add_patch(Rectangle((lx,0.32),0.58,0.58,facecolor=BG[cl],edgecolor=LINE,lw=0.9))
        ax.text(lx+0.74,0.61,name,ha="left",va="center",fontsize=11,color=MUT); lx+=3.4
    fig.savefig(f"{OUT}/{out}.png",bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(f"saved {out}.png  ({0.84*Wtab:.1f}x{0.84*H:.1f} in, aspect {Wtab/H:.2f}, rows {nR})")

if "--convsep-fold0" in sys.argv:
    # separate fold-0 convsep table (kept ALONGSIDE the fold-1 prune_convsep.png, not replacing it)
    render("convsep_heavy", fold=0, out="prune_convsep_f0")
else:
    for t in (sys.argv[1:] or list(CFG)): render(t)
