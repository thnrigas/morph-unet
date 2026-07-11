#!/usr/bin/env python3
"""Slide 7 table — Dice (Vessel/Tumour/macro) + ASSD + params for baseline vs Compact-CNN(linear)
vs the morphological variants. mean±std over folds where >1 fold exists. mpm_fast excluded (mixed)."""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
PAR={"convsep_heavy":5.99,"mpm_bottleneck":18.47,"mpm_deep":9.06,
     "mpm_full_l2":6.71,"morphunet_heavy":6.00,"mpm_balanced":30.32,
     "unet_baseline":31.03}
def stats(tag):
    dv,dt,av,at=[],[],[],[]
    for f in sorted(glob.glob(f"{RES}/{tag}_f*_scores.json")):
        r=json.load(open(f))["results"]["mean"]; v,t=r["Vessel"],r["Tumour"]
        dv.append(v["Dice"]); dt.append(t["Dice"])
        av.append(v["Avg. Symmetric Surface Distance"]); at.append(t["Avg. Symmetric Surface Distance"])
    n=len(dv); mac=[(a+b)/2 for a,b in zip(dv,dt)]
    f=lambda xs: (sum(xs)/len(xs), st.pstdev(xs) if len(xs)>1 else 0)
    return n,f(dv),f(dt),f(mac),f(av),f(at)

# (label, tag or None for baseline hardcoded, group)
ROWS=[
 ("Baseline U-Net","__base__","Baseline"),
 ("Baseline residual U-Net","unet_baseline","Baseline"),
 ("Compact-CNN  (linear)","convsep_heavy","Compact-CNN — linear"),
 ("bottleneck","mpm_bottleneck","Morphological-CNN"),
 ("deep","mpm_deep","Morphological-CNN"),
 ("full_l2","mpm_full_l2","Morphological-CNN"),
 ("heavy","morphunet_heavy","Morphological-CNN"),
 ("balanced","mpm_balanced","Morphological-CNN"),
]
def fmt(m,s,d=3): return f"{m:.{d}f}" + (f"±{s:.{d}f}" if s>0 else "")
data=[]
for lab,tag,grp in ROWS:
    if tag=="__base__":
        data.append((lab,grp,"31.07","1","0.467","0.301","0.384","7.09","54.65"))
    else:
        n,(mv,sv),(mt,st_),(mc,sc),(av,sav),(at,sat)=stats(tag)
        data.append((lab,grp,f"{PAR[tag]:.2f}",str(n),
                     fmt(mv,sv),fmt(mt,st_),fmt(mc,sc),fmt(av,sav,2),fmt(at,sat,2)))

# ---- draw ----
plt.rcParams.update({"font.family":"DejaVu Sans"})
INK="#20222b"; MUT="#5b6672"; LINE="#d9dde2"; BEST="#1f8f7a"; BAD="#c02d3b"
cols=["model","params\n(M)","folds","Dice\nVessel","Dice\nTumour","Dice\nmacro","ASSD\nVessel ↓","ASSD\nTumour ↓"]
cw=[0.185,0.075,0.06,0.135,0.135,0.135,0.11,0.11]  # sums ~0.945; wider Dice cols so ±std don't collide
x0=0.02; xs=[x0];
for w in cw: xs.append(xs[-1]+w)
nrows=len(data); ngrp=len(set(d[1] for d in data))
units=nrows+0.62*ngrp
fig_h=1.5+0.52*units
fig,ax=plt.subplots(figsize=(15.0,fig_h),dpi=200); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
top=0.90; rh=(top-0.12)/(units+1.0)
# header
yh=top
for i,c in enumerate(cols):
    ax.text(xs[i]+cw[i]/2, yh, c, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK, linespacing=0.95)
ax.plot([x0,xs[-1]],[yh-rh*0.55]*2,color=INK,lw=1.4)
# rows
best_macro=max(float(d[6].split("±")[0]) for d in data)
prev_grp=None
y=yh-rh*0.9
for d in data:
    lab,grp,par,nf,dvv,dtt,dmc,asv,ast=d
    if grp!=prev_grp:
        ax.text(x0, y+rh*0.05, grp.upper(), ha="left", va="center", fontsize=10,
                color=BEST if "linear" in grp else MUT, fontweight="bold",
                fontfamily="monospace")
        y-=rh*0.62; prev_grp=grp
    vals=[lab,par,nf,dvv,dtt,dmc,asv,ast]
    is_best = abs(float(dmc.split("±")[0])-best_macro)<1e-6
    is_bad = float(dmc.split("±")[0])<0.25
    for i,v in enumerate(vals):
        col=INK; fw="normal"; ha="center"
        if i==0: ha="left"; fw="bold" if is_best else "normal"
        if i==5:  # macro
            if is_best: col=BEST; fw="bold"
            elif is_bad: col=BAD; fw="bold"
        xx = xs[i]+ (0.004 if i==0 else cw[i]/2)
        ax.text(xx, y, v, ha=ha, va="center", fontsize=12.0, color=col,
                fontweight=fw, fontfamily="monospace" if i>0 else "sans-serif")
    ax.plot([x0,xs[-1]],[y-rh*0.5]*2,color=LINE,lw=0.7)
    y-=rh
ax.set_title("Segmentation quality vs. parameter count  ·  MSD Task08 hepatic vessel (test set)",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", x=x0, y=0.965)
ax.text(x0,0.03,"Dice: higher better · ASSD (avg symmetric surface dist, mm): lower better · "
        "±std shown when >1 fold (see 'folds' column)",
        fontsize=8.6,color=MUT)
fig.savefig(f"{RES}/../slides/figs/tbl_slide7_results.png",bbox_inches="tight",facecolor="white")
print("saved tbl_slide7_results.png")
for d in data: print(d)
