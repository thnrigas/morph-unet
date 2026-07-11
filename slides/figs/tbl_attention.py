#!/usr/bin/env python3
"""Attention-results table PNG (matched to tbl_slide7 style): baseline vs morphological vs linear
attention on the residual U-Net. mean±std over available folds. Vessel/Tumour/macro Dice + ASSD."""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
OUT="/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"

def stats(tag):
    dv,dt,av,at=[],[],[],[]
    for f in sorted(glob.glob(f"{RES}/{tag}_f*_scores.json")):
        r=json.load(open(f))["results"]["mean"]; v,t=r["Vessel"],r["Tumour"]
        dv.append(v["Dice"]); dt.append(t["Dice"])
        av.append(v["Avg. Symmetric Surface Distance"]); at.append(t["Avg. Symmetric Surface Distance"])
    n=len(dv); mac=[(a+b)/2 for a,b in zip(dv,dt)]
    f=lambda xs: (sum(xs)/len(xs), st.pstdev(xs) if len(xs)>1 else 0)
    return n,f(dv),f(dt),f(mac),f(av),f(at)

ROWS=[("Baseline residual U-Net","unet_baseline"),
      ("Morphological attention","unet_morphattn"),
      ("Linear attention (skip γ₀=0.5)","unet_linattn_g05")]
def fmt(m,s,d=3): return f"{m:.{d}f}" + (f"±{s:.{d}f}" if s>0 else "")
data=[]
for lab,tag in ROWS:
    n,(mv,sv),(mt,st_),(mc,sc),(av,sav),(at,sat)=stats(tag)
    data.append((lab,str(n),fmt(mv,sv),fmt(mt,st_),fmt(mc,sc),fmt(av,sav,2),fmt(at,sat,2)))

plt.rcParams.update({"font.family":"DejaVu Sans"})
INK="#20222b"; MUT="#5b6672"; LINE="#d9dde2"; BEST="#1f8f7a"
cols=["model","folds","Dice\nVessel","Dice\nTumour","Dice\nmacro","ASSD\nVessel ↓","ASSD\nTumour ↓"]
cw=[0.315,0.07,0.13,0.13,0.13,0.105,0.105]
x0=0.02; xs=[x0]
for w in cw: xs.append(xs[-1]+w)
nrows=len(data)
fig_h=1.4+0.62*(nrows+1)
fig,ax=plt.subplots(figsize=(14.0,fig_h),dpi=200); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
top=0.88; rh=(top-0.16)/(nrows+1.0)
yh=top
for i,c in enumerate(cols):
    ax.text(xs[i]+cw[i]/2, yh, c, ha="center", va="center", fontsize=13,
            fontweight="bold", color=INK, linespacing=0.95)
ax.plot([x0,xs[-1]],[yh-rh*0.55]*2,color=INK,lw=1.4)
best_v=max(float(d[2].split("±")[0]) for d in data)
y=yh-rh
for d in data:
    lab,nf,dvv,dtt,dmc,asv,ast=d
    vals=[lab,nf,dvv,dtt,dmc,asv,ast]
    is_best=abs(float(dvv.split("±")[0])-best_v)<1e-6
    for i,v in enumerate(vals):
        col=INK; fw="normal"; ha="center"
        if i==0: ha="left"; fw="bold"
        if i==2 and is_best: col=BEST; fw="bold"
        xx=xs[i]+(0.004 if i==0 else cw[i]/2)
        ax.text(xx, y, v, ha=ha, va="center", fontsize=13, color=col,
                fontweight=fw, fontfamily="monospace" if i>0 else "sans-serif")
    ax.plot([x0,xs[-1]],[y-rh*0.5]*2,color=LINE,lw=0.7)
    y-=rh
ax.set_title("Morphological vs. linear attention on the residual U-Net  ·  MSD Task08 hepatic vessel (test set)",
             fontsize=12.5, fontweight="bold", color=INK, loc="left", x=x0, y=0.97)
ax.text(x0,0.05,"Dice: higher better · ASSD (mm): lower better · ±std shown when >1 fold (see 'folds')",
        fontsize=9,color=MUT)
fig.savefig(f"{OUT}/tbl_attention_results.png",bbox_inches="tight",facecolor="white")
print("saved tbl_attention_results.png"); [print(d) for d in data]
