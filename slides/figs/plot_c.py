#!/usr/bin/env python3
"""Plot C — channel retention by resolution group at aggressive pruning (keep=0.1, global).
Morphological heavy vs linear convsep, averaged over the lin/act/fb criteria (error bar = spread
across criteria). The two profiles are near-identical -> deep-stage redundancy is a property of the
UNet + task, NOT of morphology. convsep is the control that proves it.
"""
import json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES="/home/Kasimatis/Documents/kasimat/morph-unet/results"
HIRES={"enc1","enc2","dec1","dec2"}; DEEP={"enc4","center","dec4"}
GROUPS=["HIGH-res","mid","DEEP / center"]
def grp(L):
    st=L.split(".")[0]
    return "HIGH-res" if st in HIRES else ("DEEP / center" if st in DEEP else "mid")

def retention(tag,fold):
    """returns {group: [pct over criteria]}"""
    out=collections.defaultdict(list)
    for crit in ["ling","actg","fbg"]:
        d=json.load(open(f"{RES}/{tag}_prune-{crit}-k10_f{fold}_prune.json"))
        aft,bef=d["layer_widths"],d["layer_in_before"]
        acc=collections.defaultdict(lambda:[0,0])
        for L in aft: acc[grp(L)][0]+=aft[L]; acc[grp(L)][1]+=bef[L]
        for g in GROUPS: out[g].append(100*acc[g][0]/acc[g][1])
    return out

morph=retention("morphunet_heavy","0"); lin=retention("convsep_heavy","1")
m_mean=[np.mean(morph[g]) for g in GROUPS]; m_err=[np.std(morph[g]) for g in GROUPS]
l_mean=[np.mean(lin[g])   for g in GROUPS]; l_err=[np.std(lin[g])   for g in GROUPS]

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
                     "axes.edgecolor":"#4a4a52","axes.linewidth":1.0})
INK="#20222b"; C_M="#1f8f7a"; C_L="#2f6fd6"
fig,ax=plt.subplots(figsize=(9.0,5.6),dpi=200)
x=np.arange(3); w=0.36
b1=ax.bar(x-w/2,m_mean,w,yerr=m_err,capsize=5,color=C_M,edgecolor=INK,lw=1.0,
          label="morphological (heavy)",zorder=3,error_kw=dict(elinewidth=1.3))
b2=ax.bar(x+w/2,l_mean,w,yerr=l_err,capsize=5,color=C_L,edgecolor=INK,lw=1.0,
          label="linear (heavy)",zorder=3,error_kw=dict(elinewidth=1.3))
for bars,means,errs in ((b1,m_mean,m_err),(b2,l_mean,l_err)):
    for r,mv,ev in zip(bars,means,errs):
        ax.text(r.get_x()+r.get_width()/2, mv+ev+0.7, f"{mv:.0f}%", ha="center",
                va="bottom", fontsize=11, fontweight="bold", color=INK)

ax.set_xticks(x); ax.set_xticklabels(
    ["HIGH-res\nenc1·enc2·dec1·dec2","mid\nenc3·dec3","DEEP / center\nenc4·center·dec4"],fontsize=11)
ax.set_ylabel("% channels retained  (global prune, keep 0.1)",fontsize=12.5)
ax.set_ylim(0,26)
ax.set_title("Both blocks prune the deep bottleneck ~3× harder than the high-res stages —\n"
             "so the redundancy is the U-Net's, not morphology's",
             fontsize=13.5,fontweight="bold",loc="left",color=INK,pad=12)
ax.legend(frameon=False,fontsize=11.5,loc="upper right")
ax.grid(True,axis="y",color="#e7e7ec",lw=0.8,zorder=0)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
# arrow annotating the essential vs redundant reading
ax.annotate("kept → essential", (0-w/2, m_mean[0]), xytext=(0-w/2, 24),
            ha="center", fontsize=10, color="#5b5e69")
ax.annotate("pruned away → redundant", (2, 12), xytext=(2, 15.5),
            ha="center", fontsize=10, color="#5b5e69")
fig.tight_layout()
out=f"{RES}/../slides/figs/plot_c_retention_by_resolution.png"
fig.savefig(out,bbox_inches="tight",facecolor="white")
print("saved",out)
print("morph",[f"{v:.1f}" for v in m_mean],"| convsep",[f"{v:.1f}" for v in l_mean])
