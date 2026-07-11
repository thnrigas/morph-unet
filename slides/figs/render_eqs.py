#!/usr/bin/env python3
# Render the math-heavy pieces (equations + scoring-schemes table) as crisp transparent PNGs
# so the PPTX keeps correct notation while the rest of the deck stays native/editable.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/Kasimatis/Documents/kasimat/morph-unet/slides/figs"
NAVY = "#0F244F"; GREY = "#4A4A4A"; GOLD = "#C9A227"

def render_eq(tex, name, fontsize=26, color=NAVY):
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, tex, fontsize=fontsize, color=color)
    fig.savefig(f"{OUT}/{name}.png", dpi=230, transparent=True,
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("saved", name)

# slide 2 — parameter-count reduction (mathtext has no \underbrace -> compose with under-labels)
figp, axp = plt.subplots(figsize=(11.0, 1.55), dpi=230)
axp.axis("off"); axp.set_xlim(0, 1); axp.set_ylim(0, 1)
axp.text(0.23, 0.62, r"$3\cdot 3\cdot C_{\mathrm{in}}\cdot C_{\mathrm{out}}$", fontsize=27, color=NAVY, ha="center", va="center")
axp.text(0.23, 0.14, "standard conv", fontsize=13, color=GREY, ha="center", va="center")
axp.annotate("", xy=(0.53, 0.62), xytext=(0.44, 0.62),
             arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.0))
axp.text(0.79, 0.62, r"$3\cdot 3\cdot C_{\mathrm{in}}+C_{\mathrm{in}}\cdot C_{\mathrm{out}}$", fontsize=27, color=NAVY, ha="center", va="center")
axp.text(0.79, 0.14, "compact block", fontsize=13, color=GOLD, ha="center", va="center", fontweight="bold")
figp.savefig(f"{OUT}/eq_params.png", dpi=230, transparent=True, bbox_inches="tight", pad_inches=0.06)
plt.close(figp); print("saved eq_params")

# slide 3 — MPM neuron
render_eq(r"$\delta(x)=\max_k\,(x_k+w_k)\qquad\varepsilon(x)=\min_k\,(x_k+w_k)"
          r"\qquad\mathrm{MPM}(x)=\alpha\,[\,\delta(x)+\varepsilon(x)\,]$",
          "eq_mpm", fontsize=24)

# slide 8 — the four base scoring schemes (formula table)
rows = [
    ("l1x1",  r"$\| \mathrm{proj}_i\|\cdot|\alpha_i|\cdot \mathrm{spread}(SE_i)$",       "μόνο weights · data-free"),
    ("lin",   r"$\| \mathrm{proj}_i\|\cdot|\alpha_i|$",                                   "linear pathway · morphology-agnostic"),
    ("act",   r"$\| \mathrm{proj}_i\|\cdot \mathbb{E}\,|\mathrm{morph}_i(x)|$",           "μετρημένο activation"),
    ("morph", r"$|\alpha_i|\cdot \mathrm{spread}(SE_i)\cdot r_i$",                         "morphology-specific  ( $r_i$ = off-centre win-rate )"),
]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12.0, 3.1), dpi=230)
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
x_sch, x_form, x_note = 0.015, 0.135, 0.62
# header
yh = 0.90
ax.text(x_sch, yh, "scheme", fontsize=16, color=NAVY, fontweight="bold", fontfamily="monospace", va="center")
ax.text(x_form, yh, "score ανά input channel", fontsize=16, color=NAVY, fontweight="bold", va="center")
ax.plot([0.01, 0.99], [yh-0.09]*2, color=GOLD, lw=2.2)
rh = 0.185
for i, (sch, form, note) in enumerate(rows):
    y = yh - 0.24 - i*rh
    ax.text(x_sch, y, sch, fontsize=15.5, color=NAVY, fontweight="bold", fontfamily="monospace", va="center")
    ax.text(x_form, y, form, fontsize=19, color=NAVY, va="center")
    ax.text(x_note, y, note, fontsize=11.5, color=GREY, va="center")
    if i < len(rows)-1:
        ax.plot([0.01, 0.99], [y-rh/2]*2, color="#D8DCE3", lw=0.8)
fig.savefig(f"{OUT}/schemes_table.png", bbox_inches="tight", pad_inches=0.05, transparent=True)
plt.close(fig)
print("saved schemes_table")
