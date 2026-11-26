import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

DARK   = "#1a1a2e"
MID    = "#16213e"
LEAD   = "#0f3460"
ACCENT1= "#e94560"   # region 1 – warm red
ACCENT2= "#0f9b8e"   # region 2 – teal
LEAD_L = "#4a90d9"   # lead colour (light blue)
ARROW  = "#f5a623"   # arrow / electron colour
TEXT   = "#f0f0f0"
SUBTEXT= "#aaaaaa"
WHITE  = "#ffffff"

fig = plt.figure(figsize=(14, 7), facecolor=DARK)
fig.patch.set_facecolor(DARK)

ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_facecolor(DARK)
ax.axis("off")

# ── Helper: draw a row of lattice sites ────────────────────────────────────
def draw_chain(ax, x_start, x_end, y, n_sites, color, zorder=3, alpha=1.0):
    xs = np.linspace(x_start, x_end, n_sites)
    # hopping lines
    for i in range(len(xs)-1):
        ax.plot([xs[i], xs[i+1]], [y, y],
                color=color, lw=1.8, alpha=alpha*0.6, zorder=zorder-1)
    # sites
    for x in xs:
        circle = plt.Circle((x, y), 0.13, color=color,
                             zorder=zorder, alpha=alpha)
        ax.add_patch(circle)
        circle2 = plt.Circle((x, y), 0.13, fill=False,
                              edgecolor=WHITE, lw=0.5,
                              zorder=zorder+1, alpha=alpha*0.5)
        ax.add_patch(circle2)
    return xs

# ── Section labels background boxes ────────────────────────────────────────
def label_box(ax, x, y, text, color, fontsize=10, text_color=WHITE):
    bbox = dict(boxstyle="round,pad=0.4", facecolor=color,
                edgecolor="none", alpha=0.85)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", bbox=bbox, zorder=10)

# ═══════════════════════════════════════════════════════════════════════════
# TOP HALF – schematic diagram
# ═══════════════════════════════════════════════════════════════════════════
y_chain = 4.8
x_lead_L_start = 0.4
x_lead_L_end   = 2.8
x_reg1_start   = 3.0
x_reg1_end     = 5.6
x_reg2_start   = 5.8
x_reg2_end     = 8.4
x_lead_R_start = 8.6
x_lead_R_end   = 11.0

N_lead = 8
N_reg  = 9

# background highlight bands for regions
for xb, xe, col in [(x_reg1_start-0.15, x_reg1_end+0.15, ACCENT1),
                    (x_reg2_start-0.15, x_reg2_end+0.15, ACCENT2)]:
    rect = FancyBboxPatch((xb, y_chain-0.55), xe-xb, 1.1,
                           boxstyle="round,pad=0.08",
                           facecolor=col, edgecolor="none",
                           alpha=0.18, zorder=1)
    ax.add_patch(rect)

# chains
draw_chain(ax, x_lead_L_start, x_lead_L_end, y_chain, N_lead,  LEAD_L)
xs1 = draw_chain(ax, x_reg1_start,   x_reg1_end,   y_chain, N_reg,   ACCENT1)
xs2 = draw_chain(ax, x_reg2_start,   x_reg2_end,   y_chain, N_reg,   ACCENT2)
draw_chain(ax, x_lead_R_start, x_lead_R_end, y_chain, N_lead,  LEAD_L)

# connecting hops between sections
for (x0, x1) in [(x_lead_L_end, x_reg1_start),
                 (x_reg1_end,   x_reg2_start),
                 (x_reg2_end,   x_lead_R_start)]:
    ax.plot([x0, x1], [y_chain, y_chain],
            color=WHITE, lw=1.5, ls="--", alpha=0.35, zorder=2)

# ── orbital mixing squiggles above region sites ────────────────────────────
def draw_mixing_arc(ax, x, y, color, scale=0.3):
    """Small arc above site to suggest orbital mixing."""
    theta = np.linspace(0, np.pi, 40)
    rx, ry = scale*0.7, scale*0.5
    ax.plot(x + rx*np.cos(theta), y + 0.18 + ry*np.sin(theta),
            color=color, lw=1.2, alpha=0.7, zorder=5)

for x in xs1[::2]:
    draw_mixing_arc(ax, x, y_chain, ACCENT1)
for x in xs2[::2]:
    draw_mixing_arc(ax, x, y_chain, ACCENT2)

# ── Section labels ─────────────────────────────────────────────────────────
label_box(ax, (x_lead_L_start+x_lead_L_end)/2, y_chain-1.1,
          "Lead (L)", LEAD_L, fontsize=9)
label_box(ax, (x_reg1_start+x_reg1_end)/2,     y_chain-1.1,
          "Region 1\n$g_1,\\, \\chi_{xy}$", ACCENT1, fontsize=9)
label_box(ax, (x_reg2_start+x_reg2_end)/2,     y_chain-1.1,
          "Region 2\n$g_2,\\, \\chi_{xy}$", ACCENT2, fontsize=9)
label_box(ax, (x_lead_R_start+x_lead_R_end)/2, y_chain-1.1,
          "Lead (R)", LEAD_L, fontsize=9)

# dashed vertical separators
for xv in [x_reg1_start-0.15, x_reg1_end+0.15,
           x_reg2_start-0.15, x_reg2_end+0.15]:
    ax.plot([xv, xv], [y_chain-0.6, y_chain+0.6],
            color=WHITE, lw=0.8, ls=":", alpha=0.3, zorder=2)

# ── Electron arrow (incoming) ──────────────────────────────────────────────
ax.annotate("", xy=(x_lead_L_end-0.3, y_chain+0.55),
            xytext=(x_lead_L_start+0.1, y_chain+0.55),
            arrowprops=dict(arrowstyle="-|>", color=ARROW,
                            lw=2.0, mutation_scale=14),
            zorder=8)
ax.text((x_lead_L_start+x_lead_L_end)/2, y_chain+0.82,
        "electron\n$L_z=0$ (quadrupole)",
        ha="center", va="bottom", fontsize=8.5,
        color=ARROW, fontstyle="italic")

# ── Orbital state labels along chain ──────────────────────────────────────
state_y = y_chain + 1.35
for xpos, label, col in [
    ((x_lead_L_start+x_lead_L_end)/2,   "$L_z = 0$\nquadrupole",  LEAD_L),
    ((x_reg1_start+x_reg1_end)/2,        "$L_z = \\pm1$\ndipole",   ACCENT1),
    ((x_reg2_start+x_reg2_end)/2,        "$L_z = 0$ or $\\pm1$?\nfiltering?", ACCENT2),
    ((x_lead_R_start+x_lead_R_end)/2,    "$L_z = 0$ or $\\pm1$\noutcome",     LEAD_L),
]:
    ax.text(xpos, state_y, label, ha="center", va="bottom",
            fontsize=8, color=col, alpha=0.92,
            bbox=dict(facecolor=DARK, edgecolor=col,
                      alpha=0.5, boxstyle="round,pad=0.3", lw=0.8))

# transition arrows between state labels
for x0, x1 in [(x_lead_L_end, x_reg1_start),
               (x_reg1_end,   x_reg2_start),
               (x_reg2_end,   x_lead_R_start)]:
    ax.annotate("", xy=(x1+0.1, state_y+0.3),
                xytext=(x0-0.1, state_y+0.3),
                arrowprops=dict(arrowstyle="-|>", color=WHITE,
                                lw=1.0, mutation_scale=10, alpha=0.4),
                zorder=6)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(7.0, 6.75,
        "Two-Region Tight-Binding Model for Orbital Filtering",
        ha="center", va="top", fontsize=13, color=WHITE,
        fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# BOTTOM HALF – orbital state cartoons (px, py, dxy)
# ═══════════════════════════════════════════════════════════════════════════
y_orb = 1.55
orb_xs = [2.2, 5.2, 8.2, 11.2]
orb_labels = ["$p_x$\n(dipole)", "$p_y$\n(dipole)",
              "$d_{xy}$\n(quadrupole)", "$L_z=0$ mix\n(quadrupole)"]
orb_colors = [ACCENT1, ACCENT1, ACCENT2, LEAD_L]

def draw_px(ax, cx, cy, r=0.55, color=ACCENT1):
    theta = np.linspace(0, np.pi, 80)
    ax.fill(cx + r*np.cos(theta)*0.55, cy + r*np.sin(theta),
            color=color, alpha=0.7)
    ax.fill(cx - r*np.cos(theta)*0.55, cy - r*np.sin(theta),
            color=color, alpha=0.35)

def draw_py(ax, cx, cy, r=0.55, color=ACCENT1):
    theta = np.linspace(0, np.pi, 80)
    ax.fill(cx + r*np.sin(theta), cy + r*np.cos(theta)*0.55,
            color=color, alpha=0.7)
    ax.fill(cx - r*np.sin(theta), cy - r*np.cos(theta)*0.55,
            color=color, alpha=0.35)

def draw_dxy(ax, cx, cy, r=0.55, color=ACCENT2):
    for sign in [1, -1]:
        theta = np.linspace(0, np.pi/2, 60)
        xs_ = sign * r * np.cos(theta) * np.sin(theta) * 2
        ys_ = sign * r * (np.cos(theta)**2 - np.sin(theta)**2)
        ax.fill(cx + xs_, cy + ys_, color=color, alpha=0.7)
        ax.fill(cx - xs_, cy - ys_, color=color, alpha=0.35)

def draw_quadrupole_mix(ax, cx, cy, r=0.55, color=LEAD_L):
    # superposition circle with inner rings
    for rad, a in [(r, 0.25), (r*0.65, 0.35), (r*0.3, 0.5)]:
        circle = plt.Circle((cx, cy), rad, fill=False,
                             edgecolor=color, lw=1.5, alpha=a)
        ax.add_patch(circle)
    # cross
    for dx, dy in [(r*0.8, 0), (0, r*0.8), (-r*0.8, 0), (0, -r*0.8)]:
        ax.plot([cx, cx+dx], [cy, cy+dy],
                color=color, lw=1.2, alpha=0.6)

drawers = [draw_px, draw_py, draw_dxy, draw_quadrupole_mix]
for i, (cx, lbl, col, drawer) in enumerate(
        zip(orb_xs, orb_labels, orb_colors, drawers)):
    drawer(ax, cx, y_orb, color=col)
    ax.text(cx, y_orb - 0.85, lbl, ha="center", va="top",
            fontsize=8.5, color=col)

# mixing arrows between orbital cartoons
for i in range(len(orb_xs)-1):
    mid_x = (orb_xs[i] + orb_xs[i+1]) / 2
    ax.annotate("", xy=(orb_xs[i+1]-0.7, y_orb),
                xytext=(orb_xs[i]+0.7,   y_orb),
                arrowprops=dict(arrowstyle="<->", color=WHITE,
                                lw=1.2, mutation_scale=12, alpha=0.4))

ax.text(7.0, 0.18,
        "Orbital states involved in the quadrupole $\\leftrightarrow$ dipole conversion",
        ha="center", va="bottom", fontsize=9,
        color=SUBTEXT, fontstyle="italic")

# dividing line between top and bottom
ax.plot([0.3, 13.7], [2.85, 2.85],
        color=WHITE, lw=0.6, alpha=0.15, ls="--")

# ── Chi_xy label ───────────────────────────────────────────────────────────
ax.text(11.6, y_chain+0.1,
        "$\\chi_{xy}$ =\norbital\nmixing\nmatrix",
        ha="left", va="center", fontsize=8,
        color=WHITE, alpha=0.55,
        bbox=dict(facecolor=DARK, edgecolor=WHITE,
                  alpha=0.2, boxstyle="round,pad=0.3", lw=0.5))

plt.savefig("/mnt/user-data/outputs/two_region_model_figure.pdf",
            dpi=300, bbox_inches="tight", facecolor=DARK)
plt.savefig("/mnt/user-data/outputs/two_region_model_figure.png",
            dpi=300, bbox_inches="tight", facecolor=DARK)
print("Saved.")
