"""
Fig. 1 concept schematic: two-segment chain with g(x) = g1 (d1 sites), g2 (d2 sites).
(a) chain + coupling profile, quadrupole |p_y> injected.
(b) g2/g1 = -1  -> pseudospin echo, Delta phi = 0, <Lz> = 0      (no filtering)
(c) g2/g1 != -1 tuned so Delta phi = pi/2 -> <Lz> = +-1           (filtering)
Output: fig1_concept.pdf (PRB two-column width).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle, FancyArrowPatch

plt.rcParams.update({
    "font.family": "STIXGeneral", "font.size": 12, "font.weight": "bold",
    "mathtext.fontset": "custom", "mathtext.rm": "STIXGeneral:bold",
    "mathtext.it": "STIXGeneral:italic:bold", "mathtext.bf": "STIXGeneral:italic:bold",
    "mathtext.default": "it",
    "axes.linewidth": 0.6,
})
C1, C2, CL = "#c0392b", "#2e6fb7", "#8c8c8c"      # g1, g2, leads
POS, NEG = "#e8a33d", "#6a4c93"                    # orbital lobe signs


# ---------------------------------------------------------------- helpers
def p_orbital(ax, x, y, s=0.35, vertical=True, z=5):
    """Dumbbell p orbital (p_y if vertical, p_x otherwise)."""
    dx, dy = (0, s * 0.62) if vertical else (s * 0.62, 0)
    w, h = (s * 0.55, s) if vertical else (s, s * 0.55)
    ax.add_patch(Ellipse((x + dx, y + dy), w, h, fc=POS, ec="k", lw=0.5, zorder=z))
    ax.add_patch(Ellipse((x - dx, y - dy), w, h, fc=NEG, ec="k", lw=0.5, zorder=z))


def lz_orbital(ax, x, y, r=0.38, z=5):
    """p_x -/+ i p_y: ring with circulation arrow (orbital dipole)."""
    ax.add_patch(Circle((x, y), r, fill=False, ec=POS, lw=4.5, zorder=z))
    ax.add_patch(Circle((x, y), r, fill=False, ec="k", lw=0.4, zorder=z + 1))
    t = np.linspace(np.pi * 0.15, np.pi * 1.55, 60)
    ax.plot(x + 1.45 * r * np.cos(t), y + 1.45 * r * np.sin(t), c="k", lw=0.8, zorder=z)
    ax.annotate("", xy=(x + 1.45 * r * np.cos(t[-1] + .2), y + 1.45 * r * np.sin(t[-1] + .2)),
                xytext=(x + 1.45 * r * np.cos(t[-1]), y + 1.45 * r * np.sin(t[-1])),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="k", mutation_scale=8))


def spiral_arc(ax, th0, dth, r0, r1, color, lw=1.8):
    """Arc on the pseudospin circle; angle measured from +sigma_y axis (horizontal)."""
    t = np.linspace(th0, th0 + dth, 300)
    r = np.linspace(r0, r1, t.size)
    X, Y = r * np.cos(t), r * np.sin(t)
    ax.plot(X, Y, c=color, lw=lw, solid_capstyle="round", zorder=4)
    ax.annotate("", xy=(X[-1], Y[-1]), xytext=(X[-6], Y[-6]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11), zorder=4)
    return X[-1], Y[-1], t[-1]


def pseudospin_panel(ax, phi1, phi2, out, label, cond, lz_text, verdict):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.6, 4.5); ax.set_ylim(-1.55, 1.55)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="0.6", lw=0.9, ls="--"))
    ax.plot([-1.1, 1.1], [0, 0], c="0.8", lw=0.5); ax.plot([0, 0], [-1.1, 1.1], c="0.8", lw=0.5)
    ax.text(0, 1.12, r"$|p_y\rangle$", ha="center", va="bottom", fontsize=11)
    ax.text(0, -1.12, r"$|p_x\rangle$", ha="center", va="top", fontsize=11)
    ax.plot(0, 1, "o", c="k", ms=4, zorder=8)
    _, _, th = spiral_arc(ax, np.pi / 2, phi1, 1.0, 0.88, C1)
    x, y, _ = spiral_arc(ax, th, phi2, 0.80, 0.70, C2)
    ax.plot(x, y, "o", mfc="w", mec="k", ms=4.5, zorder=8)
    ax.add_patch(FancyArrowPatch((1.45, 0), (2.45, 0), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.0, color="k"))
    if out == "py":
        p_orbital(ax, 3.4, 0.1, s=0.42, vertical=True)
    else:
        lz_orbital(ax, 3.4, 0.1)
    ax.text(3.4, -1.1, lz_text, ha="center", fontsize=11.5)
    ax.text(-1.6, 1.55, label, ha="left", va="top", fontsize=12.5, weight="bold")
    ax.text(1.95, 1.4, cond, ha="center", va="top", fontsize=11.5)
    ax.text(1.95, 0.22, verdict, ha="center", fontsize=11.5,
            color="k" if out == "py" else C1)


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(3.4, 5.1))
gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.0, 1.0], hspace=0.06,
                      left=0.01, right=0.99, top=0.99, bottom=0.01)
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

# ---- (a) chain
ax = axA; ax.axis("off"); ax.set_aspect("equal")
nL, n1, n2, nR = 2, 4, 4, 2
xs = np.arange(nL + n1 + n2 + nR) * 1.0 + 0.5
ax.set_xlim(-1.9, xs[-1] + 1.2); ax.set_ylim(-1.75, 3.75)
cols = [CL] * nL + [C1] * n1 + [C2] * n2 + [CL] * nR
x1a, x1b = xs[nL] - 0.5, xs[nL + n1 - 1] + 0.5
x2a, x2b = xs[nL + n1] - 0.5, xs[nL + n1 + n2 - 1] + 0.5
ax.add_patch(Rectangle((x1a, -0.5), x1b - x1a, 1.0, fc=C1, alpha=0.10, ec="none"))
ax.add_patch(Rectangle((x2a, -0.5), x2b - x2a, 1.0, fc=C2, alpha=0.10, ec="none"))
ax.plot([xs[0] - 0.8, xs[-1] + 0.8], [0, 0], c="k", lw=0.8, zorder=1)
for x, c in zip(xs, cols):
    ax.add_patch(Circle((x, 0), 0.3, fc=c, ec="k", lw=0.5, zorder=3))
for a, b, lab, c in [(x1a, x1b, r"$d_1$", C1), (x2a, x2b, r"$d_2$", C2)]:
    ax.annotate("", xy=(a + 0.05, -0.85), xytext=(b - 0.05, -0.85),
                arrowprops=dict(arrowstyle="|-|", lw=0.7, color=c, mutation_scale=3))
    ax.text((a + b) / 2, -1.0, lab, ha="center", va="top", color=c, fontsize=12)
# g(x) profile
y0, h = 2.55, 0.6
ax.plot([xs[0] - 0.8, x1a, x1a, x1b, x1b], [y0, y0, y0 + h, y0 + h, y0], c="k", lw=1.0)
ax.plot([x1b, x2a, x2a, x2b, x2b, xs[-1] + 0.8], [y0, y0, y0 - h, y0 - h, y0, y0], c="k", lw=1.0)
for f in (0.55, 1.45):
    ax.plot([x2a, x2b], [y0 - f * h] * 2, c=C2, lw=0.9, ls="--")
ax.annotate("", xy=(x2b + 0.4, y0 - 1.6 * h), xytext=(x2b + 0.4, y0 - 0.4 * h),
            arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.18", lw=1.0, color=C2, mutation_scale=12, shrinkA=0, shrinkB=0))
ax.text(xs[0] - 0.8, y0 + 0.1, r"$g(x)$", fontsize=12, va="bottom")
ax.text((x1a + x1b) / 2, y0 + h + 0.08, r"$g_1$", ha="center", va="bottom", color=C1, fontsize=12)
ax.text((x2a + x2b) / 2, y0 - 1.45 * h - 0.08, r"$g_2$", ha="center", va="top", color=C2, fontsize=12)
# input
p_orbital(ax, -1.35, 0.0, s=0.42, vertical=True)
ax.text(-1.35, -0.8, r"$|p_y\rangle$", ha="center", va="top", fontsize=11.5)
ax.add_patch(FancyArrowPatch((-0.95, 0.0), (-0.3, 0.0), arrowstyle="-|>", mutation_scale=9, lw=1.0))
ax.add_patch(FancyArrowPatch((xs[-1] + 0.4, 0.0), (xs[-1] + 1.15, 0.0), arrowstyle="-|>",
                             mutation_scale=9, lw=1.0))
ax.text(-1.9, 3.75, "(a)", va="top", fontsize=12.5, weight="bold")

# ---- (b), (c) pseudospin precession
phi1 = 1.4 * np.pi
pseudospin_panel(axB, phi1, -phi1, "py", "(b)", r"$g_2=-g_1$",
                 r"$\langle L_z\rangle=0$", "no filtering")
pseudospin_panel(axC, phi1, -phi1 + np.pi / 2, "lz", "(c)", r"$g_2\neq-g_1$",
                 r"$\langle L_z\rangle=\pm1$", "filtering")

#fig.savefig("fig1_concept.pdf")
#fig.savefig("fig1_concept.png", dpi=250)
plt.show()