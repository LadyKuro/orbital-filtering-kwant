"""
Fig. 2, panel 4: transmitted <Lz> vs coupling imbalance at fixed energy,
for the four input states (two quadrupole, two dipole).

Fix g1 and E; sweep g2 through the imbalance delta = g1 + g2 (delta = 0 is
the antisymmetric echo point g2 = -g1). Ballistic pseudospin readouts:
    |py> : <Lz> = -sin(dphi)      |px> : <Lz> = +sin(dphi)
    |+z> : <Lz> = +cos(dphi)      |-z> : <Lz> = -cos(dphi)
with dphi = d[dk(g1,E) + dk(g2,E)],
     dk(g)a = arccos((eps-g-E)/2t) - arccos((eps+g-E)/2t).

At the echo (delta = 0): quadrupoles cross zero (no Lz generated,
character preserved), dipoles are preserved at +/-1. Away from it the two
families sweep in quadrature -> imbalance is the filter/anti-filter knob.

Kwant overlay hook: if kwant_lz_vs_delta.npz exists (array 'delta' plus
'lz_py','lz_px','lz_pz_p','lz_pz_m'), points are drawn over the lines.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

# ------------------------------------------------------------------ model
t, a, eps = 1.0, 1.0, 0.0
g1, d, E0 = 0.5, 60, 0.0
delta_max = 0.16                     # imbalance window (a few fringes)

def dk(g, E):
    return (np.arccos(np.clip((eps - g - E) / (2 * t), -1, 1))
            - np.arccos(np.clip((eps + g - E) / (2 * t), -1, 1))) / a

delta = np.linspace(-delta_max, delta_max, 1000)
g2 = delta - g1                      # g2 = -g1 + delta
dphi = d * (dk(g1, E0) + dk(g2, E0))

# ------------------------------------------------------------------ curves
C_Q = "#4477AA"   # quadrupole family (blue)
C_D = "#EE6677"   # dipole family (red)

curves = [
    (r"$|p_y\rangle$", -np.sin(dphi), C_Q, "-",  1.3),
    (r"$|p_x\rangle$", +np.sin(dphi), C_Q, "--", 1.1),
    (r"$|{+}z\rangle$", +np.cos(dphi), C_D, "-",  1.3),
    (r"$|{-}z\rangle$", -np.cos(dphi), C_D, "--", 1.1),
]

fig, ax = plt.subplots(figsize=(3.375, 2.4), constrained_layout=True)

for lab, y, col, ls, lw in curves:
    ax.plot(delta, y, ls, color=col, lw=lw, label=lab)

# optional exact-scattering overlay
if os.path.exists("kwant_lz_vs_delta.npz"):
    dat = np.load("kwant_lz_vs_delta.npz")
    keymap = {r"$|p_y\rangle$": "lz_py", r"$|p_x\rangle$": "lz_px",
              r"$|{+}z\rangle$": "lz_pz_p", r"$|{-}z\rangle$": "lz_pz_m"}
    for lab, y, col, ls, lw in curves:
        k = keymap[lab]
        if k in dat:
            ax.plot(dat["delta"][::15], dat[k][::15], "o", ms=2.2,
                    mfc="none", mew=0.6, color=col, zorder=5)

for yv in (-1, 0, 1):
    ax.axhline(yv, color="#CCCCCC", lw=0.5, ls=":", zorder=0)
ax.axvline(0.0, color="#999999", lw=0.6, ls="--", zorder=0)
ax.annotate("echo\n($g_2=-g_1$)", (0.004, -0.30), fontsize=6.2,
            color="#555555", ha="left", va="center")

ax.set_xlim(-delta_max, delta_max)
ax.set_ylim(-1.18, 1.18)
ax.set_xlabel(r"imbalance $\;\delta = g_1 + g_2$")
ax.set_ylabel(r"$\langle L_z\rangle_{\mathrm{out}}$")
ax.set_title(rf"$g_1={g1},\ E={E0}$", fontsize=8, pad=3)
ax.legend(loc="upper right", fontsize=6.6, frameon=True, framealpha=1.0,
          facecolor="white", edgecolor="none", ncol=2,
          columnspacing=1.0, handlelength=1.5, labelspacing=0.25)
ax.tick_params(direction="in", top=True, right=True, length=2.5)

fig.savefig("/home/claude/fig2_lz_vs_imbalance.png", dpi=300)
fig.savefig("/home/claude/fig2_lz_vs_imbalance.pdf")
print("dphi(delta=0) =", d * (dk(g1, E0) + dk(-g1, E0)))
print("done")
