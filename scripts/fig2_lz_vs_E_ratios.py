"""
Fig. 2 panel: transmitted <Lz> vs energy for selected g2/g1 ratios.
Input state |py> (quadrupole), two segments d1 = d2 = d, ballistic
pseudospin readout  <Lz> = -sin(dphi),  dphi = d[dk(g1,E) + dk(g2,E)],
dk(g)a = arccos((eps-g-E)/2t) - arccos((eps+g-E)/2t).

Ratios chosen from the stripe ladder at band center:
  r = -1                    echo line, <Lz> = 0 at every E (no filtering)
  r = (-g1+delta_0)/g1      n=0 conversion stripe, <Lz> ~ -1, flattest
  r = (-g1+delta_1)/g1      n=1 stripe, <Lz> ~ +1, dispersion ~3x larger
  r = +1                    homogeneous reference (rapid oscillation)

Kwant overlay hook: if kwant_lz_vs_E.npz exists (arrays 'E' and one
'lz_r<label>' per ratio), points are drawn over the ballistic lines.
"""

import os
import numpy as np
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

# Tol bright
C = {"echo": "#000000", "n0": "#347AC1", "n1": "#A53D49", "hom": "#C4B339"}

# ------------------------------------------------------------------ model
t, a, eps = 1.0, 1.0, 0.0
g1, d = 0.5, 60

def dk(g, E):
    return (np.arccos(np.clip((eps - g - E) / (2 * t), -1, 1))
            - np.arccos(np.clip((eps + g - E) / (2 * t), -1, 1))) / a

def dphi(g2, E):
    return d * (dk(g1, E) + dk(g2, E))

# exact stripe positions at band center (dphi = pi/2 + n pi);
# n = 0 is the first stripe above the echo line (delta > 0, dphi = +pi/2),
# n = -1 the first below it (delta < 0, dphi = -pi/2).
def stripe(n, E0=eps):
    f = lambda g2: dphi(g2, E0) - (np.pi / 2 + n * np.pi)
    pred = np.pi / (d * ((1 / np.sqrt(1 - ((eps + g1 - E0) / 2 / t) ** 2)
                          + 1 / np.sqrt(1 - ((eps - g1 - E0) / 2 / t) ** 2))
                         / (2 * t * a)))
    if n >= 0:
        lo, hi = -g1 + n * 0.3 * pred, -g1 + (n + 2) * 1.5 * pred
    else:
        lo, hi = -g1 + (n - 1) * 1.5 * pred, -g1 + (n + 1) * 0.7 * pred
    return brentq(f, lo, hi)

g2_n0 = stripe(0)
g2_nm1 = stripe(-1)
print(f"stripe g2 values: n=0 -> {g2_n0:.4f} (r={g2_n0/g1:.3f}), "
      f"n=-1 -> {g2_nm1:.4f} (r={g2_nm1/g1:.3f})")

# propagating window for both channels in both segments (|E -/+ g| < 2t)
Emax = 2 * t - max(abs(g1), abs(g2_nm1), abs(g2_n0)) - 0.02
E = np.linspace(-Emax, Emax, 1200)

curves = [
    ("echo", -g1,    r"$g_2/g_1=-1$",                  "-", 1.4),
#    ("n0",   g2_n0,  rf"$g_2/g_1={g2_n0/g1:.3f}$ ($n=1$)",    "-", 1.2),
    ("n1",   g2_nm1, rf"$g_2/g_1={g2_nm1/g1:.3f}$",  "-", 1.2),
    ("hom",  +g1,    r"$g_2/g_1=+1$ (uniform)",               "-", 1.0),
]

print(g2_n0/g1)
print(g2_nm1/g1)
# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(3.375, 2.5),
                       gridspec_kw=dict(left=0.145, right=0.975,
                                        top=0.89, bottom=0.14))

for key, g2, lab, ls, lw in curves:
    lz = np.cos(dphi(g2, E))
    ax.plot(E, lz, ls, color=C[key], lw=lw, label=lab,
            alpha=0.55 if key == "hom" else 1.0)

# optional exact-scattering overlay
if os.path.exists("kwant_lz_vs_E.npz"):
    dat = np.load("kwant_lz_vs_E.npz")
    for key, g2, lab, ls, lw in curves:
        k = f"lz_r{key}"
        if k in dat:
            ax.plot(dat["E"][::20], dat[k][::20], "o", ms=2.2,
                    mfc="none", mew=0.6, color=C[key], zorder=5)

for y in (-1, 0, 1):
    ax.axhline(y, color="#BBBBBB", lw=0.5, ls=":", zorder=0)
ax.axvline(eps, color="#BBBBBB", lw=0.5, ls="--", zorder=0)
ax.set_title(r"injected state $|p_y\rangle+i|p_x\rangle $", fontsize=8)
ax.set_xlim(-Emax, Emax)
ax.set_ylim(-1.18, 1.18)
ax.set_xlabel(r"$E/t$")
ax.set_ylabel(r"$\langle L_z\rangle_{\mathrm{out}}$")
ax.legend(loc="lower left", fontsize=6.3, frameon=True, framealpha=0.9,
          facecolor="white", edgecolor="none",
          borderaxespad=0.2, handlelength=1.6, labelspacing=0.25)
ax.annotate("no filtering", (0.62 * Emax, 1.03), fontsize=6.5,
            color="#444444", fontweight="bold")
ax.annotate("filtering", (-0.13 * Emax, 0.07), fontsize=6.5,
            color=C["n0"], fontweight="bold")
ax.tick_params(direction="in", top=True, right=True, length=2.5)


#fig.savefig("plots/fig2_lz_vs_E_ratios_cos.png", dpi=300)
fig.savefig("plots/fig2_lz_vs_E_ratios_cos.pdf")
print("done")