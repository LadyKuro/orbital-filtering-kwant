"""
Fig. 2 maps: transmitted <Lz> over the (g2/g1, E) plane at fixed g1,
for a quadrupole input |py> and a dipole input |+z>.

Ballistic pseudospin phase (two segments, d1 = d2 = d):
    dphi(g2,E) = d [ dk(g1,E) + dk(g2,E) ],
    dk(g)a = arccos((eps-g-E)/2t) - arccos((eps+g-E)/2t).
Readouts:  |py> -> <Lz> = -sin(dphi);   |+z> -> <Lz> = +cos(dphi).

Both channels must propagate in both segments; the two-channel region is
|E| + max(|g1|,|g2|) < 2t. Outside it one mode is evanescent (polarizer
regime) and the ballistic cosine no longer applies -> masked (grey).

Overlays:
  * echo line g2/g1 = -1 (dashed): quadrupole <Lz> = 0 at every E.
  * conversion loci dphi = pi/2 + n pi (thin solid): full conversion;
    they fan out from the echo line with |E| and refocus at band centre
    E = eps, the geometric statement of the stripe ladder.

Kwant overlay: replace the analytic fields by loading exact-scattering
.npz maps of the same shape (see hook at the bottom).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})

# ------------------------------------------------------------------ model
t, a, eps = 1.0, 1.0, 0.0
g1, d = 0.5, 60

r = np.linspace(-2.2, 1.4, 601)          # g2 / g1
g2 = r * g1
E = np.linspace(-1.8, 1.8, 601)
RR, EE = np.meshgrid(r, E)
G2 = RR * g1

def dk(g, Egrid):
    return (np.arccos(np.clip((eps - g - Egrid) / (2 * t), -1, 1))
            - np.arccos(np.clip((eps + g - Egrid) / (2 * t), -1, 1))) / a

DPHI = d * (dk(g1, EE) + dk(G2, EE))

lz_py = -np.sin(DPHI)          # quadrupole input
lz_pz = np.cos(DPHI)           # dipole input

# two-channel propagating mask (both segments, both channels)
gmax = np.maximum(np.abs(g1), np.abs(G2))
prop = (np.abs(EE) + gmax) < (2 * t - 1e-9)
lz_py = np.where(prop, lz_py, np.nan)
lz_pz = np.where(prop, lz_pz, np.nan)
DPHI_m = np.where(prop, DPHI, np.nan)

# ------------------------------------------------------------------ plot
fig, axes = plt.subplots(2, 1, figsize=(3.375, 5.0), sharex=True,
                         constrained_layout=True)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#DDDDDD")
norm = Normalize(-1, 1)

panels = [
    (axes[0], lz_py, r"injected $|p_y\rangle$"),
    (axes[1], lz_pz, r"injected $|p_y\rangle + i |p_x\rangle$"),
]

# conversion loci: dphi = pi/2 + n pi
lo = np.nanmin(DPHI_m); hi = np.nanmax(DPHI_m)
nmin = int(np.floor((lo - np.pi / 2) / np.pi))
nmax = int(np.ceil((hi - np.pi / 2) / np.pi))
levels = [(np.pi / 2 + n * np.pi) for n in range(nmin, nmax + 1)]

for ax, field, title in panels:
    im = ax.pcolormesh(r, E, field, cmap=cmap, norm=norm, shading="auto",
                       rasterized=True)
    # echo line
    ax.axvline(-1.0, color="k", lw=0.8, ls=(0, (4, 2)))
    # conversion loci
    ax.contour(RR, EE, DPHI_m, levels=levels, colors="k",
               linewidths=0.4, alpha=0.55)
    ax.set_ylabel(r"$E/t$")
    ax.set_ylim(E.min(), E.max())
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(direction="in", top=True, right=True, length=2.5,
                   color="#888888")

axes[1].set_xlabel(r"$g_2/g_1$")
axes[1].set_xlim(r.min(), r.max())

cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.9,
                    ticks=[-1, 0, 1], pad=0.02)
cbar.set_label(r"$\langle L_z\rangle_{\mathrm{out}}$")
cbar.ax.tick_params(length=2.5)

fig.savefig("plots/fig2_maps_ratio_E.png", dpi=300)
fig.savefig("plots/fig2_maps_ratio_E.pdf")
print("dphi range:", lo, hi, " n:", nmin, nmax)
print("done")

# ---- Kwant overlay hook -------------------------------------------------
# To use exact scattering instead of the ballistic cosine, save arrays of
# shape (len(E), len(r)) named 'lz_py' and 'lz_pz' (plus 'r','E') to
# kwant_maps_ratio_E.npz and swap:
#     dat = np.load("kwant_maps_ratio_E.npz")
#     lz_py, lz_pz = dat["lz_py"], dat["lz_pz"]
# keeping the same mask/overlay code.
