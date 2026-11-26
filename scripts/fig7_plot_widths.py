"""
Fig. 7: regions of finite and vanishing transmitted orbital polarization
L_z^max in the (E, theta) plane, one panel per lead width W.

Reads the per-width maps written by fig7_compute_widths.py:
    data/Lz_map_w{W}_L{L_central}.npz   (thetas, energies, Lz_map, params)
No Kwant computation is done here.

Colours: bright = L_z^max <= EPS (zero), dark = L_z^max > EPS (finite),
grey = pixel whose computation failed (NaN), so a failure is never shown
as a physical zero.

Moved from notebooks/spectrum_calculations.ipynb (cell 10) and
replot_Lz_widths.py (load_width_map).

Run from the repository root:  python scripts/fig7_plot_widths.py
"""

import ast
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

# =======================================================================
# Configuration
# =======================================================================
WIDTHS = [2, 3, 4, 5, 6, 10, 15, 20, 50, 100]
L_CENTRAL = 400          # must match BASE_PARAMS['L_central'] in fig7_compute_widths.py
EPS = 1e-12              # threshold separating zero from finite L_z^max
DATA_DIR = "data"
FIG_DIR = "figures"

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.linewidth": 0.6,
})


# =======================================================================
# Loading
# =======================================================================
def load_width_map(width, L_central=L_CENTRAL, data_dir=DATA_DIR):
    """Load the saved (thetas, energies, Lz_map, params) for one width."""
    path = os.path.join(data_dir, f"Lz_map_w{width}_L{L_central}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data for width={width}: {path}")

    with np.load(path, allow_pickle=True) as d:
        thetas = d["thetas"]
        energies = d["energies"]
        Lz_map = d["Lz_map"]
        # params was saved via np.array(str(params)) -> a dict literal
        params = ast.literal_eval(str(d["params"]))
    return thetas, energies, Lz_map, params


# =======================================================================
# Figure
# =======================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    base_cmap = plt.get_cmap("magma")
    colors = base_cmap(np.array([0.95, 0.2]))       # [zero, finite]
    c_nan = "#9e9e9e"
    cmap = ListedColormap([colors[0], colors[1], c_nan])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)    # 0 zero, 1 finite, 2 NaN

    n_cols = len(WIDTHS) // 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 7),
                             sharey=True, constrained_layout=True)

    for ax, w in zip(axes.flatten(), WIDTHS):
        path = os.path.join(DATA_DIR, f"Lz_map_w{w}_L{L_CENTRAL}.npz")
        if not os.path.exists(path):
            print(f"[W={w:2d}] SKIPPED - file not found: {path}")
            continue
        thetas, energies, Lz_map, params = load_width_map(w)

        category = np.where(np.isnan(Lz_map), 2, (Lz_map > EPS).astype(int))
        n_nan = int((category == 2).sum())
        if n_nan:
            print(f"[W={w:2d}] {n_nan} failed pixels (grey)")

        X, Y = np.meshgrid(energies, thetas / np.pi)
        ax.scatter(X.flatten(), Y.flatten(), c=category.flatten(),
                   cmap=cmap, norm=norm, marker="s", s=0.4, rasterized=True)
        ax.set_title(f"$W={w}$", fontsize=18, fontweight="bold")
        ax.set_xlabel("$E$", fontsize=18, fontweight="bold")
        ax.tick_params(labelsize=17)

    for row in range(n_rows):
        axes[row, 0].set_ylabel(r"$\theta / \pi$", fontsize=17, fontweight="bold")

    cb_cmap = ListedColormap([colors[1], colors[0]])       # the two physical phases
    sm = ScalarMappable(cmap=cb_cmap, norm=BoundaryNorm([0, 1, 2], 2))
    cbar = fig.colorbar(sm, ax=axes, location="left",
                        ticks=[0.5, 1.5], shrink=0.55, pad=0.02, aspect=28)
    cbar.ax.set_yticklabels([r"$\langle L_z \rangle \neq 0$" "\n" "transmitted",
                             r"$\langle L_z \rangle = 0$" "\n" "quenched"], fontsize=20)
    fig.suptitle(r"$\qquad \qquad \qquad \qquad \qquad$ Regions of finite and "
                 r"cancelled transmitted orbital polarization in the $(E, \theta)$ plane",
                 fontsize=25, fontweight="bold")
    fig.get_layout_engine().set(h_pad=0.15)

    fig.savefig(os.path.join(FIG_DIR, "fig7_phase_diagram_widths.pdf"),
                bbox_inches="tight", dpi=500)


if __name__ == "__main__":
    main()
