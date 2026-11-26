"""
Fig. 5: eigenvalue spectrum of the isolated rotated central region as a
function of the rotation angle theta. Two panels: W = 2 and W = 4,
L_central = 200.

For each width the spectra are computed, saved to
    data/spectrum_vs_angle_w{W}_L{L_central}.npz
and plotted to figures/fig5_spectrum_w{W}_L{L_central}.png.
Set COMPUTE = False to re-plot from saved data only.

Moved from notebooks/spectrum_calculations.ipynb (cells 2, 4, 5).
Parameters now match Figs. 6-7 (Vpi = -0.3; the notebook used +0.4).

Run from the repository root:  python scripts/fig5_spectrum_vs_theta.py
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from orbfilter.rotated_2d import Rotated2D

# =======================================================================
# Configuration
# =======================================================================
PARAMS = dict(
    onsite_energy=0.0,
    Vpi=-0.3,
    Vsigma=1.0,
    L=400,             # only L_central matters for the isolated region
    L_central=200,
    a=1.0,
    theta=0.0,         # overwritten in the scan
    coupling=0.5,      # not used for the isolated region
)
WIDTHS = [2, 4]
ANGLES = np.linspace(0, np.pi / 2, 401)
COMPUTE = True
DATA_DIR = "data"
FIG_DIR = "figures"

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})


def data_path(width):
    return os.path.join(DATA_DIR,
                        f"spectrum_vs_angle_w{width}_L{PARAMS['L_central']}.npz")


# =======================================================================
# Compute
# =======================================================================
def compute_spectra(width):
    """Eigenvalues of the finite rotated region for every angle in ANGLES.
    The number of sites (hence eigenvalues) changes with theta, so the
    spectra are stored as separate arrays eval_0, eval_1, ..."""
    params = dict(PARAMS, width=width)
    all_evals = []
    for theta in ANGLES:
        wire = Rotated2D(**dict(params, theta=theta))
        all_evals.append(wire.middle_spectrum())
    np.savez(data_path(width),
             **{f"eval_{i}": arr for i, arr in enumerate(all_evals)},
             angle=np.array(ANGLES))
    print(f"[W={width}] spectra saved to {data_path(width)}")


# =======================================================================
# Plot
# =======================================================================
def plot_spectra(width):
    data = np.load(data_path(width))
    all_evals = [data[f"eval_{i}"] for i in range(len(data.files) - 1)]
    angles = data["angle"] / np.pi

    plt.figure(figsize=(8, 5.04))
    for t, evals in zip(angles, all_evals):
        plt.plot(evals, [t] * len(evals), ",", color="navy", ms=0.7,
                 alpha=0.25, rasterized=True)
    plt.xlabel(r"$E$", fontsize=19)
    plt.ylabel(r"Rotation angle $\theta/\pi$", fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(
        FIG_DIR, f"fig5_spectrum_w{width}_L{PARAMS['L_central']}.png"), dpi=600)
    plt.close()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    for w in WIDTHS:
        if COMPUTE:
            compute_spectra(w)
        plot_spectra(w)
