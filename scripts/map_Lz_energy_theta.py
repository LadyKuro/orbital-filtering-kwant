"""
2D map of the transmitted orbital polarization envelope L_z^max in the
(energy, rotation angle) plane for the Rotated2D junction.

Axis convention (this version):
    x-axis = energy E
    y-axis = rotation angle theta/pi

Each pixel is one smatrix call (cheap after the buffer shrink), and the
grid is embarrassingly parallel -> joblib over all (theta, E) pairs.

The plot overlays the analytic sub-band edges of the square leads (from
the orbital-resolved dispersions). With energy on the x-axis these are
theta-independent VERTICAL lines:
- a py or px channel closing removes one coherence pair A_p from the sum;
- if at some (theta, E) *either* all py or all px channels are closed,
  L_z^max = 0 identically (nothing for Lz to couple).

Assumes: Rotated2D class (corrected version), transmitted_Lz_envelope
from scan_Lz_theta_example.py (with the nuc = wf.shape[0]//norb fix).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from models.rotated_junction_2d import Rotated2D
from scan_Lz_theta_example import Lz, transmitted_Lz_envelope

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.6,
})


# =======================================================================
# Lead sub-band edges (analytic, from the appendix dispersions)
# =======================================================================
def lead_subband_edges(params):
    """
    Band edges of every (orbital, transverse channel) mode of the square
    lead:  E_edge = eps + 2*t_perp*cos(q_p) -/+ 2*|t_x|  (bottom/top).
    Returns a list of (E_bottom, E_top, label) --- theta-independent,
    they appear as VERTICAL lines on the (E, theta) map.
    """
    e = params["onsite_energy"]
    v1, v2 = params["Vpi"], params["Vsigma"]
    W = params["width"]

    edges = []
    for p in range(1, W + 1):
        cq = np.cos(p * np.pi / (W + 1))
        for label, tx, tp in (("pz", v1, v1),
                              ("py", v1, v2),
                              ("px", v2, v1)):
            center = e + 2 * tp * cq
            edges.append((center - 2 * abs(tx),
                          center + 2 * abs(tx),
                          f"{label},p={p}"))
    return edges


# =======================================================================
# The 2D scan
# =======================================================================
def _one_pixel(params, theta, energy):
    wire = Rotated2D(**dict(params, theta=theta))
    wire.build()
    try:
        return transmitted_Lz_envelope(wire, energy=energy)
    except Exception:
        # e.g. no propagating modes at this energy -> no transmission
        return 0.0


def map_Lz_energy_theta(params, energies, thetas, n_jobs=-1, verbose=5):
    """
    Returns Lz_map with shape (len(thetas), len(energies)) ---
    row = angle, column = energy, ready for
    pcolormesh(energies, thetas, Lz_map).
    """
    grid = [(th, en) for th in thetas for en in energies]
    vals = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one_pixel)(params, th, en) for th, en in grid)
    return np.array(vals).reshape(len(thetas), len(energies))


# =======================================================================
# Plotting
# =======================================================================
def plot_Lz_map(energies, thetas, Lz_map, params,
                show_edges=False, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5.1))

    pcm = ax.pcolormesh(energies, thetas / np.pi, Lz_map,
                        cmap="magma_r", shading="auto",
                        vmin=0.0, vmax=max(Lz_map.max(), 1e-12))
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$L_z^{\max}$", fontsize=17)
    cbar.ax.tick_params(labelsize=16)

    if show_edges:
        drawn = set()
        for (Eb, Et, label) in lead_subband_edges(params):
            for E in (Eb, Et):
                if energies.min() <= E <= energies.max():
                    key = round(E, 9)
                    if key in drawn:        # degenerate edges: draw once
                        continue
                    drawn.add(key)
                    ax.axvline(E, color="cyan", lw=0.7, ls="--", alpha=0.7)

    ax.set_xlabel(r"$E$", fontsize=17)
    ax.tick_params(axis='both', labelsize=17)
    ax.set_ylabel(r"Rotation angle $\theta/\pi$", fontsize=17)
    ax.set_title(
        f"Transmitted orbital polarization, "
        f"W = {params['width']}, L = {params['L_central']}",
        fontsize=17)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=500)
#        fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight", dpi=500)
    return fig, ax

def plot_Lz_phase_diagram(energies, thetas, Lz_map, params,
                 save_path=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    pd = Lz_map > 1e-6

    pcm = ax.pcolormesh(energies, thetas / np.pi, pd.astype(float),
                        cmap="inferno", shading="auto",
                        vmin=0.0, vmax=max(Lz_map.max(), 1e-12))
    
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$L_z^{\max}$", fontsize=15)


    ax.set_xlabel(r"$E$", fontsize=15)
    ax.set_ylabel(r"Rotation angle $\theta/\pi$", fontsize=15)
    ax.set_title(
        f"Transmitted orbital polarization, "
        f"W = {params['width']}, L2 = {params['L_central']}",
        fontsize=15)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=500)
    return fig, ax

# =======================================================================
# Example run
# =======================================================================
if __name__ == "__main__":
    params = dict(
        onsite_energy=0.0,
        Vpi=-0.3,          # <-- your values
        Vsigma=1.0,
        L=440,             # 2*20 buffer + 400 central
        L_central=400,
        width=4,
        a=1.0,
        coupling=0.5,
        theta=0.0,
    )

    # start coarse (~30 min-scale grids come later)
    thetas = np.linspace(0.0, np.pi / 2, 401)
    energies = np.linspace(-1.5, 1.5, 401)   # adjust to your bandwidth

    Lz_map = map_Lz_energy_theta(params, energies, thetas, n_jobs=-1)

    np.savez(f"data/Lz_map_w{params['width']}_L{params['L_central']}.npz",
             thetas=thetas, energies=energies, Lz_map=Lz_map,
             axes="rows=thetas, cols=energies",
             params=np.array(str(params)))

    plot_Lz_map(energies, thetas, Lz_map, params,
                save_path=f"plots/Lz_map_w{params['width']}.png")
    plt.show()