"""
(g, E) map for the 1D orbital chain: inject |+z> from the left, read the
orbital character of the *transmitted* state in the right lead.

Configurations via `sign`:
  sign = +1 : g1 = g2 = g      (homogeneous chain of length d1+d2)
  sign = -1 : g1 = +g, g2 = -g (the refocusing two-segment configuration)

Because the leads are orbital-isotropic, the transmitted state is a single
orbital vector phi_out (one k, no lead precession), obtained from the
S-matrix in the mode basis:  c_out = t @ c_in,  phi_out = U_out @ c_out.
Its <Lz> is x-independent -> a clean scalar per (g, E) pixel.

Ballistic (no-reflection) prediction for the homogeneous case:
  Lz_out = cos(Delta_phi_tot),
  Delta_phi_tot = [k_-(g,E) - k_+(g,E)] * (d1 + d2),
  cos(k_pm a) = (eps +/- g - E) / (2 t_hop)     [hopping -t convention]
For sign = -1 the ballistic prediction is Lz_out = 1 identically (equal
segments unwind each other); deviations from 1 map the interface
reflections in (g, E).

Requires the orbfilter package.
"""

import numpy as np
import matplotlib.pyplot as plt

from orbfilter.chain_1d import chi_z, map_g_energy, ballistic_Lz


# ----------------------------------------------------------------------
#  Plot
# ----------------------------------------------------------------------
def plot_map(gs, energies, Z, title, cmap="RdBu_r", vmin=-1, vmax=1,
             save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    pcm = ax.pcolormesh(energies, gs, Z.T, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.02)
    ax.set_xlabel("Energy E", fontsize=13)
    ax.set_ylabel(r"coupling $g$", fontsize=13)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=500)
    return fig, ax


if __name__ == "__main__":
    params = dict(t=1.0, eps=0.0, d1=60, d2=60, chi=chi_z, a=1.0)
    gs = np.linspace(0.0, 0.8, 81)
    energies = np.linspace(-1.6, 1.6, 121)   # inside the lead band (-2t, 2t)

    # --- homogeneous: fringes, compare with ballistic formula -----------
    Lz_h, Qd_h, T_h = map_g_energy(params, gs, energies, sign=+1)
    plot_map(gs, energies, Lz_h,
             r"homogeneous $g_1=g_2=g$: transmitted $\langle L_z\rangle$, "
             r"input $|{+}z\rangle$",
             save_path="plots/map_gE_homogeneous_Lz.png")
    b_Lz = ballistic_Lz(params, gs, energies)
    plot_map(gs, energies, b_Lz,
             "ballistic prediction $\\cos\\Delta\\phi_{tot}$ (no reflections)",
             save_path="plots/map_gE_ballistic.png")
    plot_map(gs, energies, Lz_h - b_Lz,
             "kwant - ballistic prediction ",
             save_path="plots/comparison.png")
    


    plt.show()
