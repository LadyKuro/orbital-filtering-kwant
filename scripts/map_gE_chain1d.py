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

Requires: orbital_chain_1d.py in the same directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from orbfilter.chain_1d import OrbitalChain1D, STATES, Lz, Q_ops, chi_z


# ----------------------------------------------------------------------
#  Transmitted orbital state from the S-matrix
# ----------------------------------------------------------------------
def transmitted_state(wire, phi_in, energy):
    """
    Returns (phi_out, T): orbital vector of the transmitted state in the
    right lead (unnormalized; |phi_out|^2 relates to transmitted flux)
    and the total transmission probability for this input.
    Returns (None, 0.0) if no propagating modes.
    """
    import kwant
    sm = kwant.smatrix(wire.syst, energy)
    t = sm.submatrix(1, 0)                    # (n_out_R, n_in_L)
    n_out, n_in = t.shape
    if n_out == 0 or n_in == 0:
        return None, 0.0

    # incoming modes of lead 0: first n_in columns; outgoing of lead 1:
    # columns n_out: (same convention validated in the 2D envelope code)
    u_in = sm.lead_info[0].wave_functions[:wire.norb, :n_in]
    u_out = sm.lead_info[1].wave_functions[:wire.norb, n_out:]

    phi_in = np.asarray(phi_in, complex)
    phi_in = phi_in / np.linalg.norm(phi_in)
    c_in, *_ = np.linalg.lstsq(u_in, phi_in, rcond=None)
    flux_in = np.sum(np.abs(c_in) ** 2)

    c_out = t @ c_in
    T = float(np.sum(np.abs(c_out) ** 2) / flux_in)   # transmitted / incident flux
    phi_out = u_out @ c_out
    return phi_out, T


def pixel(params, g, energy, sign, phi_in):
    wire = OrbitalChain1D(g1=g, g2=sign * g, **params)
    wire.build()
    phi_out, T = transmitted_state(wire, phi_in, energy)
    if phi_out is None or np.linalg.norm(phi_out) < 1e-12:
        return np.nan, np.nan, T
    n2 = np.vdot(phi_out, phi_out).real
    lz = np.vdot(phi_out, Lz @ phi_out).real / n2
    qd = np.vdot(phi_out, Q_ops["Lx^2-Ly^2"] @ phi_out).real / n2
    return lz, qd, T


# ----------------------------------------------------------------------
#  Map
# ----------------------------------------------------------------------
def map_g_energy(params, gs, energies, sign=+1, state="+z",
                 n_jobs=-1, verbose=5):
    phi_in = STATES[state]
    grid = [(g, en) for en in energies for g in gs]
    res = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(pixel)(params, g, en, sign, phi_in) for g, en in grid)
    res = np.array(res).reshape(len(energies), len(gs), 3)
    return res[..., 0], res[..., 1], res[..., 2]   # Lz, Qdiag, T


def ballistic_Lz(params, gs, energies):
    """cos(Delta_phi_tot) for the homogeneous chain, NaN where evanescent."""
    t, eps = params["t"], params["eps"]
    d = params["d1"] + params["d2"]
    G, E = np.meshgrid(gs, energies)
    cp = (eps + G - E) / (2 * t)
    cm = (eps - G - E) / (2 * t)
    out = np.full_like(G, np.nan, dtype=float)
    ok = (np.abs(cp) < 1) & (np.abs(cm) < 1)
    out[ok] = np.cos((np.arccos(cm[ok]) - np.arccos(cp[ok])) * d)
    return out


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
