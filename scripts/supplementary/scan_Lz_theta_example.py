"""
Theta-scan of the transmitted orbital polarization for the Rotated2D junction.

Compares two observables at each rotation angle theta:

  1. local_max  -- max_x |<Lz>(x)| from the binned profile in region III
                   (finite window; includes near-interface transients)
  2. env_max    -- analytic far-field envelope L_z^max = (2/N) sum_p A_p
                   built from the S-matrix coherences (Eq. (envelope) of
                   the appendix derivation; x-independent, asymptotic)

Where the curves agree the local peak is representative of the asymptotic
precession; where local_max exceeds env_max the excess is an interface
transient (evanescent contributions absent from the propagating-mode sum).

Requires the orbfilter package.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from orbfilter.rotated_2d import Rotated2D, Lz

# =======================================================================
#  Local binned maximum in region III (as before, index-safe version)
# =======================================================================
def local_Lz_max(wire, energy=0.0, bin_coeff=1.0, buffer_cells=10,
                 operator=Lz):
    bin_x, Otot = wire.compute_operator_binned(
        operator=operator, energy=energy, lead=0,
        binning_coeff=bin_coeff, summed_over_modes=True, normed=True)
    x_cut = wire.a * (wire.L1 + wire.L2) + buffer_cells * wire.a
    ind = np.searchsorted(bin_x, x_cut)
    if ind >= Otot.shape[1]:
        raise ValueError("Buffer places the cut beyond the last bin; "
                         "decrease buffer_cells or increase L.")
    return np.max(np.abs(Otot[:, ind:]))


# =======================================================================
#  The scan
# =======================================================================
def scan_Lz_vs_theta(params, angles, energy=0.5, bin_coeff=1.0,
                     buffer_cells=10, verbose=True):
    local_max = np.empty(len(angles))
    env_max = np.empty(len(angles))

    for k, theta in enumerate(angles):
        p = dict(params, theta=theta)
        wire = Rotated2D(**p)
        wire.build()

        local_max[k] = local_Lz_max(wire, energy=energy,
                                    bin_coeff=bin_coeff,
                                    buffer_cells=buffer_cells)
        env_max[k] = wire.transmitted_Lz_envelope(energy=energy)

        if verbose and (k % 10 == 0 or k == len(angles) - 1):
            print(f"[{k+1:3d}/{len(angles)}] theta = {theta:.4f}  "
                  f"local = {local_max[k]:.4f}  env = {env_max[k]:.4f}")

    return np.asarray(angles), local_max, env_max


# =======================================================================
#  Plotting
# =======================================================================
def plot_Lz_scan(angles, local_max, env_max, energy, width,
                 save_path=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(angles / np.pi, local_max, 'o-', ms=3.5, color='purple',
            label=r'local peak, region III  $\max_x|\langle L_z\rangle|$')
    ax.plot(angles / np.pi, env_max, 's-', ms=3.5, color='teal',
            label=r'far-field envelope  $L_z^{\max}$ (S-matrix)')
    ax.fill_between(angles / np.pi, env_max, local_max,
                    where=local_max > env_max, alpha=0.15,
                    color='purple', label='interface transient')
    ax.set_xlabel(r"Rotation angle $\theta/\pi$", fontsize=13)
    ax.set_ylabel(r"$|L_z|$", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.set_title(f"E = {energy}, width = {width}", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=500)
    return fig, ax


# =======================================================================
#  Example run
# =======================================================================
if __name__ == "__main__":
    os.makedirs("figures/supplementary", exist_ok=True)
    params = dict(
        onsite_energy=0.0,
        Vpi=-0.3,
        Vsigma=1.0,
        L=400,
        L_central=300,
        width=4,
        a=1.0,
        coupling=0.5,
        interface_cutoff=1.2,   # r_c in units of a (None = nearest bond only)
        theta=0.0,         # overwritten in the scan
    )
    energy = 0.5
    angles = np.linspace(0.0, np.pi / 2, 101)   # 201 once you're happy

    angles, local_max, env_max = scan_Lz_vs_theta(
        params, angles, energy=energy, bin_coeff=1.0, buffer_cells=10)

    plot_Lz_scan(angles, local_max, env_max,
                 energy=energy, width=params["width"],
                 save_path=f"figures/supplementary/Lz_scan_w{params['width']}_e{energy}.png")
    plt.show()

    # --- optional single-angle diagnostic: overlay the analytic
    #     sinusoid structure onto the binned profile -------------------
    # theta0 = angles[np.argmax(env_max)]
    # wire = Rotated2D(**dict(params, theta=theta0)); wire.build()
    # bin_x, Otot = wire.compute_operator_binned(
    #     operator=Lz, energy=energy, lead=0, summed_over_modes=True)
    # plt.plot(bin_x, Otot[0], lw=0.8)
    # plt.axvline(wire.a * (wire.L1 + wire.L2), color='k', ls='--', lw=0.8)
    # plt.axhline(wire.transmitted_Lz_envelope(energy), color='teal', ls=':')
    # plt.axhline(-wire.transmitted_Lz_envelope(energy), color='teal', ls=':')
    # plt.xlabel('x'); plt.ylabel(r'$\langle L_z\rangle(x)$'); plt.show()
