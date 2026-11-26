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

Assumes the Rotated2D class with the corrected SK matrices, the
query_ball_point interface, and compute_operator_binned as previously
rewritten. Requires: numpy, matplotlib, kwant.
"""

import numpy as np
import matplotlib.pyplot as plt
import kwant
from models.rotated_junction_2d import Rotated2D

# --- the orbital operator, paper Eq. (2), basis {pz, py, px} -----------
Lz = np.array([[0, 0, 0],
               [0, 0, -1j],
               [0, 1j, 0]])


# =======================================================================
#  Analytic envelope from the S-matrix (closed form, no x-scan needed)
# =======================================================================
def transmitted_Lz_envelope(wire, energy=0.0, operator=Lz):
    """
    L_z^max = (2/N) * sum_p |rho_{yp,xp}|  for the unpolarized equal-flux
    mixture of incoming left-lead modes, where rho = t t^dagger is the
    transmitted coherence matrix and the sum runs over pairs of outgoing
    right-lead modes (n, m) coupled by the operator (py <-> px, same
    transverse channel p -- automatically selected by the matrix element).

    Implementation note: rather than classifying modes by orbital/channel
    by hand, we use the operator matrix elements M_nm = <u_n|Lz|u_m>
    between outgoing modes. Pairs not coupled by Lz (different transverse
    channel, or involving pz) have M_nm = 0 by the orthogonality argument
    of the derivation, so the generic sum

        L_env = (2/N) * sum_{n<m} |rho_mn| * |M_nm| / sqrt(N_nn N_mm)

    reduces to the appendix formula without manual bookkeeping, and stays
    correct if the lead is ever modified (e.g. onsite disorder in tests).
    Returns 0.0 if the right lead carries no propagating modes.
    """
    sm = kwant.smatrix(wire.syst, energy)
    t = sm.submatrix(1, 0)                        # (n_out_R, n_in_L)
    n_out = t.shape[0]
    if n_out == 0:
        return 0.0

    modes = sm.lead_info[1]
    wf = modes.wave_functions[:, n_out:]          # outgoing modes only
    nuc = wf.shape[0] // wire.norb
    u = wf.T.reshape(n_out, nuc, wire.norb)       # (n, sites_uc, norb)

    rho = t @ t.conj().T                          # transmitted coherences
    M = np.einsum('nsi,ij,msj->nm', u.conj(), operator, u)
    G = np.einsum('nsi,msi->nm', u.conj(), u)     # mode overlaps (diag ~ norms)

    N = np.real(np.sum(np.diag(rho) * np.diag(G)))
    if N <= 1e-14:
        return 0.0

    # envelope: sum of |coherence * matrix element| over distinct pairs;
    # factor 2 because (n,m) and (m,n) are the conjugate halves of the
    # same sinusoid (cf. 2 Im[...] in the derivation)
    env = 0.0
    for n in range(n_out):
        for m in range(n + 1, n_out):
            env += 2.0 * np.abs(rho[m, n] * M[n, m])
    return env / N


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
        env_max[k] = transmitted_Lz_envelope(wire, energy=energy)

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
#  Example run (matches your earlier parameters)
# =======================================================================
if __name__ == "__main__":
    params = dict(
        onsite_energy=0.0,
        Vpi=-0.3,          # <-- your values here
        Vsigma=1.0,
        L=400,
        L_central=300,
        width=4,
        a=1.0,
        coupling=0.5,
        theta=0.0,         # overwritten in the scan
    )
    energy = 0.5
    angles = np.linspace(0.0, np.pi / 2, 101)   # 201 once you're happy

    angles, local_max, env_max = scan_Lz_vs_theta(
        params, angles, energy=energy, bin_coeff=1.0, buffer_cells=10)

    plot_Lz_scan(angles, local_max, env_max,
                 energy=energy, width=params["width"],
                 save_path=f"plots/Lz_scan_w{params['width']}_e{energy}.png")
    plt.show()

    # --- optional single-angle diagnostic: overlay the analytic
    #     sinusoid structure onto the binned profile -------------------
    # theta0 = angles[np.argmax(env_max)]
    # wire = Rotated2D(**dict(params, theta=theta0)); wire.build()
    # bin_x, Otot = wire.compute_operator_binned(
    #     operator=Lz, energy=energy, lead=0, summed_over_modes=True)
    # plt.plot(bin_x, Otot[0], lw=0.8)
    # plt.axvline(wire.a * (wire.L1 + wire.L2), color='k', ls='--', lw=0.8)
    # plt.axhline(transmitted_Lz_envelope(wire, energy), color='teal', ls=':')
    # plt.axhline(-transmitted_Lz_envelope(wire, energy), color='teal', ls=':')
    # plt.xlabel('x'); plt.ylabel(r'$\langle L_z\rangle(x)$'); plt.show()
