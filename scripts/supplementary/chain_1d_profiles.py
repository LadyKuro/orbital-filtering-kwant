"""
Spatial profiles <L>(x), <Q>(x) along the 1D chain for two examples:
  (a) homogeneous chain g1 = g2, quadrupole input |p_y>
  (b) two segments +/- g, dipole input |+z>

Moved from the __main__ block of orbfilter/chain_1d.py; only the output
path was changed (now figures/chain_1d_profiles.png).

Run from the repository root:  python scripts/supplementary/chain_1d_profiles.py
"""
import os

import matplotlib.pyplot as plt

from orbfilter.chain_1d import OrbitalChain1D, STATES, chi_z


if __name__ == "__main__":
    E = 0.7

    # --- (a) homogeneous chain, inject pure quadrupole |py> --------------
    hom = OrbitalChain1D(g1=0.3, g2=0.3, d1=60, d2=60, chi=chi_z)
    hom.build()
    x, prof = hom.profiles(STATES["py"], energy=E)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for name in ("Lz",):
        axes[0].plot(x, prof[name], label=name)
    for name in ("Q_xy (chi_z)", "Lx^2-Ly^2"):
        axes[1].plot(x, prof[name], label=name)
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("dipole"); axes[1].set_ylabel("quadrupole")
    axes[1].set_xlabel("x")
    fig.suptitle(f"homogeneous g0={hom.g1}, input |py>, E={E}")

    # --- (b) two segments +/- g0, inject dipole |+z> ---------------------
    two = OrbitalChain1D(g1=0.3, g2=-0.3, d1=60, d2=60, chi=chi_z)
    two.build()
    x, prof = two.profiles(STATES["+z"], energy=E)

    fig2, axes2 = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes2[0].plot(x, prof["Lz"], label="Lz")
    axes2[1].plot(x, prof["Lx^2-Ly^2"], label="Lx^2-Ly^2")
    axes2[1].plot(x, prof["Q_xy (chi_z)"], label="Q_xy")
    for ax in axes2:
        ax.axvline(two.d1, color="k", ls="--", lw=0.8)
        ax.legend(); ax.grid(alpha=0.3)
    axes2[1].set_xlabel("x")
    dphi = two.precession_angle(E, two.g1, two.d1)
    fig2.suptitle(f"segments +/-{two.g1}, input |+z>, "
                  f"ballistic precession per segment = {dphi:.2f} rad")
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/chain_1d_profiles.png", bbox_inches="tight", dpi=500)

    plt.show()
