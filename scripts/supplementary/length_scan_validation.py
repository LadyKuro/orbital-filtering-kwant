"""
Length-dependence validation of the ballistic formula.

For the homogeneous chain (g1 = g2 = g, total length d), the appendix
predicts
    <Lz>_out = cos( [k_-(g) - k_+(g)] * d ),
    cos(k_pm a) = (eps +/- g - E) / (2 t).

This script maps the numerical <Lz>_out on a (d, g) grid at a few
representative energies and compares with the formula in three ways:
  1. side-by-side (d, g) maps, numerics vs analytic;
  2. difference map (Fabry-Perot corrections);
  3. line cuts <Lz>(d) at fixed g with the cosine overlaid -- the most
     direct test of the LINEAR-in-d phase accumulation, including the
     1/d fringe spacing.

Fringe-position check: along d at fixed (g, E) the zeros of <Lz> should
sit at d_n = (pi/2 + n pi)/Dk. Positions test the phase formula; any
amplitude reduction tests the equal-transmission assumption (appendix,
validity section) -- these are separable diagnostics.

Requires the orbfilter package.
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from orbfilter.chain_1d import OrbitalChain1D, STATES, Lz, chi_z, transmitted_state


# ----------------------------------------------------------------------
def delta_k(params, g, E):
    """k_-(g) - k_+(g); NaN if either channel is evanescent."""
    t, eps = params["t"], params["eps"]
    cp = (eps + g - E) / (2 * t)
    cm = (eps - g - E) / (2 * t)
    if abs(cp) >= 1 or abs(cm) >= 1:
        return np.nan
    return (np.arccos(cm) - np.arccos(cp)) / params["a"]


def pixel(params, d, g, energy):
    # homogeneous chain of total length d: split arbitrarily, g1 = g2
    wire = OrbitalChain1D(g1=g, g2=g, d1=d // 2, d2=d - d // 2, **params)
    wire.build()
    phi_out, _ = transmitted_state(wire, STATES["+z"], energy)
    if phi_out is None:
        return np.nan
    n2 = np.vdot(phi_out, phi_out).real
    if n2 < 1e-12:
        return np.nan
    return np.vdot(phi_out, Lz @ phi_out).real / n2


def map_d_g(params, ds, gs, energy, n_jobs=-1):
    grid = [(d, g) for g in gs for d in ds]
    vals = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(pixel)(params, int(d), g, energy) for d, g in grid)
    return np.array(vals).reshape(len(gs), len(ds))   # rows = g, cols = d


def analytic_d_g(params, ds, gs, energy):
    out = np.full((len(gs), len(ds)), np.nan)
    for i, g in enumerate(gs):
        dk = delta_k(params, g, energy)
        if np.isnan(dk):
            continue
        out[i, :] = np.cos(dk * np.asarray(ds) * params["a"])
    return out


# ----------------------------------------------------------------------
def figure_for_energy(params, ds, gs, energy, save_prefix=None):
    num = map_d_g(params, ds, gs, energy)
    ana = analytic_d_g(params, ds, gs, energy)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    opts = dict(shading="auto", rasterized=True)
    p0 = axes[0].pcolormesh(ds, gs, num, cmap="RdBu_r",
                            vmin=-1, vmax=1, **opts)
    axes[0].set_title(f"numerics, E = {energy}")
    axes[1].pcolormesh(ds, gs, ana, cmap="RdBu_r", vmin=-1, vmax=1, **opts)
    axes[1].set_title(r"ballistic $\cos(\Delta k\, d)$")
    dmax = np.nanmax(np.abs(num - ana))
    p2 = axes[2].pcolormesh(ds, gs, num - ana, cmap="PiYG",
                            vmin=-dmax, vmax=dmax, **opts)
    axes[2].set_title(f"difference (max {dmax:.2f})")
    for ax in axes:
        ax.set_xlabel("total length d (sites)")
    axes[0].set_ylabel("g")
    fig.colorbar(p0, ax=axes[:2], pad=0.02)
    fig.colorbar(p2, ax=axes[2], pad=0.02)
    if save_prefix:
        fig.savefig(f"{save_prefix}_maps_E{energy}.png",
                    bbox_inches="tight", dpi=400)

    # ---- line cuts: <Lz>(d) at a few g, cosine overlaid ----------------
    g_cuts = [gs[len(gs) // 5], gs[len(gs) // 2], gs[-len(gs) // 5]]
    fig2, axes2 = plt.subplots(len(g_cuts), 1, figsize=(8, 7.5),
                               sharex=True)
    for ax, g in zip(axes2, g_cuts):
        i = np.argmin(np.abs(gs - g))
        ax.plot(ds, num[i], "o", ms=3, label="numerics")
        dk = delta_k(params, gs[i], energy)
        if not np.isnan(dk):
            dd = np.linspace(ds[0], ds[-1], 600)
            ax.plot(dd, np.cos(dk * dd * params["a"]), "-", lw=1,
                    label=r"$\cos(\Delta k\, d)$")
            # predicted zeros d_n = (pi/2 + n pi)/dk
            n_max = int(dk * ds[-1] / np.pi)
            zeros = [(np.pi / 2 + n * np.pi) / dk for n in range(n_max + 1)]
            for z in zeros:
                ax.axvline(z, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(rf"$\langle L_z\rangle$, g={gs[i]:.2f}")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes2[-1].set_xlabel("total length d (sites)")
    fig2.suptitle(f"E = {energy}: dotted = predicted conversion points "
                  r"$d_n=(\pi/2+n\pi)/\Delta k$")
    if save_prefix:
        fig2.savefig(f"{save_prefix}_cuts_E{energy}.png",
                     bbox_inches="tight", dpi=400)
    return num, ana


if __name__ == "__main__":
    params = dict(t=1.0, eps=0.0, chi=chi_z, a=1.0)
    ds = np.arange(4, 200, 2)          # total lengths (even, for d//2 split)
    gs = np.linspace(0.02, 0.7, 69)

    # representative energies: band center; generic; approaching band edge
    for E in (0.0, 0.7, 1.5):
        figure_for_energy(params, ds, gs, E, save_prefix="plots/dscan")

    plt.show()
