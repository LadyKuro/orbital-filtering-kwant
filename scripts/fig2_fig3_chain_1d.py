"""
Figs. 2 and 3: two-segment 1D chain (Sec. II).

  Fig. 2  <Lz>_out vs E for several g2/g1 ratios, inputs |p_y> and |+z>,
          Kwant scattering with the analytic (ballistic) overlay.
  Fig. 3  <Lz>_out over the (g2/g1, E) plane at fixed g1 (Kwant).

The scattering is done by orbfilter.chain_1d (OrbitalChain1D,
transmitted_state); this script only loops over parameters and plots.
`params` is a dict with keys t, eps, d1, d2,
e.g.  params = dict(t=1.0, eps=0.0, d1=50, d2=50).

Run from the repository root:  python scripts/fig2_fig3_chain_1d.py

Sign convention of the analytic overlays (paper Eqs. (7)-(8)):
    |p_y> -> -sin(dphi)   |p_x> -> +sin(dphi)
    |+z>  -> +cos(dphi)   |-z>  -> -cos(dphi)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
from orbfilter.chain_1d import OrbitalChain1D, transmitted_state, STATES, Lz

# ---- Tol bright palette / revtex-ish style ---------------------------
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 8, "axes.linewidth": 0.6})
C_Q, C_D = "#4477AA", "#EE6677"          # quadrupole / dipole families

# ======================================================================
#  thin wrapper: one (g1,g2,d1,d2,E,state) -> <Lz>
# ======================================================================
def lz_point(params, g1, g2, energy, phi_in, d1=None, d2=None):
    kw = dict(params)
    if d1 is not None:
        kw["d1"] = d1
    if d2 is not None:
        kw["d2"] = d2
    wire = OrbitalChain1D(g1=g1, g2=g2, **kw)
    wire.build()
    phi_out, T = transmitted_state(wire, phi_in, energy)
    if phi_out is None or np.linalg.norm(phi_out) < 1e-12:
        return np.nan
    n2 = np.vdot(phi_out, phi_out).real
    return np.vdot(phi_out, Lz @ phi_out).real / n2


# ----- analytic reference (exact-scattering signs) --------------------
def _dphi(params, g1, g2, energy, d1=None, d2=None):
    t, eps = params["t"], params["eps"]
    d1 = params["d1"] if d1 is None else d1
    d2 = params["d2"] if d2 is None else d2
    cp1 = (eps + g1 - energy) / (2 * t); cm1 = (eps - g1 - energy) / (2 * t)
    cp2 = (eps + g2 - energy) / (2 * t); cm2 = (eps - g2 - energy) / (2 * t)
    if max(abs(cp1), abs(cm1), abs(cp2), abs(cm2)) >= 1:
        return np.nan
    return ((np.arccos(cm1) - np.arccos(cp1)) * d1
            + (np.arccos(cm2) - np.arccos(cp2)) * d2)

_ANALYTIC = {"py": lambda p: -np.sin(p), "px": lambda p: +np.sin(p),
             "+z": lambda p: +np.cos(p), "-z": lambda p: -np.cos(p)}

def analytic_lz(params, g1, g2, energy, state, d1=None, d2=None):
    p = _dphi(params, g1, g2, energy, d1, d2)
    return np.nan if np.isnan(p) else _ANALYTIC[state](p)


# ======================================================================
#  (1) <Lz> vs E for several g2/g1 ratios, one input state
# ======================================================================
# role-keyed palette matching the standalone ratio figure
C_RATIO = {"echo": "#000000", "n0": "#347AC1", "n1": "#B8414F",
           "nm1": "#B13D4B", "hom": "#D09E38"}

def fig_lz_vs_E(params, g1, ratios, state="py", energies=None,
                n_jobs=-1, overlay=True, save=None):
    """ratios: dict role -> g2/g1, with role in {echo, n0, n1, nm1, hom}
    (role sets the colour, matching the standalone figure).  Loops your
    transmitted_state over E.  <Lz> is the Kwant S-matrix readout; for |p_y>
    that is a sine curve, for |+z> a cosine (cf. Fig. 2)."""
    if energies is None:
        rmax = max(abs(r) for r in ratios.values())
        Emax = 2 * params["t"] - g1 * rmax - 0.02
        energies = np.linspace(-Emax, Emax, 1200)
    else:
        Emax = energies.max()
    phi = STATES[state]

    fig, ax = plt.subplots(figsize=(3.375, 2.5),
                           gridspec_kw=dict(left=0.145, right=0.975,
                                            top=0.89, bottom=0.14))
    for role, r in ratios.items():
        g2 = r * g1
        col = C_RATIO.get(role, "#347AC1")
        lw = 1.4 if role == "echo" else (1.2 if role == "hom" else 1.2)
        alpha = 0.6 if role == "hom" else 1.0
        lab = _ratio_label(role, r)
        lz = Parallel(n_jobs=n_jobs)(
            delayed(lz_point)(params, g1, g2, e, phi) for e in energies)
        ax.plot(energies, lz, "-", color=col, lw=lw, label=lab, alpha=alpha)
        if overlay:
            la = [analytic_lz(params, g1, g2, e, state) for e in energies]
            ax.plot(energies, la, dashes=(4, 6), color="#239C0A", lw=1.2,
                    alpha=0.8, label="analytic" if role == "echo" else None)

    for yv in (-1, 0, 1):
        ax.axhline(yv, color="#BBBBBB", lw=0.5, ls=":", zorder=0)
    ax.axvline(params["eps"], color="#BBBBBB", lw=0.5, ls="--", zorder=0)
#    ttl = {"py": r"|p_y\rangle", "px": r"|p_x\rangle",
#           "+z": r"|p_y\rangle+i|p_x\rangle",
#           "-z": r"|p_y\rangle-i|p_x\rangle"}[state]
    ttl = {"py": r"$\langle L_z\rangle = 0$", "px": r"$\langle L_z\rangle = 0$",
           "+z": r"$\langle L_z\rangle = 1$",
           "-z": r"$\langle L_z\rangle = -1$"}[state]

    ax.set_title(rf"injected state with {ttl}", fontsize=9.2)
    ax.set_xlim(-Emax, Emax)
    ax.set_ylim(-1.16, 1.16)
    ax.set_xlabel(r"$E/t$", fontsize=10)
    ax.set_ylabel(r"$\langle L_z\rangle_{\mathrm{out}}$", fontsize=10)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="none", borderaxespad=0.2,
              handlelength=1.6, labelspacing=0.25)
    # annotations: for |+z> the echo preserves the dipole (no filtering),
    # the stripes convert; for |p_y> it is the reverse.
    if state in ("+z", "-z"):
        ax.annotate("no filtering", (-0.15 * Emax, 1.03), fontsize=6.5,
                    color="#444444", fontweight="bold")
        ax.annotate("filtering", (-0.11 * Emax, 0.15), fontsize=6.5,
                    color=C_RATIO["n1"], fontweight="bold")
    else:
        ax.annotate("no filtering", (-0.15 * Emax, 0.05), fontsize=6.5,
                    color="#444444", fontweight="bold")
        ax.annotate("filtering", (-0.11 * Emax, 1.03), fontsize=6.5,
                    color=C_RATIO["n1"], fontweight="bold")
    ax.tick_params(direction="in", top=True, right=True, length=2.5)
    if save:
        fig.savefig(save, dpi=300); fig.savefig(save.replace(".png", ".pdf"))
    return fig, ax


def _ratio_label(role, r):
    if role == "echo":
        return r"$g_2/g_1=-1$"
    if role == "hom":
        return r"$g_2/g_1=+1$"
    return rf"$g_2/g_1={r:.2f}$"



# ======================================================================
#  (2) Kwant <Lz>(g2/g1, E) map, used by fig_maps_ratio_E(engine='kwant')
# ======================================================================
def map_g2_energy(params, g1, ratios, energies, state, n_jobs=-1):
    """<Lz>(g2/g1, E) from transmitted_state (g1 fixed, g2 varied)."""
    phi = STATES[state]
    grid = [(r, e) for e in energies for r in ratios]
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(lz_point)(params, g1, r * g1, e, phi) for r, e in grid)
    return np.array(res).reshape(len(energies), len(ratios))


# ======================================================================
#  Stripe (conversion-locus) ratios for GENERAL lengths d1, d2
# ======================================================================
def stripe_ratio(params, g1, n, d1=None, d2=None, E0=None,
                 rmin=-3.0, rmax=1.0, nscan=4000):
    """g2/g1 of the n-th conversion locus (dphi = pi/2 + n*pi) at energy E0
    (default band centre), for arbitrary segment lengths d1, d2.  Found by
    scanning dphi(g2) for the sign change of [dphi - (pi/2 + n*pi)].
    Returns np.nan if that locus is absent in [rmin, rmax]."""
    from scipy.optimize import brentq
    eps = params["eps"]
    E0 = eps if E0 is None else E0
    target = np.pi / 2 + n * np.pi
    rs = np.linspace(rmin, rmax, nscan)
    f = np.array([_dphi(params, g1, r * g1, E0, d1, d2) for r in rs]) - target
    out = []
    for i in range(len(rs) - 1):
        if np.isfinite(f[i]) and np.isfinite(f[i + 1]) and f[i] * f[i + 1] < 0:
            root = brentq(lambda r: _dphi(params, g1, r * g1, E0, d1, d2)
                          - target, rs[i], rs[i + 1])
            out.append(root)
    if not out:
        return np.nan
    # return the locus closest to the echo line g2/g1 = -1
    return min(out, key=lambda r: abs(r + 1.0))


def flanking_stripe_ratios(params, g1, d1=None, d2=None, E0=None):
    """Convenience: {echo, n=0, n=-1, uniform} ratios for given lengths.
    For d1 != d2 the echo (dphi = 0) is no longer at g2/g1 = -1; the true
    dphi = 0 locus is returned as 'echo' instead."""
    from scipy.optimize import brentq
    eps = params["eps"]
    E0 = eps if E0 is None else E0
    # dphi = 0 locus (refocus / no filtering)
    rs = np.linspace(-3, 1, 4000)
    f = np.array([_dphi(params, g1, r * g1, E0, d1, d2) for r in rs])
    echo = -1.0
    for i in range(len(rs) - 1):
        if np.isfinite(f[i]) and np.isfinite(f[i + 1]) and f[i] * f[i + 1] < 0:
            echo = brentq(lambda r: _dphi(params, g1, r * g1, E0, d1, d2),
                          rs[i], rs[i + 1]); break
    return {"echo": echo,
            "n0": stripe_ratio(params, g1, 0, d1, d2, E0),
            "nm1": stripe_ratio(params, g1, -1, d1, d2, E0),
            "hom": 1.0}


# ======================================================================
#  Two-panel (g2/g1, E) maps, |py> and |+z|, styled + length-aware
# ======================================================================
def fig_maps_ratio_E(params, g1, r=None, energies=None, d1=None, d2=None,
                     engine="analytic", py_sign=-1, mask=True,
                     n_jobs=-1, save=None):
    """Transmitted <Lz> over (g2/g1, E) for |py> (top) and |+z> (bottom).

    engine='analytic' -> ballistic fields (fast); 'kwant' -> loops your
    transmitted_state.  d1, d2 override params (different lengths); the
    conversion-loci contours and the two-channel mask are recomputed for
    those lengths, so the stripes move correctly when d1 != d2.

    py_sign=-1 gives <Lz> = -sin(dphi) for |py>, as in Eq. (7); it only
    affects engine='analytic'.  The dashed vertical marks g2/g1 = -1; when d1 != d2 the
    true zero-<Lz> locus is drawn as a solid black line instead.
    """
    t, eps = params["t"], params["eps"]
    d1 = params["d1"] if d1 is None else d1
    d2 = params["d2"] if d2 is None else d2
    if r is None:
        r = np.linspace(-2., 1.3, 801)
    if energies is None:
        energies = np.linspace(-1.5, 1.5, 801)
    RR, EE = np.meshgrid(r, energies)
    G2 = RR * g1

    def dk(g, Eg):
        return (np.arccos(np.clip((eps - g - Eg) / (2 * t), -1, 1))
                - np.arccos(np.clip((eps + g - Eg) / (2 * t), -1, 1)))
    DPHI = dk(g1, EE) * d1 + dk(G2, EE) * d2                 # length-aware

    # ---- fields ----
    if engine == "analytic":
        lz_py = py_sign * np.sin(DPHI)
        lz_pz = np.cos(DPHI)
    elif engine == "kwant":
        pj = dict(params); pj["d1"], pj["d2"] = d1, d2
        lz_py = map_g2_energy(pj, g1, r, energies, "py", n_jobs)
        lz_pz = map_g2_energy(pj, g1, r, energies, "+z", n_jobs)
    else:
        raise ValueError("engine must be 'analytic' or 'kwant'")

    # ---- two-channel (polarizer) mask ----
    if mask:
        gmax = np.maximum(np.abs(g1), np.abs(G2))
        prop = (np.abs(EE) + gmax) < (2 * t - 1e-9)
        lz_py = np.where(prop, lz_py, np.nan)
        lz_pz = np.where(prop, lz_pz, np.nan)
        DPHI = np.where(prop, DPHI, np.nan)

    # ---- conversion loci dphi = pi/2 + n pi ----
    lo, hi = np.nanmin(DPHI), np.nanmax(DPHI)
    levels = [np.pi / 2 + n * np.pi
              for n in range(int(np.floor((lo - np.pi / 2) / np.pi)),
                             int(np.ceil((hi - np.pi / 2) / np.pi)) + 1)]

    # ---- plot (style copied from the standalone map figure) ----
    fig, axes = plt.subplots(2, 1, figsize=(3.375, 5.0), sharex=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DDDDDD")
    norm = Normalize(-1, 1)
    panels = [(axes[0], lz_py, r"injected state with $\langle L_z\rangle = 0$"),
              (axes[1], lz_pz, r"injected state with $\langle L_z\rangle = 1$")]
    same_len = (d1 == d2)
    for ax, field, title in panels:
        im = ax.pcolormesh(r, energies, field, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        if same_len:
            ax.axvline(-1.0, color="k", lw=0.8, ls=(0, (4, 2)))   # echo
        else:
            ax.contour(RR, EE, DPHI, levels=[0.0], colors="k",
                       linewidths=0.9)                           # true null
        ax.contour(RR, EE, DPHI, levels=levels, colors="k",
                   linewidths=0.4, alpha=0.55)
        ax.set_ylabel(r"$E/t$", fontsize=10)
        ax.set_ylim(energies.min(), energies.max())
        ax.set_title(title, fontsize=9.2, pad=3)
        ax.tick_params(direction="in", top=True, right=True, length=2.5,
                       color="#888888")
    axes[1].set_xlabel(r"$g_2/g_1$", fontsize=10)
    axes[1].set_xlim(-1.2, 1.05)
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.9,
                        ticks=[-1, 0, 1], pad=0.02)
    cbar.set_label(r"$\langle L_z\rangle_{\mathrm{out}}$", fontsize=10)
    cbar.ax.tick_params(length=2.5)
    if save:
        fig.savefig(save, dpi=300)
        fig.savefig(save.replace(".png", ".pdf"))
    print(f"dphi range: {lo:.2f} {hi:.2f}  d1={d1} d2={d2} engine={engine}")
    return fig, axes


# ======================================================================
#  Driver: produces Figs. 2 and 3
# ======================================================================
if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    params = dict(t=1.0, eps=0.0, d1=50, d2=50)

    # length-aware stripe ratios (recomputed for whatever d1, d2 you set)
    print("equal lengths d1=d2=50:", flanking_stripe_ratios(params, 0.5))
    print("unequal d1=70,d2=50:  ",
          flanking_stripe_ratios(params, 0.5, d1=70, d2=50))

    # (1) lines vs E, role-keyed ratios (colours match the standalone fig)
    rr = flanking_stripe_ratios(params, 0.5)
    ratios_py = {"echo": rr["echo"], #"n0": rr["n0"],
                 "nm1": rr["nm1"], "hom": 1.0}
    fig_lz_vs_E(params, 0.5, ratios_py, state="py",
                save="figures/fig2_lz_vs_E_py.png")
    fig_lz_vs_E(params, 0.5, ratios_py, state="+z",
                save="figures/fig2_lz_vs_E_pz.png")
    print("ratios finished")
    print("starting kwant")
    fig_maps_ratio_E(params, 0.5,
                     engine="kwant", save="figures/fig3_maps_ratio_E.png")
    print("done")