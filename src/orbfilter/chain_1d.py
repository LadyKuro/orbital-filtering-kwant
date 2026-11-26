"""
1D three-orbital chain (Sec. II of the paper) with
mirror-symmetry-breaking orbital coupling g0 * chi, in one or two segments.

Physics summary (chi = chi_z case):
  pz decouples; in the {py, px} subspace the operators
      sigma_x = chi_z = LxLy + LyLx     (off-diagonal quadrupole Q_xy)
      sigma_y = Lz                       (dipole)
      sigma_z = Lx^2 - Ly^2              (diagonal quadrupole)
  close a pseudospin algebra, and the coupling g0*chi_z makes the orbital
  character precess around the sigma_x axis with spatial rate
      dphi/dx = k_minus - k_plus,   cos(k_pm a) = (eps +/- g0 - E) / (2 t).
  A second segment with -g0 reverses the precession -> dipole/quadrupole
  reconversion. Full dipole->quadrupole conversion after one segment of
  length d requires (k_- - k_+) d = pi/2 + n pi.

Basis: {pz, py, px}. Leads are orbital-isotropic (t * I),
so all orbital superpositions are degenerate incoming states and <L>, <Q>
are constant in the leads (no lead precession).

Requires: numpy, kwant, joblib.
"""

import numpy as np
import kwant
from joblib import Parallel, delayed

# ----------------------------------------------------------------------
#  Orbital operators, basis {pz, py, px}
# ----------------------------------------------------------------------
Lx = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
Ly = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
Lz = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])

def anticomm(A, B):
    return A @ B + B @ A

chi_x = anticomm(Ly, Lz)          # breaks M_x, C_2x
chi_y = anticomm(Lx, Lz)          # breaks M_y, C_2y
chi_z = anticomm(Lx, Ly)          # breaks M_z, C_2z  (py<->px mixer)

# quadrupole set tracked in profiles: 3 off-diagonal + 2 diagonal
Q_ops = {
    "Q_yz (chi_x)": chi_x,
    "Q_xz (chi_y)": chi_y,
    "Q_xy (chi_z)": chi_z,
    "Lx^2-Ly^2":    Lx @ Lx - Ly @ Ly,
    "3Lz^2-L^2":    3 * (Lz @ Lz) - 2 * np.eye(3),   # L(L+1)=2
}
L_ops = {"Lx": Lx, "Ly": Ly, "Lz": Lz}

# convenient orbital input states
def ket(*amps):
    v = np.array(amps, dtype=complex)
    return v / np.linalg.norm(v)

STATES = {
    "pz":  ket(1, 0, 0),
    "py":  ket(0, 1, 0),
    "px":  ket(0, 0, 1),
    "+z":  ket(0, 1, +1j),     # (|py> + i|px>)/sqrt2 : Lz = +1
    "-z":  ket(0, 1, -1j),     # Lz = -1
    "+x":  ket(-1j, 1, 0),     # (-i|pz> + |py>)/sqrt2 : Lx = +1
    "+y":  ket(-1j, 0, 1),     # (-i|pz> + |px>)/sqrt2 : Ly = +1
}
# NOTE: |+x> = (-i|py> + |pz>)/sqrt2 in terms of Lx matrix above; verify:
# Lx @ STATES["+x"] should equal +1 * STATES["+x"]. Same for +y with Ly.
for name, op in (("+x", Lx), ("+y", Ly), ("+z", Lz)):
    v = STATES[name]
    assert np.allclose(op @ v, v), f"state {name} is not a +1 eigenstate"


# ----------------------------------------------------------------------
#  Model
# ----------------------------------------------------------------------
class OrbitalChain1D:
    """
    Leads:   H = eps*I on site, hopping -t*I (orbital-isotropic).
    Region:  same, plus onsite g(x) * chi, with
             g(x) = g1 in segment 1 (length d1), g2 in segment 2 (length d2).
             Homogeneous case: set d2 = 0 (or g2 = g1).
    """

    def __init__(self, t=1.0, eps=0.0, g1=0.5, g2=-0.5,
                 d1=60, d2=60, chi=chi_z, a=1.0):
        self.t, self.eps, self.a = t, eps, a
        self.g1, self.g2, self.d1, self.d2 = g1, g2, d1, d2
        self.chi = np.asarray(chi)
        self.norb = 3
        self.syst = None

    def build(self):
        lat = kwant.lattice.chain(self.a, norbs=3)
        syst = kwant.Builder()
        I3 = np.eye(3)

        L = self.d1 + self.d2
        for x in range(L):
            g = self.g1 if x < self.d1 else self.g2
            syst[lat(x)] = self.eps * I3 + g * self.chi
        for x in range(L - 1):
            syst[lat(x), lat(x + 1)] = -self.t * I3

        for sym_vec, edge in (((-self.a,), 0), ((self.a,), L - 1)):
            lead = kwant.Builder(kwant.TranslationalSymmetry(sym_vec))
            lead[lat(0)] = self.eps * I3
            lead[lat(0), lat(1)] = -self.t * I3
            syst.attach_lead(lead, add_cells=0)

        self.syst = syst.finalized()
        return self.syst

    # ------------------------------------------------------------------
    #  Injection of an arbitrary orbital state
    # ------------------------------------------------------------------
    def scattering_state(self, phi_in, energy=0.0, lead=0):
        """
        Scattering state for an incoming state with orbital vector phi_in
        (3-component, any normalization) in the given lead.

        Works because the isotropic lead makes all incoming modes at a
        given energy degenerate with equal |velocity| -> identical flux
        normalization -> superposition coefficients are the projections
        of phi_in on the incoming-mode orbital vectors.

        Returns psi with shape (n_sites, 3).
        """
        if self.syst is None:
            raise RuntimeError("call build() first")
        phi_in = np.asarray(phi_in, dtype=complex)
        phi_in = phi_in / np.linalg.norm(phi_in)

        wf = kwant.wave_function(self.syst, energy)
        psi_modes = wf(lead)                       # (n_in_modes, n_sites*3)

        flead = self.syst.leads[lead]
        prop, _ = flead.modes(energy)
        v = prop.velocities
        incoming = np.where(v < 0)[0]              # Kwant: incoming v < 0
        # orbital vectors of the incoming modes (1 site per lead cell)
        U = prop.wave_functions[:, incoming]       # (3, n_in)
        U = U[:self.norb, :]

        if U.shape[1] != psi_modes.shape[0]:
            raise RuntimeError("mode bookkeeping mismatch; check ordering")

        # coefficients c solving U c = phi_in (U unitary up to normalization)
        c, *_ = np.linalg.lstsq(U, phi_in, rcond=None)

        psi = np.tensordot(c, psi_modes, axes=(0, 0))
        return psi.reshape(-1, self.norb)

    # ------------------------------------------------------------------
    #  Profiles
    # ------------------------------------------------------------------
    def profiles(self, phi_in, energy=0.0, lead=0, operators=None):
        """
        Per-site normalized expectation values <O>(x) for a dict of
        orbital operators. Returns (x_positions, {name: values}).
        """
        if operators is None:
            operators = {**L_ops, **Q_ops}
        psi = self.scattering_state(phi_in, energy, lead)
        dens = np.einsum("si,si->s", psi.conj(), psi).real
        x = np.array([s.pos[0] for s in self.syst.sites])
        order = np.argsort(x)
        out = {}
        for name, O in operators.items():
            val = np.einsum("si,ij,sj->s", psi.conj(), np.asarray(O), psi).real
            out[name] = (val / dens)[order]
        return x[order], out

    # ------------------------------------------------------------------
    #  Analytic reference (homogeneous chi_z segment, no reflections)
    # ------------------------------------------------------------------
    def precession_angle(self, energy, g, length):
        """(k_- - k_+) * length for a chi_z segment; ballistic estimate."""
        kp = np.arccos(np.clip((self.eps + g - energy) / (2 * self.t), -1, 1))
        km = np.arccos(np.clip((self.eps - g - energy) / (2 * self.t), -1, 1))
        return (km - kp) * length * self.a

def transmitted_state(wire, phi_in, energy):
    """
    Returns (phi_out, T): orbital vector of the transmitted state in the
    right lead (unnormalized; |phi_out|^2 relates to transmitted flux)
    and the total transmission probability for this input.
    Returns (None, 0.0) if no propagating modes.
    """
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
