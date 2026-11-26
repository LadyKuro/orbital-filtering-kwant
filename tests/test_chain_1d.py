"""Tests for the 1D two-segment chain (orbfilter.chain_1d, Sec. II)."""

import numpy as np
import pytest

from orbfilter.chain_1d import (OrbitalChain1D, transmitted_state, STATES,
                                Lx, Ly, Lz, chi_z, Q_ops)

PY_PX = np.ix_([1, 2], [1, 2])          # {p_y, p_x} block of the {p_z, p_y, p_x} basis
SX = np.array([[0, 1], [1, 0]])
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]])


# ----------------------------------------------------------------------
#  Orbital operators
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name, op", [("+x", Lx), ("+y", Ly), ("+z", Lz)])
def test_dipole_states_are_plus_one_eigenstates(name, op):
    v = STATES[name]
    assert np.allclose(op @ v, v)


def test_pseudospin_algebra_in_py_px_subspace():
    """chi_z, Lz and Lx^2 - Ly^2 act as sigma_x, sigma_y, sigma_z on {p_y, p_x}."""
    assert np.allclose(chi_z[PY_PX], SX)
    assert np.allclose(Lz[PY_PX], SY)
    assert np.allclose(Q_ops["Lx^2-Ly^2"][PY_PX], SZ)


def test_chi_z_does_not_touch_pz():
    assert np.allclose(chi_z[0, :], 0) and np.allclose(chi_z[:, 0], 0)


# ----------------------------------------------------------------------
#  Ballistic phase
# ----------------------------------------------------------------------
def test_precession_angle_is_odd_in_g():
    wire = OrbitalChain1D()
    for E in (-0.6, 0.0, 0.4):
        assert np.isclose(wire.precession_angle(E, 0.3, 50),
                          -wire.precession_angle(E, -0.3, 50))


# ----------------------------------------------------------------------
#  Scattering: the g2 = -g1 echo is an exact invariant (Fig. 3, dashed line)
# ----------------------------------------------------------------------
def _lz_out(wire, state, E):
    phi_out, T = transmitted_state(wire, STATES[state], E)
    return np.vdot(phi_out, Lz @ phi_out).real / np.vdot(phi_out, phi_out).real, T


@pytest.mark.parametrize("E", [-0.5, 0.2, 0.7])
def test_echo_quadrupole_input_gives_zero_lz(E):
    wire = OrbitalChain1D(g1=0.3, g2=-0.3, d1=40, d2=40)
    wire.build()
    lz, T = _lz_out(wire, "py", E)
    assert T > 0
    assert abs(lz) < 1e-6


@pytest.mark.parametrize("E", [-0.5, 0.2, 0.7])
def test_echo_dipole_input_is_preserved(E):
    wire = OrbitalChain1D(g1=0.3, g2=-0.3, d1=40, d2=40)
    wire.build()
    lz, _ = _lz_out(wire, "+z", E)
    assert abs(lz - 1) < 1e-6


def test_transmission_is_a_probability():
    wire = OrbitalChain1D(g1=0.3, g2=0.3, d1=40, d2=40)
    wire.build()
    for E in (-1.0, 0.0, 0.9):
        _, T = transmitted_state(wire, STATES["py"], E)
        assert -1e-10 <= T <= 1 + 1e-10
