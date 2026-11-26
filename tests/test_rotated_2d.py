"""Tests for the rotated 2D junction (orbfilter.rotated_2d, Sec. III).

Small systems (W = 3, L_central = 20) so the suite runs in seconds.
"""

import numpy as np
import pytest
import kwant

from orbfilter.rotated_2d import Rotated2D

PARAMS = dict(onsite_energy=0.0, Vpi=-0.3, Vsigma=1.0, L=30, L_central=20,
              width=3, a=1.0, theta=0.0, coupling=0.5, interface_cutoff=1.2)
E_TEST = 0.3


def make(**kw):
    wire = Rotated2D(**dict(PARAMS, **kw))
    wire.build()
    return wire


# ----------------------------------------------------------------------
#  Slater-Koster matrices (no system needed)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.0, 0.3, np.pi / 6, 1.1, np.pi / 2, 2.5])
def test_interface_hopping_matches_bulk_sk_at_distance_a(phi):
    """t_direction(phi, d=a) / coupling must equal the bulk matrix t1(phi):
    same Slater-Koster form, including the sign of the p_y-p_x element."""
    w = Rotated2D(**PARAMS)
    assert np.allclose(w.t_direction(phi, w.a) / w.coupling, w.t1(phi))


@pytest.mark.parametrize("angle", [0.0, 0.4, np.pi / 6, 1.2])
def test_y_hopping_is_x_hopping_rotated_by_90_degrees(angle):
    w = Rotated2D(**PARAMS)
    assert np.allclose(w.t2(angle), w.t1(angle + np.pi / 2))


def test_width_one_is_rejected():
    with pytest.raises(ValueError):
        Rotated2D(**dict(PARAMS, width=1)).build()


# ----------------------------------------------------------------------
#  Symmetry limits of the junction
# ----------------------------------------------------------------------
@pytest.mark.parametrize("cutoff", [None, 1.2])
@pytest.mark.parametrize("theta", [0.0, np.pi / 2])
def test_no_orbital_polarization_without_rotation(theta, cutoff):
    """theta = 0, pi/2: the central lattice coincides with the lead lattice,
    all hoppings are orbital-diagonal, so L_z^max vanishes."""
    wire = make(theta=theta, interface_cutoff=cutoff)
    assert abs(wire.transmitted_Lz_envelope(energy=E_TEST)) < 1e-10


def test_theta_and_theta_plus_90_degrees_are_the_same_junction():
    """The square lattice is C4 symmetric: R(theta) and R(theta + pi/2)
    give the same sites and the same hoppings."""
    a = make(theta=0.3).transmitted_Lz_envelope(energy=E_TEST)
    b = make(theta=0.3 + np.pi / 2).transmitted_Lz_envelope(energy=E_TEST)
    assert np.isclose(a, b, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("cutoff", [None, 1.2])
def test_perfect_lead_limit(cutoff):
    """theta = 0 and coupling = 1: the junction is an unbroken lead, so every
    open channel is fully transmitted (requires a <= r_c < sqrt(2) a)."""
    wire = make(theta=0.0, coupling=1.0, interface_cutoff=cutoff)
    sm = kwant.smatrix(wire.syst, E_TEST)
    n_open = sm.num_propagating(0)
    assert n_open > 0
    assert np.isclose(sm.transmission(1, 0), n_open, atol=1e-8)


def test_rotation_produces_finite_polarization_somewhere():
    """Sanity check that the observable is not identically zero."""
    vals = [make(theta=th).transmitted_Lz_envelope(energy=E)
            for th in (0.3, 0.5) for E in (-0.8, 0.3, 0.9)]
    assert max(vals) > 1e-3
