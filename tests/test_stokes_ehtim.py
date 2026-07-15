"""Cross-check our Stokes<->product conventions against ehtim's authoritative
``observing/pol_conventions.py`` (Chael 2026, which cites "TMS Ch. 4").

This is the rigorous pin: rather than trust our hand-entered coefficient table,
we assert it reproduces ehtim's transforms bit-for-bit over random Stokes
vectors. If either side ever changes its convention, CI fails here. Marked
``ehtim`` so it runs only where the [data] extra is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pol = pytest.importorskip("ehtim.observing.pol_conventions")

from neuraldmd.physics.stokes import (  # noqa: E402  (after importorskip by design)
    CIRCULAR_PRODUCTS,
    LINEAR_PRODUCTS,
    STOKES_ORDER,
    evpa,
    stokes_pauli_matrices,
    stokes_to_products_matrix,
)

# ehtim's transforms emit a once-per-session MixedPolConventionWarning; silence it.
pytestmark = [pytest.mark.ehtim, pytest.mark.filterwarnings("ignore")]


def _random_stokes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, n))  # rows: I, Q, U, V


def test_circular_products_match_ehtim():
    """(RR, RL, LR, LL) from our matrix == ehtim.stokes_to_circ, exactly."""
    i, q, u, v = _random_stokes()
    M = stokes_to_products_matrix(CIRCULAR_PRODUCTS, STOKES_ORDER)  # rows RR,RL,LR,LL
    ours = M @ np.stack([i, q, u, v]).astype(complex)
    rr, ll, rl, lr = pol.stokes_to_circ(i, q, u, v)  # ehtim order: rr, ll, rl, lr
    np.testing.assert_allclose(ours[0], rr, atol=1e-12)  # RR
    np.testing.assert_allclose(ours[1], rl, atol=1e-12)  # RL
    np.testing.assert_allclose(ours[2], lr, atol=1e-12)  # LR
    np.testing.assert_allclose(ours[3], ll, atol=1e-12)  # LL


def test_linear_products_match_ehtim():
    """(XX, XY, YX, YY) from our matrix == ehtim.stokes_to_lin, exactly."""
    i, q, u, v = _random_stokes(seed=1)
    M = stokes_to_products_matrix(LINEAR_PRODUCTS, STOKES_ORDER)  # rows XX,XY,YX,YY
    ours = M @ np.stack([i, q, u, v]).astype(complex)
    xx, yy, xy, yx = pol.stokes_to_lin(i, q, u, v)  # ehtim order: xx, yy, xy, yx
    np.testing.assert_allclose(ours[0], xx, atol=1e-12)  # XX
    np.testing.assert_allclose(ours[1], xy, atol=1e-12)  # XY
    np.testing.assert_allclose(ours[2], yx, atol=1e-12)  # YX
    np.testing.assert_allclose(ours[3], yy, atol=1e-12)  # YY


def test_pauli_coherency_matrix_is_ehtim_linear_brightness():
    """B = sum_s S_s sigma_s == [[XX, XY], [YX, YY]] from ehtim (TMS Eq. 4.28)."""
    sig = stokes_pauli_matrices()
    rng = np.random.default_rng(2)
    for i, q, u, v in rng.normal(size=(20, 4)):
        B = i * sig["I"] + q * sig["Q"] + u * sig["U"] + v * sig["V"]
        xx, yy, xy, yx = pol.stokes_to_lin(i, q, u, v)
        np.testing.assert_allclose(B, [[xx, xy], [yx, yy]], atol=1e-12)


def test_roundtrip_through_ehtim_circular():
    """ehtim.circ_to_stokes o (our stokes->circ) == identity."""
    i, q, u, v = _random_stokes(seed=3)
    M = stokes_to_products_matrix(CIRCULAR_PRODUCTS, STOKES_ORDER)
    rr, rl, lr, ll = M @ np.stack([i, q, u, v]).astype(complex)
    i2, q2, u2, v2 = pol.circ_to_stokes(rr, ll, rl, lr)  # ehtim arg order rr,ll,rl,lr
    np.testing.assert_allclose([i2.real, q2.real, u2.real, v2.real], [i, q, u, v], atol=1e-12)


def test_evpa_matches_ehtim_definition():
    """Our evpa == 0.5*angle(Q + iU), ehtim's electric-vector position angle."""
    rng = np.random.default_rng(4)
    q, u = rng.normal(size=(2, 100))
    ours = evpa(q, u)
    ref = 0.5 * np.angle(q + 1j * u)  # ehtim image.py convention
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_rime_jones_factoring_matches_ehtim():
    """Pin the RIME Jones convention we will implement in Phase 7: J = G (I + D),
    G = diag(g1, g2), (I + D) = [[1, d1], [d2, 1]] -- identical to
    ehtim.pol_conventions.jones_matrix (TMS Eqs. 4.46-4.47)."""
    g1, g2, d1, d2 = 0.9 + 0.1j, 1.1 - 0.05j, 0.03 + 0.01j, -0.02 + 0.04j
    G = np.array([[g1, 0], [0, g2]], complex)
    IpD = np.array([[1, d1], [d2, 1]], complex)
    np.testing.assert_allclose(pol.jones_matrix(g1, g2, d1, d2), G @ IpD, atol=1e-12)


def test_rime_forward_corruption_inverts():
    """The forward RIME V' = J1 B J2^H is undone by ehtim's inverse-Jones
    correction, on a full IQUV coherency. Grounds the Phase-7 forward model."""
    sig = stokes_pauli_matrices()
    i, q, u, v = 1.0, 0.2, -0.15, 0.05
    B = i * sig["I"] + q * sig["Q"] + u * sig["U"] + v * sig["V"]  # linear brightness
    J1 = pol.jones_matrix(0.95 + 0.05j, 1.02 - 0.03j, 0.02, -0.01)
    J2 = pol.jones_matrix(1.03 + 0.02j, 0.97 + 0.04j, -0.015, 0.025)
    V_obs = J1 @ B @ J2.conj().T  # forward corruption (TMS Eq. 4.52)
    V_corr = pol.apply_inverse_jones_to_coherency(V_obs, J1, J2)
    np.testing.assert_allclose(V_corr, B, atol=1e-12)
