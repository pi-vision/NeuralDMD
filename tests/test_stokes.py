"""Analytic tests for Stokes <-> correlation-product conversions."""

from __future__ import annotations

import numpy as np

from neuraldmd.physics.stokes import (
    CIRCULAR_PRODUCTS,
    LINEAR_PRODUCTS,
    STOKES_ORDER,
    evpa,
    linear_polarized_intensity,
    products_to_stokes_matrix,
    stokes_pauli_matrices,
    stokes_to_products_matrix,
)


def _prod(stokes_vec, products, stokes=STOKES_ORDER):
    """Map a Stokes vector to correlation products.

    Parameters
    ----------
    stokes_vec : sequence of complex
        Stokes values, ordered as ``stokes``.
    products : tuple of str
        Correlation products to produce.
    stokes : tuple of str, optional
        Ordering of ``stokes_vec`` (default IQUV).

    Returns
    -------
    numpy.ndarray
        The product vector ``M @ stokes_vec``.
    """
    return stokes_to_products_matrix(products, stokes) @ np.asarray(stokes_vec, complex)


def test_pure_stokes_to_circular():
    """Each pure Stokes state maps to the expected circular-hand pattern."""
    # products order: (RR, RL, LR, LL)
    np.testing.assert_allclose(_prod([1, 0, 0, 0], CIRCULAR_PRODUCTS), [1, 0, 0, 1])  # I
    np.testing.assert_allclose(_prod([0, 1, 0, 0], CIRCULAR_PRODUCTS), [0, 1, 1, 0])  # Q
    np.testing.assert_allclose(_prod([0, 0, 1, 0], CIRCULAR_PRODUCTS), [0, 1j, -1j, 0])  # U
    np.testing.assert_allclose(_prod([0, 0, 0, 1], CIRCULAR_PRODUCTS), [1, 0, 0, -1])  # V


def test_pure_stokes_to_linear():
    """Each pure Stokes state maps to the expected linear-hand pattern."""
    # products order: (XX, XY, YX, YY)
    np.testing.assert_allclose(_prod([1, 0, 0, 0], LINEAR_PRODUCTS), [1, 0, 0, 1])  # I
    np.testing.assert_allclose(_prod([0, 1, 0, 0], LINEAR_PRODUCTS), [1, 0, 0, -1])  # Q
    np.testing.assert_allclose(_prod([0, 0, 1, 0], LINEAR_PRODUCTS), [0, 1, 1, 0])  # U
    np.testing.assert_allclose(_prod([0, 0, 0, 1], LINEAR_PRODUCTS), [0, 1j, -1j, 0])  # V


def test_v_zero_gives_equal_parallel_hands():
    """With V=0 the parallel hands are equal (RR == LL == I)."""
    rng = np.random.default_rng(0)
    I, Q, U = rng.normal(size=3)
    rr, rl, lr, ll = _prod([I, Q, U, 0.0], CIRCULAR_PRODUCTS)
    np.testing.assert_allclose(rr, ll)  # V=0 -> RR == LL
    np.testing.assert_allclose(rr, I)


def test_circular_roundtrip_is_identity():
    """Stokes->circular->Stokes is the identity for the full 4-product basis."""
    fwd = stokes_to_products_matrix(CIRCULAR_PRODUCTS, STOKES_ORDER)  # 4x4
    inv = products_to_stokes_matrix(STOKES_ORDER, CIRCULAR_PRODUCTS)
    np.testing.assert_allclose(inv @ fwd, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(fwd @ inv, np.eye(4), atol=1e-12)


def test_iqu_subset_drops_v():
    """An (I, Q, U)-only sky yields a 4x3 map with no V column."""
    # a sky with only (I, Q, U) -> RR carries no V term
    M = stokes_to_products_matrix(CIRCULAR_PRODUCTS, ("I", "Q", "U"))
    assert M.shape == (4, 3)
    rr, rl, lr, ll = M @ np.array([2.0, 0, 0], complex)  # I=2, no V -> RR=LL=2
    np.testing.assert_allclose([rr, ll], [2.0, 2.0])
    np.testing.assert_allclose([rl, lr], [0.0, 0.0])


def test_pauli_matrices_build_linear_coherency():
    """B = sum_s S_s sigma_s equals the [[XX, XY], [YX, YY]] coherency matrix."""
    sig = stokes_pauli_matrices()
    I, Q, U, V = 0.7, 0.2, -0.3, 0.1
    B = I * sig["I"] + Q * sig["Q"] + U * sig["U"] + V * sig["V"]
    xx, xy, yx, yy = _prod([I, Q, U, V], LINEAR_PRODUCTS)
    np.testing.assert_allclose(B, np.array([[xx, xy], [yx, yy]]))


def test_evpa():
    """EVPA takes the expected values for (Q, U) on the coordinate axes."""
    np.testing.assert_allclose(evpa(np.array(1.0), np.array(0.0)), 0.0)
    np.testing.assert_allclose(evpa(np.array(0.0), np.array(1.0)), np.pi / 4)
    np.testing.assert_allclose(evpa(np.array(-1.0), np.array(0.0)), np.pi / 2)
    np.testing.assert_allclose(evpa(np.array(0.0), np.array(-1.0)), -np.pi / 4)


def test_linear_polarized_intensity():
    """P = sqrt(Q^2 + U^2) (3-4-5 triangle)."""
    np.testing.assert_allclose(linear_polarized_intensity(np.array(3.0), np.array(4.0)), 5.0)


def test_unknown_product_raises():
    """Requesting an unrecognized product raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="unknown products"):
        stokes_to_products_matrix(("RR", "ZZ"))
