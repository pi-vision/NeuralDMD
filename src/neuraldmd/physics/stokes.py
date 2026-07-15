"""Stokes parameters and their mapping to interferometric correlation products.

Conventions (IAU / EHT circular feeds), ported from the resolve pipeline:

Coherency products in terms of Stokes (I, Q, U, V):

    circular feeds (R, L):        linear feeds (X, Y):
        RR = I + V                    XX = I + Q
        LL = I - V                    YY = I - Q
        RL = Q + iU                   XY = U + iV
        LR = Q - iU                   YX = U - iV

So for perfectly calibrated data a pure-I source gives RR = LL = I and
RL = LR = 0, while V = 0 gives RR = LL. EVPA (electric-vector position angle)
is chi = 1/2 * atan2(U, Q).

These are pure, jax-friendly constant matrices; nothing here imports ehtim.
"""

from __future__ import annotations

import numpy as np

STOKES_ORDER: tuple[str, ...] = ("I", "Q", "U", "V")
CIRCULAR_PRODUCTS: tuple[str, ...] = ("RR", "RL", "LR", "LL")
LINEAR_PRODUCTS: tuple[str, ...] = ("XX", "XY", "YX", "YY")

# product = sum_s _PRODUCT_COEFFS[product][s] * Stokes_s
_PRODUCT_COEFFS: dict[str, dict[str, complex]] = {
    "RR": {"I": 1, "V": 1},
    "LL": {"I": 1, "V": -1},
    "RL": {"Q": 1, "U": 1j},
    "LR": {"Q": 1, "U": -1j},
    "XX": {"I": 1, "Q": 1},
    "YY": {"I": 1, "Q": -1},
    "XY": {"U": 1, "V": 1j},
    "YX": {"U": 1, "V": -1j},
}


def stokes_to_products_matrix(
    products: tuple[str, ...], stokes: tuple[str, ...] = STOKES_ORDER
) -> np.ndarray:
    """Matrix ``M`` (n_products, n_stokes) with ``product_vec = M @ stokes_vec``.

    Parameters
    ----------
    products : tuple of str
        Correlation products to produce, e.g. ``("RR", "RL", "LR", "LL")``.
    stokes : tuple of str
        Stokes parameters present in the sky model, in order (default IQUV).
    """
    unknown = set(products) - _PRODUCT_COEFFS.keys()
    if unknown:
        raise ValueError(f"unknown products: {sorted(unknown)}")
    M = np.zeros((len(products), len(stokes)), dtype=np.complex128)
    for i, p in enumerate(products):
        for s, c in _PRODUCT_COEFFS[p].items():
            if s in stokes:
                M[i, stokes.index(s)] = c
    return M


def products_to_stokes_matrix(stokes: tuple[str, ...], products: tuple[str, ...]) -> np.ndarray:
    """Least-squares inverse of :func:`stokes_to_products_matrix`.

    For a complete product basis (e.g. all four circular products) this is the
    exact inverse; otherwise it is the Moore-Penrose pseudo-inverse.
    """
    return np.linalg.pinv(stokes_to_products_matrix(products, stokes))


def stokes_pauli_matrices() -> dict[str, np.ndarray]:
    """The 2x2 basis matrices B = sum_s S_s * sigma_s for the coherency matrix.

    Used by the RIME chain (J_i B J_j^H). I->identity, Q,U,V->Pauli-like.
    """
    return {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "Q": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        "U": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "V": np.array([[0, 1j], [-1j, 0]], dtype=np.complex128),
    }


def evpa(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Electric-vector position angle chi = 1/2 * atan2(U, Q), in radians."""
    return 0.5 * np.arctan2(U, Q)


def linear_polarized_intensity(Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """P = sqrt(Q^2 + U^2). Note: NOT divided by I (we never divide by I)."""
    return np.sqrt(Q**2 + U**2)
