"""Stokes parameters and their mapping to interferometric correlation products.

Conventions follow Thompson, Moran & Swenson (TMS), *Interferometry and
Synthesis in Radio Astronomy*, 3rd ed. (Springer, 2017), Chapter 4 -- the
IAU / IEEE (engineering, exp(+j*omega*t)) circular-feed convention with
R = (X + iY)/sqrt(2), L = (X - iY)/sqrt(2).

Coherency products in terms of Stokes (I, Q, U, V):

    circular feeds (R, L):        linear feeds (X, Y):
        RR = I + V                    XX = I + Q
        LL = I - V                    YY = I - Q
        RL = Q + iU                   XY = U + iV
        LR = Q - iU                   YX = U - iV

Provenance / cross-checks (all three agree exactly):
  * TMS Eq. (4.28) gives the linear coherencies directly:
        <Ex Ex*> = 1/2 (I + Q),   <Ey Ey*> = 1/2 (I - Q),
        <Ex Ey*> = 1/2 (U + jV),  <Ey Ex*> = 1/2 (U - jV).
  * TMS Eq. (4.29) (Morris/Weiler general formula) with circular-feed
    ellipticities chi_R = -pi/4, chi_L = +pi/4 yields RR = I + V, LL = I - V,
    and, after field-rotation derotation, RL = Q + jU, LR = Q - jU.
  * ehtim ``observing/pol_conventions.py`` (Chael 2026, which itself cites
    "TMS Ch. 4"): ``stokes_to_circ_cross(q,u) = (q + 1j*u, q - 1j*u)`` etc.
    ``tests/test_stokes_ehtim.py`` asserts our matrices equal ehtim's numerically.

So for perfectly calibrated data a pure-I source gives RR = LL = I and
RL = LR = 0, while V = 0 gives RR = LL. EVPA (electric-vector position angle)
is chi = 1/2 * atan2(U, Q) (TMS/IAU; matches ehtim ``0.5*angle(Q + iU)``).

Sign caveat: V's sign is tied to the circular basis choice. We use the IAU/IEEE
engineering convention R = (X + iY)/sqrt(2) (positive V = RCP, giving RR = I + V);
the opposite "physics" basis R = (X - iY)/sqrt(2) flips the sign of V. See
``docs/conventions.md``.

RIME grounding (used by ``physics/rime.py``, Phase 7): TMS Eqs. (4.44)-(4.52)
give the measurement equation V'_mn = J_m V_mn J_n^H, where V_mn is the true sky
coherency matrix and the per-station Jones matrix factors as J = G (I + D) R with
G = diag(g_1, g_2) (Eq. 4.47), (I + D) = [[1, d_1], [d_2, 1]] (Eq. 4.46, leakage),
and R = diag(e^{-j*psi}, e^{+j*psi}) (Eq. 4.45, feed/parallactic rotation) -- the
same factoring as ehtim ``pol_conventions.jones_matrix``.

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
    """The 2x2 basis matrices sigma_s for the *linear-feed* coherency matrix.

    The sky brightness (coherency) matrix is
        B = sum_s S_s * sigma_s = [[I + Q, U + iV], [U - iV, I - Q]]
              = [[XX, XY], [YX, YY]]   (TMS Eq. 4.28, linear (X, Y) basis),
    i.e. exactly ehtim's ``_coherency_matrix`` in the XY basis. Used by the RIME
    chain V'_mn = J_m B J_n^H (TMS Eq. 4.52). For circular feeds either build B in
    this linear basis and put the feed transform inside J (F = BASIS_LIN_TO_CIRC),
    or work directly with the circular coherency [[RR, RL], [LR, LL]]; the two are
    equivalent. I->identity, Q,U,V->Pauli-like.
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
