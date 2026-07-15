"""Characterize the Zernike bank: masked-QR orthonormality, basis count,
low-|m| mode selection, disk mask.
"""

from __future__ import annotations

import numpy as np
from _impl import build_zernike_targets, make_xy_grid, zernike_complex_basis


def test_masked_qr_orthonormal():
    xy = make_xy_grid(24, 24, np.pi, np.pi)
    Q, nm, mask = zernike_complex_basis(xy, radius=1.0, max_n=4, do_masked_qr=True)
    G = np.asarray(Q).conj().T @ np.asarray(Q)
    np.testing.assert_allclose(G, np.eye(Q.shape[1]), atol=1e-4)


def test_basis_count_for_max_n():
    # #(n,m): n<=max_n, |m|<=n, n-|m| even. max_n=4 -> 1+2+3+4+5 = 15
    xy = make_xy_grid(20, 20, np.pi, np.pi)
    _, nm, _ = zernike_complex_basis(xy, radius=1.0, max_n=4, do_masked_qr=False)
    assert len(nm) == 15
    assert (0, 0) in nm and (4, 4) in nm


def test_pick_prefers_low_azimuthal_order():
    _, picked, _, _ = build_zernike_targets(24, 24, 1.0, np.pi, np.pi, r=5, max_n=8)
    assert picked[0] == (0, 0)  # the disk (piston) mode first
    assert all(abs(m) <= 3 for (_, m) in picked)


def test_mask_is_unit_disk():
    xy = make_xy_grid(24, 24, np.pi, np.pi)
    _, _, mask = zernike_complex_basis(xy, radius=np.pi / 2, max_n=2)
    mask = np.asarray(mask)
    rho = np.sqrt((np.asarray(xy) ** 2).sum(1)) / (np.pi / 2)
    np.testing.assert_array_equal(mask, (rho <= 1.0).astype(np.float32))
