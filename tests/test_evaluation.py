"""Characterize evaluation helpers: grid parity with the loader, mode sorting,
PSNR, and the evaluate_chi2 <-> loss_fn duplication (must agree exactly).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _impl import (
    NeuralDMD,
    calc_psnr,
    evaluate_chi2,
    loss_fn,
    pixel_grid_coords,
    sort_modes_by_lambda,
)


def test_pixel_grid_matches_loader(tiny_obs):
    grid = pixel_grid_coords(tiny_obs.H, tiny_obs.W, tiny_obs.fov, tiny_obs.fov)
    np.testing.assert_allclose(grid, tiny_obs.loader_grid(), rtol=1e-6)


def test_sort_modes_by_lambda():
    W = jnp.asarray(np.arange(6).reshape(2, 3) + 0j)  # (P=2, r=3)
    # |exp(Omega)| = exp(Re Omega): choose Re = [-1, 0, -0.5] -> order [1, 2, 0]
    Omega = jnp.asarray([-1.0, 0.0, -0.5]) + 0j
    b = jnp.asarray([10.0, 20.0, 30.0]) + 0j
    Ws, Os, bs = sort_modes_by_lambda(W, Omega, b)
    assert np.allclose(np.asarray(bs), [20.0, 30.0, 10.0])
    assert np.allclose(np.real(np.asarray(Os)), [0.0, -0.5, -1.0])
    assert np.allclose(np.asarray(Ws)[:, 0], np.asarray(W)[:, 1])


def test_calc_psnr():
    a = np.zeros((4, 4), np.float32)
    assert np.isinf(calc_psnr(a, a))
    b = np.full((4, 4), 0.1, np.float32)
    # mse = 0.01, max=1 -> 10*log10(1/0.01) = 20 dB
    np.testing.assert_allclose(calc_psnr(a, b, max_pixel_value=1.0), 20.0, rtol=1e-6)


def test_evaluate_chi2_truth_is_self_consistent(tiny_obs):
    # targets = A @ truth by construction -> truth reconstructs at ~0 misfit
    I = tiny_obs.movie[: tiny_obs.T_obs].reshape(tiny_obs.T_obs, tiny_obs.P).T  # (P, T_obs)
    out = evaluate_chi2(I.astype(np.float32), tiny_obs.data_dir)
    assert out["chi2_vis"] < 1e-3
    assert out["chi2_amp"] < 1e-3


def test_evaluate_chi2_matches_loss_fn(tiny_obs):
    m = NeuralDMD(r=3, key=jax.random.PRNGKey(4), num_frequencies=2)
    xy = jnp.asarray(tiny_obs.loader_grid())
    T = tiny_obs.T_obs
    times = jnp.asarray(tiny_obs.times[:T])
    fmax, fmin = 1.0, 0.0

    I_tot, _, _ = m.reconstruct(xy, times, frame_max=fmax, frame_min=fmin)  # (P, T)
    ev = evaluate_chi2(np.asarray(I_tot), tiny_obs.data_dir)

    frames = jnp.asarray(tiny_obs.movie[:T].reshape(T, tiny_obs.P))
    _, (_, c_vis, c_amp, c_cp) = loss_fn(
        m,
        xy,
        frames,
        jnp.asarray(tiny_obs.targets),
        jnp.asarray(tiny_obs.sigmas),
        jnp.asarray(tiny_obs.masks),
        jnp.asarray(tiny_obs.amp_targets),
        jnp.asarray(tiny_obs.amp_sigmas),
        jnp.asarray(tiny_obs.cp_targets),
        jnp.asarray(tiny_obs.cp_sigmas),
        jnp.asarray(tiny_obs.cp_masks),
        jnp.asarray(tiny_obs.tris),
        jnp.asarray(tiny_obs.A),
        times,
        fmax,
        fmin,
    )
    np.testing.assert_allclose(ev["chi2_vis"], float(c_vis), rtol=1e-4)
    np.testing.assert_allclose(ev["chi2_amp"], float(c_amp), rtol=1e-4)
    np.testing.assert_allclose(ev["chi2_cp"], float(c_cp), rtol=1e-4)
