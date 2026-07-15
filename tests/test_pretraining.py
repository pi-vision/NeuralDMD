"""Characterize pretraining pieces: radius of gyration, the scale-invariant
column residual, the Zernike alignment loss, and a short alignment run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _impl import (
    NeuralDMD,
    _best_complex_scale_residual,
    build_zernike_targets,
    pretrain_model,
    radius_of_gyration,
    zernike_alignment_loss,
)


def test_radius_of_gyration_gaussian():
    # a 2D isotropic Gaussian exp(-r^2/2 s^2) has <r^2> = 2 s^2 -> Rg = sqrt(2) s
    H = W = 128
    fov = np.pi
    s = 0.15
    xs = (np.arange(W) - W / 2) * (fov / W)
    ys = (np.arange(H) - H / 2) * (fov / H)
    X, Y = np.meshgrid(xs, ys)
    frame = np.exp(-(X**2 + Y**2) / (2 * s**2))
    video = frame[None].astype(np.float32)  # (1, H, W)

    rg, drg = radius_of_gyration(video, fov, fov)
    np.testing.assert_allclose(rg, np.sqrt(2.0) * s, rtol=2e-2)
    assert drg >= 0.0


def test_best_complex_scale_residual():
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=8) + 1j * rng.normal(size=8))
    # y proportional to x -> residual ~ 0
    y = (2.0 - 1.5j) * x
    assert float(_best_complex_scale_residual(x, y)) < 1e-8
    # y orthogonal to x -> optimal scale is 0 -> residual = ||y||^2
    y2 = jnp.asarray(rng.normal(size=8) + 1j * rng.normal(size=8))
    y2 = y2 - (jnp.vdot(x, y2) / jnp.vdot(x, x)) * x  # project out x
    np.testing.assert_allclose(
        float(_best_complex_scale_residual(x, y2)), float(jnp.sum(jnp.abs(y2) ** 2)), rtol=1e-5
    )


def test_alignment_loss_scale_invariant():
    _, _, _, xy = build_zernike_targets(16, 16, 1.0, np.pi, np.pi, r=5, max_n=6)
    Z, _, _, _ = build_zernike_targets(16, 16, 1.0, np.pi, np.pi, r=6, max_n=6)  # r+1 cols
    assert float(zernike_alignment_loss(Z, Z)) < 1e-6
    assert float(zernike_alignment_loss((3.0 + 2.0j) * Z, Z)) < 1e-6  # per-column scale free
    shuffled = jnp.asarray(np.asarray(Z)[:, ::-1].copy())
    assert float(zernike_alignment_loss(shuffled, Z)) > 1e-3


@pytest.mark.slow
def test_pretrain_reduces_alignment_loss():
    r = 5
    Z, _, _, xy = build_zernike_targets(16, 16, 1.0, np.pi, np.pi, r=r + 1, max_n=6)
    model = NeuralDMD(r=r, key=jax.random.PRNGKey(0), num_frequencies=2)
    _, losses = pretrain_model(
        model, xy, Z, num_steps=60, lr=1e-3, key=jax.random.PRNGKey(1), print_every=1000
    )
    assert losses[-1] < losses[0]
