"""Characterize the NeuralDMD model: forward shapes, spectral ranges, the
zero-initialized amplitude head, gauge fixing, and reconstruct == manual sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _impl import NeuralDMD


def _model(r=3, key=0, **kw):
    return NeuralDMD(r=r, key=jax.random.PRNGKey(key), num_frequencies=2, **kw)


def test_forward_shapes_and_dtypes():
    m = _model(r=3)
    P = 20
    xy = jnp.asarray(np.random.default_rng(1).normal(size=(P, 2)), dtype=jnp.float32)
    W0, W, Omega, b0, b = m(xy)
    assert W0.shape == (P, 1)
    assert W.shape == (P, 3) and jnp.iscomplexobj(W)
    assert Omega.shape == (3,) and jnp.iscomplexobj(Omega)
    assert b0.shape == (1,)
    assert b.shape == (3,) and jnp.iscomplexobj(b)


def test_spectrum_ranges():
    # alpha = -sigmoid(raw) in (-1, 0); theta in (theta_min, theta_max)
    m = _model(r=8, theta_min=0.0, theta_max=1.0)
    _, _, Omega, _, _ = m(jnp.zeros((4, 2)))
    alpha, theta = jnp.real(Omega), jnp.imag(Omega)
    assert bool(jnp.all(alpha < 0)) and bool(jnp.all(alpha > -1))
    assert bool(jnp.all(theta > 0)) and bool(jnp.all(theta < 1))


def test_zero_init_amplitude_head():
    # b-head is zero-initialized -> outputs are analytically fixed at init:
    #   b0   = softplus(0)                       = ln 2               ~= 0.6931
    #   b_k  = softplus(0)*init_mag*exp(i*pi*tanh(0)) = ln2 * 0.1     ~= 0.0693
    m = _model(r=5)
    _, _, _, b0, b = m(jnp.zeros((4, 2)))
    np.testing.assert_allclose(np.asarray(b0), [np.log(2.0)], rtol=1e-5)
    b_init = np.log(2.0) * 0.1
    np.testing.assert_allclose(np.asarray(b), np.full(5, b_init, np.complex64), atol=1e-6)


def test_gauge_fix_unit_rms():
    # each dynamic mode is normalized to unit RMS over the pixel batch
    m = _model(r=4)
    xy = jnp.asarray(np.random.default_rng(2).normal(size=(64, 2)), dtype=jnp.float32)
    _, W, _, _, _ = m(xy)
    rms = np.sqrt(np.mean(np.abs(np.asarray(W)) ** 2, axis=0))
    np.testing.assert_allclose(rms, np.ones(4), rtol=1e-4)


def test_reconstruct_matches_manual_einsum():
    m = _model(r=3)
    xy = jnp.asarray(np.random.default_rng(3).normal(size=(30, 2)), dtype=jnp.float32)
    times = jnp.asarray(np.linspace(0, 1, 5), dtype=jnp.float32)
    fmax, fmin = 2.0, 0.5

    I_tot, I_stat, I_dyn = m.reconstruct(xy, times, frame_max=fmax, frame_min=fmin)

    W0, W, Omega, b0, b = m(xy)
    lam = np.exp(np.asarray(Omega)[:, None] * np.asarray(times)[None, :] * m.t_scale)
    stat = np.asarray(W0)[:, 0:1] * np.asarray(b0)[0]
    dyn = 2 * np.real(np.einsum("pr,rt,r->pt", np.asarray(W), lam, np.asarray(b)))
    scale = fmax - fmin
    np.testing.assert_allclose(np.asarray(I_stat), stat * scale + fmin, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(I_dyn), dyn * scale, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(I_tot), (stat + dyn) * scale + fmin, rtol=1e-4, atol=1e-5)


def test_output_size_and_static_fields():
    m = _model(r=7)
    assert m.output_size == 2 * 7 + 1
    assert m.r == 7
    assert m.t_scale == 200.0
