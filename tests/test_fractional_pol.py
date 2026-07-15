"""FractionalPolNeuralDMD: polarization tied to I (P<=I by construction),
unpolarized initialization, and end-to-end training.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.physics.stokes import linear_polarized_intensity
from neuraldmd.polarized import FractionalPolNeuralDMD
from neuraldmd.training import make_polarized_optimizer, polarized_train_step

MODEL_KW = dict(
    hidden_size=32, num_layers=2, num_frequencies=2,
    temporal_latent_dim=16, temporal_hidden=32, temporal_layers=2,
)


def _xy_times(p=64, t=4):
    xy = jnp.asarray(np.random.default_rng(0).normal(size=(p, 2)))
    return xy, jnp.linspace(0.0, 1.0, t)


def test_stokes_fields_keys_shapes_modes():
    """stokes_fields returns I/Q/U images and one mode tuple per sub-network."""
    m = FractionalPolNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    xy, times = _xy_times()
    images, modes = m.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})
    assert set(images) == {"I", "Q", "U"}
    for arr in images.values():
        assert arr.shape == (64, 4)
    assert len(modes) == 4  # intensity, frac, cos2xi, sin2xi


def test_unpolarized_at_init():
    """outshift makes m_l ~ 0 at init, so the source starts ~unpolarized."""
    m = FractionalPolNeuralDMD(
        ("I", "Q", "U"), r=3, key=jax.random.PRNGKey(1), outshift=10.0, **MODEL_KW
    )
    xy, times = _xy_times()
    images, _ = m.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})
    p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
    abs_i = np.abs(np.asarray(images["I"]))
    assert np.max(p / (abs_i + 1e-6)) < 1e-2  # fractional pol ~0 everywhere


def test_p_le_scaled_i_by_construction():
    """For ANY parameters, P = m_l*|I| <= scaling_ml*|I| (the physical bound holds)."""
    cap = 0.9
    for seed in range(3):
        m = FractionalPolNeuralDMD(
            ("I", "Q", "U"), r=3, key=jax.random.PRNGKey(seed),
            outshift=0.0, scaling_ml=cap, **MODEL_KW,
        )
        xy, times = _xy_times()
        images, _ = m.stokes_fields(xy, times, {"I": 2.0}, {"I": 0.0})
        p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
        abs_i = np.abs(np.asarray(images["I"]))
        assert np.all(p <= cap * abs_i + 1e-5), f"P > {cap}|I| at seed {seed}"


def test_pol_vanishes_where_i_vanishes():
    """Q, U -> 0 where I -> 0 (polarization only where there is total flux)."""
    m = FractionalPolNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    xy, times = _xy_times()
    images, _ = m.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})
    abs_i = np.abs(np.asarray(images["I"]))
    p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
    faint = abs_i < 0.01 * abs_i.max()
    if faint.any():
        assert np.max(p[faint]) < 0.01 * abs_i.max() + 1e-6


def test_v_channel_present_only_when_requested():
    """The circular sub-model exists iff V is in the Stokes set."""
    key = jax.random.PRNGKey(3)
    with_v = FractionalPolNeuralDMD(("I", "Q", "U", "V"), r=2, key=key, **MODEL_KW)
    without_v = FractionalPolNeuralDMD(("I", "Q", "U"), r=2, key=key, **MODEL_KW)
    assert with_v.circ is not None and without_v.circ is None
    xy, times = _xy_times()
    images, _ = with_v.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})
    assert "V" in images


def test_fractional_model_trains():
    """The fractional model composes with the loss/optimizer and reduces the loss."""
    stokes = ("I", "Q", "U")
    m = FractionalPolNeuralDMD(stokes, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    rng = np.random.default_rng(0)
    p, t, v = 64, 4, 6
    xy = jnp.asarray(rng.normal(size=(p, 2)))
    a_np = (rng.normal(size=(t, v, p)) + 1j * rng.normal(size=(t, v, p))).astype(np.complex64)
    a = jnp.asarray(a_np)
    ti = jnp.linspace(0.0, 1.0, t)
    truth = {s: jnp.asarray(rng.normal(size=(p, t))) for s in stokes}
    targets = {s: jnp.einsum("tvp,pt->tv", a, truth[s].astype(jnp.complex64)) for s in stokes}
    sig = {s: jnp.ones((t, v)) for s in stokes}
    msk = {s: jnp.ones((t, v)) for s in stokes}
    fmax = {s: 1.0 for s in stokes}
    fmin = {s: 0.0 for s in stokes}
    nopen = dict(neg_weight=0.0, w_sparse_weight=0.0, b_sparse_weight=0.0)

    opt = make_polarized_optimizer(m, initial_lr=3e-3)
    st = opt.init(eqx.filter(m, eqx.is_array))
    first = None
    for step in range(20):
        m, st, loss, _ = polarized_train_step(
            m, st, xy, targets, sig, msk, a, ti, opt, fmax, fmin, **nopen
        )
        if step == 0:
            first = float(loss)
    assert float(loss) < first
