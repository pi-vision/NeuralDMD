"""The optional shared-state arguments on ``NeuralDMD.__call__``.

A polarized container can drive a field from state it shares with its siblings:
a spectrum (``omega``), a spatial trunk (``spatial_features``), or a per-mode
on/off mask (``b_mask``). Each defaults to ``None``, and these tests pin that the
defaults reproduce the standalone forward pass exactly.

Package-only: these methods do not exist in the monolith, so this module imports
``neuraldmd`` directly rather than going through the dual-impl harness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.model import NeuralDMD, physical_intensities


def _model(r=3, key=0):
    """A small NeuralDMD for the equivalence checks."""
    return NeuralDMD(
        r=r,
        hidden_size=32,
        num_layers=2,
        num_frequencies=2,
        temporal_latent_dim=16,
        temporal_hidden=32,
        temporal_layers=2,
        key=jax.random.PRNGKey(key),
    )


def _xy(p=32):
    """Fixed pixel coordinates."""
    return jnp.asarray(np.random.default_rng(0).normal(size=(p, 2)))


def test_passing_own_omega_is_a_no_op():
    """Supplying the field's own spectrum reproduces the default forward exactly."""
    m, xy = _model(), _xy()
    alphas, thetas = m.temporal_omega()
    ref = m(xy)
    got = m(xy, omega=alphas + 1j * thetas)
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_passing_own_features_is_a_no_op():
    """Supplying the field's own trunk activations reproduces the default forward."""
    m, xy = _model(), _xy()
    feats = jax.vmap(m.spatial_features)(xy)
    ref = m(xy)
    got = m(xy, spatial_features=feats)
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)


def test_spatial_forward_matches_features_then_head():
    """spatial_forward is exactly the trunk followed by the mode head."""
    m, xy = _model(), _xy()
    direct = jax.vmap(m.spatial_forward)(xy)
    split = jax.vmap(m._spatial_from_features)(jax.vmap(m.spatial_features)(xy))
    for a, b in zip(direct, split, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_unit_mask_is_a_no_op():
    """An all-ones mask leaves the amplitudes untouched."""
    m, xy = _model(), _xy()
    ref = m(xy)[4]
    got = m(xy, b_mask=jnp.ones(m.r))[4]
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(got))


def test_masked_mode_is_removed_and_others_are_untouched():
    """A zeroed mask entry zeroes that amplitude and leaves the rest alone."""
    m, xy = _model(), _xy()
    ref = np.asarray(m(xy)[4])
    got = np.asarray(m(xy, b_mask=jnp.array([1.0, 0.0, 1.0]))[4])
    assert got[1] == 0
    np.testing.assert_array_equal(got[[0, 2]], ref[[0, 2]])


def test_masked_mode_receives_no_gradient():
    """A locked mode's amplitude parameters get exactly zero gradient."""
    m, xy = _model(), _xy()
    times = jnp.linspace(0.0, 1.0, 4)
    mask = jnp.array([1.0, 0.0, 1.0])

    def loss(model):
        img, _ = physical_intensities(model, xy, times, 1.0, 0.0, b_mask=mask)
        return jnp.sum(img**2)

    import equinox as eqx

    grad = eqx.filter_grad(loss)(m)
    # b comes from rows 1 + 2k and 2 + 2k of the amplitude head (k = mode index)
    rows = np.asarray(grad.temporal_b.head.weight)[[1 + 2 * 1, 2 + 2 * 1]]
    assert np.all(rows == 0)
    live = np.asarray(grad.temporal_b.head.weight)[[1 + 2 * 0, 2 + 2 * 0]]
    assert np.any(live != 0)


def test_zero_mask_gives_the_static_reconstruction():
    """With every dynamic mode masked off, only the static term survives."""
    m, xy = _model(), _xy()
    times = jnp.linspace(0.0, 1.0, 4)
    img, (W0, _, b0, _) = physical_intensities(m, xy, times, 1.0, 0.0, b_mask=jnp.zeros(m.r))
    static = np.asarray(W0[:, 0:1] * b0[0])
    np.testing.assert_allclose(np.asarray(img), np.broadcast_to(static, img.shape), atol=1e-6)
