"""PolarizedNeuralDMD (polarization = fraction of I): config, pytree behavior,
the physical guarantees (P<=I, unpolarized init, shared support), a faithful
Stokes-I channel, and end-to-end training.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.config import StokesConfig
from neuraldmd.model import NeuralDMD, physical_intensities
from neuraldmd.physics.stokes import linear_polarized_intensity
from neuraldmd.polarized import PolarizedNeuralDMD
from neuraldmd.training import make_polarized_optimizer, polarized_train_step

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)


def _xy_times(p=64, t=4):
    xy = jnp.asarray(np.random.default_rng(0).normal(size=(p, 2)))
    return xy, jnp.linspace(0.0, 1.0, t)


# --------------------------------------------------------------------------
# config + pytree
# --------------------------------------------------------------------------


def test_stokesconfig_validation():
    """StokesConfig defaults to IQU and rejects malformed selections."""
    assert StokesConfig().stokes == ("I", "Q", "U")
    assert StokesConfig(("I", "Q", "U", "V")).stokes == ("I", "Q", "U", "V")
    with pytest.raises(ValueError, match="non-empty"):
        StokesConfig(())
    with pytest.raises(ValueError, match="unknown"):
        StokesConfig(("I", "Z"))
    with pytest.raises(ValueError, match="duplicate"):
        StokesConfig(("I", "Q", "Q"))
    with pytest.raises(ValueError, match="Stokes I"):
        StokesConfig(("Q", "U"))


def test_accepts_stokesconfig():
    """The model accepts a StokesConfig as well as a bare tuple."""
    m = PolarizedNeuralDMD(StokesConfig(("I", "V")), r=2, key=jax.random.PRNGKey(5), **MODEL_KW)
    assert m.stokes == ("I", "V")


def test_partitions_like_a_pytree():
    """eqx.partition/combine round-trips the model (needed by the optimizer)."""
    m = PolarizedNeuralDMD(("I", "Q", "U"), r=2, key=jax.random.PRNGKey(4), **MODEL_KW)
    params, static = eqx.partition(m, eqx.is_inexact_array)
    assert len(jax.tree_util.tree_leaves(params)) > 0
    restored = eqx.combine(params, static)
    assert restored.stokes == ("I", "Q", "U")


def test_stokes_fields_keys_shapes_modes():
    """stokes_fields returns I/Q/U images and one mode tuple per sub-network."""
    m = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    xy, times = _xy_times()
    images, modes = m.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})
    assert set(images) == {"I", "Q", "U"}
    for arr in images.values():
        assert arr.shape == (64, 4)
    assert len(modes) == 4  # intensity, frac, cos2xi, sin2xi


def test_v_channel_present_only_when_requested():
    """The circular sub-model exists iff V is in the Stokes set."""
    key = jax.random.PRNGKey(3)
    with_v = PolarizedNeuralDMD(("I", "Q", "U", "V"), r=2, key=key, **MODEL_KW)
    without_v = PolarizedNeuralDMD(("I", "Q", "U"), r=2, key=key, **MODEL_KW)
    assert with_v.circ is not None and without_v.circ is None
    images, _ = with_v.stokes_fields(*_xy_times(), {"I": 1.0}, {"I": 0.0})
    assert "V" in images


# --------------------------------------------------------------------------
# physical guarantees
# --------------------------------------------------------------------------


def test_i_channel_is_a_faithful_neuraldmd():
    """The I field equals a standalone NeuralDMD built from the same split key --
    the polarized model does not perturb Stokes I."""
    key = jax.random.PRNGKey(0)
    r = 4
    m = PolarizedNeuralDMD(("I", "Q", "U"), r=r, key=key, **MODEL_KW)
    ref = NeuralDMD(r=r, key=jax.random.split(key, 4)[0], **MODEL_KW)
    xy, times = _xy_times()
    images, _ = m.stokes_fields(xy, times, {"I": 1.5}, {"I": 0.0})
    ref_img, _ = physical_intensities(ref, xy, times, 1.5, 0.0)
    np.testing.assert_allclose(np.asarray(images["I"]), np.asarray(ref_img), rtol=1e-6)


def test_unpolarized_at_init():
    """outshift makes m_l ~ 0 at init, so the source starts ~unpolarized."""
    m = PolarizedNeuralDMD(
        ("I", "Q", "U"), r=3, key=jax.random.PRNGKey(1), outshift=10.0, **MODEL_KW
    )
    images, _ = m.stokes_fields(*_xy_times(), {"I": 1.0}, {"I": 0.0})
    p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
    abs_i = np.abs(np.asarray(images["I"]))
    assert np.max(p / (abs_i + 1e-6)) < 1e-2


def test_p_le_scaled_i_by_construction():
    """For ANY parameters, P = m_l*|I| <= scaling_ml*|I| (the physical bound holds)."""
    cap = 0.9
    for seed in range(3):
        m = PolarizedNeuralDMD(
            ("I", "Q", "U"),
            r=3,
            key=jax.random.PRNGKey(seed),
            outshift=0.0,
            scaling_ml=cap,
            **MODEL_KW,
        )
        images, _ = m.stokes_fields(*_xy_times(), {"I": 2.0}, {"I": 0.0})
        p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
        abs_i = np.abs(np.asarray(images["I"]))
        assert np.all(p <= cap * abs_i + 1e-5), f"P > {cap}|I| at seed {seed}"


def test_pol_vanishes_where_i_vanishes():
    """Q, U -> 0 where I -> 0 (polarization only where there is total flux)."""
    m = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    images, _ = m.stokes_fields(*_xy_times(), {"I": 1.0}, {"I": 0.0})
    abs_i = np.abs(np.asarray(images["I"]))
    p = linear_polarized_intensity(np.asarray(images["Q"]), np.asarray(images["U"]))
    faint = abs_i < 0.01 * abs_i.max()
    if faint.any():
        assert np.max(p[faint]) < 0.01 * abs_i.max() + 1e-6


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def test_model_trains():
    """The model composes with the loss/optimizer and reduces the loss."""
    stokes = ("I", "Q", "U")
    m = PolarizedNeuralDMD(stokes, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
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


# --------------------------------------------------------------------------
# golden forward pass
# --------------------------------------------------------------------------

# Captured from the pre-refactor forward pass. These pin `stokes_fields` while
# the shared-Omega / shared-trunk plumbing is threaded through it: that work
# must not move a single output. Each tuple is
# (img[0, 0], img[7, 2], img[-1, -1], img.mean(), img.std()).
GOLDEN_FORWARD = {
    "fractional": {
        "I": (-1.38729119, -0.669709265, 0.236030936, -0.822372794, 0.697149158),
        "Q": (0.301011741, 5.74086989e-05, 0.0372040682, 0.0654011071, 0.0995814279),
        "U": (-0.0121491598, -0.0608569235, 0.02988014, -0.0373474583, 0.0860256255),
    },
    "direct": {
        "I": (-1.38729119, -0.669709265, 0.236030936, -0.822372794, 0.697149158),
        "Q": (1.07651234, -0.454574436, 0.940788031, -0.224100128, 1.00840151),
        "U": (-1.44087112, -0.000659632031, 0.973104835, -0.400629342, 0.970730722),
    },
    "iscaled": {
        "I": (-1.38729119, -0.669709265, 0.236030936, -0.822372794, 0.697149158),
        "Q": (-0.853832066, 0.196962118, 0.131260499, 0.147672057, 0.579255283),
        "U": (1.03288805, 0.000294507743, 0.13473089, 0.326774538, 0.54517591),
    },
    "expm": {
        "I": (-1.38729119, -0.669709265, 0.236030936, -0.822372794, 0.697149158),
        "Q": (-0.691972435, 0.196962103, 0.117701188, 0.141675875, 0.523117006),
        "U": (0.926179051, 0.000285811315, 0.121744312, 0.304236829, 0.501046538),
    },
    "expm_full": {
        "I": (1.07628942, 1.00423467, 2.52018285, 1.53175402, 1.31218517),
        "Q": (0.536846519, -0.295346349, 1.2567358, -0.0373856463, 1.37919176),
        "U": (-0.718548834, -0.000428576488, 1.29990554, -0.10353341, 0.934732914),
    },
}


@pytest.mark.parametrize("chart", sorted(GOLDEN_FORWARD))
def test_golden_forward_unchanged(chart):
    """Every chart's forward pass still matches the recorded reference."""
    m = PolarizedNeuralDMD(
        ("I", "Q", "U"), r=3, key=jax.random.PRNGKey(0), pol_param=chart, **MODEL_KW
    )
    xy, times = _xy_times()
    images, _ = m.stokes_fields(xy, times, {"I": 1.5}, {"I": 0.0})
    for stokes, expected in GOLDEN_FORWARD[chart].items():
        a = np.asarray(images[stokes])
        got = (float(a[0, 0]), float(a[7, 2]), float(a[-1, -1]), float(a.mean()), float(a.std()))
        np.testing.assert_allclose(
            got, expected, rtol=1e-6, atol=1e-7, err_msg=f"{chart}/{stokes} drifted"
        )
