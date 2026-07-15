"""PolarizedNeuralDMD: I-only parity with the scalar model, plus shapes,
independence, config validation, and equinox pytree behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.config import StokesConfig
from neuraldmd.model import NeuralDMD
from neuraldmd.polarized import PolarizedNeuralDMD

# small nets keep the tests fast; kwargs shared by container and reference model
MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)


def _xy_times(p=20, t=5):
    """Deterministic ``(P, 2)`` coordinates and ``(T,)`` times for a forward pass."""
    xy = jnp.asarray(np.random.default_rng(0).normal(size=(p, 2)), dtype=float)
    return xy, jnp.linspace(0.0, 1.0, t)


def test_i_only_matches_legacy_neuraldmd():
    """An I-only container reconstructs identically to a standalone NeuralDMD
    built from the same split key -- the wrapper adds no Stokes-I behavior."""
    key = jax.random.PRNGKey(0)
    r = 4
    pol = PolarizedNeuralDMD(("I",), r=r, key=key, **MODEL_KW)
    ref = NeuralDMD(r=r, key=jax.random.split(key, 1)[0], **MODEL_KW)

    xy, times = _xy_times()
    got = pol.reconstruct(xy, times)["I"]
    want = ref.reconstruct(xy, times)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_reconstruct_keys_and_shapes():
    """reconstruct returns (intensities, static, dynamic) per Stokes; the static
    component is time-invariant (P, 1), the others (P, T)."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(1), **MODEL_KW)
    xy, times = _xy_times()
    out = pol.reconstruct(xy, times)
    assert set(out) == {"I", "Q", "U"}
    for intensities, static, dynamic in out.values():
        assert intensities.shape == (20, 5)
        assert static.shape == (20, 1)  # W0*b0 does not depend on t
        assert dynamic.shape == (20, 5)


def test_call_returns_per_stokes_mode_tuples():
    """__call__ returns each sub-model's (W0, W, Omega, b0, b)."""
    r = 2
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=r, key=jax.random.PRNGKey(3), **MODEL_KW)
    xy, _ = _xy_times()
    out = pol(xy)
    assert set(out) == {"I", "Q", "U"}
    W0, W, Omega, b0, b = out["Q"]
    assert W.shape == (20, r) and Omega.shape == (r,) and b.shape == (r,)


def test_stokes_have_independent_parameters():
    """Different Stokes get different split keys -> different spatial networks."""
    pol = PolarizedNeuralDMD(("I", "Q"), r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    xy, _ = _xy_times()
    wi = pol.models["I"].spatial_forward(xy[0])[1]
    wq = pol.models["Q"].spatial_forward(xy[0])[1]
    assert not np.allclose(np.asarray(wi), np.asarray(wq))


def test_container_partitions_like_a_pytree():
    """eqx.partition/combine round-trips the container (needed by the optimizer)."""
    import equinox as eqx

    pol = PolarizedNeuralDMD(("I", "Q"), r=2, key=jax.random.PRNGKey(4), **MODEL_KW)
    params, static = eqx.partition(pol, eqx.is_inexact_array)
    n_arrays = len(jax.tree_util.tree_leaves(params))
    assert n_arrays > 0  # trainable leaves exist for both Stokes
    restored = eqx.combine(params, static)
    assert restored.stokes == ("I", "Q")
    xy, times = _xy_times()
    np.testing.assert_allclose(
        np.asarray(restored.reconstruct(xy, times)["Q"][0]),
        np.asarray(pol.reconstruct(xy, times)["Q"][0]),
    )


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


def test_container_accepts_stokesconfig():
    """PolarizedNeuralDMD accepts a StokesConfig as well as a bare tuple."""
    pol = PolarizedNeuralDMD(StokesConfig(("I", "V")), r=2, key=jax.random.PRNGKey(5), **MODEL_KW)
    assert pol.stokes == ("I", "V")
