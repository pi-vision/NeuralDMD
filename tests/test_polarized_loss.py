"""polarized_loss_fn: I-only parity with the scalar loss_fn, per-Stokes summation,
the optional P<=I penalty, and jittability.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.losses import loss_fn, polarized_loss_fn
from neuraldmd.model import NeuralDMD
from neuraldmd.polarized import PolarizedNeuralDMD

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)


def _batch(p=16, t=4, v=6, k=2, seed=0):
    """Fabricate a tiny, self-consistent loss batch (both legacy and polarized)."""
    rng = np.random.default_rng(seed)
    c = lambda shp: (rng.normal(size=shp) + 1j * rng.normal(size=shp)).astype(np.complex64)  # noqa: E731
    pos = lambda shp: (np.abs(rng.normal(size=shp)) + 0.1).astype(np.float32)  # noqa: E731
    tri = np.zeros((t, k, 3, 2), np.int32)
    tri[..., 0] = rng.integers(0, v, size=(t, k, 3))
    tri[..., 1] = 1
    return dict(
        xy=jnp.asarray(rng.normal(size=(p, 2))),
        A=jnp.asarray(c((t, v, p))),
        vt=jnp.asarray(c((t, v))),
        vs=jnp.asarray(pos((t, v))),
        vm=jnp.asarray((rng.random((t, v)) > 0.2).astype(np.float32)),
        ti=jnp.linspace(0.0, 1.0, t),
        fb=jnp.asarray(rng.normal(size=(t, p))),
        at=jnp.asarray(pos((t, v))),
        as_=jnp.asarray(pos((t, v))),
        ct=jnp.asarray(rng.normal(size=(t, k)).astype(np.float32)),
        cs=jnp.asarray(pos((t, k))),
        cm=jnp.asarray(np.ones((t, k), np.float32)),
        tri=jnp.asarray(tri),
    )


def test_i_only_matches_legacy_loss():
    """polarized_loss_fn on ("I",) equals loss_fn's total and chi2_vis exactly."""
    key = jax.random.PRNGKey(0)
    r = 4
    pol = PolarizedNeuralDMD(("I",), r=r, key=key, **MODEL_KW)
    ref = NeuralDMD(r=r, key=jax.random.split(key, 1)[0], **MODEL_KW)
    d = _batch()
    w = dict(neg_weight=1.0, w_sparse_weight=0.5, b_sparse_weight=0.5)

    legacy_total, legacy_aux = loss_fn(
        ref, d["xy"], d["fb"], d["vt"], d["vs"], d["vm"], d["at"], d["as_"],
        d["ct"], d["cs"], d["cm"], d["tri"], d["A"], d["ti"], 1.2, 0.0, **w,
    )
    pol_total, pol_aux = polarized_loss_fn(
        pol, d["xy"], {"I": d["vt"]}, {"I": d["vs"]}, {"I": d["vm"]},
        d["A"], d["ti"], {"I": 1.2}, {"I": 0.0}, **w,
    )
    np.testing.assert_allclose(float(pol_total), float(legacy_total), rtol=1e-6)
    np.testing.assert_allclose(float(pol_aux["chi2_vis"]["I"]), float(legacy_aux[1]), rtol=1e-6)


def test_total_sums_per_stokes_chi2_plus_penalties():
    """Total equals sum of per-Stokes chi2 plus the (non-negative) penalties."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(1), **MODEL_KW)
    d = _batch()
    tgt = {s: d["vt"] for s in ("I", "Q", "U")}
    sig = {s: d["vs"] for s in ("I", "Q", "U")}
    msk = {s: d["vm"] for s in ("I", "Q", "U")}
    fmax = {s: 1.0 for s in ("I", "Q", "U")}
    fmin = {s: 0.0 for s in ("I", "Q", "U")}
    total, aux = polarized_loss_fn(
        pol, d["xy"], tgt, sig, msk, d["A"], d["ti"], fmax, fmin,
        neg_weight=1.0, w_sparse_weight=0.0, b_sparse_weight=0.0,
    )
    assert set(aux["chi2_vis"]) == {"I", "Q", "U"}
    chi2_sum = sum(float(v) for v in aux["chi2_vis"].values())
    # with sparsity off, total = chi2_sum + neg_I (>=0); p_penalty is 0 (weight 0)
    assert float(total) >= chi2_sum - 1e-6
    assert float(aux["neg_I"]) >= 0.0
    assert float(aux["p_penalty"]) == 0.0  # disabled by default


def test_p_le_i_penalty_only_adds_load():
    """Enabling the P<=I penalty can only increase the total; it is 0 for I-only."""
    d = _batch()
    # I-only: no polarized Stokes -> penalty must be exactly 0 even if weighted
    poli = PolarizedNeuralDMD(("I",), r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    _, auxi = polarized_loss_fn(
        poli, d["xy"], {"I": d["vt"]}, {"I": d["vs"]}, {"I": d["vm"]},
        d["A"], d["ti"], {"I": 1.0}, {"I": 0.0}, p_le_i_weight=10.0,
    )
    assert float(auxi["p_penalty"]) == 0.0

    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(3), **MODEL_KW)
    args = (
        pol, d["xy"], {s: d["vt"] for s in "IQU"}, {s: d["vs"] for s in "IQU"},
        {s: d["vm"] for s in "IQU"}, d["A"], d["ti"],
        {s: 1.0 for s in "IQU"}, {s: 0.0 for s in "IQU"},
    )
    t_off, _ = polarized_loss_fn(*args, p_le_i_weight=0.0)
    t_on, aux_on = polarized_loss_fn(*args, p_le_i_weight=5.0)
    assert float(aux_on["p_penalty"]) >= 0.0
    assert float(t_on) >= float(t_off) - 1e-6


def test_polarized_loss_is_jittable():
    """The loss compiles and runs under eqx.filter_jit (needed for training)."""
    pol = PolarizedNeuralDMD(("I", "Q"), r=2, key=jax.random.PRNGKey(4), **MODEL_KW)
    d = _batch()
    jitted = eqx.filter_jit(polarized_loss_fn)
    total, aux = jitted(
        pol, d["xy"], {s: d["vt"] for s in "IQ"}, {s: d["vs"] for s in "IQ"},
        {s: d["vm"] for s in "IQ"}, d["A"], d["ti"],
        {s: 1.0 for s in "IQ"}, {s: 0.0 for s in "IQ"},
    )
    assert np.isfinite(float(total))
    assert set(aux["chi2_vis"]) == {"I", "Q"}


# --------------------------------------------------------------------------
# Circular basis (per-product chi-squared): model Stokes-vis -> RR/LL/RL/LR
# --------------------------------------------------------------------------

_PRODUCTS = ("RR", "LL", "RL", "LR")


def _model_products(pol, d, fmax, fmin, products=_PRODUCTS):
    """Independently recompute the modeled correlation-product visibilities."""
    from neuraldmd.model import physical_intensities
    from neuraldmd.physics.stokes import stokes_to_products_matrix

    vis = {}
    for s in pol.stokes:
        img, _ = physical_intensities(pol.models[s], d["xy"], d["ti"], fmax[s], fmin[s])
        vis[s] = np.einsum("tvp,pt->tv", np.asarray(d["A"]), np.asarray(img).astype(np.complex64))
    m = stokes_to_products_matrix(tuple(products), pol.stokes)
    out = {}
    for i, p in enumerate(products):
        out[p] = sum(m[i, j] * vis[s] for j, s in enumerate(pol.stokes))
    return out


def _iqu_scaling():
    """Unit per-Stokes scaling dicts for an IQU model."""
    return {s: 1.0 for s in "IQU"}, {s: 0.0 for s in "IQU"}


def test_circular_perfect_fit_is_zero():
    """With product targets equal to the modeled products, every per-product
    chi-squared is ~0 -- the circular fidelity term is correct at the truth."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(7), **MODEL_KW)
    d = _batch()
    fmax, fmin = _iqu_scaling()
    prod = _model_products(pol, d, fmax, fmin)
    tgt = {p: jnp.asarray(prod[p], dtype=jnp.complex64) for p in _PRODUCTS}
    sig = {p: jnp.ones_like(d["vs"]) for p in _PRODUCTS}
    msk = {p: jnp.ones_like(d["vm"]) for p in _PRODUCTS}
    _, aux = polarized_loss_fn(
        pol, d["xy"], tgt, sig, msk, d["A"], d["ti"], fmax, fmin,
        basis="circular", neg_weight=0.0, w_sparse_weight=0.0, b_sparse_weight=0.0,
    )
    for p in _PRODUCTS:
        np.testing.assert_allclose(float(aux["chi2_vis"][p]), 0.0, atol=1e-4)


def test_circular_product_chi2_matches_manual():
    """One product's chi-squared equals an independent hand computation."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(8), **MODEL_KW)
    d = _batch()
    fmax, fmin = _iqu_scaling()
    tgt = {p: d["vt"] * (1.0 + 0.1 * i) for i, p in enumerate(_PRODUCTS)}
    sig = {p: d["vs"] for p in _PRODUCTS}
    msk = {p: d["vm"] for p in _PRODUCTS}
    _, aux = polarized_loss_fn(
        pol, d["xy"], tgt, sig, msk, d["A"], d["ti"], fmax, fmin,
        basis="circular", neg_weight=0.0, w_sparse_weight=0.0, b_sparse_weight=0.0,
    )
    prod = _model_products(pol, d, fmax, fmin)
    p = "RL"
    m, s, t = np.asarray(msk[p]), np.asarray(sig[p]), np.asarray(tgt[p])
    manual = float(
        np.sum(np.abs(prod[p] - t) ** 2 * m / s**2) / (2.0 * np.sum(m))
    )
    np.testing.assert_allclose(float(aux["chi2_vis"][p]), manual, rtol=1e-4)


def test_circular_v_absent_makes_rr_and_ll_chi2_equal():
    """With no modeled V, RR and LL model visibilities coincide (both = vis_I),
    so their chi-squared against a shared target are equal."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(9), **MODEL_KW)
    d = _batch()
    fmax, fmin = _iqu_scaling()
    tgt = {p: d["vt"] for p in _PRODUCTS}
    sig = {p: d["vs"] for p in _PRODUCTS}
    msk = {p: d["vm"] for p in _PRODUCTS}
    _, aux = polarized_loss_fn(
        pol, d["xy"], tgt, sig, msk, d["A"], d["ti"], fmax, fmin, basis="circular",
    )
    np.testing.assert_allclose(
        float(aux["chi2_vis"]["RR"]), float(aux["chi2_vis"]["LL"]), rtol=1e-6
    )


def test_circular_basis_jittable():
    """The circular-basis loss compiles and runs under eqx.filter_jit."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=2, key=jax.random.PRNGKey(10), **MODEL_KW)
    d = _batch()
    fmax, fmin = _iqu_scaling()
    tgt = {p: d["vt"] for p in _PRODUCTS}
    sig = {p: d["vs"] for p in _PRODUCTS}
    msk = {p: d["vm"] for p in _PRODUCTS}
    jitted = eqx.filter_jit(polarized_loss_fn)
    total, aux = jitted(
        pol, d["xy"], tgt, sig, msk, d["A"], d["ti"], fmax, fmin, basis="circular"
    )
    assert np.isfinite(float(total))
    assert set(aux["chi2_vis"]) == set(_PRODUCTS)
    assert aux["basis"] == "circular"


def test_invalid_basis_raises():
    """An unknown fidelity basis fails fast."""
    pol = PolarizedNeuralDMD(("I", "Q", "U"), r=2, key=jax.random.PRNGKey(11), **MODEL_KW)
    d = _batch()
    fmax, fmin = _iqu_scaling()
    with pytest.raises(ValueError, match="basis must be"):
        polarized_loss_fn(
            pol, d["xy"], {s: d["vt"] for s in "IQU"}, {s: d["vs"] for s in "IQU"},
            {s: d["vm"] for s in "IQU"}, d["A"], d["ti"], fmax, fmin, basis="linear",
        )
