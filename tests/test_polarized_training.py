"""Polarized training wiring: a mini-train reduces every per-Stokes chi-squared,
the hierarchical mode freezes chosen Stokes, and the circular basis trains too.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.losses import polarized_loss_fn
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
STOKES = ("I", "Q", "U")
_NOPEN = dict(neg_weight=0.0, w_sparse_weight=0.0, b_sparse_weight=0.0)  # isolate the chi2


def _fittable_batch(stokes=STOKES, p=20, t=4, v=6, seed=0):
    """A batch whose targets are exact visibilities of a fixed truth image, so a
    perfect fit (chi2 -> 0) exists and gradient descent must reduce chi2."""
    rng = np.random.default_rng(seed)
    xy = jnp.asarray(rng.normal(size=(p, 2)))
    a = (rng.normal(size=(t, v, p)) + 1j * rng.normal(size=(t, v, p))).astype(np.complex64)
    a = jnp.asarray(a)
    ti = jnp.linspace(0.0, 1.0, t)
    truth = {s: jnp.asarray(rng.normal(size=(p, t))) for s in stokes}
    targets = {s: jnp.einsum("tvp,pt->tv", a, truth[s].astype(jnp.complex64)) for s in stokes}
    sig = {s: jnp.ones((t, v)) for s in stokes}
    msk = {s: jnp.ones((t, v)) for s in stokes}
    fmax = {s: 1.0 for s in stokes}
    fmin = {s: 0.0 for s in stokes}
    return xy, a, ti, targets, sig, msk, fmax, fmin


def _chi2s(model, xy, targets, sig, msk, a, ti, fmax, fmin, **kw):
    _, aux = polarized_loss_fn(model, xy, targets, sig, msk, a, ti, fmax, fmin, **_NOPEN, **kw)
    return {k: float(v) for k, v in aux["chi2_vis"].items()}


def test_minitrain_decreases_every_stokes_chi2():
    """30 AdamW steps reduce chi2 for I, Q, and U simultaneously."""
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    xy, a, ti, targets, sig, msk, fmax, fmin = _fittable_batch()
    opt = make_polarized_optimizer(model, initial_lr=3e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    init = _chi2s(model, xy, targets, sig, msk, a, ti, fmax, fmin)
    for _ in range(30):
        model, opt_state, _, _ = polarized_train_step(
            model, opt_state, xy, targets, sig, msk, a, ti, opt, fmax, fmin, **_NOPEN
        )
    final = _chi2s(model, xy, targets, sig, msk, a, ti, fmax, fmin)
    for s in STOKES:
        assert final[s] < init[s], f"{s}: {final[s]:.4g} !< {init[s]:.4g}"


def test_freeze_intensity_holds_i_but_trains_pol():
    """freeze_intensity=True holds the intensity field bit-identical while a
    polarization field (frac) still moves."""
    # outshift=0 so m_l ~ 0.5 and the pol gradient is healthy in a few steps
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(1), outshift=0.0, **MODEL_KW)
    xy, a, ti, targets, sig, msk, fmax, fmin = _fittable_batch(seed=1)
    opt = make_polarized_optimizer(model, initial_lr=1e-2)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    before = eqx.filter(model, eqx.is_array)
    for _ in range(5):
        model, opt_state, _, _ = polarized_train_step(
            model,
            opt_state,
            xy,
            targets,
            sig,
            msk,
            a,
            ti,
            opt,
            fmax,
            fmin,
            freeze_intensity=True,
            **_NOPEN,
        )
    after = eqx.filter(model, eqx.is_array)

    def leaves(m, attr):
        return jax.tree_util.tree_leaves(getattr(m, attr))

    i_unchanged = all(
        np.allclose(np.asarray(x), np.asarray(y))
        for x, y in zip(leaves(before, "intensity"), leaves(after, "intensity"), strict=True)
    )
    frac_changed = any(
        not np.allclose(np.asarray(x), np.asarray(y))
        for x, y in zip(leaves(before, "frac"), leaves(after, "frac"), strict=True)
    )
    assert i_unchanged, "frozen intensity field changed"
    assert frac_changed, "trainable pol field did not change"


def test_pretrain_stokes_i_touches_only_intensity():
    """pretrain_stokes_i aligns the intensity field (loss decreases) and leaves the
    polarization fields bit-identical."""
    from neuraldmd.pretraining import pretrain_stokes_i

    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    truth_i = np.abs(np.random.default_rng(0).normal(size=(4, 16, 16)))

    def leaves(m, attr):
        return jax.tree_util.tree_leaves(getattr(m, attr))

    pre, losses = pretrain_stokes_i(model, truth_i, num_steps=25, lr=1e-3)
    assert losses[-1] < losses[0]
    for attr in ("frac", "cos2xi", "sin2xi"):
        assert all(
            np.allclose(np.asarray(a), np.asarray(b))
            for a, b in zip(leaves(model, attr), leaves(pre, attr), strict=True)
        )
    assert any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(leaves(model, "intensity"), leaves(pre, "intensity"), strict=True)
    )


def test_optimizer_variants_build_and_step():
    """adam/adamax and an exponential-decay schedule all build and take a step."""
    from neuraldmd.training import make_polarized_optimizer

    model = PolarizedNeuralDMD(("I", "Q"), r=2, key=jax.random.PRNGKey(5), **MODEL_KW)
    xy, a, ti, targets, sig, msk, fmax, fmin = _fittable_batch(stokes=("I", "Q"), seed=5)
    for name in ("adam", "adamax"):
        opt = make_polarized_optimizer(model, optimizer=name, lr_decay_rate=0.5, lr_decay_steps=10)
        st = opt.init(eqx.filter(model, eqx.is_inexact_array))
        _, _, loss, _ = polarized_train_step(
            model, st, xy, targets, sig, msk, a, ti, opt, fmax, fmin, **_NOPEN
        )
        assert np.isfinite(float(loss))


def test_circular_basis_training_reduces_loss():
    """A few steps in the circular (per-product) basis reduce the total loss."""
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    xy, a, ti, _, _, _, fmax, fmin = _fittable_batch(seed=2)
    # product-basis targets from a fixed truth so a good fit exists
    rng = np.random.default_rng(9)
    prods = ("RR", "LL", "RL", "LR")

    def _rand_vis():
        v = rng.normal(size=(4, 6)) + 1j * rng.normal(size=(4, 6))
        return jnp.asarray(v.astype(np.complex64))

    tgt = {p: _rand_vis() for p in prods}
    sig = {p: jnp.ones((4, 6)) for p in prods}
    msk = {p: jnp.ones((4, 6)) for p in prods}
    opt = make_polarized_optimizer(model, initial_lr=3e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    first = None
    for step in range(20):
        model, opt_state, loss, _ = polarized_train_step(
            model, opt_state, xy, tgt, sig, msk, a, ti, opt, fmax, fmin, basis="circular", **_NOPEN
        )
        if step == 0:
            first = float(loss)
    assert float(loss) < first


def _max_leaf_change(a, b):
    """Largest absolute change across all array leaves of two model subtrees."""
    la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array))
    return max(float(jnp.abs(x - y).max()) for x, y in zip(la, lb, strict=True))


def test_r_pol_starves_polarization_capacity():
    """``r_pol`` gives the polarization fields fewer DMD modes than Stokes I."""
    model = PolarizedNeuralDMD(STOKES, r=8, r_pol=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    assert model.intensity.r == 8
    assert model.frac.r == model.cos2xi.r == model.sin2xi.r == 3


def test_pol_scale_zero_freezes_polarization():
    """``pol_scale=0`` holds the polarization fields fixed while Stokes I updates."""
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    xy, a, ti, tgt, sig, msk, fmax, fmin = _fittable_batch(seed=1)
    opt = make_polarized_optimizer(model, initial_lr=3e-3)
    st = opt.init(eqx.filter(model, eqx.is_array))
    new, *_ = polarized_train_step(
        model, st, xy, tgt, sig, msk, a, ti, opt, fmax, fmin, pol_scale=jnp.asarray(0.0), **_NOPEN
    )
    assert _max_leaf_change(model.frac, new.frac) == 0.0
    assert _max_leaf_change(model.cos2xi, new.cos2xi) == 0.0
    assert _max_leaf_change(model.intensity, new.intensity) > 0.0


def test_freeze_intensity_freezes_i_not_pol():
    """``freeze_intensity`` holds Stokes I fixed while the polarization updates."""
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    xy, a, ti, tgt, sig, msk, fmax, fmin = _fittable_batch(seed=1)
    opt = make_polarized_optimizer(model, initial_lr=3e-3)
    st = opt.init(eqx.filter(model, eqx.is_array))
    new, *_ = polarized_train_step(
        model, st, xy, tgt, sig, msk, a, ti, opt, fmax, fmin, freeze_intensity=True, **_NOPEN
    )
    assert _max_leaf_change(model.intensity, new.intensity) == 0.0
    assert _max_leaf_change(model.frac, new.frac) > 0.0
