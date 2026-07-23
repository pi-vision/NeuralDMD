"""Sharing a temporal bank (and optionally a spatial trunk) across Stokes.

The physics question these settings exist to answer is whether polarization
oscillates at the same frequencies as total intensity. ``couple='pol'`` ties the
polarization fields to each other, ``couple='all'`` adds Stokes I, and
``n_shared < r`` leaves each field private modes so the data can disagree.

Package-only: imports ``neuraldmd`` directly rather than the dual-impl harness.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.losses import polarized_loss_fn
from neuraldmd.polarized import PolarizedNeuralDMD

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)
CHARTS = ("fractional", "direct", "iscaled", "expm", "expm_full")
R = 3


def _model(couple="none", n_shared=None, share_trunk=False, chart="fractional", key=0, **kw):
    """A small polarized model with the requested coupling."""
    return PolarizedNeuralDMD(
        ("I", "Q", "U"),
        r=R,
        key=jax.random.PRNGKey(key),
        pol_param=chart,
        couple=couple,
        n_shared=n_shared,
        share_trunk=share_trunk,
        **{**MODEL_KW, **kw},
    )


def _xy_times(p=32, t=4):
    """Fixed coordinates and normalized times."""
    xy = jnp.asarray(np.random.default_rng(0).normal(size=(p, 2)))
    return xy, jnp.linspace(0.0, 1.0, t)


def _fields(m):
    """Per-Stokes image cubes for a model."""
    xy, times = _xy_times()
    return m.stokes_fields(xy, times, {"I": 1.5}, {"I": 0.0})[0]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_rejects_unknown_couple():
    """An unrecognized coupling mode is refused."""
    with pytest.raises(ValueError, match="couple must be"):
        _model(couple="both")


def test_rejects_share_trunk_without_a_bank_owner():
    """Trunk sharing needs a coupled field to own the trunk."""
    with pytest.raises(ValueError, match="share_trunk"):
        _model(couple="none", share_trunk=True)


def test_rejects_out_of_range_n_shared():
    """n_shared must lie within the mode count."""
    with pytest.raises(ValueError, match="n_shared"):
        _model(couple="all", n_shared=R + 1)


def test_r_pol_is_forced_and_warned_when_coupled():
    """A shared bank has one rank, so a differing r_pol is overridden."""
    with pytest.warns(UserWarning, match="r_pol"):
        m = PolarizedNeuralDMD(
            ("I", "Q", "U"),
            r=R,
            key=jax.random.PRNGKey(0),
            couple="all",
            r_pol=1,
            **MODEL_KW,
        )
    assert m.frac.r == R


def test_init_is_identical_across_couple_settings():
    """Coupling changes the forward pass, never the initialization."""
    a = _model(couple="none")
    b = _model(couple="all")
    c = _model(couple="pol", share_trunk=True)
    for x, y in zip(
        jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(b, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(
        jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(c, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# --------------------------------------------------------------------------
# the partition reduces to the uncoupled model at its endpoints
# --------------------------------------------------------------------------


@pytest.mark.parametrize("couple", ["pol", "all"])
def test_n_shared_zero_is_the_uncoupled_model(couple):
    """With no shared modes, every chart output matches couple='none' exactly."""
    ref = _fields(_model(couple="none"))
    got = _fields(_model(couple=couple, n_shared=0))
    for s in ref:
        np.testing.assert_array_equal(np.asarray(ref[s]), np.asarray(got[s]))


def test_intensity_is_untouched_by_pol_coupling():
    """couple='pol' leaves Stokes I exactly as the uncoupled model produces it."""
    ref = _fields(_model(couple="none"))
    got = _fields(_model(couple="pol"))
    np.testing.assert_array_equal(np.asarray(ref["I"]), np.asarray(got["I"]))


def test_intensity_is_untouched_when_it_owns_the_bank():
    """couple='all' hands Stokes I its own spectrum, so its image cannot move."""
    ref = _fields(_model(couple="none"))
    got = _fields(_model(couple="all"))
    np.testing.assert_allclose(np.asarray(ref["I"]), np.asarray(got["I"]), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("chart", CHARTS)
def test_every_chart_composes_with_coupling(chart):
    """Coupling changes where Omega comes from, not the per-chart algebra."""
    imgs = _fields(_model(couple="all", chart=chart))
    for s in ("I", "Q", "U"):
        assert imgs[s].shape == (32, 4)
        assert np.all(np.isfinite(np.asarray(imgs[s])))


# --------------------------------------------------------------------------
# the coupling is real: gradients actually flow into the shared bank
# --------------------------------------------------------------------------


def _crosshand_loss(m):
    """Chi-squared on RL/LR only -- a purely polarization-driven objective."""
    xy, times = _xy_times()
    rng = np.random.default_rng(1)
    t, v, p = 4, 5, 32
    a = jnp.asarray(rng.normal(size=(t, v, p)) + 1j * rng.normal(size=(t, v, p)))
    keys = ("RL", "LR")
    tgt = {k: jnp.asarray(rng.normal(size=(t, v)) + 0j) for k in keys}
    sig = {k: jnp.ones((t, v)) for k in keys}
    msk = {k: jnp.ones((t, v)) for k in keys}
    loss, _ = polarized_loss_fn(
        m,
        xy,
        tgt,
        sig,
        msk,
        a,
        times,
        {"I": 1.5},
        {"I": 0.0},
        basis="circular",
        products=keys,
        neg_weight=0.0,
        w_sparse_weight=0.0,
        b_sparse_weight=0.0,
    )
    return loss


def _omega_grad(m, field):
    """Gradient of the cross-hand loss w.r.t. one field's spectrum latent."""
    grad = eqx.filter_grad(_crosshand_loss)(m)
    return np.asarray(getattr(grad, field).temporal_omega.latent)


def test_polarization_data_reaches_the_shared_bank():
    """Under couple='all', cross-hand data shapes the Stokes-I spectrum.

    Uses ``direct``, the one chart whose Q and U do not multiply the I field, so
    a gradient from RL/LR into I's spectrum can only have come through the bank.
    """
    uncoupled = _omega_grad(_model(couple="none", chart="direct"), "intensity")
    coupled = _omega_grad(_model(couple="all", chart="direct"), "intensity")
    assert np.all(uncoupled == 0), "sanity: RL/LR cannot reach I's spectrum in `direct`"
    assert np.any(coupled != 0), "coupling did not connect pol data to the shared bank"


@pytest.mark.parametrize("chart", ["fractional", "iscaled", "expm", "expm_full"])
def test_i_tying_charts_already_link_pol_data_to_the_i_spectrum(chart):
    """Charts that build Q,U as multiples of I couple the two without a shared bank.

    Their cross-hand chi-squared depends on the I field, hence on its spectrum,
    which is why the coupling check above has to use ``direct``.
    """
    grad = _omega_grad(_model(couple="none", chart=chart), "intensity")
    assert np.any(grad != 0)


def test_borrowed_spectrum_receives_no_gradient():
    """A field that borrows the whole bank stops training its own spectrum."""
    grad = _omega_grad(_model(couple="all", n_shared=R), "frac")
    assert np.all(grad == 0)


def test_private_modes_keep_their_own_spectrum_trainable():
    """With n_shared < r, a field's remaining private modes still learn."""
    grad = _omega_grad(_model(couple="all", n_shared=R - 1), "frac")
    assert np.any(grad != 0)


def test_shared_trunk_is_driven_by_polarization_and_pol_trunks_go_quiet():
    """Trunk sharing routes pol gradients into the owner's trunk, not the borrower's."""
    m = _model(couple="all", share_trunk=True)
    grad = eqx.filter_grad(_crosshand_loss)(m)
    owner = np.asarray(grad.intensity.mlp.in_proj.weight)
    borrower_trunk = np.asarray(grad.frac.mlp.in_proj.weight)
    borrower_head = np.asarray(grad.frac.mlp.out_head.weight)
    assert np.any(owner != 0), "shared trunk received no polarization gradient"
    assert np.all(borrower_trunk == 0), "borrowed trunk should be inert"
    assert np.any(borrower_head != 0), "per-field head must stay trainable"


# --------------------------------------------------------------------------
# aligned_modes
# --------------------------------------------------------------------------


def test_aligned_modes_requires_a_shared_bank():
    """Mode indices are not comparable without a shared spectrum."""
    with pytest.raises(ValueError, match="shared bank"):
        _model(couple="none").aligned_modes(*_xy_times())


def test_aligned_modes_reports_the_bank_and_every_coupled_field():
    """The returned spectrum is the owner's, and each coupled field is present."""
    m = _model(couple="all")
    xy, times = _xy_times()
    out = m.aligned_modes(xy, times)
    alphas, thetas = m.intensity.temporal_omega()
    np.testing.assert_array_equal(np.asarray(out["Omega"]), np.asarray(alphas + 1j * thetas))
    assert out["n_shared"] == R
    assert set(out["fields"]) == {"intensity", "frac", "cos2xi", "sin2xi"}
    for modes in out["fields"].values():
        w0, w, b0, b = modes
        assert w.shape == (32, R)
        assert b.shape == (R,)


def test_aligned_modes_excludes_intensity_under_pol_coupling():
    """couple='pol' shares a bank among the polarization fields only."""
    out = _model(couple="pol").aligned_modes(*_xy_times())
    assert "intensity" not in out["fields"]
