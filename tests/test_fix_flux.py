"""The total flux is supplied, not fitted.

Flux is degenerate with the global station-gain amplitude. A soft penalty does not
resolve it (measured: at flux_weight=1000 a global gain scale survived and the fitted
gains scored worse than assuming no gains at all). Fixing the flux structurally removes
the degree of freedom, so there is nothing for a gain scale to trade against.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.data.lightcurve import CO_LOCATED_SITES, measure_lightcurve
from neuraldmd.polarized import PolarizedNeuralDMD

S = ("I", "Q", "U")
KW = dict(hidden_size=16, num_layers=1, num_frequencies=2)


def _model(fix_flux=None, seed=0):
    return PolarizedNeuralDMD(S, r=2, key=jax.random.PRNGKey(seed), fix_flux=fix_flux, **KW)


def _fields(model, n_t=5, npix=8):
    g = jnp.linspace(-0.5, 0.5, npix)
    xy = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    times = jnp.linspace(0.0, 1.0, n_t)
    return model.stokes_fields(xy, times, {"I": 1.0}, {"I": 0.0})


def test_scalar_fix_flux_is_exact_every_frame():
    """sum(I) == fix_flux exactly, for every frame -- not approximately."""
    img, _ = _fields(_model(fix_flux=2.7))
    np.testing.assert_allclose(np.asarray(jnp.sum(img["I"], axis=0)), 2.7, rtol=1e-5)


def test_fix_flux_holds_regardless_of_the_networks():
    """Different random weights -> same total. The constraint is structural, so no
    choice of parameters can violate it (this is what a penalty cannot promise)."""
    for seed in range(4):
        img, _ = _fields(_model(fix_flux=1.3, seed=seed))
        np.testing.assert_allclose(np.asarray(jnp.sum(img["I"], axis=0)), 1.3, rtol=1e-5)


def test_unfixed_flux_is_free_to_wander():
    """Control: without fix_flux the total is whatever the networks say, and varies
    with the seed. This is the freedom that the gain scale exploits."""
    tot = [float(jnp.sum(_fields(_model(seed=s))[0]["I"][:, 0])) for s in range(4)]
    assert np.std(tot) > 1e-6, "totals identical across seeds -- test is not probing"


def test_fix_flux_preserves_polarization_fractions_exactly():
    """All Stokes are scaled by ONE factor, so every pol fraction is untouched. If flux
    fixing changed Q/I or U/I it would corrupt the very thing we recover."""
    free, _ = _fields(_model())
    fixed, _ = _fields(_model(fix_flux=2.7))
    for s in ("Q", "U"):
        np.testing.assert_allclose(
            np.asarray(fixed[s] / fixed["I"]), np.asarray(free[s] / free["I"]), rtol=1e-4
        )
    # EVPA follows, since it is a function of those two ratios only
    evpa_free = 0.5 * np.arctan2(
        np.asarray(free["U"] / free["I"]), np.asarray(free["Q"] / free["I"])
    )
    evpa_fix = 0.5 * np.arctan2(
        np.asarray(fixed["U"] / fixed["I"]), np.asarray(fixed["Q"] / fixed["I"])
    )
    np.testing.assert_allclose(evpa_fix, evpa_free, atol=1e-5)


def test_a_negative_total_flips_the_field_rather_than_breaking_the_constraint():
    """Documents the sign convention. An UNTRAINED field can integrate negative; the
    rescale keeps the sign, so sum(I) == flux still holds EXACTLY and the field is
    merely negated. That negation is a gauge the raw networks absorb -- but note it
    shifts EVPA computed as 0.5*atan2(U, Q) by 90 degrees, so EVPA must be taken from
    the Q/I, U/I ratios. In real fits the disk pretrain starts sum(I) > 0."""
    free, _ = _fields(_model())
    fixed, _ = _fields(_model(fix_flux=2.7))
    total_free = float(jnp.sum(free["I"][:, 0]))
    assert total_free < 0, "fixture no longer exercises the negative-total branch"
    # constraint still exact despite the negative total
    np.testing.assert_allclose(float(jnp.sum(fixed["I"][:, 0])), 2.7, rtol=1e-5)
    # and the field is negated, not blown up (the clip bug produced sum(I) ~ -1e8)
    scale = float(jnp.sum(fixed["I"][:, 0]) / jnp.sum(free["I"][:, 0]))
    assert scale < 0 and abs(scale) < 1.0


def test_lightcurve_is_interpolated_onto_batch_times():
    """A per-frame curve is sampled at the batch's normalized times, so a minibatch
    (a subset of frames) and the export path (all frames) agree on the same curve."""
    lc = [1.0, 2.0, 3.0]  # linear over t in [0, 1]
    model = _model(fix_flux=lc)
    g = jnp.linspace(-0.5, 0.5, 8)
    xy = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    # midpoint of the curve must interpolate to 2.0
    img, _ = model.stokes_fields(xy, jnp.array([0.0, 0.5, 1.0]), {"I": 1.0}, {"I": 0.0})
    np.testing.assert_allclose(np.asarray(jnp.sum(img["I"], axis=0)), [1.0, 2.0, 3.0], rtol=1e-4)
    # a batch hitting an off-grid time gets the interpolated value, not a neighbour
    img2, _ = model.stokes_fields(xy, jnp.array([0.25]), {"I": 1.0}, {"I": 0.0})
    np.testing.assert_allclose(float(jnp.sum(img2["I"])), 1.5, rtol=1e-4)


def test_fix_flux_is_differentiable_and_does_not_kill_gradients():
    """The rescale is a reparameterization, not a stop_gradient: the networks must
    still learn shape. A zero gradient here would silently freeze the image."""
    model = _model(fix_flux=2.7)
    g = jnp.linspace(-0.5, 0.5, 8)
    xy = jnp.stack(jnp.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def shape_loss(m):
        img, _ = m.stokes_fields(xy, jnp.linspace(0, 1, 3), {"I": 1.0}, {"I": 0.0})
        target = jnp.zeros_like(img["I"]).at[0].set(2.7)  # all flux in one pixel
        return jnp.sum((img["I"] - target) ** 2)

    import equinox as eqx

    grads = eqx.filter_grad(shape_loss)(model)
    leaves = [g for g in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))]
    tot = sum(float(jnp.sum(jnp.abs(x))) for x in leaves)
    assert tot > 0, "flux fixing killed the gradient to the shape"


@pytest.mark.parametrize("bad", [0.0, -1.0, [1.0, 0.0], [np.nan]])
def test_fix_flux_rejects_unphysical_values(bad):
    """A zero/negative/NaN flux would make the rescale blow up or flip the image."""
    with pytest.raises(ValueError, match="positive and finite"):
        _model(fix_flux=bad)


def test_co_located_groups_are_real_eht_sites():
    """The intra-site premise: these pairs share a mountain, so their baseline is
    metres and the source is unresolved. If this table is wrong the measured flux is
    wrong, silently."""
    flat = [s for g in CO_LOCATED_SITES for s in g]
    assert len(flat) == len(set(flat)), "a station cannot be at two sites"
    assert frozenset({"ALMA", "APEX"}) in CO_LOCATED_SITES  # Chajnantor
    assert any({"SMA", "JCMT"} <= g for g in CO_LOCATED_SITES)  # Maunakea


def test_measure_lightcurve_needs_intra_site_baselines():
    """An array with no co-located pair cannot supply its own flux scale, and must say
    so rather than return a plausible wrong number."""

    class FakeOp:
        stokes = ("I",)
        stations = ("LMT", "SPT", "PV")  # no two share a site
        targets = {"I": np.ones((2, 3), complex)}
        masks = {"I": np.ones((2, 3))}
        bl_station_ids = np.zeros((2, 3, 2), int)

    with pytest.raises(ValueError, match="no intra-site baselines"):
        measure_lightcurve(FakeOp())
