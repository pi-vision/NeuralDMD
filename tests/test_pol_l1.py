"""L1 (total polarized flux) prior: exact value, wiring, and the degeneracy it breaks.

The on-ring m=2 EVPA swirl and the off-ring polarized haze are near-degenerate in
chi2 at the sampled (u,v); they differ in polarized *flux*. These tests pin the
penalty to an independent recomputation, check it is position-independent (the
property that distinguishes it from ``compact_pol_weight``, a radial moment), and
guard the ``sqrt(0)`` NaN that the P<=I penalty hit when pol is tied to I.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
STOKES = ("I", "Q", "U")


def _batch(p=16, t=4, v=6, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        xy=jnp.asarray(rng.normal(size=(p, 2))),
        A=jnp.asarray((rng.normal(size=(t, v, p)) + 1j * rng.normal(size=(t, v, p))).astype("c8")),
        vt=jnp.asarray((rng.normal(size=(t, v)) + 1j * rng.normal(size=(t, v))).astype("c8")),
        vs=jnp.asarray((np.abs(rng.normal(size=(t, v))) + 0.1).astype("f4")),
        vm=jnp.asarray((rng.random((t, v)) > 0.2).astype("f4")),
        ti=jnp.linspace(0.0, 1.0, t),
    )


def _call(model, d, **kw):
    ss = model.stokes
    return polarized_loss_fn(
        model,
        d["xy"],
        {s: d["vt"] for s in ss},
        {s: d["vs"] for s in ss},
        {s: d["vm"] for s in ss},
        d["A"],
        d["ti"],
        {s: 1.0 for s in ss},
        {s: 0.0 for s in ss},
        neg_weight=0.0,
        w_sparse_weight=0.0,
        b_sparse_weight=0.0,
        **kw,
    )


def test_pol_l1_matches_independent_recomputation():
    """aux['pol_l1_penalty'] == mean over frames of sum over pixels of sqrt(Q^2+U^2).

    Recomputed from the model's own Stokes fields, so an axis slip (summing frames
    / averaging pixels) or a missing Stokes would fail.
    """
    d = _batch()
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    _, aux = _call(model, d, pol_l1_weight=1.0)

    images, _ = model.stokes_fields(
        d["xy"], d["ti"], {s: 1.0 for s in STOKES}, {s: 0.0 for s in STOKES}
    )
    q, u = np.asarray(images["Q"]), np.asarray(images["U"])  # (P_pix, T)
    expected = np.mean(np.sum(np.sqrt(q**2 + u**2 + 1e-12), axis=0))

    assert float(aux["pol_l1_penalty"]) == np.float32(expected)


def test_pol_l1_off_by_default_and_zero_for_i_only():
    """Default weight leaves it disabled; an I-only model has no pol flux at all."""
    d = _batch()
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(1), **MODEL_KW)
    _, aux = _call(model, d)
    assert float(aux["pol_l1_penalty"]) == 0.0

    ionly = PolarizedNeuralDMD(("I",), r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    _, aux_i = _call(ionly, d, pol_l1_weight=10.0)
    assert float(aux_i["pol_l1_penalty"]) == 0.0


def test_pol_l1_enters_total_with_its_weight():
    """total(w) - total(0) == w * penalty -- catches an unwired or misweighted term."""
    d = _batch()
    model = PolarizedNeuralDMD(STOKES, r=3, key=jax.random.PRNGKey(3), **MODEL_KW)
    t0, _ = _call(model, d, pol_l1_weight=0.0)
    w = 2.5
    tw, aux = _call(model, d, pol_l1_weight=w)
    # float32 accumulation over the whole loss -> compare to single precision
    assert np.isclose(float(tw) - float(t0), w * float(aux["pol_l1_penalty"]), rtol=1e-5)


def test_pol_l1_is_position_independent_unlike_compact_pol():
    """Equal polarized flux at large vs small radius costs the SAME L1.

    This is the property that separates it from ``compact_pol_weight`` (a radial
    second moment), which was measured to weight off-ring pol only ~2x -- too weak
    against the haze's chi2 reward. Uses the penalty formula on controlled fields.
    """
    npix, flux = 32, 4.0
    yy, xx = np.mgrid[0:npix, 0:npix]
    r = np.hypot(xx - (npix - 1) / 2, yy - (npix - 1) / 2)

    def field(rmin, rmax):
        """Annulus carrying exactly `flux` of polarized intensity (Q only)."""
        m = (r >= rmin) & (r < rmax)
        f = np.zeros((npix, npix))
        f[m] = flux / m.sum()
        return f

    inner, outer = field(3, 6), field(11, 14)  # same total flux, different radii
    l1 = lambda f: np.sum(np.sqrt(f**2 + 1e-12))  # noqa: E731
    # rtol 1e-3, not tighter: the eps guard floors every empty pixel at sqrt(1e-12)
    # = 1e-6, so annuli covering different pixel counts differ by ~1e-3 over a
    # 32x32 grid. That floor is the only position-dependence L1 has -- and it is
    # ~3 orders below the radial moment's, which is the point of the comparison.
    assert np.isclose(l1(inner), l1(outer), rtol=1e-3)  # L1: identical
    assert np.isclose(l1(inner), flux, rtol=1e-3)  # and equals the flux

    r2 = (r**2 / np.mean(r**2))[..., None]
    moment = lambda f: np.sum(f[..., None] * r2)  # noqa: E731
    assert moment(outer) > 3.0 * moment(inner)  # radial moment: strongly differs


def test_pol_l1_gradient_finite_where_pol_vanishes():
    """d/dQ sqrt(Q^2+U^2+eps) is finite at Q=U=0.

    Unguarded, sqrt(0) has infinite derivative; with pol tied to I (``iscaled``)
    Q and U vanish exactly where I crosses zero, which produced a training NaN in
    the P<=I penalty. The same eps guards this prior.
    """
    g = jax.grad(lambda qu: jnp.sum(jnp.sqrt(qu[0] ** 2 + qu[1] ** 2 + 1e-12)))
    at_zero = np.asarray(g(jnp.zeros((2, 5))))
    assert np.all(np.isfinite(at_zero))

    unguarded = jax.grad(lambda qu: jnp.sum(jnp.sqrt(qu[0] ** 2 + qu[1] ** 2)))
    assert not np.all(np.isfinite(np.asarray(unguarded(jnp.zeros((2, 5))))))
