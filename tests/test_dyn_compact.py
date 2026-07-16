"""Dynamic-mode compactness: confine time-varying structure, leave the static ring alone.

Measured on the truth: variability lives entirely on the ring (0.355 inside 20-34 uas,
0.003 beyond 50 uas). The reconstruction leaks 0.379 off-source -- 126x too much -- so
Stokes I reads as a clean static ring plus low-frequency mush and the orbiting hotspot
is lost. ``compact_weight`` cannot fix this cleanly: it penalizes TOTAL I, so the ring
(r~28 uas) and the haze (r>50 uas) differ by only ~(60/28)^2 ~ 4.6x and the ring gets
squeezed too. This prior touches only the DYNAMIC modes W, never the static W0.
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
S = ("I", "Q", "U")
P = ("RR", "LL", "RL", "LR")


def _batch(p=24, t=4, v=6, seed=0):
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


def test_off_by_default():
    d = _batch()
    m = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    _, aux = _call(m, d)
    assert float(aux["dyn_compact_penalty"]) == 0.0


def test_enters_total_with_its_weight():
    """total(w) - total(0) == w * penalty -- catches an unwired or misweighted term."""
    d = _batch()
    m = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(1), **MODEL_KW)
    t0, _ = _call(m, d, dyn_compact_weight=0.0)
    w = 3.0
    tw, aux = _call(m, d, dyn_compact_weight=w)
    assert np.isclose(float(tw) - float(t0), w * float(aux["dyn_compact_penalty"]), rtol=1e-5)


def test_matches_independent_recomputation():
    """penalty == sum(|W_I| * r2_normalized) / r, recomputed from the model's own modes."""
    d = _batch()
    m = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(2), **MODEL_KW)
    _, aux = _call(m, d, dyn_compact_weight=1.0)

    _, modes = m.stokes_fields(d["xy"], d["ti"], {s: 1.0 for s in S}, {s: 0.0 for s in S})
    w_dyn = np.asarray(modes[0][1])  # intensity sub-network, dynamic spatial modes (P, r)
    xy = np.asarray(d["xy"])
    r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
    r2 = r2 / (np.mean(r2) + 1e-12)
    expected = np.sum(np.abs(w_dyn) * r2[:, None]) / w_dyn.shape[1]
    assert np.isclose(float(aux["dyn_compact_penalty"]), expected, rtol=1e-5)


def test_uses_the_dynamic_modes_not_the_static_mode():
    """The penalty must be built from W (dynamic), never W0 (static).

    W0 and W are slices of ONE shared spatial MLP's output, so there is no separate
    static head to assert a zero gradient on. Instead pin the value: recomputing with
    W0 in place of W must NOT reproduce the penalty.
    """
    d = _batch()
    m = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(3), **MODEL_KW)
    _, aux = _call(m, d, dyn_compact_weight=1.0)

    _, modes = m.stokes_fields(d["xy"], d["ti"], {s: 1.0 for s in S}, {s: 0.0 for s in S})
    w0, w_dyn = np.asarray(modes[0][0]), np.asarray(modes[0][1])
    xy = np.asarray(d["xy"])
    r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
    r2 = r2 / (np.mean(r2) + 1e-12)

    from_dyn = np.sum(np.abs(w_dyn) * r2[:, None]) / w_dyn.shape[1]
    from_static = np.sum(np.abs(w0) * r2[:, None]) / max(w0.shape[1], 1)
    got = float(aux["dyn_compact_penalty"])
    assert np.isclose(got, from_dyn, rtol=1e-5), "penalty is not the dynamic-mode moment"
    assert not np.isclose(got, from_static, rtol=1e-3), (
        "penalty matches the STATIC mode moment -- it would squeeze the ring"
    )


def test_prefers_compact_variability_over_spread_variability():
    """Given equal dynamic amplitude, structure at large radius must cost more.

    Uses the penalty formula on controlled fields: this is the invariant that makes
    the prior select an on-ring orbiting hotspot over an off-source haze.
    """
    npix = 16
    yy, xx = np.mgrid[0:npix, 0:npix]
    c = (npix - 1) / 2
    r = np.hypot(xx - c, yy - c).ravel()
    r2 = r**2 / np.mean(r**2)

    amp = 4.0

    def mode(rmin, rmax):
        m = (r >= rmin) & (r < rmax)
        w = np.zeros(len(r))
        w[m] = amp / m.sum()  # same total dynamic amplitude
        return w[:, None]

    on_ring, off_source = mode(2, 5), mode(6, 8)
    pen = lambda w: float(np.sum(np.abs(w) * r2[:, None]) / w.shape[1])  # noqa: E731
    assert pen(off_source) > 2.0 * pen(on_ring), (
        f"off-source variability must cost more: {pen(off_source):.3f} vs {pen(on_ring):.3f}"
    )
