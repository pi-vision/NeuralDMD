"""Unit tests for StationGains (visibility-domain per-station calibration)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuraldmd.calibration import PRODUCT_HANDS, StationGains


def _random_vis(t, m, seed=0):
    """A (t, m) complex64 visibility array with unit-scale entries."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))).astype(np.complex64)


def _set(gains, amp_raw=None, phase=None):
    """Return ``gains`` with the given parameter arrays substituted."""
    if amp_raw is not None:
        gains = eqx.tree_at(lambda g: g.amp_raw, gains, jnp.asarray(amp_raw))
    if phase is not None:
        gains = eqx.tree_at(lambda g: g.phase, gains, jnp.asarray(phase))
    return gains


def _random_gains(n_st, t, n_hands=1, use_phase=True, amp_bounds=(0.5, 2.0), seed=0):
    """A StationGains with random in-bounds amplitudes and phases."""
    g = StationGains(n_st, t, n_hands=n_hands, use_phase=use_phase, amp_bounds=amp_bounds)
    ka, kp = jax.random.split(jax.random.PRNGKey(seed))
    shape = (n_st, t, n_hands)
    g = _set(g, amp_raw=jax.random.normal(ka, shape) * 0.5)
    if use_phase:
        g = _set(g, phase=jax.random.normal(kp, shape) * 0.7)
    return g


def test_init_gains_are_identity():
    """At initialization all gains are exactly unit, so apply is a no-op."""
    t, m = 4, 5
    g = StationGains(n_stations=3, n_times=t)
    np.testing.assert_allclose(np.asarray(g.amplitudes()), 1.0, rtol=1e-6)
    ids = np.zeros((t, m, 2), np.int32)
    ids[..., 0], ids[..., 1] = 0, 1
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    np.testing.assert_allclose(out, vis, rtol=1e-6, atol=1e-6)


def test_amplitude_bounds_and_gradient():
    """Amplitudes stay strictly inside amp_bounds even for extreme raw values,
    and the gradient with respect to the raw parameter never vanishes."""
    g = StationGains(3, 2, amp_bounds=(0.9, 1.1))
    g = _set(g, amp_raw=jnp.array([[[25.0], [-25.0]], [[0.0], [0.0]], [[1.0], [-1.0]]]))
    amp = np.asarray(g.amplitudes())
    assert amp.max() <= 1.1 and amp.min() >= 0.9

    def amp_of(raw):
        gg = _set(g, amp_raw=jnp.full((3, 2, 1), raw))
        return jnp.sum(gg.amplitudes())

    for raw in (0.0, 8.0, -8.0):  # including near-saturated values
        assert float(jax.grad(amp_of)(raw)) > 0.0


def test_amp_bounds_must_bracket_one():
    """Bounds that do not bracket 1 are rejected (init could not be unit gain)."""
    with pytest.raises(ValueError):
        StationGains(2, 2, amp_bounds=(1.1, 1.5))


def test_pad_neutrality():
    """Padded baseline slots (station id -1) pass through unchanged."""
    t, m = 3, 4
    g = _random_gains(3, t, use_phase=False, seed=0)
    ids = -np.ones((t, m, 2), np.int32)  # every slot padded
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    np.testing.assert_allclose(out, vis, rtol=1e-6, atol=1e-6)


def test_gather_matches_manual():
    """apply == V_ij * g_i * conj(g_j) computed element-by-element by hand."""
    t, m = 2, 3
    g = _random_gains(4, t, seed=1)
    ids = np.array([[[0, 1], [2, 3], [1, 3]], [[0, 2], [1, 2], [0, 3]]], np.int32)
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    gg = np.asarray(g.station_gains())[:, :, 0]  # (n_st, t)
    exp = np.empty_like(vis)
    for ti in range(t):
        for mi in range(m):
            i, j = ids[ti, mi]
            exp[ti, mi] = vis[ti, mi] * gg[i, ti] * np.conj(gg[j, ti])
    np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5)


def test_closure_phase_invariance():
    """Station-based gains (amp + phase) leave the closure phase unchanged."""
    t = 2
    g = _random_gains(3, t, seed=2)
    # triangle 0-1-2 with baselines (0,1),(1,2),(2,0)
    ids = np.array([[[0, 1], [1, 2], [2, 0]]] * t, np.int32).reshape(t, 3, 2)
    vis = _random_vis(t, 3, seed=3)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    cp0 = np.angle(vis[:, 0] * vis[:, 1] * vis[:, 2])
    cp1 = np.angle(out[:, 0] * out[:, 1] * out[:, 2])
    resid = np.angle(np.exp(1j * (cp1 - cp0)))  # wrapped difference
    np.testing.assert_allclose(resid, 0.0, atol=1e-4)


def test_roundtrip_inverse_x64():
    """apply then apply(inverse=True) recovers the input to ~float64 precision."""
    jax.config.update("jax_enable_x64", True)
    try:
        t, m = 3, 5
        g = StationGains(4, t, use_phase=True, amp_bounds=(0.5, 2.0))
        ka, kp = jax.random.split(jax.random.PRNGKey(4))
        g = _set(
            g,
            amp_raw=jax.random.normal(ka, (4, t, 1), dtype=jnp.float64) * 0.5,
            phase=jax.random.normal(kp, (4, t, 1), dtype=jnp.float64) * 0.5,
        )
        ids = np.array([[[0, 1], [2, 3], [1, 3], [0, 2], [-1, -1]]] * t, np.int32).reshape(t, m, 2)
        rng = np.random.default_rng(5)
        vis = rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))
        corr = g.apply(jnp.asarray(vis), jnp.asarray(ids))
        back = g.apply(corr, jnp.asarray(ids), inverse=True)
        np.testing.assert_allclose(np.asarray(back), vis, rtol=1e-10, atol=1e-12)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_baseline_reversal_conjugation():
    """Reversing a baseline (swap stations, conjugate vis) commutes with apply:
    apply(conj(V), ids_swapped, hands_swapped) == conj(apply(V, ids, hands))."""
    t, m = 2, 4
    g = _random_gains(4, t, n_hands=2, seed=6)
    ids = np.array([[[0, 1], [2, 3], [1, 3], [0, 2]]] * t, np.int32).reshape(t, m, 2)
    vis = _random_vis(t, m, seed=7)
    for hands in ((0, 0), (0, 1)):
        fwd = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids), hands=hands))
        rev = np.asarray(
            g.apply(jnp.asarray(np.conj(vis)), jnp.asarray(ids[..., ::-1]), hands=hands[::-1])
        )
        np.testing.assert_allclose(rev, np.conj(fwd), rtol=1e-5, atol=1e-5)


def test_per_hand_product_mapping():
    """With n_hands=2 the cross-hand products mix hands: RL gets g_R,i*conj(g_L,j)."""
    t, m = 2, 3
    g = _random_gains(4, t, n_hands=2, seed=8)
    ids = np.array([[[0, 1], [2, 3], [1, 2]]] * t, np.int32).reshape(t, m, 2)
    vis = _random_vis(t, m, seed=9)
    gg = np.asarray(g.station_gains())  # (n_st, t, 2), hand 0 = R, 1 = L
    for prod, (hi_, hj_) in PRODUCT_HANDS.items():
        out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids), hands=(hi_, hj_)))
        exp = np.empty_like(vis)
        for ti in range(t):
            for mi in range(m):
                i, j = ids[ti, mi]
                exp[ti, mi] = vis[ti, mi] * gg[i, ti, hi_] * np.conj(gg[j, ti, hj_])
        np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5, err_msg=prod)
    # the two hands genuinely differ, so RR- and LL-corrupted vis differ
    rr = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids), hands=(0, 0)))
    ll = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids), hands=(1, 1)))
    assert np.abs(rr - ll).max() > 1e-3


def test_time_indices_minibatch_gather():
    """A time minibatch with explicit frame indices uses those frames' gains."""
    n_t = 5
    g = _random_gains(3, n_t, use_phase=False, seed=10)
    ids_full = np.array([[[0, 1], [1, 2]]] * n_t, np.int32).reshape(n_t, 2, 2)
    vis_full = _random_vis(n_t, 2, seed=11)
    full = np.asarray(g.apply(jnp.asarray(vis_full), jnp.asarray(ids_full)))
    # shuffled two-frame minibatch
    sel = np.array([3, 0])
    batch = np.asarray(
        g.apply(
            jnp.asarray(vis_full[sel]), jnp.asarray(ids_full[sel]), time_indices=jnp.asarray(sel)
        )
    )
    np.testing.assert_allclose(batch, full[sel], rtol=1e-6, atol=1e-6)
    # without time_indices the same call is silently wrong -- assert it differs
    naive = np.asarray(g.apply(jnp.asarray(vis_full[sel]), jnp.asarray(ids_full[sel])))
    assert np.abs(naive - full[sel]).max() > 1e-4


def test_gradient_matches_finite_differences():
    """Autodiff of a chi2-like scalar through the padded gather matches central
    finite differences for both amplitude and phase parameters (x64)."""
    jax.config.update("jax_enable_x64", True)
    try:
        t, m = 2, 3
        g = StationGains(3, t, use_phase=True, amp_bounds=(0.5, 2.0))
        ka, kp = jax.random.split(jax.random.PRNGKey(12))
        g = _set(
            g,
            amp_raw=jax.random.normal(ka, (3, t, 1), dtype=jnp.float64) * 0.4,
            phase=jax.random.normal(kp, (3, t, 1), dtype=jnp.float64) * 0.6,
        )
        ids = np.array([[[0, 1], [1, 2], [-1, -1]]] * t, np.int32).reshape(t, m, 2)
        rng = np.random.default_rng(13)
        vis = rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))
        target = rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))

        def loss(gg):
            out = gg.apply(jnp.asarray(vis), jnp.asarray(ids))
            return jnp.sum(jnp.abs(out - jnp.asarray(target)) ** 2)

        grads = eqx.filter_grad(loss)(g)
        eps = 1e-6
        for attr in ("amp_raw", "phase"):
            base = np.asarray(getattr(g, attr))
            gan = np.asarray(getattr(grads, attr))
            for idx in ((0, 0, 0), (1, 1, 0), (2, 0, 0)):
                plus, minus = base.copy(), base.copy()
                plus[idx] += eps
                minus[idx] -= eps
                fd = (
                    float(loss(_set(g, **{attr: plus}))) - float(loss(_set(g, **{attr: minus})))
                ) / (2 * eps)
                np.testing.assert_allclose(
                    gan[idx], fd, rtol=1e-6, atol=1e-8, err_msg=f"{attr}{idx}"
                )
    finally:
        jax.config.update("jax_enable_x64", False)
