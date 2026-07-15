"""Unit tests for StationGains (visibility-domain per-station calibration)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.calibration import StationGains


def _random_vis(t, m, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))).astype(np.complex64)


def test_init_gains_are_identity():
    """At initialisation all gains are unit, so apply is a no-op."""
    t, m = 4, 5
    g = StationGains(n_stations=3, n_times=t)
    ids = np.zeros((t, m, 2), np.int32)
    ids[..., 0], ids[..., 1] = 0, 1
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    np.testing.assert_allclose(out, vis, rtol=1e-6, atol=1e-6)


def test_amplitude_clip():
    """Amplitudes are hard-clipped to amp_bounds; in-bounds values pass through."""
    g = StationGains(3, 2, amp_bounds=(0.9, 1.1))
    g = eqx.tree_at(
        lambda mm: mm.log_amp, g,
        jnp.array([[5.0, -5.0], [0.0, 0.0], [0.05, -0.05]]),
    )
    amp = np.asarray(g.amplitudes())
    assert amp.max() <= 1.1 + 1e-6
    assert amp.min() >= 0.9 - 1e-6
    assert abs(amp[2, 0] - np.exp(0.05)) < 1e-5  # in-bounds -> not clipped


def test_pad_neutrality():
    """Padded baseline slots (station id -1) pass through unchanged."""
    t, m = 3, 4
    g = StationGains(3, t, amp_bounds=(0.5, 2.0))
    g = eqx.tree_at(
        lambda mm: mm.log_amp, g,
        jax.random.normal(jax.random.PRNGKey(0), (3, t)) * 0.1,
    )
    ids = -np.ones((t, m, 2), np.int32)  # every slot padded
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    np.testing.assert_allclose(out, vis, rtol=1e-6, atol=1e-6)


def test_gather_matches_manual():
    """apply == V_ij * g_i * conj(g_j) computed element-by-element by hand."""
    t, m = 2, 3
    g = StationGains(4, t, use_phase=True, amp_bounds=(0.5, 2.0))
    ka, kp = jax.random.split(jax.random.PRNGKey(1))
    g = eqx.tree_at(lambda mm: mm.log_amp, g, jax.random.normal(ka, (4, t)) * 0.1)
    g = eqx.tree_at(lambda mm: mm.phase, g, jax.random.normal(kp, (4, t)) * 0.3)
    ids = np.array(
        [[[0, 1], [2, 3], [1, 3]], [[0, 2], [1, 2], [0, 3]]], np.int32
    )  # (t, m, 2)
    vis = _random_vis(t, m)
    out = np.asarray(g.apply(jnp.asarray(vis), jnp.asarray(ids)))
    gg = np.asarray(g.station_gains())  # (n_st, t)
    exp = np.empty_like(vis)
    for ti in range(t):
        for mi in range(m):
            i, j = ids[ti, mi]
            exp[ti, mi] = vis[ti, mi] * gg[i, ti] * np.conj(gg[j, ti])
    np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5)


def test_closure_phase_invariance():
    """Station-based gains (amp + phase) leave the closure phase unchanged."""
    t = 2
    g = StationGains(3, t, use_phase=True, amp_bounds=(0.5, 2.0))
    ka, kp = jax.random.split(jax.random.PRNGKey(2))
    g = eqx.tree_at(lambda mm: mm.log_amp, g, jax.random.normal(ka, (3, t)) * 0.2)
    g = eqx.tree_at(lambda mm: mm.phase, g, jax.random.normal(kp, (3, t)) * 0.7)
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
        g = eqx.tree_at(
            lambda mm: mm.log_amp, g,
            jax.random.normal(ka, (4, t), dtype=jnp.float64) * 0.2,
        )
        g = eqx.tree_at(
            lambda mm: mm.phase, g,
            jax.random.normal(kp, (4, t), dtype=jnp.float64) * 0.5,
        )
        ids = np.array(
            [[[0, 1], [2, 3], [1, 3], [0, 2], [-1, -1]]] * t, np.int32
        ).reshape(t, m, 2)
        rng = np.random.default_rng(5)
        vis = rng.standard_normal((t, m)) + 1j * rng.standard_normal((t, m))
        corr = g.apply(jnp.asarray(vis), jnp.asarray(ids))
        back = g.apply(corr, jnp.asarray(ids), inverse=True)
        np.testing.assert_allclose(np.asarray(back), vis, rtol=1e-10, atol=1e-12)
    finally:
        jax.config.update("jax_enable_x64", False)
