"""Station gains wired into the polarized likelihood: switch, gradients, invariants.

The gains are an optional field on the model (like ``circ``), so they are solved by the
same ``filter_value_and_grad`` as the sky -- calibration and imaging are one
optimization. These tests pin the properties that make that safe:

  * off by default (M2 data are gain-free by construction),
  * gradients actually reach BOTH gain amplitudes and phases,
  * R and L are separate at ``n_hands=2`` (the physical case),
  * closure phases are gain-invariant -- the canary that says the gains are being
    applied as a per-station product and not absorbing sky structure.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.calibration import PRODUCT_HANDS, StationGains
from neuraldmd.losses import calculate_closure_phases, polarized_loss_fn
from neuraldmd.polarized import PolarizedNeuralDMD, with_gains

S = ("I", "Q", "U")
P = ("RR", "LL", "RL", "LR")
MODEL_KW = dict(hidden_size=16, num_layers=2, num_frequencies=2)


def _setup(t=4, m=6, pix=16, nst=3, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        xy=jnp.asarray(rng.normal(size=(pix, 2))),
        A=jnp.asarray(
            (rng.normal(size=(t, m, pix)) + 1j * rng.normal(size=(t, m, pix))).astype("c8")
        ),
        tgt={
            k: jnp.asarray((rng.normal(size=(t, m)) + 1j * rng.normal(size=(t, m))).astype("c8"))
            for k in P
        },
        sig={k: jnp.ones((t, m)) for k in P},
        msk={k: jnp.ones((t, m)) for k in P},
        bl=jnp.asarray(rng.integers(0, nst, size=(t, m, 2)).astype(np.int32)),
        fidx=jnp.arange(t),
        ti=jnp.linspace(0.0, 1.0, t),
        nst=nst,
        t=t,
    )


def _loss(model, d):
    return polarized_loss_fn(
        model,
        d["xy"],
        d["tgt"],
        d["sig"],
        d["msk"],
        d["A"],
        d["ti"],
        {s: 1.0 for s in S},
        {s: 0.0 for s in S},
        basis="circular",
        products=P,
        neg_weight=0.0,
        w_sparse_weight=0.0,
        b_sparse_weight=0.0,
        bl_station_ids=d["bl"],
        frame_indices=d["fidx"],
    )


def _model(d, gains=None, seed=0):
    m = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(seed), **MODEL_KW)
    return with_gains(m, gains) if gains is not None else m


def test_gains_off_by_default():
    """A plain model solves no calibration -- M2 must be untouched by this feature."""
    d = _setup()
    m = _model(d)
    assert m.gains is None
    loss, _ = _loss(m, d)
    assert np.isfinite(float(loss))


def test_identity_gains_do_not_change_the_loss():
    """Gains initialise to g=1, so attaching them must be a no-op at step 0.

    If this fails the switch is not neutral and enabling it would perturb the fit
    before a single gradient step.
    """
    d = _setup()
    g = StationGains(n_stations=d["nst"], n_times=d["t"], n_hands=2, use_phase=True)
    assert np.allclose(np.asarray(g.station_gains()), 1.0)
    l_off, _ = _loss(_model(d), d)
    l_on, _ = _loss(_model(d, g), d)
    assert float(l_on) == float(l_off)


def test_gradients_reach_amplitudes_and_phases():
    """Complex gains: BOTH amp and phase must receive gradient, and the sky too."""
    d = _setup()
    g = StationGains(n_stations=d["nst"], n_times=d["t"], n_hands=2, use_phase=True)
    grads = eqx.filter_grad(lambda mm: _loss(mm, d)[0])(_model(d, g))
    assert float(np.abs(np.asarray(grads.gains.amp_raw)).max()) > 0.0
    assert float(np.abs(np.asarray(grads.gains.phase)).max()) > 0.0, "phases are not solved"
    sky = jax.tree_util.tree_leaves(eqx.filter(grads.intensity, eqx.is_array))
    assert max(float(np.abs(np.asarray(x)).max()) for x in sky) > 0.0, "sky stopped training"


def test_R_and_L_gains_are_separate_at_two_hands():
    """n_hands=2 must give R and L independent gradients; n_hands=1 ties them.

    RR uses (0,0) and LL uses (1,1), so perturbing hand 0 alone must move RR and
    leave LL alone -- otherwise the hands are secretly shared.
    """
    d = _setup()
    g2 = StationGains(n_stations=d["nst"], n_times=d["t"], n_hands=2, use_phase=False)
    assert g2.amp_raw.shape[-1] == 2, "n_hands=2 must allocate two hands"
    assert PRODUCT_HANDS["RR"] == (0, 0) and PRODUCT_HANDS["LL"] == (1, 1)
    assert PRODUCT_HANDS["RL"] == (0, 1) and PRODUCT_HANDS["LR"] == (1, 0)

    # bump hand 0 only; RR (hand 0) must change, LL (hand 1) must not
    bumped = eqx.tree_at(lambda x: x.amp_raw, g2, g2.amp_raw.at[:, :, 0].add(0.5))
    base = {k: float(v) for k, v in _loss(_model(d, g2), d)[1]["chi2_vis"].items()}
    got = {k: float(v) for k, v in _loss(_model(d, bumped), d)[1]["chi2_vis"].items()}
    assert got["RR"] != base["RR"], "hand-0 gain did not affect RR"
    assert got["LL"] == base["LL"], "hand-0 gain leaked into LL -- hands are not separate"


def test_closure_phases_are_gain_invariant():
    """THE canary: gains cancel around a triangle.

    Station gains multiply V_ij by g_i conj(g_j), so any closed triangle is
    unchanged. If a gain fit improves closure phases, it is absorbing SKY structure
    into the gains rather than calibration -- the failure mode that matters.
    """
    rng = np.random.default_rng(0)
    t, m, nst = 3, 6, 4
    vis = jnp.asarray((rng.normal(size=(t, m)) + 1j * rng.normal(size=(t, m))).astype("c8"))
    # three baselines forming a closed triangle over stations 0-1-2
    bl = np.full((t, m, 2), -1, np.int32)
    bl[:, 0] = (0, 1)
    bl[:, 1] = (1, 2)
    bl[:, 2] = (0, 2)
    tri = np.zeros((t, 1, 3, 2), np.int32)
    tri[..., 0] = np.array([0, 1, 2])  # baseline indices
    tri[..., 1] = np.array([1, 1, -1])  # signs: V01 * V12 * conj(V02)
    g = StationGains(n_stations=nst, n_times=t, n_hands=1, use_phase=True)
    g = eqx.tree_at(lambda x: x.phase, g, jnp.asarray(rng.normal(size=g.phase.shape) * 0.7))
    g = eqx.tree_at(lambda x: x.amp_raw, g, jnp.asarray(rng.normal(size=g.amp_raw.shape) * 0.7))

    cp_before = calculate_closure_phases(vis, jnp.asarray(tri))
    cp_after = calculate_closure_phases(
        g.apply(vis, jnp.asarray(bl), hands=(0, 0), time_indices=jnp.arange(t)), jnp.asarray(tri)
    )
    assert np.allclose(
        np.angle(np.asarray(cp_before)), np.angle(np.asarray(cp_after)), atol=1e-4
    ), (
        "closure phases moved under a pure station-gain change -- gains are not being "
        "applied as g_i conj(g_j)"
    )
