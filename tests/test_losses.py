"""Characterize loss_fn arithmetic + closure phases against an independent
numpy reimplementation on the tiny_obs fixture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from _impl import NeuralDMD, calculate_closure_phases, loss_fn, sparsity_loss


def _model(r=3):
    return NeuralDMD(r=r, key=jax.random.PRNGKey(7), num_frequencies=2)


def _batch(tobs, tb=3):
    def g(a):
        return jnp.asarray(a[:tb])

    frames = jnp.asarray(tobs.movie[:tb].reshape(tb, tobs.P))
    return dict(
        frame_batch=frames,
        A_batch=g(tobs.A),
        vis_target_batch=g(tobs.targets),
        vis_sigma_batch=g(tobs.sigmas),
        vis_mask_batch=g(tobs.masks),
        amp_target_batch=g(tobs.amp_targets),
        amp_sigma_batch=g(tobs.amp_sigmas),
        cp_target_batch=g(tobs.cp_targets),
        cp_sigma_batch=g(tobs.cp_sigmas),
        cp_mask_batch=g(tobs.cp_masks),
        triangles=g(tobs.tris),
        time_indices=jnp.asarray(tobs.times[:tb]),
    )


def test_loss_fn_matches_numpy(tiny_obs):
    m = _model(r=3)
    xy = jnp.asarray(tiny_obs.loader_grid())
    b = _batch(tiny_obs, tb=3)
    fmax, fmin = 1.0, 0.0

    total, (rec, chi2_vis, chi2_amp, chi2_cp) = loss_fn(
        m,
        xy,
        b["frame_batch"],
        b["vis_target_batch"],
        b["vis_sigma_batch"],
        b["vis_mask_batch"],
        b["amp_target_batch"],
        b["amp_sigma_batch"],
        b["cp_target_batch"],
        b["cp_sigma_batch"],
        b["cp_mask_batch"],
        b["triangles"],
        b["A_batch"],
        b["time_indices"],
        fmax,
        fmin,
    )

    # independent numpy reimplementation using the same model outputs
    W0, W, Omega, b0, bb = (np.asarray(x) for x in m(xy))
    t = np.asarray(b["time_indices"])
    lam = np.exp(Omega[:, None] * t[None, :] * m.t_scale)
    I = W0[:, 0:1] * b0[0] + 2 * np.real(np.einsum("pr,rt,r->pt", W, lam, bb))
    I = I * (fmax - fmin) + fmin
    A = np.asarray(b["A_batch"])
    tgt = np.asarray(b["vis_target_batch"])
    sig = np.asarray(b["vis_sigma_batch"])
    msk = np.asarray(b["vis_mask_batch"])
    vis_pred = np.einsum("tvp,pt->tv", A, I.astype(np.complex64))

    chi2_vis_np = np.sum(np.abs(vis_pred - tgt) ** 2 * msk / sig**2) / (2 * np.sum(msk))
    neg_np = np.sum(np.maximum(-I, 0.0) ** 2)
    rec_np = np.sum(np.abs(np.asarray(b["frame_batch"]) - I.T))
    amp = np.asarray(b["amp_target_batch"])
    amps = np.asarray(b["amp_sigma_batch"])
    chi2_amp_np = np.sum(((np.abs(vis_pred) - amp) / amps) ** 2 * msk) / np.sum(msk)
    spars_np = (np.mean(np.abs(W0)) + np.mean(np.abs(W))) + (
        np.mean(np.abs(b0)) + np.mean(np.abs(bb))
    )
    total_np = chi2_vis_np + neg_np + spars_np

    np.testing.assert_allclose(float(chi2_vis), chi2_vis_np, rtol=1e-4)
    np.testing.assert_allclose(float(chi2_amp), chi2_amp_np, rtol=1e-4)
    np.testing.assert_allclose(float(rec), rec_np, rtol=1e-4)
    np.testing.assert_allclose(float(total), total_np, rtol=1e-4)


def test_padded_visibilities_ignored(tiny_obs):
    # corrupting masked-out (padded) targets must not change chi2_vis
    m = _model(r=3)
    xy = jnp.asarray(tiny_obs.loader_grid())
    b = _batch(tiny_obs, tb=3)
    args = [
        b["frame_batch"],
        b["vis_target_batch"],
        b["vis_sigma_batch"],
        b["vis_mask_batch"],
        b["amp_target_batch"],
        b["amp_sigma_batch"],
        b["cp_target_batch"],
        b["cp_sigma_batch"],
        b["cp_mask_batch"],
        b["triangles"],
        b["A_batch"],
        b["time_indices"],
    ]
    _, (_, chi2a, _, _) = loss_fn(m, xy, *args, 1.0, 0.0)

    corrupt = np.asarray(b["vis_target_batch"]).copy()
    corrupt[:, tiny_obs.n_real :] += 1e3 + 1e3j  # padded columns
    args[1] = jnp.asarray(corrupt)
    _, (_, chi2b, _, _) = loss_fn(m, xy, *args, 1.0, 0.0)
    np.testing.assert_allclose(float(chi2a), float(chi2b), rtol=1e-6)


def test_closure_phase_sign_conjugation():
    # calculate_closure_phases: normalized bispectrum with conj where sign<0
    rng = np.random.default_rng(0)
    T, V = 2, 6
    vis = (rng.normal(size=(T, V)) + 1j * rng.normal(size=(T, V))).astype(np.complex64)
    tris = np.array([[[0, 1], [1, 1], [2, -1]], [[3, 1], [4, -1], [5, 1]]], np.int32)
    tris = np.broadcast_to(tris, (T, 2, 3, 2))

    out = np.asarray(calculate_closure_phases(jnp.asarray(vis), jnp.asarray(tris)))

    exp = np.zeros((T, 2), np.complex64)
    for t in range(T):
        for j, legs in enumerate(tris[t]):
            prod = 1.0 + 0j
            for bl, sign in legs:
                prod *= np.conj(vis[t, bl]) if sign < 0 else vis[t, bl]
            exp[t, j] = prod / (np.abs(prod) + 1e-6)
    np.testing.assert_allclose(out, exp, rtol=1e-4, atol=1e-5)


def test_sparsity_loss():
    W0 = jnp.asarray([[1.0], [-3.0]])
    W = jnp.asarray([[1 + 0j, -0j], [0 + 4j, 2 + 0j]])
    got = float(sparsity_loss(W0, W))
    exp = float(np.mean(np.abs(np.asarray(W0))) + np.mean(np.abs(np.asarray(W))))
    np.testing.assert_allclose(got, exp, rtol=1e-6)
