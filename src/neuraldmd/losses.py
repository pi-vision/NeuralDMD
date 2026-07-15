"""Visibility chi-squared loss, mode sparsity, and closure-phase diagnostics.

Only ``chi2_vis`` (a reduced chi-squared, per real degree of freedom) drives the
gradient; amplitude and closure-phase chi-squared are returned as diagnostics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sparsity_loss(W0: jax.Array, W: jax.Array) -> jax.Array:
    """L1 penalty encouraging sparse spatial modes / amplitudes."""
    return jnp.mean(jnp.abs(W0)) + jnp.mean(jnp.abs(W))


def calculate_closure_phases(vis_pred: jax.Array, triangles: jax.Array) -> jax.Array:
    """Predicted bispectrum phasors for the stored (index, sign) triangles."""
    idxs = triangles[..., 0]  # (T_b, n_tri, 3)
    signs = triangles[..., 1]

    bl1, bl2, bl3 = idxs[..., 0], idxs[..., 1], idxs[..., 2]
    s1, s2, s3 = signs[..., 0], signs[..., 1], signs[..., 2]

    V1 = jnp.take_along_axis(vis_pred, bl1, axis=-1)
    V2 = jnp.take_along_axis(vis_pred, bl2, axis=-1)
    V3 = jnp.take_along_axis(vis_pred, bl3, axis=-1)
    V1 = jnp.where(s1 < 0, jnp.conj(V1), V1)
    V2 = jnp.where(s2 < 0, jnp.conj(V2), V2)
    V3 = jnp.where(s3 < 0, jnp.conj(V3), V3)

    phasor = V1 * V2 * V3
    return phasor / (jnp.abs(phasor) + 1e-6)


def loss_fn(
    model,
    xy,
    frame_batch,
    vis_target_batch,
    vis_sigma_batch,
    vis_mask_batch,
    amp_target_batch,
    amp_sigma_batch,
    cp_target_batch,
    cp_sigma_batch,
    cp_mask_batch,
    triangles,
    A_batch,
    time_indices,
    frame_max,
    frame_min,
    neg_weight: float = 1.0,
    w_sparse_weight: float = 1.0,
    b_sparse_weight: float = 1.0,
):
    """Visibility chi-squared + negativity + sparsity.

    ``frame_batch`` (ground-truth frames) is used only for the reconstruction
    diagnostic in aux -- it does not influence the gradient.
    """
    W0, W, Omega, b0, b = model(xy)
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * model.t_scale)
    I_stat = W0[:, 0:1] * b0[0]  # (P, 1), time-independent
    I_dyn = 2 * jnp.real(jnp.einsum("pr,rt,r->pt", W, lambda_exp, b))
    intensities = I_stat + I_dyn  # (P, T_b), normalized units

    # map to physical units before applying the measurement operator
    intensities = intensities * (frame_max - frame_min) + frame_min
    negative_penalty = jnp.sum(jax.nn.relu(-intensities) ** 2)
    reconstruction_loss = jnp.sum(jnp.abs(frame_batch - intensities.T))

    vis_pred = jnp.einsum("tvp,pt->tv", A_batch, intensities.astype(jnp.complex64))

    # Each complex visibility carries 2 degrees of freedom (Re + Im, each with
    # variance sigma^2), so divide by 2N: a perfect model gives chi2_vis ~ 1.
    vis_diff = jnp.abs(vis_pred - vis_target_batch)
    chi2_vis = jnp.sum(vis_diff**2 * vis_mask_batch / vis_sigma_batch**2) / (
        2.0 * jnp.sum(vis_mask_batch)
    )

    # amplitude / closure-phase chi-squared: diagnostics only
    amp_pred = jnp.abs(vis_pred)
    amp_res2 = ((amp_pred - amp_target_batch) / amp_sigma_batch) ** 2
    chi2_amp = jnp.sum(amp_res2 * vis_mask_batch) / jnp.sum(vis_mask_batch)

    phasor_pred = calculate_closure_phases(vis_pred, triangles)
    phasor_target = jnp.exp(1j * cp_target_batch)
    phasor_res = jnp.abs(phasor_pred - phasor_target) ** 2  # = 2 (1 - cos dpsi)
    chi2_cp = jnp.sum(phasor_res * cp_mask_batch / cp_sigma_batch**2) / jnp.sum(cp_mask_batch)

    total = (
        chi2_vis
        + neg_weight * negative_penalty
        + w_sparse_weight * sparsity_loss(W0, W)
        + b_sparse_weight * sparsity_loss(b0, b)
    )
    return total, (reconstruction_loss, chi2_vis, chi2_amp, chi2_cp)
