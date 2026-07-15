"""Visibility chi-squared loss, mode sparsity, and closure-phase diagnostics.

Only ``chi2_vis`` (a reduced chi-squared, per real degree of freedom) drives the
gradient; amplitude and closure-phase chi-squared are returned as diagnostics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics.stokes import stokes_to_products_matrix


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


def _physical_intensities(model, xy, time_indices, frame_max, frame_min):
    """Reconstruct one scalar model's physical-unit intensities and mode arrays.

    Mirrors the inline reconstruction in :func:`loss_fn` exactly (so the
    Stokes-I path is bit-identical).

    Parameters
    ----------
    model : NeuralDMD
        A single scalar model.
    xy : jax.Array
        ``(P, 2)`` pixel coordinates.
    time_indices : jax.Array
        ``(T,)`` normalized times of the frame batch.
    frame_max, frame_min : float
        Output scaling ``intensities * (frame_max - frame_min) + frame_min``.

    Returns
    -------
    intensities : jax.Array
        ``(P, T)`` physical-unit intensities.
    modes : tuple
        ``(W0, W, b0, b)`` for the sparsity penalties.
    """
    W0, W, Omega, b0, b = model(xy)
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * model.t_scale)
    i_stat = W0[:, 0:1] * b0[0]
    i_dyn = 2 * jnp.real(jnp.einsum("pr,rt,r->pt", W, lambda_exp, b))
    intensities = (i_stat + i_dyn) * (frame_max - frame_min) + frame_min
    return intensities, (W0, W, b0, b)


def _vis_chi2(vis_pred, target, sigma, mask):
    """Reduced complex-visibility chi-squared (per real dof: divide by 2*sum(mask))."""
    diff2 = jnp.abs(vis_pred - target) ** 2
    return jnp.sum(diff2 * mask / sigma**2) / (2.0 * jnp.sum(mask))


def polarized_loss_fn(
    model,
    xy,
    targets: dict,
    sigmas: dict,
    masks: dict,
    A_batch,
    time_indices,
    frame_max: dict,
    frame_min: dict,
    *,
    basis: str = "stokes",
    products: tuple[str, ...] = ("RR", "LL", "RL", "LR"),
    neg_weight: float = 1.0,
    w_sparse_weight: float = 1.0,
    b_sparse_weight: float = 1.0,
    p_le_i_weight: float = 0.0,
):
    """Data-fidelity loss for a :class:`PolarizedNeuralDMD`, Stokes or circular basis.

    The gradient-driving term is a sum of reduced chi-squared, each divided by
    ``2 * sum(mask)`` as in :func:`loss_fn`, plus a negativity penalty on **Stokes
    I only** (Q, U, V are signed), per-net sparsity, and an optional soft
    ``P <= I`` penalty (default off). These image-domain penalties do not depend
    on the fidelity basis.

    Two fidelity bases:

    * ``basis="stokes"`` -- chi-squared of each modeled Stokes visibility against
      the corresponding Stokes target. Simple; assumes the data were converted to
      Stokes visibilities (which correlates the per-hand thermal noise).
    * ``basis="circular"`` -- the modeled Stokes visibilities are combined into
      correlation products (default RR, LL, RL, LR) via
      :func:`neuraldmd.physics.stokes.stokes_to_products_matrix`, and chi-squared
      is taken against the native product data with their independent per-hand
      sigma. This is the noise-faithful comparison for real interferometric data.

    With ``model.stokes == ("I",)``, ``basis="stokes"``, and matching
    weights/scaling this returns exactly the :func:`loss_fn` total (parity gate).

    Parameters
    ----------
    model : PolarizedNeuralDMD
        The polarized container.
    xy : jax.Array
        ``(P, 2)`` pixel coordinates.
    targets, sigmas, masks : dict of str -> jax.Array
        Visibility targets, 1-sigma errors, and 0/1 masks, each ``(T, V)``. Keyed
        by Stokes for ``basis="stokes"`` and by product for ``basis="circular"``.
    A_batch : jax.Array
        ``(T, V, P)`` image->visibility operator (shared across Stokes).
    time_indices : jax.Array
        ``(T,)`` normalized frame times.
    frame_max, frame_min : dict of str -> float
        Per-Stokes output scaling (Stokes I uses the physical intensity range;
        signed Stokes typically use ``frame_min = 0`` with a symmetric ``frame_max``).
    basis : {"stokes", "circular"}
        Fidelity basis (see above).
    products : tuple of str
        Correlation products for ``basis="circular"`` (default RR, LL, RL, LR).
    neg_weight, w_sparse_weight, b_sparse_weight : float
        Weights for the I-negativity and per-net sparsity penalties.
    p_le_i_weight : float
        Weight for the optional ``sum(relu(sqrt(Q^2+U^2+V^2) - I)^2)`` penalty
        (default 0.0 -> disabled).

    Returns
    -------
    total : jax.Array
        Scalar loss.
    aux : dict
        ``{"chi2_vis": {key: value}, "neg_I": value, "p_penalty": value,
        "basis": basis}`` -- ``chi2_vis`` keyed by Stokes or product per ``basis``.
    """
    phys: dict[str, jax.Array] = {}
    vis_stokes: dict[str, jax.Array] = {}
    sparse_total = 0.0
    for s in model.stokes:
        intensities, (W0, W, b0, b) = _physical_intensities(
            model.models[s], xy, time_indices, frame_max[s], frame_min[s]
        )
        phys[s] = intensities
        vis_stokes[s] = jnp.einsum("tvp,pt->tv", A_batch, intensities.astype(jnp.complex64))
        sparse_total = (
            sparse_total
            + w_sparse_weight * sparsity_loss(W0, W)
            + b_sparse_weight * sparsity_loss(b0, b)
        )

    if basis == "stokes":
        chi2 = {s: _vis_chi2(vis_stokes[s], targets[s], sigmas[s], masks[s]) for s in model.stokes}
    elif basis == "circular":
        # constant (products, model.stokes are static) -> folded at trace time
        m_mat = jnp.asarray(
            stokes_to_products_matrix(tuple(products), model.stokes), dtype=jnp.complex64
        )
        chi2 = {}
        for i, p in enumerate(products):
            vis_p = sum(m_mat[i, j] * vis_stokes[s] for j, s in enumerate(model.stokes))
            chi2[p] = _vis_chi2(vis_p, targets[p], sigmas[p], masks[p])
    else:
        raise ValueError(f"basis must be 'stokes' or 'circular', got {basis!r}")

    neg_i = jnp.sum(jax.nn.relu(-phys["I"]) ** 2)  # negativity: Stokes I only

    pol = [s for s in model.stokes if s in ("Q", "U", "V")]
    if p_le_i_weight and pol:
        p_sq = sum(phys[s] ** 2 for s in pol)
        p_penalty = jnp.sum(jax.nn.relu(jnp.sqrt(p_sq) - phys["I"]) ** 2)
    else:
        p_penalty = jnp.asarray(0.0)

    total = sum(chi2.values()) + neg_weight * neg_i + p_le_i_weight * p_penalty + sparse_total
    return total, {"chi2_vis": chi2, "neg_I": neg_i, "p_penalty": p_penalty, "basis": basis}
