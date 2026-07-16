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
    flux_target: float | None = None,
    flux_weight: float = 1.0,
    compact_weight: float = 0.0,
    compact_pol_weight: float = 0.0,
    pol_support_weight: float = 0.0,
    pol_support_tau: float = 0.05,
    pol_l1_weight: float = 0.0,
    dyn_compact_weight: float = 0.0,
):
    """Data-fidelity loss for a :class:`PolarizedNeuralDMD`, Stokes or circular basis.

    The gradient-driving term is a sum of reduced chi-squared, each divided by
    ``2 * sum(mask)`` as in :func:`loss_fn`, plus a negativity penalty on **Stokes
    I only** (Q, U, V are signed), per-net sparsity, an optional soft ``P <= I``
    penalty (default off), and an optional total-flux (lightcurve) anchor
    (default off). These image-domain penalties do not depend on the fidelity
    basis.

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
    flux_target : float or None
        Known total flux [Jy]. When set, adds
        ``flux_weight * mean_t(((sum_p I[p, t] - flux_target) / flux_target)^2)``
        -- a lightcurve anchor. The array has no zero-spacing baseline, so total
        flux is only weakly constrained by the data and can otherwise leak into
        a large-scale haze (and, later, into station gain amplitudes). ``None``
        disables (default).
    flux_weight : float
        Weight of the total-flux anchor.
    compact_weight : float
        Weight of the compactness prior -- the flux-weighted mean squared radius
        (source size) of Stokes I. Suppresses off-source haze that the short
        baselines cannot constrain. ``0`` disables (default).
    compact_pol_weight : float
        Weight of the same second-moment prior applied to the polarized intensity
        ``P = sqrt(Q^2+U^2(+V^2))``. Suppresses the off-source *polarized* haze
        that direct (untied) Q,U fields otherwise dump into the cross-hand null
        space, forcing the pol onto the ring where its azimuthal (EVPA) structure
        is actually constrained. ``0`` disables (default).
    pol_support_weight : float
        Weight of the polarized *support* prior ``mean(P * exp(-I / tau))``, which
        penalizes polarized flux where Stokes I is faint. Confines pol to I's
        bright support (both off-ring and the dark ring-center) -- the constraint
        the fractional ``P = m_l * I`` parameterization enforces by construction,
        supplied explicitly here for direct Q,U fields. ``0`` disables (default).
    pol_support_tau : float
        Gate scale as a fraction of PEAK I (I is normalized to its max), so it is
        dataset-independent: pol is suppressed where ``I < ~tau * I_peak``. Smaller
        ``tau`` = harder gate (default ``0.05``).
    pol_l1_weight : float
        Weight of the L1 (total polarized flux) prior ``mean(sum(P))`` -- the
        classic RML sparsity regularizer applied to P. The on-ring m=2 swirl and
        the off-ring haze are near-degenerate in chi2 at the sampled (u,v) but not
        in polarized flux: the ring buys the same chi2 with several times less
        flux, so penalizing flux per se selects it. Unlike ``compact_pol_weight``
        it does not depend on radius, and unlike ``pol_support_weight`` it has no
        I-gate (hence no barrier from I and no path to game it via I). ``0``
        disables (default).

    Returns
    -------
    total : jax.Array
        Scalar loss.
    aux : dict
        ``{"chi2_vis": {key: value}, "neg_I": value, "p_penalty": value,
        "flux_penalty": value, "basis": basis}`` -- ``chi2_vis`` keyed by Stokes
        or product per ``basis``.
    """
    images, modes = model.stokes_fields(xy, time_indices, frame_max, frame_min)
    vis_stokes = {
        s: jnp.einsum("tvp,pt->tv", A_batch, images[s].astype(jnp.complex64)) for s in model.stokes
    }
    sparse_total = 0.0
    for w0, w, b0, b in modes:
        sparse_total = (
            sparse_total
            + w_sparse_weight * sparsity_loss(w0, w)
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

    neg_i = jnp.sum(jax.nn.relu(-images["I"]) ** 2)  # negativity: Stokes I only

    pol = [s for s in model.stokes if s in ("Q", "U", "V")]
    if p_le_i_weight and pol:
        # eps inside the sqrt is REQUIRED: with pol tied to I (iscaled), Q,U vanish
        # exactly where I crosses zero, so p_sq=0 there and an unguarded sqrt has an
        # infinite derivative -- combined with relu'(<0)=0 it yields 0*inf = NaN,
        # which blew up otherwise-healthy runs the instant a pixel reached I=0.
        p_sq = sum(images[s] ** 2 for s in pol)
        p_penalty = jnp.sum(jax.nn.relu(jnp.sqrt(p_sq + 1e-12) - images["I"]) ** 2)
    else:
        p_penalty = jnp.asarray(0.0)

    if flux_target is not None:
        # total-flux (lightcurve) anchor: images are Jy/pixel, so the per-frame
        # pixel sum is the model's zero-spacing flux
        tot_flux = jnp.sum(images["I"], axis=0)  # (T,)
        flux_penalty = jnp.mean(((tot_flux - flux_target) / flux_target) ** 2)
    else:
        flux_penalty = jnp.asarray(0.0)

    if compact_weight or compact_pol_weight or dyn_compact_weight:
        # compactness prior: the second moment about the field center (radially-
        # weighted total flux). Off-source haze (large radius) lives in the
        # short-baseline null space and is otherwise unconstrained; this penalizes
        # peripheral flux *absolutely* -- unlike a flux-normalized source-size,
        # which the model games by brightening the center instead of removing the
        # haze. ``r2`` is normalized to O(1) so the weight is unit-independent.
        r2 = xy[:, 0] ** 2 + xy[:, 1] ** 2
        r2 = (r2 / (jnp.mean(r2) + 1e-12))[:, None]  # (P, 1), ~O(1)
    if compact_weight:
        compact_penalty = jnp.mean(jnp.sum(jax.nn.relu(images["I"]) * r2, axis=0))
    else:
        compact_penalty = jnp.asarray(0.0)

    if dyn_compact_weight:
        # DYNAMIC compactness: the radial second moment of the Stokes-I *dynamic*
        # spatial modes only. In `I = W0.b0 + 2 Re sum_j W[:,j] e^{Omega_j t} b_j`,
        # variability at pixel p comes from |W[p, j]| -- so penalizing |W| * r^2
        # confines the TIME-VARYING structure to small radius while leaving the
        # static ring (W0) completely untouched. `compact_weight` cannot do this:
        # it penalizes total I, so the ring at r~28 uas and the dynamic haze at
        # r>50 uas differ by only ~4.6x in r^2 and the ring gets squeezed too.
        # Measured motivation: the truth's variability is entirely on-ring (0.355
        # on-ring, 0.003 beyond 50 uas); the recon leaks 0.379 off-source (126x).
        # modes[0] is the intensity sub-network in every parameterization branch.
        w_dyn = modes[0][1]  # (P_pix, r) dynamic spatial modes of Stokes I
        dyn_compact_penalty = jnp.sum(jnp.abs(w_dyn) * r2) / w_dyn.shape[1]
    else:
        dyn_compact_penalty = jnp.asarray(0.0)

    if (compact_pol_weight or pol_support_weight or pol_l1_weight) and pol:
        p_mag = jnp.sqrt(sum(images[s] ** 2 for s in pol) + 1e-12)  # (P_pix, T)

    if pol_l1_weight and pol:
        # L1 (total polarized flux) prior -- the classic RML sparsity regularizer,
        # applied to P rather than I. The cross-hand null space lets the model buy
        # chi2 either with the true on-ring m=2 swirl or with a diffuse off-ring
        # haze; the two are near-degenerate at the sampled (u,v), so chi2 alone
        # cannot choose. They differ sharply in *flux efficiency*: the ring buys
        # the same chi2 with several times less polarized flux than the haze, so
        # penalizing total P per unit flux favours the ring -- unlike a radial
        # moment (only ~2x stronger off-ring than on-ring, empirically too weak)
        # and without the I-gate's optimization barrier. eps inside the sqrt is
        # REQUIRED (see p_penalty above): with pol tied to I, Q,U vanish exactly
        # where I crosses zero and an unguarded sqrt gives 0*inf = NaN.
        pol_l1_penalty = jnp.mean(jnp.sum(p_mag, axis=0))
    else:
        pol_l1_penalty = jnp.asarray(0.0)

    if compact_pol_weight and pol:
        # polarization compactness: the SAME second moment applied to the linear
        # (+circular) polarized intensity P = sqrt(Q^2+U^2(+V^2)). With direct
        # (untied) Q,U fields the cross-hand null space lets pol flux escape
        # off-source as a haze while the ring is left under-polarized and
        # structureless (m=0 EVPA); removing that off-ring escape forces the pol
        # onto the ring, where RL/LR can only be satisfied by the true azimuthal
        # (m>=2) EVPA structure. Uncheatable (absolute, not flux-normalized).
        # NB radius-weighting alone still permits pol in the (faint) ring CENTER;
        # ``pol_support_weight`` below gates on I-brightness instead, forbidding it.
        compact_pol_penalty = jnp.mean(jnp.sum(p_mag * r2, axis=0))
    else:
        compact_pol_penalty = jnp.asarray(0.0)

    if pol_support_weight and pol:
        # polarization SUPPORT prior: penalize polarized flux P where Stokes I is
        # faint, P * exp(-I / tau). Confines pol to I's bright support (the ring),
        # suppressing both the off-ring haze AND pol leaking into the dark
        # ring-center -- unlike a radial moment, which the model games by pooling
        # pol at small radius. This is what the fractional (P = m_l * I)
        # parameterization gets for free; direct Q,U need it explicitly. ``tau`` is
        # a fraction of the PEAK brightness (I normalized to its max), so the gate
        # is dataset-independent: pol is suppressed where I < ~tau * I_peak.
        # I enters the gate ONLY as a passive reference (stop_gradient): otherwise
        # exp(-I/tau) has d/dI < 0 and the model games the penalty by INFLATING I
        # (brightening the source to switch off the gate) instead of shrinking P --
        # which destroys the Stokes-I reconstruction. This penalty may only push P
        # down, never push I up.
        i_abs = jax.lax.stop_gradient(jnp.abs(images["I"]))
        i_peak = jnp.max(i_abs) + 1e-12
        gate = jnp.exp(-i_abs / (pol_support_tau * i_peak))
        # sum over pixels, mean over frames -- consistent with the other priors
        support_penalty = jnp.mean(jnp.sum(p_mag * gate, axis=0))
    else:
        support_penalty = jnp.asarray(0.0)

    total = (
        sum(chi2.values())
        + neg_weight * neg_i
        + p_le_i_weight * p_penalty
        + flux_weight * flux_penalty
        + compact_weight * compact_penalty
        + dyn_compact_weight * dyn_compact_penalty
        + compact_pol_weight * compact_pol_penalty
        + pol_support_weight * support_penalty
        + pol_l1_weight * pol_l1_penalty
        + sparse_total
    )
    return total, {
        "chi2_vis": chi2,
        "neg_I": neg_i,
        "p_penalty": p_penalty,
        "flux_penalty": flux_penalty,
        "compact_penalty": compact_penalty,
        "dyn_compact_penalty": dyn_compact_penalty,
        "compact_pol_penalty": compact_pol_penalty,
        "support_penalty": support_penalty,
        "pol_l1_penalty": pol_l1_penalty,
        "basis": basis,
    }
