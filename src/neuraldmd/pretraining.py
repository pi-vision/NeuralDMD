"""Disk-template pretraining: initialize NeuralDMD's spatial modes.

Fitting sparse visibilities from a cold start is badly non-convex. The fix is
to warm-start the spatial network so that its modes already span a smooth,
orthogonal, disk-supported basis of about the right size:

1. estimate the source size from the data (radius of gyration),
2. build complex Zernike targets on a disk of that size (zernike_bank.py),
3. train the spatial network so its gauge-fixed modes [W0, W] align with the
   targets, up to a per-mode complex scale.

Only the spatial network receives gradients (the alignment loss does not
depend on the temporal nets), so the resulting "disk template" checkpoint is
a pure spatial-mode initialization. It depends only on grid size, disk
radius, and model hyperparameters — one template can be reused across
datasets with similar source extent.
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .model import physical_intensities
from .zernike import build_zernike_targets


def radius_of_gyration(video, fov_x=np.pi, fov_y=np.pi):
    """Median flux-weighted RMS radius of a (T, H, W) movie.

    For real data, estimate the source extent from the visibility-amplitude
    fall-off instead.
    """
    T, H, W = video.shape
    sx, sy = fov_x / W, fov_y / H
    x = (np.arange(W) - W / 2.0) * sx
    y = (np.arange(H) - H / 2.0) * sy
    X, Y = np.meshgrid(x, y)

    Rg = np.empty(T)
    for t in range(T):
        I = np.clip(video[t], 0.0, None)
        F = I.sum() + 1e-12
        cx = (X * I).sum() / F
        cy = (Y * I).sum() / F
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        Rg[t] = np.sqrt((r2 * I).sum() / F + 1e-12)

    Rg_med = np.median(Rg)
    dRg = np.sqrt(2.0) * np.median(np.abs(Rg - Rg_med))
    return Rg_med, dRg


def _best_complex_scale_residual(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-12):
    """min over complex a of ||a*x - y||^2 (a* = x^H y / x^H x)."""
    xy = jnp.vdot(x, y)
    xx = jnp.vdot(x, x) + eps
    a = xy / xx
    r = a * x - y
    return jnp.sum(jnp.abs(r) ** 2)


def zernike_alignment_loss(W_cat: jnp.ndarray, Z_targets: jnp.ndarray):
    """Sum over modes of the scale-invariant misfit between W and Z columns.

    W_cat: (P, r+1) — model modes [W0, W] on the sampled pixels
    Z_targets: (P, r+1) — target Zernike columns in the same order
    """
    col_loss = jax.vmap(_best_complex_scale_residual, in_axes=(1, 1))(W_cat, Z_targets)
    return jnp.sum(col_loss)


def pretrain_loss_fn(model, xy, Z_targets):
    W0, W = jax.vmap(model.spatial_forward)(xy)
    W0, W = model._gauge_fix(W0, W)
    return zernike_alignment_loss(jnp.concatenate([W0, W], axis=1), Z_targets)


@eqx.filter_jit
def _pretrain_step(model, opt_state, xy, Z_targets, optimizer):
    loss, grads = eqx.filter_value_and_grad(pretrain_loss_fn)(model, xy, Z_targets)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def pretrain_model(
    model,
    xy,
    Z_targets,
    num_steps=2000,
    lr=1e-4,
    key=None,
    jitter=0.01,
    print_every=200,
):
    """Align the spatial network's modes with the Zernike targets.

    xy: (P, 2) full pixel grid (same convention as DMDDataLoader.pixel_coords)
    Z_targets: (P, r+1) complex — column 0 is the target for the static mode
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    xy = jnp.asarray(xy)
    Z_targets = jnp.asarray(Z_targets)

    losses = []
    with tqdm(total=num_steps) as pbar:
        for step in range(num_steps):
            key, subkey = jax.random.split(key)
            xy_noisy = xy + jax.random.normal(subkey, xy.shape) * jitter
            model, opt_state, loss = _pretrain_step(
                model, opt_state, xy_noisy, Z_targets, optimizer
            )
            losses.append(float(loss))
            pbar.set_postfix(loss=f"{float(loss):.4f}")
            pbar.update(1)
            if (step + 1) % print_every == 0:
                print(f"step {step + 1}/{num_steps}  alignment loss {float(loss):.5f}", flush=True)

    return model, losses


def pretrain_stokes_i(
    polarized_model,
    truth_i,
    fov=np.pi,
    num_steps=2000,
    lr=1e-4,
    radius_scale=1.5,
    max_n=8,
    key=None,
):
    """Disk-template pretrain the Stokes-I sub-model of a ``PolarizedNeuralDMD``.

    Estimates the source size from ``truth_i`` (radius of gyration), builds
    Zernike targets on a disk of ``radius_scale * Rg``, and aligns the I
    sub-model's spatial modes to them -- exactly the Stokes-I baseline recipe,
    applied only to the ``"I"`` sub-model. Q/U/V are left at their zero-init
    (signed fields start near zero, which is correct). This gives the polarized
    fit a well-conditioned image prior that a bare chi2 fit lacks.

    Parameters
    ----------
    polarized_model : PolarizedNeuralDMD
        The model whose ``models["I"]`` sub-model is pretrained.
    truth_i : numpy.ndarray
        ``(T, H, W)`` Stokes-I reference movie (for the size estimate). For real
        data, estimate the extent from visibility amplitudes instead.
    fov : float
        Network coordinate extent (must match the loader's ``fov_x``/``fov_y``).
    num_steps, lr, radius_scale, max_n : pretraining hyperparameters.
    key : jax.Array or None
        PRNG key for the coordinate jitter.

    Returns
    -------
    model : PolarizedNeuralDMD
        A copy with the ``"I"`` sub-model replaced by the pretrained one.
    losses : list of float
        Alignment-loss history.
    """
    model_i = polarized_model.i_submodel
    _, height, width = np.asarray(truth_i).shape
    r_g, _ = radius_of_gyration(truth_i, fov_x=fov, fov_y=fov)
    z_targets, _picked, _mask, xy = build_zernike_targets(
        height,
        width,
        radius_scale * r_g,
        fov,
        fov,
        model_i.r + 1,
        max_n=max_n,
        prefer_ms=(0, 1, 2, 3),
    )
    trained_i, losses = pretrain_model(model_i, xy, z_targets, num_steps=num_steps, lr=lr, key=key)
    return polarized_model.replace_i_submodel(trained_i), losses


def pretrain_log_intensity(
    polarized_model,
    truth_i,
    fov=np.pi,
    num_steps=2000,
    lr=1e-3,
    radius_scale=1.5,
    floor=1e-3,
    max_n=8,
    key=None,
):
    """Log-space disk-template pretrain for an ``expm_full`` ``PolarizedNeuralDMD``.

    For ``pol_param="expm_full"`` the intensity sub-model is ``s = log I`` (total
    intensity ``I = e^s cosh(p)``). The standard :func:`pretrain_stokes_i` targets a
    *linear* disk (~0 outside the source), which in log space would leave ``e^0 = 1``
    full-brightness background flux everywhere. Instead we fit ``s`` directly, in
    image space, to ``log(disk + floor)`` so that ``e^s ~ disk`` (small outside).
    The disk radius is estimated from ``truth_i`` exactly as in the linear pretrain.

    Parameters
    ----------
    polarized_model : PolarizedNeuralDMD
        Model whose ``"I"`` (log-intensity) sub-model is pretrained.
    truth_i : numpy.ndarray
        ``(T, H, W)`` Stokes-I reference (for the size estimate only).
    fov, num_steps, lr, radius_scale, max_n : hyperparameters (mirror
        :func:`pretrain_stokes_i`).
    floor : float
        Outside-disk intensity; the log target is ``log(floor)`` there (``1e-3`` ->
        about ``-6.9``, i.e. ``e^s ~ 0.1%`` of the disk value off-source).
    key : jax.Array or None
        Unused (kept for signature parity); the log target is defined on the fixed
        pixel grid.

    Returns
    -------
    model : PolarizedNeuralDMD
        A copy with the pretrained log-intensity sub-model.
    losses : list of float
        Image-space MSE history.
    """
    model_i = polarized_model.i_submodel
    _, height, width = np.asarray(truth_i).shape
    r_g, _ = radius_of_gyration(truth_i, fov_x=fov, fov_y=fov)
    _z, _picked, mask, xy = build_zernike_targets(
        height,
        width,
        radius_scale * r_g,
        fov,
        fov,
        model_i.r + 1,
        max_n=max_n,
        prefer_ms=(0, 1, 2, 3),
    )
    xy = jnp.asarray(xy)
    log_target = jnp.log(jnp.asarray(mask, dtype=jnp.float32).reshape(-1) + floor)  # (P,)
    t0 = jnp.zeros((1,), dtype=jnp.float32)

    def loss_fn(m):
        s, _ = physical_intensities(m, xy, t0, 1.0, 0.0)  # (P, 1) raw log-I field
        return jnp.mean((s[:, 0] - log_target) ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model_i, eqx.is_array))

    @eqx.filter_jit
    def step(m, opt_state_):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, opt_state_ = optimizer.update(grads, opt_state_, eqx.filter(m, eqx.is_array))
        return eqx.apply_updates(m, updates), opt_state_, loss

    losses = []
    with tqdm(total=num_steps) as pbar:
        for i in range(num_steps):
            model_i, opt_state, loss = step(model_i, opt_state)
            losses.append(float(loss))
            pbar.update(1)
            if (i + 1) % 200 == 0:
                print(f"log-pretrain step {i + 1}/{num_steps}  mse {float(loss):.5f}", flush=True)

    return polarized_model.replace_i_submodel(model_i), losses


def save_template(model, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"disk_template_r{model.r}_f{model.num_frequencies}.eqx")
    eqx.tree_serialise_leaves(path, model)
    print(f"Saved disk template to {path}")
    return path
