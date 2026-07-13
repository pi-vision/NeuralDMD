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

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from tqdm import tqdm


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
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
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
                print(f"step {step+1}/{num_steps}  alignment loss {float(loss):.5f}", flush=True)

    return model, losses


def save_template(model, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(
        models_dir, f"disk_template_r{model.r}_f{model.num_frequencies}.eqx"
    )
    eqx.tree_serialise_leaves(path, model)
    print(f"Saved disk template to {path}")
    return path
