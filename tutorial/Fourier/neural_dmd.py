"""NeuralDMD for Fourier-domain (interferometric) data.

The model decomposes a movie as

    I(x, t) = W0(x) b0  +  2 Re[ sum_k W_k(x) b_k exp(Omega_k * t_scale * t) ]

where the static mode W0 and the complex spatial modes W_k are the outputs of
a coordinate network (spatial ResidualMLP over positionally-encoded pixel
coordinates), and the continuous spectrum Omega_k = alpha_k + i theta_k and
amplitudes b_k are produced by two small latent-vector networks. Because the
image is real, each complex mode implicitly carries its conjugate twin — `r`
counts complex modes directly (no explicit conjugate stacking).

Training fits the observed complex visibilities: the predicted movie is pushed
through the per-frame forward operators A_t and compared to data with a
chi-squared loss, plus a negativity penalty and L1 sparsity on the modes and
amplitudes. Amplitude and closure-phase chi-squared are tracked as diagnostics.
"""

import os

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

# Building blocks now live in the package (Phase-2 refactor); re-exported here
# so `from neural_dmd import ...` keeps working during the transition.
from neuraldmd.encoding import SinusoidalEncoding
from neuraldmd.networks import ResBlock, ResidualMLP, zero_init_linear


from neuraldmd.model import NeuralDMD, TemporalOmegaMLP, TemporalBMLP


# -------------------------
# Loss
# -------------------------
def sparsity_loss(W0, W):
    """L1 penalty encouraging sparse spatial modes / amplitudes."""
    return jnp.mean(jnp.abs(W0)) + jnp.mean(jnp.abs(W))


def calculate_closure_phases(vis_pred, triangles):
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
    neg_weight=1.0,
    w_sparse_weight=1.0,
    b_sparse_weight=1.0,
):
    """Visibility chi-squared + negativity + sparsity.

    frame_batch (ground-truth frames) is used only for the reconstruction
    diagnostic in aux — it does not influence the gradient.
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
    chi2_cp = jnp.sum(
        phasor_res * cp_mask_batch / cp_sigma_batch**2
    ) / jnp.sum(cp_mask_batch)

    total = (
        chi2_vis
        + neg_weight * negative_penalty
        + w_sparse_weight * sparsity_loss(W0, W)
        + b_sparse_weight * sparsity_loss(b0, b)
    )
    return total, (reconstruction_loss, chi2_vis, chi2_amp, chi2_cp)


# -------------------------
# Training
# -------------------------
@eqx.filter_jit
def train_step(
    model,
    opt_state,
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
    optimizer,
    frame_max,
    frame_min,
):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
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
    )
    updates, opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def train_epoch_jit(model, opt_state, batch_list, optimizer, key, frame_max, frame_min):
    (
        frame_batches,
        pixel_coords,
        As_batches,
        targets_batches,
        sigmas_batches,
        mask_batches,
        time_batches,
        amp_batches,
        amp_sigma_batches,
        cp_batches,
        cp_sigma_batches,
        cp_mask_batches,
        tri_batches,
    ) = batch_list

    def scan_fn(carry, batch_idx):
        model, opt_state, key = carry
        key, subkey = jax.random.split(key)

        # small coordinate jitter regularizes the coordinate network
        noise = jax.random.normal(subkey, shape=pixel_coords.shape) * 0.01
        xy_noisy = pixel_coords + noise

        model, opt_state, loss, aux = train_step(
            model,
            opt_state,
            xy_noisy,
            frame_batches[batch_idx],
            targets_batches[batch_idx],
            sigmas_batches[batch_idx],
            mask_batches[batch_idx],
            amp_batches[batch_idx],
            amp_sigma_batches[batch_idx],
            cp_batches[batch_idx],
            cp_sigma_batches[batch_idx],
            cp_mask_batches[batch_idx],
            tri_batches[batch_idx],
            As_batches[batch_idx],
            time_batches[batch_idx],
            optimizer,
            frame_max,
            frame_min,
        )
        rec_loss, chi2_vis, chi2_amp, chi2_cp = aux
        return (model, opt_state, key), (loss, rec_loss, chi2_vis, chi2_amp, chi2_cp)

    num_batches = frame_batches.shape[0]
    (model, opt_state, _), logs = jax.lax.scan(
        scan_fn, (model, opt_state, key), jnp.arange(num_batches)
    )
    return model, opt_state, tuple(jnp.mean(x) for x in logs)


class PlateauScheduler:
    def __init__(self, initial_lr, factor=1.0, patience=500, min_lr=1e-8):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = jnp.inf
        self.epochs_since_improvement = 0

    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.patience:
            new_lr = max(self.lr * self.factor, self.min_lr)
            if new_lr < self.lr:
                print(f"Reducing learning rate to {new_lr:.6f}", flush=True)
                self.lr = new_lr
                self.epochs_since_improvement = 0
        return self.lr


def plot_losses(history, output_dir, skip_first=2):
    epochs = range(skip_first + 1, len(history["total"]) + 1)
    plt.figure(figsize=(10, 6))
    for name, values in history.items():
        if name == "rec":
            continue  # different scale; plot separately if needed
        plt.plot(epochs, values[skip_first:], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Losses over training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.close()


def train_model(
    model,
    train_loader,
    num_epochs,
    key,
    models_dir,
    plots_dir,
    frame_max,
    frame_min,
    initial_lr=3e-4,
    weight_decay=1e-4,
    lr_factor=1.0,
    lr_patience=500,
    print_every=50,
    plot_every=500,
    early_stop_chi2=1.0,
    early_stop_epochs=3,
):
    """Train on visibilities; see loss_fn for the objective.

    Early stopping: chi2_vis is a reduced chi-squared (per real degree of
    freedom), so ~1.0 means the model fits the data at the noise level and
    pushing further only fits noise. Training stops once the epoch-mean
    chi2_vis stays at or below early_stop_chi2 for early_stop_epochs
    consecutive epochs (set early_stop_chi2=None to disable). Note that with
    an inflated error budget (e.g. fractional systematic noise added to
    sigma), the ground truth itself sits below 1 — lower the threshold if you
    want a tighter fit.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    scheduler = PlateauScheduler(initial_lr, factor=lr_factor, patience=lr_patience)
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=initial_lr, weight_decay=weight_decay
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    ckpt_path = os.path.join(
        models_dir, f"trained_model_r{model.r}_f{model.num_frequencies}.eqx"
    )
    history = {"total": [], "chi2_vis": [], "chi2_amp": [], "chi2_cp": [], "rec": []}
    best_loss = jnp.inf
    epochs_at_noise_level = 0

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            epoch_data = train_loader.get_epoch_data(epoch)
            model, opt_state, (loss, rec, chi2_vis, chi2_amp, chi2_cp) = train_epoch_jit(
                model, opt_state, epoch_data, optimizer, key, frame_max, frame_min
            )

            history["total"].append(float(loss))
            history["chi2_vis"].append(float(chi2_vis))
            history["chi2_amp"].append(float(chi2_amp))
            history["chi2_cp"].append(float(chi2_cp))
            history["rec"].append(float(rec))

            new_lr = scheduler.step(loss)
            # inject_hyperparams keeps the lr inside opt_state; updating it
            # here propagates through the jitted epoch as a traced leaf
            opt_state.hyperparams["learning_rate"] = jnp.asarray(
                new_lr, dtype=jnp.float32
            )
            pbar.set_postfix(
                loss=f"{float(loss):.4f}",
                chi2_vis=f"{float(chi2_vis):.3f}",
                chi2_cp=f"{float(chi2_cp):.3f}",
            )
            pbar.update(1)

            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}  loss={float(loss):.5f}  "
                    f"chi2_vis={float(chi2_vis):.3f}  chi2_amp={float(chi2_amp):.3f}  "
                    f"chi2_cp={float(chi2_cp):.3f}  rec={float(rec):.1f}  "
                    f"lr={scheduler.lr:.2e}",
                    flush=True,
                )

            if loss < best_loss:
                best_loss = loss
                eqx.tree_serialise_leaves(ckpt_path, model)

            if early_stop_chi2 is not None:
                if float(chi2_vis) <= early_stop_chi2:
                    epochs_at_noise_level += 1
                else:
                    epochs_at_noise_level = 0
                if epochs_at_noise_level >= early_stop_epochs:
                    print(
                        f"Early stop at epoch {epoch+1}: chi2_vis = "
                        f"{float(chi2_vis):.3f} <= {early_stop_chi2} for "
                        f"{early_stop_epochs} consecutive epochs (noise level "
                        f"reached; continuing would fit noise).",
                        flush=True,
                    )
                    break

            if epoch > 4 and (epoch + 1) % plot_every == 0:
                plot_losses(history, plots_dir)

    plot_losses(history, plots_dir)
    print(f"Best checkpoint saved to {ckpt_path}")
    return model, history
