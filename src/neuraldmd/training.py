"""Training loop: jitted step/epoch, an LR-on-plateau scheduler, and the driver."""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from .losses import loss_fn, polarized_loss_fn


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
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
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


def make_polarized_optimizer(
    model,
    initial_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    *,
    optimizer: str = "adamw",
    lr_decay_rate: float = 1.0,
    lr_decay_steps: int = 1000,
):
    """Build the optimizer for a :class:`PolarizedNeuralDMD` (single transform).

    Intensity freezing (the hierarchical study mode) is applied in
    :func:`polarized_train_step` via ``freeze_intensity`` -- the Stokes-I
    sub-model's update is zeroed there, an exact freeze including weight decay.

    Parameters
    ----------
    model : PolarizedNeuralDMD
        Model whose parameter tree the optimizer state is shaped to.
    initial_lr : float
        Initial learning rate.
    weight_decay : float
        Decoupled weight decay (``adamw`` only).
    optimizer : {"adamw", "adam", "adamax"}
        Update rule (``"adamax"`` is often more robust for imaging fields).
    lr_decay_rate : float
        If < 1.0, use ``optax.exponential_decay(initial_lr, lr_decay_steps,
        lr_decay_rate)`` (per-step annealing); 1.0 keeps the LR constant.
    lr_decay_steps : int
        Transition steps for the exponential decay.

    Returns
    -------
    optax.GradientTransformation
        Initialize with ``opt.init(eqx.filter(model, eqx.is_array))``.
    """
    lr = (
        optax.exponential_decay(initial_lr, lr_decay_steps, lr_decay_rate)
        if lr_decay_rate < 1.0
        else initial_lr
    )
    if optimizer == "adamax":
        return optax.adamax(learning_rate=lr)
    if optimizer == "adam":
        return optax.adam(learning_rate=lr)
    if optimizer == "adamw":
        return optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def _zero_intensity_update(updates):
    """Zero the update for the Stokes-I (``intensity``) sub-model (exact freeze)."""
    zeroed = jax.tree_util.tree_map(jnp.zeros_like, updates.intensity)
    return eqx.tree_at(lambda u: u.intensity, updates, zeroed)


def _scale_pol_update(updates, factor):
    """Scale the polarization sub-models' updates by ``factor`` in ``[0, 1]``.

    A soft learning-rate warmup/freeze for the fractional-pol fields (``frac``,
    ``cos2xi``, ``sin2xi``, and ``circ`` when present) while Stokes I converges:
    ``factor=0`` freezes polarization, ``factor=1`` trains it at the full rate.

    Parameters
    ----------
    updates : PolarizedNeuralDMD
        Filtered optimizer-update pytree (same structure as the model params).
    factor : float or jax.Array
        Multiplicative scale applied to every polarization-field leaf.

    Returns
    -------
    PolarizedNeuralDMD
        ``updates`` with the polarization sub-model leaves scaled by ``factor``.
    """

    def scaled(sub):
        return jax.tree_util.tree_map(lambda x: x * factor, sub)

    updates = eqx.tree_at(lambda u: u.frac, updates, scaled(updates.frac))
    updates = eqx.tree_at(lambda u: u.cos2xi, updates, scaled(updates.cos2xi))
    updates = eqx.tree_at(lambda u: u.sin2xi, updates, scaled(updates.sin2xi))
    if updates.circ is not None:
        updates = eqx.tree_at(lambda u: u.circ, updates, scaled(updates.circ))
    return updates


@eqx.filter_jit
def polarized_train_step(
    model,
    opt_state,
    xy,
    targets,
    sigmas,
    masks,
    A_batch,
    time_indices,
    optimizer,
    frame_max,
    frame_min,
    *,
    freeze_intensity: bool = False,
    pol_scale=1.0,
    basis: str = "stokes",
    products: tuple[str, ...] = ("RR", "LL", "RL", "LR"),
    neg_weight: float = 1.0,
    w_sparse_weight: float = 1.0,
    b_sparse_weight: float = 1.0,
    p_le_i_weight: float = 0.0,
):
    """One AdamW step on a :class:`PolarizedNeuralDMD` via :func:`polarized_loss_fn`.

    Differentiates the loss with respect to the whole model pytree. The
    polarization-field updates are scaled by ``pol_scale`` (a soft LR
    warmup/freeze) and, when ``freeze_intensity`` is set, the Stokes-I
    (``intensity``) sub-model's update is zeroed (held fixed) -- the hierarchical
    study mode (fit polarization on a frozen I). Pass ``pol_scale`` as a JAX
    scalar so the ramp does not retrigger ``eqx.filter_jit``; the other
    keyword-only arguments are static.

    Returns
    -------
    model, opt_state, loss, aux
        The updated model and optimizer state, the scalar loss, and the
        :func:`polarized_loss_fn` aux dict.
    """

    def loss_wrap(m):
        return polarized_loss_fn(
            m,
            xy,
            targets,
            sigmas,
            masks,
            A_batch,
            time_indices,
            frame_max,
            frame_min,
            basis=basis,
            products=products,
            neg_weight=neg_weight,
            w_sparse_weight=w_sparse_weight,
            b_sparse_weight=b_sparse_weight,
            p_le_i_weight=p_le_i_weight,
        )

    (loss, aux), grads = eqx.filter_value_and_grad(loss_wrap, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    updates = _scale_pol_update(updates, pol_scale)  # soft pol LR warmup/freeze
    if freeze_intensity:
        updates = _zero_intensity_update(updates)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def polarized_train_epoch(
    model,
    opt_state,
    epoch_data,
    optimizer,
    key,
    frame_max,
    frame_min,
    *,
    freeze_intensity: bool = False,
    pol_scale=1.0,
    basis: str = "stokes",
    products: tuple[str, ...] = ("RR", "LL", "RL", "LR"),
    neg_weight: float = 1.0,
    w_sparse_weight: float = 1.0,
    b_sparse_weight: float = 1.0,
    p_le_i_weight: float = 0.0,
):
    """Scan :func:`polarized_train_step` over one epoch's batches.

    ``epoch_data`` is a
    ``(pixel_coords, As, targets, sigmas, masks, times)`` tuple from
    :meth:`neuraldmd.data.loader.PolarizedDMDDataLoader.get_epoch_data`, with the
    per-key dicts batched along a leading batch axis. Coordinates get small jitter
    each step (as in :func:`train_epoch_jit`).

    Returns
    -------
    model, opt_state, mean_loss, mean_chi2
        ``mean_chi2`` is a dict keyed like the data (Stokes or product).
    """
    pixel_coords, as_b, tgt_b, sig_b, msk_b, times_b = epoch_data
    keys = tuple(tgt_b.keys())

    def scan_fn(carry, i):
        model, opt_state, key = carry
        key, subkey = jax.random.split(key)
        xy = pixel_coords + jax.random.normal(subkey, shape=pixel_coords.shape) * 0.01
        model, opt_state, loss, aux = polarized_train_step(
            model,
            opt_state,
            xy,
            {k: tgt_b[k][i] for k in keys},
            {k: sig_b[k][i] for k in keys},
            {k: msk_b[k][i] for k in keys},
            as_b[i],
            times_b[i],
            optimizer,
            frame_max,
            frame_min,
            freeze_intensity=freeze_intensity,
            pol_scale=pol_scale,
            basis=basis,
            products=products,
            neg_weight=neg_weight,
            w_sparse_weight=w_sparse_weight,
            b_sparse_weight=b_sparse_weight,
            p_le_i_weight=p_le_i_weight,
        )
        return (model, opt_state, key), (loss, aux["chi2_vis"])

    n_batches = as_b.shape[0]
    (model, opt_state, _), (loss_log, chi2_log) = jax.lax.scan(
        scan_fn, (model, opt_state, key), jnp.arange(n_batches)
    )
    mean_chi2 = {k: jnp.mean(v) for k, v in chi2_log.items()}
    return model, opt_state, jnp.mean(loss_log), mean_chi2


def train_polarized_model(
    model,
    loader,
    num_epochs,
    key,
    models_dir,
    frame_max,
    frame_min,
    *,
    basis: str = "stokes",
    products: tuple[str, ...] = ("RR", "LL", "RL", "LR"),
    freeze_intensity: bool = False,
    pol_warmup_epochs: int = 0,
    freeze_i_after: int | None = None,
    initial_lr: float = 3e-4,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    lr_decay_rate: float = 1.0,
    lr_decay_steps: int = 1000,
    neg_weight: float = 1.0,
    w_sparse_weight: float = 1.0,
    b_sparse_weight: float = 1.0,
    p_le_i_weight: float = 0.0,
    print_every: int = 50,
    early_stop_chi2: float | None = None,
    early_stop_epochs: int = 3,
    fold_epoch_key: bool = True,
):
    """Train a :class:`PolarizedNeuralDMD` on a
    :class:`~neuraldmd.data.loader.PolarizedDMDDataLoader`.

    Objective is :func:`polarized_loss_fn` in the chosen ``basis``. Early stopping
    triggers when EVERY per-key chi-squared stays at or below ``early_stop_chi2``
    for ``early_stop_epochs`` consecutive epochs (``None`` disables). Use a
    threshold below 1 when systematic noise inflates sigma -- the truth then sits
    at chi^2 < 1, so stopping at 1 leaves the images under-sharpened. The best
    (lowest worst-product chi^2) model is checkpointed and returned.

    ``pol_warmup_epochs`` linearly ramps the polarization learning rate from 0 to
    full over the first epochs so Stokes I converges before polarization is fit;
    ``freeze_i_after`` additionally hard-freezes Stokes I from that epoch onward
    (fit polarization on a fixed I, as in a staged reconstruction).

    Returns
    -------
    model, history
        ``history`` has ``"total"`` (list) and ``"chi2"`` (dict of per-key lists).
    """
    os.makedirs(models_dir, exist_ok=True)
    optimizer = make_polarized_optimizer(
        model,
        initial_lr=initial_lr,
        weight_decay=weight_decay,
        optimizer=optimizer_name,
        lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    ckpt_path = os.path.join(models_dir, "polarized_model.eqx")

    history = {"total": [], "chi2": {k: [] for k in loader.keys}}
    best_metric = float("inf")
    at_noise = 0

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            epoch_data = loader.get_epoch_data(epoch)
            epoch_key = jax.random.fold_in(key, epoch) if fold_epoch_key else key
            # pol-LR curriculum: ramp polarization updates 0 -> 1 over the warmup
            # so Stokes I converges first, then fit polarization at full rate
            pol_scale = jnp.asarray(
                min(1.0, (epoch + 1) / pol_warmup_epochs) if pol_warmup_epochs > 0 else 1.0,
                dtype=jnp.float32,
            )
            # optional hard freeze of the (by now converged) I during the pol tail
            freeze_i = freeze_intensity or (freeze_i_after is not None and epoch >= freeze_i_after)
            model, opt_state, loss, chi2 = polarized_train_epoch(
                model,
                opt_state,
                epoch_data,
                optimizer,
                epoch_key,
                frame_max,
                frame_min,
                freeze_intensity=freeze_i,
                pol_scale=pol_scale,
                basis=basis,
                products=products,
                neg_weight=neg_weight,
                w_sparse_weight=w_sparse_weight,
                b_sparse_weight=b_sparse_weight,
                p_le_i_weight=p_le_i_weight,
            )
            history["total"].append(float(loss))
            for k in loader.keys:
                history["chi2"][k].append(float(chi2[k]))
            mean_chi2 = float(sum(chi2.values()) / len(chi2))
            max_chi2 = float(max(chi2.values()))

            pbar.set_postfix(loss=f"{float(loss):.4f}", chi2=f"{mean_chi2:.3f}")
            pbar.update(1)
            if (epoch + 1) % print_every == 0:
                per_key = "  ".join(f"chi2_{k}={float(chi2[k]):.3f}" for k in loader.keys)
                print(
                    f"Epoch {epoch + 1}/{num_epochs}  loss={float(loss):.5f}  {per_key}",
                    flush=True,
                )

            # track the best on the worst-product chi2 so the saved model targets
            # the gate (all chi2 -> 1), not an epoch where the polarized products
            # merely overfit below the noise floor while I is still poor
            if max_chi2 < best_metric:
                best_metric = max_chi2
                eqx.tree_serialise_leaves(ckpt_path, model)

            if early_stop_chi2 is not None:
                # per-product: every key must be at/below threshold (not the mean),
                # so I keeps sharpening even after the pol products reach the floor
                at_noise = at_noise + 1 if max_chi2 <= early_stop_chi2 else 0
                if at_noise >= early_stop_epochs:
                    print(
                        f"Early stop at epoch {epoch + 1}: all chi2 <= {early_stop_chi2}",
                        flush=True,
                    )
                    break

    # evaluate/return the BEST checkpoint, not the last: late-stage LR noise makes
    # the final epoch a poor snapshot, and the driver evaluates the returned model
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    print(f"Best checkpoint (max chi2 {best_metric:.3f}) restored from {ckpt_path}")
    return model, history


class PlateauScheduler:
    """Reduce the LR by ``factor`` after ``patience`` non-improving epochs.

    Note: the default ``factor=1.0`` makes this a **no-op** (``new_lr == lr`` so
    the ``new_lr < lr`` guard never fires; the LR is held constant). Set
    ``factor < 1.0`` (e.g. 0.5) to actually anneal on plateau.
    """

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
    fold_epoch_key=True,
):
    """Train on visibilities; see loss_fn for the objective.

    Early stopping: chi2_vis is a reduced chi-squared (per real degree of
    freedom), so ~1.0 means the model fits the data at the noise level and
    pushing further only fits noise. Training stops once the epoch-mean chi2_vis
    stays at or below early_stop_chi2 for early_stop_epochs consecutive epochs
    (set early_stop_chi2=None to disable). Note that with an inflated error
    budget (e.g. fractional systematic noise added to sigma), the ground truth
    itself sits below 1 -- lower the threshold if you want a tighter fit.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    scheduler = PlateauScheduler(initial_lr, factor=lr_factor, patience=lr_patience)
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=initial_lr, weight_decay=weight_decay
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    ckpt_path = os.path.join(models_dir, f"trained_model_r{model.r}_f{model.num_frequencies}.eqx")
    history = {"total": [], "chi2_vis": [], "chi2_amp": [], "chi2_cp": [], "rec": []}
    best_loss = jnp.inf
    epochs_at_noise_level = 0

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            epoch_data = train_loader.get_epoch_data(epoch)
            # Fold the epoch into the key so each epoch's coordinate-jitter RNG
            # stream is distinct (previously the same key was reused every epoch).
            epoch_key = jax.random.fold_in(key, epoch) if fold_epoch_key else key
            model, opt_state, (loss, rec, chi2_vis, chi2_amp, chi2_cp) = train_epoch_jit(
                model, opt_state, epoch_data, optimizer, epoch_key, frame_max, frame_min
            )

            history["total"].append(float(loss))
            history["chi2_vis"].append(float(chi2_vis))
            history["chi2_amp"].append(float(chi2_amp))
            history["chi2_cp"].append(float(chi2_cp))
            history["rec"].append(float(rec))

            new_lr = scheduler.step(loss)
            # inject_hyperparams keeps the lr inside opt_state; updating it here
            # propagates through the jitted epoch as a traced leaf
            opt_state.hyperparams["learning_rate"] = jnp.asarray(new_lr, dtype=jnp.float32)
            pbar.set_postfix(
                loss=f"{float(loss):.4f}",
                chi2_vis=f"{float(chi2_vis):.3f}",
                chi2_cp=f"{float(chi2_cp):.3f}",
            )
            pbar.update(1)

            if (epoch + 1) % print_every == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}  loss={float(loss):.5f}  "
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
                        f"Early stop at epoch {epoch + 1}: chi2_vis = "
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
