#!/usr/bin/env python
"""WS-B Stage 1: the Stokes-I-only, self-calibrated baseline.

Reproduces the base NeuralDMD Stokes-I recipe (tutorial NB01-04) as a single
CLI, on **self-calibrated** synthetic visibilities generated from a movie
(``Config.rlgaincal=True`` by default). This is the clean anchor for the
incremental rebuild: the base ``loss_fn`` carries no compact / dyn-compact
terms, so a clean ring here with *only* early stopping establishes the true
minimal baseline before gains and polarization are added in later stages.

Pipeline: generate (if needed) -> Zernike disk-template pretrain -> train_model
with early stopping at the noise level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import h5py
import jax
import numpy as np


def load_gt_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a ground-truth Stokes-I movie hdf5 as ``(frames, times)``.

    Parameters
    ----------
    path : Path
        HDF5 written by :func:`neuraldmd.data.generation.save_movie` (datasets
        ``frames``/``images``/``I`` for the cube and ``times`` for the clock).

    Returns
    -------
    frames : np.ndarray
        ``(T, H, W)`` real Stokes-I cube.
    times : np.ndarray
        ``(T,)`` frame times.
    """
    with h5py.File(path, "r") as f:
        cube_key = next(k for k in ("frames", "images", "I", "movie") if k in f)
        frames = np.asarray(f[cube_key][:], dtype=np.float64)
        times = np.asarray(f["times"][:], dtype=np.float64)
    return frames, times


def main() -> None:
    """Run the Stage-1 Stokes-I baseline end to end."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs-dir", type=Path, required=True, help="observation directory (obs products)")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth Stokes-I cube hdf5 (datasets I, times)")
    ap.add_argument("--out", type=Path, required=True, help="output dir for models/plots")
    ap.add_argument("--r", type=int, default=10)
    ap.add_argument("--frequencies", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--time-fraction", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--early-stop-chi2", type=float, default=1.0, help="<=0 disables early stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from neuraldmd.data.loader import DMDDataLoader
    from neuraldmd.model import NeuralDMD
    from neuraldmd.pretraining import pretrain_model, radius_of_gyration, save_template
    from neuraldmd.training import train_model
    from neuraldmd.zernike import build_zernike_targets

    fov = np.pi
    models_dir = args.out / "models"
    plots_dir = args.out / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    frames, times = load_gt_cube(args.gt)
    height, width = frames.shape[1:]
    frame_max, frame_min = float(frames.max()), float(frames.min())
    times = (times - times.min()) / (times.max() - times.min())
    print(f"[stage1] cube {frames.shape}  I in [{frame_min:.3g}, {frame_max:.3g}]")

    # ---- Zernike disk-template pretrain (NB02) ----
    key = jax.random.PRNGKey(args.seed)
    rg, _ = radius_of_gyration(frames, fov_x=fov, fov_y=fov)
    r_disk = 1.5 * rg
    z_targets, picked, _, xy = build_zernike_targets(
        height, width, r_disk, fov, fov, args.r + 1, max_n=8, prefer_ms=(0, 1, 2, 3)
    )
    print(f"[stage1] R_g={rg:.3f} R_disk={r_disk:.3f}  Zernike targets {z_targets.shape}")
    model = NeuralDMD(args.r, key=key, num_frequencies=args.frequencies)
    model, _ = pretrain_model(model, xy, z_targets, num_steps=args.pretrain_steps, lr=1e-4, key=key)
    save_template(model, str(models_dir))

    # ---- train Stokes-I on the self-cal visibilities (NB03) ----
    train_loader = DMDDataLoader(
        frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        data_dir=str(args.obs_dir),
        times=times,
        fov_x=fov,
        fov_y=fov,
        time_fraction=args.time_fraction,
    )
    model, history = train_model(
        model,
        train_loader,
        args.epochs,
        key,
        models_dir=str(models_dir),
        plots_dir=str(plots_dir),
        frame_max=frame_max,
        frame_min=frame_min,
        initial_lr=args.lr,
        print_every=100,
        plot_every=500,
        early_stop_chi2=(args.early_stop_chi2 if args.early_stop_chi2 > 0 else None),
        early_stop_epochs=3,
    )
    eqx.tree_serialise_leaves(str(models_dir / f"trained_model_r{args.r}_f{args.frequencies}.eqx"), model)
    print(f"[stage1] done; final chi2_vis ~ {history.get('chi2_vis', ['?'])[-1] if history else '?'}")


if __name__ == "__main__":
    main()
