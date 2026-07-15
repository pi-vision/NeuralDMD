#!/usr/bin/env python
"""Milestone M2: train a polarized (I, Q, U) NeuralDMD on a synthetic dataset.

Generates a polarized m-ring + hot-spot dataset (or reuses one), trains
``PolarizedNeuralDMD`` with :func:`neuraldmd.training.train_polarized_model`
(simultaneous I/Q/U, circular basis by default), then evaluates against the
ground-truth cubes and checks the M2 gates. Writes metrics + a P/EVPA summary
figure. Run under the ``ndmd`` env; on GPU via ``slurm/train_pol.slurm``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--npix", type=int, default=50)
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--fov-uas", type=float, default=200.0)
    ap.add_argument("--frac-pol", type=float, default=0.2, help="ring linear pol fraction")
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--r", type=int, default=8, help="Number of complex DMD modes per Stokes")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--basis", default="circular", choices=["stokes", "circular"])
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-data", action="store_true", help="Reuse an existing data/ obs_dir")
    # Stokes-I disk-template pretraining (image prior)
    ap.add_argument("--no-pretrain", action="store_true", help="Skip the Stokes-I pretrain")
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--pretrain-lr", type=float, default=1e-4)
    ap.add_argument(
        "--pretrain-radius",
        type=float,
        default=1.0,
        help="disk-template radius as a multiple of the gyration radius",
    )
    # optimizer / LR schedule (exponential annealing when decay-rate < 1)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "adamax"])
    ap.add_argument("--lr-decay-rate", type=float, default=1.0, help="<1 enables exp decay")
    ap.add_argument("--lr-decay-steps", type=int, default=2000)
    # fractional-pol controls
    ap.add_argument("--scaling-ml", type=float, default=1.0, help="Cap on linear pol fraction")
    ap.add_argument("--outshift", type=float, default=2.0, help="m_l sigmoid bias (small init)")
    # polarization regularization / curriculum (anti-overfit levers)
    ap.add_argument(
        "--r-pol",
        type=int,
        default=None,
        help="DMD modes for the pol fields (default = --r; use < r to starve capacity)",
    )
    ap.add_argument(
        "--pol-frequencies",
        type=int,
        default=None,
        help="positional-encoding frequencies for the pol fields (spatial band-limit)",
    )
    ap.add_argument(
        "--pol-hidden",
        type=int,
        default=None,
        help="hidden width of the pol spatial MLPs (default = --hidden-size)",
    )
    ap.add_argument(
        "--pol-layers",
        type=int,
        default=None,
        help="depth of the pol spatial MLPs (default = --num-layers)",
    )
    ap.add_argument(
        "--pol-warmup-epochs",
        type=int,
        default=0,
        help="ramp the pol learning rate 0->1 over N epochs (I converges first)",
    )
    ap.add_argument(
        "--freeze-i-after",
        type=int,
        default=None,
        help="hard-freeze Stokes I from this epoch on (fit pol on a fixed I)",
    )
    # two-stage curriculum: RR/LL-only I fit (pol frozen), then pol on frozen I
    ap.add_argument(
        "--i-only-epochs",
        type=int,
        default=0,
        help="stage-A epochs fitting only RR,LL with pol frozen (0 = single joint stage)",
    )
    ap.add_argument(
        "--pol-lr",
        type=float,
        default=None,
        help="stage-B (pol) initial learning rate (default = --lr)",
    )
    # total-flux (lightcurve) anchor: no zero-spacing baseline measures this
    ap.add_argument(
        "--flux", type=float, default=None, help="known total flux [Jy] (anchor off=None)"
    )
    ap.add_argument("--flux-weight", type=float, default=1.0, help="total-flux anchor weight")
    # evaluation: also report metrics after restoring both cubes to this beam
    ap.add_argument("--blur-uas", type=float, default=15.0, help="metric beam FWHM [uas]")
    # per-product early stop: stop once ALL products <= this; <1 so images sharpen
    ap.add_argument("--early-stop-chi2", type=float, default=0.8)
    return ap.parse_args()


def main():
    args = parse_args()
    from neuraldmd import evaluation as ev
    from neuraldmd.data.generation import generate_polarized_dataset
    from neuraldmd.data.loader import PolarizedDMDDataLoader
    from neuraldmd.data.movies import save_movie_hdf5, to_ehtim_movie
    from neuraldmd.data.observations import ObsProducts
    from neuraldmd.polarized import PolarizedNeuralDMD
    from neuraldmd.pretraining import pretrain_stokes_i
    from neuraldmd.training import train_polarized_model

    print("jax devices:", jax.devices(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    stokes = ("I", "Q", "U")

    if args.reuse_data and (data_dir / "manifest.json").exists():
        print(f"Reusing dataset at {data_dir}", flush=True)
        op = ObsProducts.from_obs_dir(data_dir)
    else:
        print("Generating polarized dataset ...", flush=True)
        op = generate_polarized_dataset(
            data_dir,
            npix=args.npix,
            fov_uas=args.fov_uas,
            num_frames=args.num_frames,
            linpol_frac=args.frac_pol,
            stokes=stokes,
            basis=args.basis,
            seed=args.seed,
        )
    print(f"Dataset keys={op.stokes}  A={op.A.shape}", flush=True)

    truth = np.load(data_dir / "truth_pol.npz")
    truth_cubes = {s: truth[s] for s in stokes}
    frame_max = {
        "I": float(truth["I"].max()),
        "Q": float(np.abs(truth["Q"]).max()),
        "U": float(np.abs(truth["U"]).max()),
    }
    frame_min = {s: 0.0 for s in stokes}

    loader = PolarizedDMDDataLoader(
        op, npix=args.npix, batch_size=args.batch_size, epochs=args.epochs, fov_x=np.pi, fov_y=np.pi
    )
    # spatial band-limit for the pol fields: smaller/lower-frequency nets than I
    pol_kwargs = {
        k: v
        for k, v in {
            "num_frequencies": args.pol_frequencies,
            "hidden_size": args.pol_hidden,
            "num_layers": args.pol_layers,
        }.items()
        if v is not None
    }
    model = PolarizedNeuralDMD(
        stokes,
        r=args.r,
        key=jax.random.PRNGKey(args.seed),
        outshift=args.outshift,
        scaling_ml=args.scaling_ml,
        r_pol=args.r_pol,
        pol_model_kwargs=pol_kwargs or None,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    if not args.no_pretrain:
        print(f"Pretraining Stokes-I disk template ({args.pretrain_steps} steps) ...", flush=True)
        model, _ = pretrain_stokes_i(
            model,
            truth_cubes["I"],
            num_steps=args.pretrain_steps,
            lr=args.pretrain_lr,
            radius_scale=args.pretrain_radius,
            key=jax.random.PRNGKey(args.seed + 2),
        )

    common = dict(
        models_dir=str(out / "models"),
        frame_max=frame_max,
        frame_min=frame_min,
        basis=args.basis,
        optimizer_name=args.optimizer,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        flux_target=args.flux,
        flux_weight=args.flux_weight,
        early_stop_chi2=args.early_stop_chi2,
        print_every=200,
    )
    total_epochs = 0
    if args.i_only_epochs > 0:
        # stage A: fit Stokes I alone on the parallel hands; pol frozen at its
        # near-unpolarized init (no pol residual can corrupt I, and vice versa)
        if args.basis != "circular":
            raise SystemExit("--i-only-epochs requires --basis circular")
        print(f"Stage A: I-only (RR,LL), pol frozen, {args.i_only_epochs} epochs ...", flush=True)
        model, hist_a = train_polarized_model(
            model,
            loader,
            num_epochs=args.i_only_epochs,
            key=jax.random.PRNGKey(args.seed + 1),
            initial_lr=args.lr,
            products=("RR", "LL"),
            freeze_pol=True,
            **common,
        )
        total_epochs += len(hist_a["total"])
        print(f"Stage B: pol on frozen I (all products), {args.epochs} epochs ...", flush=True)
        model, hist = train_polarized_model(
            model,
            loader,
            num_epochs=args.epochs,
            key=jax.random.PRNGKey(args.seed + 3),
            initial_lr=args.pol_lr if args.pol_lr is not None else args.lr,
            products=op.stokes,
            freeze_intensity=True,
            **common,
        )
    else:
        extra = {"products": op.stokes} if args.basis == "circular" else {}
        print(
            f"Training ({args.basis} basis, {args.optimizer}, {args.epochs} epochs) ...", flush=True
        )
        model, hist = train_polarized_model(
            model,
            loader,
            num_epochs=args.epochs,
            key=jax.random.PRNGKey(args.seed + 1),
            initial_lr=args.lr,
            pol_warmup_epochs=args.pol_warmup_epochs,
            freeze_i_after=args.freeze_i_after,
            **common,
            **extra,
        )
    total_epochs += len(hist["total"])

    recon = ev.reconstruct_polarized_cubes(model, args.npix, truth["times"], frame_max, frame_min)
    nrmse = ev.polarized_nrmse(recon, truth_cubes)
    evpa_err = ev.evpa_error_deg(recon, truth_cubes)
    # beam-restored metrics: the data only constrain structure to ~the array
    # resolution, so also compare after blurring both cubes to a common beam
    recon_b = ev.blur_polarized_cubes(recon, args.blur_uas, args.fov_uas)
    truth_b = ev.blur_polarized_cubes(truth_cubes, args.blur_uas, args.fov_uas)
    nrmse_b = ev.polarized_nrmse(recon_b, truth_b)
    evpa_err_b = ev.evpa_error_deg(recon_b, truth_b)
    ev.plot_polarized_summary(
        recon, truth_cubes, str(out / "pol_summary.png"), fov_uas=args.fov_uas
    )

    # export the reconstruction as an ehtim HDF5 movie (I, Q, U) for video / scoring
    try:
        recon_movie = to_ehtim_movie(
            recon["I"].astype(np.float64),
            truth["times"],
            fov_uas=args.fov_uas,
            qframes=recon["Q"].astype(np.float64),
            uframes=recon["U"].astype(np.float64),
        )
        save_movie_hdf5(recon_movie, str(out / "recon_pol.hdf5"))
    except Exception as exc:
        print(f"[warn] reconstruction hdf5 export failed: {exc}", flush=True)

    # animated GIFs (reconstruction + truth) with EVPA ticks
    try:
        ev.make_polarized_gif(recon, str(out / "recon_pol.gif"), fov_uas=args.fov_uas)
        ev.make_polarized_gif(truth_cubes, str(out / "truth_pol.gif"), fov_uas=args.fov_uas)
    except Exception as exc:
        print(f"[warn] gif export failed: {exc}", flush=True)

    final_chi2 = {k: v[-1] for k, v in hist["chi2"].items()}
    metrics = {
        "basis": args.basis,
        "epochs_run": total_epochs,
        "final_chi2": final_chi2,
        "nrmse": nrmse,
        "evpa_error_deg": evpa_err,
        "blur_uas": args.blur_uas,
        "nrmse_blurred": nrmse_b,
        "evpa_error_deg_blurred": evpa_err_b,
        "gate": {
            "chi2_in_0.8_1.2": all(0.8 <= v <= 1.2 for v in final_chi2.values()),
            "nrmse_QU_le_0.15": (nrmse["Q"] <= 0.15 and nrmse["U"] <= 0.15),
            "evpa_le_10deg": bool(evpa_err <= 10.0),
            "nrmse_QU_blurred_le_0.15": (nrmse_b["Q"] <= 0.15 and nrmse_b["U"] <= 0.15),
            "evpa_blurred_le_10deg": bool(evpa_err_b <= 10.0),
        },
    }
    (out / "m2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
