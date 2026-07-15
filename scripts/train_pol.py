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
    ap.add_argument("--frac-pol", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--r", type=int, default=8, help="Number of complex DMD modes per Stokes")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--basis", default="circular", choices=["stokes", "circular"])
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-data", action="store_true", help="Reuse an existing data/ obs_dir")
    return ap.parse_args()


def main():
    args = parse_args()
    from neuraldmd import evaluation as ev
    from neuraldmd.data.generation import generate_polarized_dataset
    from neuraldmd.data.loader import PolarizedDMDDataLoader
    from neuraldmd.data.observations import ObsProducts
    from neuraldmd.polarized import PolarizedNeuralDMD
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
            data_dir, npix=args.npix, fov_uas=args.fov_uas, num_frames=args.num_frames,
            frac_pol=args.frac_pol, stokes=stokes, basis=args.basis, seed=args.seed,
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
    model = PolarizedNeuralDMD(
        stokes, r=args.r, key=jax.random.PRNGKey(args.seed),
        hidden_size=args.hidden_size, num_layers=args.num_layers,
    )

    extra = {"products": op.stokes} if args.basis == "circular" else {}
    print(f"Training ({args.basis} basis, {args.epochs} epochs) ...", flush=True)
    model, hist = train_polarized_model(
        model, loader, num_epochs=args.epochs, key=jax.random.PRNGKey(args.seed + 1),
        models_dir=str(out / "models"), frame_max=frame_max, frame_min=frame_min,
        basis=args.basis, initial_lr=args.lr, early_stop_chi2=1.0, print_every=200, **extra,
    )

    recon = ev.reconstruct_polarized_cubes(model, args.npix, truth["times"], frame_max, frame_min)
    nrmse = ev.polarized_nrmse(recon, truth_cubes)
    evpa_err = ev.evpa_error_deg(recon, truth_cubes)
    ev.plot_polarized_summary(recon, truth_cubes, str(out / "pol_summary.png"))

    final_chi2 = {k: hist["chi2"][k][-1] for k in loader.keys}
    metrics = {
        "basis": args.basis,
        "epochs_run": len(hist["total"]),
        "final_chi2": final_chi2,
        "nrmse": nrmse,
        "evpa_error_deg": evpa_err,
        "gate": {
            "chi2_in_0.8_1.2": all(0.8 <= v <= 1.2 for v in final_chi2.values()),
            "nrmse_QU_le_0.15": (nrmse["Q"] <= 0.15 and nrmse["U"] <= 0.15),
            "evpa_le_10deg": bool(evpa_err <= 10.0),
        },
    }
    (out / "m2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
