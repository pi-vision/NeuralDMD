#!/usr/bin/env python
"""Score Stokes-I runs on the things PSNR misses: halo, variability, orbit.

PSNR is dominated by the bright ring and barely sees off-source haze or a
weak hot spot, so it ranked a visibly worse reconstruction higher (see
PROGRESS 2026-07-23). This reports, per run:

``psnr``      image-space PSNR against truth (NB04 convention)
``halo``      fraction of reconstructed flux outside the source support
              (support taken from the truth mean; truth's own value is the floor)
``dyn_amp``   on-ring temporal variability amplitude, relative to truth (1.0 = matched)
``period``    period of the highest-|b| mode, in minutes (the recovered orbit)

Run it on several runs at once to rank a sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np


def score(run: Path, truth: np.ndarray, times: np.ndarray, r: int, freqs: int) -> dict:
    """Compute halo / dynamic-amplitude / period / PSNR for one run.

    Parameters
    ----------
    run : Path
        Run directory containing ``models/trained_model_r{r}_f{freqs}.eqx``.
    truth : np.ndarray
        ``(T, H, W)`` ground-truth cube on the model grid.
    times : np.ndarray
        ``(T,)`` frame times in hours.
    r, freqs : int
        Model size, needed to rebuild the skeleton.

    Returns
    -------
    dict
        ``psnr``, ``halo``, ``dyn_amp``, ``period_min``, ``chi_note``.
    """
    from neuraldmd.evaluation import calc_psnr, pixel_grid_coords
    from neuraldmd.model import NeuralDMD

    sk = NeuralDMD(r, key=jax.random.PRNGKey(0), num_frequencies=freqs)
    model = eqx.tree_deserialise_leaves(str(run / "models" / f"trained_model_r{r}_f{freqs}.eqx"), sk)

    t, h, w = truth.shape
    tn = (times - times.min()) / (times.max() - times.min())
    fmax, fmin, tsc = float(truth.max()), float(truth.min()), float(model.t_scale)

    W0, Wm, Om, b0, b = (np.asarray(x) for x in model(jnp.asarray(pixel_grid_coords(h, w))))
    lam = np.exp(Om[:, None] * tn[None, :] * tsc)
    dyn = 2 * np.real(np.einsum("pr,rt,r->pt", Wm, lam, b))
    total = (((W0[:, 0:1] * b0[0]) + dyn) * (fmax - fmin) + fmin).T.reshape(t, h, w)
    dyn_phys = (dyn * (fmax - fmin)).T.reshape(t, h, w)

    tmean = truth.mean(0)
    off = tmean < 0.10 * tmean.max()  # off-source support
    on = tmean > 0.50 * tmean.max()  # bright ring

    pos = np.clip(total, 0, None).mean(0)
    halo = float(pos[off].sum() / max(pos.sum(), 1e-12))
    dyn_amp = float(dyn_phys[:, on].std() / max(truth[:, on].std(), 1e-12))

    norm = lambda x: (x - fmin) / (fmax - fmin)  # noqa: E731
    psnr = float(np.mean([calc_psnr(norm(total[i]), norm(truth[i])) for i in range(t)]))

    window_hr = float(times[-1] - times[0])
    omega = Om.imag * tsc / window_hr  # rad/hr
    j = int(np.argmax(np.abs(b)))
    period = float(2 * np.pi / max(abs(omega[j]), 1e-9) * 60)
    return {"psnr": psnr, "halo": halo, "dyn_amp": dyn_amp, "period_min": period}


def main() -> None:
    """Score every run given on the command line and print a ranked table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--gt", type=Path, required=True, help="truth cube on the model grid")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--frequencies", type=int, default=4)
    ap.add_argument("--truth-period-min", type=float, default=80.0, help="injected orbit, for reference")
    args = ap.parse_args()

    with h5py.File(args.gt, "r") as f:
        truth = np.asarray(f["I"][:], float)
        times = np.asarray(f["times"][:], float)

    tmean = truth.mean(0)
    off = tmean < 0.10 * tmean.max()
    print(f"{'run':22s} {'PSNR':>6s} {'halo':>7s} {'dyn_amp':>8s} {'period_min':>11s}")
    print(f"{'TRUTH (reference)':22s} {'--':>6s} {tmean[off].sum() / tmean.sum():7.3f} {1.0:8.2f} {args.truth_period_min:11.1f}")
    rows = []
    for run in args.runs:
        try:
            s = score(run, truth, times, args.r, args.frequencies)
        except Exception as exc:  # noqa: BLE001
            print(f"{run.name:22s} skipped: {exc}")
            continue
        rows.append((run.name, s))
        print(f"{run.name:22s} {s['psnr']:6.2f} {s['halo']:7.3f} {s['dyn_amp']:8.2f} {s['period_min']:11.1f}")
    if rows:
        best = max(rows, key=lambda kv: kv[1]["dyn_amp"])
        print(f"\nhighest dynamic amplitude: {best[0]} ({best[1]['dyn_amp']:.2f})")


if __name__ == "__main__":
    main()
