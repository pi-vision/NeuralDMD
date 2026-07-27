#!/usr/bin/env python
"""Evaluate a base Stokes-I NeuralDMD run: PSNR + Total/Dynamic/Static panel.

Reconstructs the physical intensity cube the same way the base ``loss_fn`` does
(``(I_stat + I_dyn) * (frame_max - frame_min) + frame_min``, with the scaling
taken from the ground-truth cube = the data, since flux is fixed not fitted),
then reports the image-space PSNR against truth (NB04 convention) and writes the
Total / Dynamic / Static truth-vs-recon figure at one frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np


def reconstruct(model, truth: np.ndarray, times_norm: np.ndarray, fmax: float, fmin: float):
    """Reconstruct physical Total / Dynamic / Static Stokes-I from a base model.

    Parameters
    ----------
    model : NeuralDMD
        Trained base model.
    truth : np.ndarray
        ``(T, H, W)`` ground-truth cube (only its shape is used here).
    times_norm : np.ndarray
        ``(T,)`` frame times normalized to ``[0, 1]``.
    fmax, fmin : float
        Physical output scaling (from the data cube).

    Returns
    -------
    total : np.ndarray
        ``(T, H, W)`` physical reconstruction.
    dynamic : np.ndarray
        ``(T, H, W)`` time-varying part (physical, no static offset).
    static : np.ndarray
        ``(H, W)`` static mode (physical).
    """
    from neuraldmd.evaluation import pixel_grid_coords

    _, h, w = truth.shape
    xy = jnp.asarray(pixel_grid_coords(h, w))
    W0, Wm, Om, b0, b = (np.asarray(x) for x in model(xy))
    tsc = float(model.t_scale)
    lam = np.exp(Om[:, None] * times_norm[None, :] * tsc)  # (r, T)
    i_stat = W0[:, 0:1] * b0[0]  # (P, 1)
    i_dyn = 2 * np.real(np.einsum("pr,rt,r->pt", Wm, lam, b))  # (P, T)
    total = ((i_stat + i_dyn) * (fmax - fmin) + fmin).T.reshape(-1, h, w)
    dynamic = (i_dyn * (fmax - fmin)).T.reshape(-1, h, w)
    static = (i_stat[:, 0] * (fmax - fmin) + fmin).reshape(h, w)
    return total, dynamic, static


def main() -> None:
    """Score one run and write its Total/Dynamic/Static figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neuraldmd.evaluation import calc_psnr
    from neuraldmd.model import NeuralDMD

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="run dir (expects models/trained_model_r{r}_f{f}.eqx)")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth cube hdf5 (datasets I, times)")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--frequencies", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None, help="output png (default <run>/eval_tds.png)")
    ap.add_argument("--frame", type=int, default=None, help="frame index for the panel (default T//4)")
    args = ap.parse_args()

    sk = NeuralDMD(args.r, key=jax.random.PRNGKey(0), num_frequencies=args.frequencies)
    ckpt = args.run / "models" / f"trained_model_r{args.r}_f{args.frequencies}.eqx"
    model = eqx.tree_deserialise_leaves(str(ckpt), sk)

    with h5py.File(args.gt, "r") as f:
        truth = np.asarray(f["I"][:], float)
        times = np.asarray(f["times"][:], float)
    tn = (times - times.min()) / (times.max() - times.min())
    fmax, fmin = float(truth.max()), float(truth.min())

    total, dynamic, static = reconstruct(model, truth, tn, fmax, fmin)
    norm = lambda x: (x - fmin) / (fmax - fmin)  # noqa: E731
    psnr = float(np.mean([calc_psnr(norm(total[i]), norm(truth[i])) for i in range(len(truth))]))
    print(f"{args.run.name}: r={args.r} f={args.frequencies}  PSNR = {psnr:.2f} dB")

    idx = args.frame if args.frame is not None else len(truth) // 4
    t_stat = truth.mean(0)
    t_dyn = truth[idx] - t_stat
    dv = float(max(np.abs(t_dyn).max(), np.abs(dynamic[idx]).max()))
    fig, ax = plt.subplots(2, 3, figsize=(10, 6.6))
    rows = [(truth[idx], t_dyn, t_stat, "truth"), (total[idx], dynamic[idx], static, "recon")]
    ik = dict(origin="upper", interpolation="bicubic")  # no display pixelation
    for row, (tot_i, dyn_i, sta_i, lbl) in enumerate(rows):
        ax[row, 0].imshow(tot_i, cmap="afmhot", vmin=0, vmax=fmax, **ik)
        ax[row, 0].set_ylabel(lbl, fontsize=12)
        ax[row, 1].imshow(dyn_i, cmap="coolwarm", vmin=-dv, vmax=dv, **ik)
        ax[row, 2].imshow(sta_i, cmap="afmhot", vmin=0, vmax=fmax, **ik)
    for j, t in enumerate(["Total", "Dynamic", "Static"]):
        ax[0, j].set_title(t)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{args.run.name}  r={args.r} f={args.frequencies}  PSNR {psnr:.1f} dB  t={times[idx]:.2f}h")
    fig.tight_layout()
    out = args.out or (args.run / "eval_tds.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
