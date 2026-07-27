#!/usr/bin/env python
"""Side-by-side: Ali's original Stokes-I pipeline vs ours, each on its own data.

Builds a 4-row figure (his truth / his reconstruction / our truth / our
reconstruction) and a 2x2 animation, all on a common render grid so the panels
are directly comparable.

Display follows ehtim: ``ehtim.Image.display`` calls ``imshow(imarr)`` with no
``origin``, i.e. matplotlib's default ``origin='upper'``, so that is used here.
Both truth and reconstruction go through the same transform, so the comparison
is unaffected by the choice -- only the on-sky orientation is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np


def reconstruct(ckpt: Path, r: int, freqs: int, times_norm: np.ndarray, fmax: float, fmin: float, npix: int):
    """Evaluate a trained base model on an ``npix`` grid over the full field.

    Parameters
    ----------
    ckpt : Path
        ``trained_model_r{r}_f{freqs}.eqx``.
    r, freqs : int
        Model size (to rebuild the skeleton).
    times_norm : np.ndarray
        ``(T,)`` frame times normalized to ``[0, 1]``.
    fmax, fmin : float
        Physical output scaling, taken from the data cube.
    npix : int
        Render resolution (the model is continuous).

    Returns
    -------
    np.ndarray
        ``(T, npix, npix)`` physical reconstruction.
    """
    from neuraldmd.evaluation import pixel_grid_coords
    from neuraldmd.model import NeuralDMD

    model = eqx.tree_deserialise_leaves(
        str(ckpt), NeuralDMD(r, key=jax.random.PRNGKey(0), num_frequencies=freqs)
    )
    W0, Wm, Om, b0, b = (np.asarray(x) for x in model(jnp.asarray(pixel_grid_coords(npix, npix))))
    lam = np.exp(Om[:, None] * times_norm[None, :] * float(model.t_scale))
    cube = ((W0[:, 0:1] * b0[0]) + 2 * np.real(np.einsum("pr,rt,r->pt", Wm, lam, b))) * (fmax - fmin) + fmin
    return cube.T.reshape(len(times_norm), npix, npix)


def load_pair(gt: Path, ckpt: Path, r: int, freqs: int, npix: int):
    """Load a truth cube and its reconstruction, both resampled to ``npix``.

    Parameters
    ----------
    gt : Path
        Ground-truth cube hdf5 (datasets ``I``, ``times``).
    ckpt : Path
        Trained checkpoint for this dataset.
    r, freqs, npix : int
        Model size and render resolution.

    Returns
    -------
    truth, recon : np.ndarray
        ``(T, npix, npix)`` each.
    """
    from scipy.ndimage import zoom

    with h5py.File(gt, "r") as f:
        truth = np.asarray(f["I"][:], float)
        times = np.asarray(f["times"][:], float)
    tn = (times - times.min()) / (times.max() - times.min())
    fmax, fmin = float(truth.max()), float(truth.min())
    recon = reconstruct(ckpt, r, freqs, tn, fmax, fmin, npix)
    k = npix / truth.shape[1]
    truth_r = np.stack([zoom(f, k, order=3) for f in truth]) if k != 1 else truth
    return truth_r, recon


def main() -> None:
    """Write the 4-row comparison figure and the 2x2 gif."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neuraldmd.evaluation import make_gif

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ali-gt", type=Path, required=True)
    ap.add_argument("--ali-ckpt", type=Path, required=True)
    ap.add_argument("--ali-r", type=int, default=10)
    ap.add_argument("--ali-f", type=int, default=2)
    ap.add_argument("--our-gt", type=Path, required=True)
    ap.add_argument("--our-ckpt", type=Path, required=True)
    ap.add_argument("--our-r", type=int, default=16)
    ap.add_argument("--our-f", type=int, default=4)
    ap.add_argument("--npix", type=int, default=128)
    ap.add_argument("--frames", type=int, nargs="*", default=[80, 200, 330])
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--ali-label", default="Ali r=10 f=2")
    ap.add_argument("--our-label", default="ours r=16 f=4")
    args = ap.parse_args()

    n = args.npix
    at, ar = load_pair(args.ali_gt, args.ali_ckpt, args.ali_r, args.ali_f, n)
    ot, orc = load_pair(args.our_gt, args.our_ckpt, args.our_r, args.our_f, n)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = [
        (at, f"truth\n({args.ali_label} data)"),
        (ar, f"recon\n{args.ali_label}"),
        (ot, f"truth\n({args.our_label} data)"),
        (orc, f"recon\n{args.our_label}"),
    ]
    fr = [i for i in args.frames if i < min(len(at), len(ot))]
    fig, ax = plt.subplots(4, 1 + len(fr), figsize=(3.0 * (1 + len(fr)), 12.2))
    for row, (cube, lbl) in enumerate(rows):
        vmax = float(np.clip(cube, 0, None).max())
        ax[row, 0].imshow(
            np.clip(cube.mean(0), 0, None), cmap="afmhot", origin="upper", interpolation="bicubic", vmax=vmax
        )
        ax[row, 0].set_ylabel(lbl, fontsize=10)
        if row == 0:
            ax[row, 0].set_title("mean I")
        for k, i in enumerate(fr):
            ax[row, 1 + k].imshow(
                np.clip(cube[i], 0, None), cmap="afmhot", origin="upper", interpolation="bicubic", vmax=vmax
            )
            if row == 0:
                ax[row, 1 + k].set_title(f"frame {i}")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Stokes-I: Ali's original pipeline vs ours (each on its own data)", fontsize=13)
    fig.tight_layout()
    out = args.outdir / "pipelines_compare.png"
    fig.savefig(out, dpi=300)

    # 2x2 animation: [his truth | his recon ; our truth | our recon]
    tn = min(len(at), len(ot))
    norm = lambda c: np.clip(c, 0, None) / max(float(np.clip(c, 0, None).max()), 1e-12)  # noqa: E731
    tiled = np.concatenate(
        [
            np.concatenate([norm(at)[:tn], norm(ar)[:tn]], axis=2),
            np.concatenate([norm(ot)[:tn], norm(orc)[:tn]], axis=2),
        ],
        axis=1,
    )
    make_gif(tiled, str(args.outdir / "pipelines_compare.gif"), fps=25, vmin=0, vmax=1)
    print(f"wrote {out} and {args.outdir / 'pipelines_compare.gif'}")
    print("gif layout:  top = Ali (truth | recon),  bottom = ours (truth | recon)")


if __name__ == "__main__":
    main()
