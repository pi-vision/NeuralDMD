#!/usr/bin/env python
"""Publication-quality Stokes-I render: continuous model vs full-res truth.

The model is a continuous coordinate network, so it can be evaluated on any grid
regardless of the (coarser) training grid. This renders it at ``--render-npix``
over a ``--fov-uas`` field of view, next to the full-resolution ground-truth
movie cropped to the same FOV, with bicubic interpolation, the ``afmhot``
colormap, and dpi=300. Use ``scripts/eval_stokesI.py`` for the PSNR number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    """Render the truth-vs-recon comparison figure and a smooth gif."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom

    from neuraldmd.evaluation import make_gif, pixel_grid_coords
    from neuraldmd.model import NeuralDMD

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gt-fullres", type=Path, required=True, help="full-res truth hdf5 (datasets I, times)")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--frequencies", type=int, required=True)
    ap.add_argument("--render-npix", type=int, default=128)
    ap.add_argument("--fov-uas", type=float, default=160.0, help="display FOV")
    ap.add_argument("--fov-uas-full", type=float, default=200.0, help="training/full-res FOV")
    ap.add_argument("--frames", type=int, nargs="*", default=[80, 200, 330])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sk = NeuralDMD(args.r, key=jax.random.PRNGKey(0), num_frequencies=args.frequencies)
    model = eqx.tree_deserialise_leaves(
        str(args.run / "models" / f"trained_model_r{args.r}_f{args.frequencies}.eqx"), sk
    )
    with h5py.File(args.gt_fullres, "r") as f:
        truth = np.asarray(f["I"][:], float)
        times = np.asarray(f["times"][:], float)
    tn = (times - times.min()) / (times.max() - times.min())
    fmax, fmin, tsc = float(truth.max()), float(truth.min()), float(model.t_scale)

    frac = args.fov_uas / args.fov_uas_full  # sub-region of the normalized grid
    npix = args.render_npix
    xy = jnp.asarray(pixel_grid_coords(npix, npix, fov_x=frac * np.pi, fov_y=frac * np.pi))
    W0, Wm, Om, b0, b = (np.asarray(x) for x in model(xy))
    lam = np.exp(Om[:, None] * tn[None, :] * tsc)
    rec = (((W0[:, 0:1] * b0[0]) + 2 * np.real(np.einsum("pr,rt,r->pt", Wm, lam, b))) * (fmax - fmin) + fmin)
    rec = rec.T.reshape(len(tn), npix, npix)

    full = truth.shape[1]
    m0 = int(round(full * (1 - frac) / 2))
    m1 = full - m0  # central crop to the display FOV
    h = args.fov_uas / 2
    ext = [-h, h, -h, h]

    def show(a, img, vmax):
        a.imshow(np.clip(img, 0, None), cmap="afmhot", origin="upper", interpolation="bicubic", vmax=vmax, extent=ext)
        a.set_xticks([])
        a.set_yticks([])

    fr = args.frames
    fig, ax = plt.subplots(2, 1 + len(fr), figsize=(3.1 * (1 + len(fr)), 6.4))
    rv = float(np.clip(rec, 0, None).max())
    show(ax[0, 0], truth.mean(0)[m0:m1, m0:m1], fmax)
    ax[0, 0].set_ylabel(f"truth {full}px", fontsize=11)
    ax[0, 0].set_title("mean I")
    for k, i in enumerate(fr):
        show(ax[0, 1 + k], truth[i, m0:m1, m0:m1], fmax)
        ax[0, 1 + k].set_title(f"frame {i}")
    show(ax[1, 0], rec.mean(0), rv)
    ax[1, 0].set_ylabel(f"recon {npix}px", fontsize=11)
    for k, i in enumerate(fr):
        show(ax[1, 1 + k], rec[i], rv)
    fig.suptitle(f"{args.run.name}  r={args.r} f={args.frequencies}  {npix}px, {args.fov_uas:g}uas, bicubic, afmhot")
    fig.tight_layout()
    out = args.out or (args.run / "hires_compare.png")
    fig.savefig(out, dpi=300)

    rec_hi = np.stack([zoom(np.clip(f, 0, None), 512 / npix, order=3) for f in rec])
    make_gif(rec_hi, str(args.run / "plots" / "recon_I_smooth.gif"), fps=25, vmin=0, vmax=rv)
    print(f"wrote {out} (dpi300) + plots/recon_I_smooth.gif")


if __name__ == "__main__":
    main()
