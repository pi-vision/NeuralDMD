#!/usr/bin/env python
"""Render aligned truth-vs-reconstruction panels for a ladder run.

Two rows (truth, aligned reconstruction), columns: mean Stokes I, mean I with
EVPA ticks, and three snapshot frames. Alignment is the integer-pixel shift that
best matches the mean I images, since phase self-calibration leaves absolute
position free. This is the by-eye check that the collaboration's beam-blurred
metrics can be too forgiving about (a filled blob can pass a blurred nxcorr).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(run: Path) -> tuple[dict, dict]:
    """Return reconstruction and truth Stokes cubes for a run.

    Parameters
    ----------
    run : Path
        Run directory with ``recon_pol.hdf5`` and ``data/truth_pol.npz``.

    Returns
    -------
    recon, truth : dict
        ``I``/``Q``/``U`` cubes ``(T, H, W)``.
    """
    with h5py.File(run / "recon_pol.hdf5") as f:
        r = {k: np.asarray(f[k][:], float) for k in ("I", "Q", "U")}
    t = np.load(run / "data" / "truth_pol.npz")
    return r, {k: t[k].astype(float) for k in ("I", "Q", "U")}


def best_shift(recon_I: np.ndarray, truth_I: np.ndarray, rng: int = 12) -> tuple[int, int]:
    """Integer-pixel shift of the reconstruction that best matches the truth.

    Parameters
    ----------
    recon_I, truth_I : np.ndarray
        Mean Stokes-I images.
    rng : int
        Search half-width in pixels.

    Returns
    -------
    tuple of int
        ``(dy, dx)``.
    """
    best = (np.inf, (0, 0))
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            e = np.linalg.norm(np.roll(np.roll(recon_I, dy, 0), dx, 1) - truth_I)
            if e < best[0]:
                best = (e, (dy, dx))
    return best[1]


def evpa_ticks(ax, I: np.ndarray, Q: np.ndarray, U: np.ndarray, step: int = 3) -> None:
    """Overlay EVPA ticks where the polarized intensity is bright.

    Parameters
    ----------
    ax : matplotlib axis
        Target axis (already showing ``I``).
    I, Q, U : np.ndarray
        Mean Stokes images.
    step : int
        Pixel stride between ticks.
    """
    P = np.hypot(Q, U)
    chi = 0.5 * np.arctan2(U, Q)
    ys, xs = np.mgrid[: I.shape[0], : I.shape[1]]
    sel = (P > 0.25 * P.max()) & (ys % step == 0) & (xs % step == 0)
    ax.quiver(
        xs[sel],
        ys[sel],
        np.sin(chi[sel]),
        np.cos(chi[sel]),
        color="cyan",
        scale=26,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        width=0.006,
    )


def render(run: Path, out: Path, frames=(20, 50, 80)) -> None:
    """Write the comparison figure for one run.

    Parameters
    ----------
    run : Path
        Run directory.
    out : Path
        Output PNG.
    frames : tuple of int
        Snapshot frame indices.
    """
    r, t = load(run)
    dy, dx = best_shift(r["I"].mean(0), t["I"].mean(0))
    r = {k: np.roll(np.roll(v, dy, 0), dx, 1) for k, v in r.items()}

    ncol = 2 + len(frames)
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))
    for row, (src, lbl) in enumerate([(t, "truth"), (r, "recon (aligned)")]):
        mean_I = src["I"].mean(0)
        axes[row, 0].imshow(mean_I, cmap="inferno", origin="lower")
        axes[row, 0].set_ylabel(lbl, fontsize=12)
        axes[row, 0].set_title("mean I")
        axes[row, 1].imshow(mean_I, cmap="inferno", origin="lower")
        evpa_ticks(axes[row, 1], mean_I, src["Q"].mean(0), src["U"].mean(0))
        axes[row, 1].set_title("mean I + EVPA")
        for j, fr in enumerate(frames):
            axes[row, 2 + j].imshow(src["I"][fr], cmap="inferno", origin="lower")
            axes[row, 2 + j].set_title(f"frame {fr}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{run.name}   (aligned shift {dx * 4},{dy * 4} uas)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main() -> None:
    """Render one figure per run directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    for r in args.runs:
        run = Path(r)
        render(run, Path(args.outdir) / f"vis_{run.name}.png")


if __name__ == "__main__":
    main()
