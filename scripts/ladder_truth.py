#!/usr/bin/env python
"""Convert an EHT validation-ladder HDF5 movie into our ``truth_pol.npz`` format.

The ladder movies are ehtim Movie HDF5 files (I/Q/U/V cubes, a times array, and
a header carrying the pixel size). Ours are evaluated on a coarser grid at the
observation's own frame times, so frames are block-summed -- which preserves
total flux, the cubes being in Jy/pixel -- and interpolated onto those times.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

UAS_PER_RAD = 1e6 * 180.0 * 3600.0 / np.pi


def load_ladder_movie(path: Path) -> dict:
    """Read a ladder HDF5 movie.

    Parameters
    ----------
    path : Path
        Path to the ``.hdf5`` movie.

    Returns
    -------
    dict
        ``I``/``Q``/``U``/``V`` cubes ``(T, H, W)``, ``times`` in hours, and
        ``fov_uas`` derived from the header pixel size.
    """
    with h5py.File(path, "r") as f:
        out = {s: np.asarray(f[s][:], dtype=np.float64) for s in ("I", "Q", "U", "V")}
        out["times_hr"] = np.asarray(f["times"][:], dtype=np.float64)
        psize_rad = float(f["header"].attrs["psize"])
    out["fov_uas"] = psize_rad * UAS_PER_RAD * out["I"].shape[-1]
    return out


def block_sum(cube: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a cube by summing ``factor x factor`` pixel blocks.

    Summing rather than averaging keeps the total flux of a Jy/pixel image fixed.

    Parameters
    ----------
    cube : np.ndarray
        ``(T, H, W)`` frames; ``H`` and ``W`` must divide by ``factor``.
    factor : int
        Block size.

    Returns
    -------
    np.ndarray
        ``(T, H // factor, W // factor)`` frames.
    """
    t, h, w = cube.shape
    if h % factor or w % factor:
        raise ValueError(f"{h}x{w} is not divisible by {factor}")
    return cube.reshape(t, h // factor, factor, w // factor, factor).sum(axis=(2, 4))


def resample_time(cube: np.ndarray, src_hr: np.ndarray, dst_hr: np.ndarray) -> np.ndarray:
    """Linearly interpolate a cube onto new frame times.

    Parameters
    ----------
    cube : np.ndarray
        ``(T, H, W)`` frames.
    src_hr, dst_hr : np.ndarray
        Source and destination times, in hours.

    Returns
    -------
    np.ndarray
        ``(len(dst_hr), H, W)`` frames.
    """
    t, h, w = cube.shape
    flat = cube.reshape(t, -1)
    out = np.empty((len(dst_hr), flat.shape[1]), dtype=np.float64)
    for p in range(flat.shape[1]):
        out[:, p] = np.interp(dst_hr, src_hr, flat[:, p])
    return out.reshape(len(dst_hr), h, w)


def main() -> None:
    """Write ``truth_pol.npz`` into an obs_dir, from a ladder movie."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("movie", help="ladder .hdf5 movie")
    ap.add_argument("--obs-dir", required=True, help="obs_dir to write truth_pol.npz into")
    ap.add_argument("--npix", type=int, default=50)
    args = ap.parse_args()

    obs_dir = Path(args.obs_dir)
    mov = load_ladder_movie(Path(args.movie))
    native = mov["I"].shape[-1]
    if native % args.npix:
        raise SystemExit(f"--npix {args.npix} must divide the movie's {native}")

    cubes = {s: block_sum(mov[s], native // args.npix) for s in ("I", "Q", "U")}

    # evaluate on the observation's own clock, so recon and truth frames line up
    times_norm = np.load(obs_dir / "times.npy").astype(np.float64)
    src = mov["times_hr"]
    dst = src[0] + times_norm * (src[-1] - src[0])
    cubes = {s: resample_time(c, src, dst) for s, c in cubes.items()}

    out = obs_dir / "truth_pol.npz"
    np.savez(
        out,
        I=cubes["I"].astype(np.float32),
        Q=cubes["Q"].astype(np.float32),
        U=cubes["U"].astype(np.float32),
        times=times_norm.astype(np.float32),
        npix=args.npix,
        fov_uas=mov["fov_uas"],
    )
    flux = cubes["I"].sum(axis=(1, 2))
    print(
        f"wrote {out}: {len(times_norm)} frames at {args.npix}px, "
        f"fov {mov['fov_uas']:.1f} uas, flux {flux.mean():.3f} +- {flux.std():.3f} Jy"
    )


if __name__ == "__main__":
    main()
