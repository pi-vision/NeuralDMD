"""Ingest EHT validation-ladder ground-truth movies.

Ladder truths are ehtim Movie HDF5 files (I/Q/U/V cubes, a times array in hours,
and a header carrying the pixel size). Scoring a fit against one means putting it
on our coarser image grid and on the observation's own frame times.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

UAS_PER_RAD = 1e6 * 180.0 * 3600.0 / np.pi


def load_ladder_movie(path: str | Path) -> dict:
    """Read a ladder HDF5 movie.

    Parameters
    ----------
    path : str or Path
        Path to the ``.hdf5`` movie.

    Returns
    -------
    dict
        ``I``/``Q``/``U``/``V`` cubes ``(T, H, W)``, ``times_hr``, and ``fov_uas``
        derived from the header pixel size.
    """
    with h5py.File(str(path), "r") as f:
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


def write_truth_npz(
    movie_path: str | Path,
    obs_dir: str | Path,
    npix: int,
    times_norm: np.ndarray,
    anchors_hr: tuple[float, float] | None = None,
) -> dict:
    """Write ``truth_pol.npz`` for a ladder movie, matched to an observation.

    Parameters
    ----------
    movie_path : str or Path
        Ladder ``.hdf5`` ground-truth movie.
    obs_dir : str or Path
        Directory to write ``truth_pol.npz`` into.
    npix : int
        Image side length; must divide the movie's native grid.
    times_norm : np.ndarray
        ``(T,)`` frame times normalized to ``[0, 1]``, as stored in the obs_dir.
    anchors_hr : tuple of float or None
        ``(t0, t1)`` hours that ``times_norm`` spans. Defaults to the movie's own
        first and last frame, which is only right when the observation covers the
        whole movie -- pass the observation's anchors otherwise.

    Returns
    -------
    dict
        ``{"fov_uas", "flux_mean", "flux_std"}`` of the written truth.
    """
    mov = load_ladder_movie(movie_path)
    native = mov["I"].shape[-1]
    if native % npix:
        raise ValueError(f"npix={npix} must divide the movie's {native}")

    cubes = {s: block_sum(mov[s], native // npix) for s in ("I", "Q", "U")}
    src = mov["times_hr"]
    t0, t1 = anchors_hr if anchors_hr is not None else (src[0], src[-1])
    dst = t0 + np.asarray(times_norm, dtype=np.float64) * (t1 - t0)
    cubes = {s: resample_time(c, src, dst) for s, c in cubes.items()}

    flux = cubes["I"].sum(axis=(1, 2))
    np.savez(
        Path(obs_dir) / "truth_pol.npz",
        I=cubes["I"].astype(np.float32),
        Q=cubes["Q"].astype(np.float32),
        U=cubes["U"].astype(np.float32),
        times=np.asarray(times_norm, dtype=np.float32),
        npix=npix,
        fov_uas=mov["fov_uas"],
    )
    return {
        "fov_uas": float(mov["fov_uas"]),
        "flux_mean": float(flux.mean()),
        "flux_std": float(flux.std()),
    }
