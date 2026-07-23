#!/usr/bin/env python
"""Score a reconstruction against its uvfits with ehtim's own closure metrics.

Real interferometric data carries station-based gain and phase errors, so a
chi-squared on complex visibilities does not measure reconstruction quality --
the validation-ladder ground truth itself scores ``chi2_vis`` in the tens of
thousands on the data it generated, while scoring ``chi2_cphase = 1.0``. Closure
phase and log closure amplitude are immune to station terms, which is why they
are the quantities the collaboration's evaluation uses.

Reports the same numbers for the ground truth when one is given, so a fit can be
read against the floor that data actually permits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DTYPES = ("cphase", "logcamp", "amp")


def _movie_on_obs_clock(path: Path, run_dir: Path):
    """Load a reconstruction movie, putting it on the observation's UT clock.

    Runs written before the exporter was fixed store normalized ``[0, 1]`` times,
    which ehtim cannot index against an observation. When that is detected the
    clock is rebuilt from the obs_dir manifest's anchors.

    Parameters
    ----------
    path : Path
        Reconstruction ``.hdf5``.
    run_dir : Path
        Run directory holding ``data/manifest.json``.

    Returns
    -------
    ehtim.movie.Movie
        Movie indexed in UT hours.
    """
    import ehtim as eh
    import h5py

    with h5py.File(path, "r") as f:
        times = np.asarray(f["times"][:], dtype=np.float64)

    if times.max() <= 1.0 + 1e-9:
        manifest = run_dir / "data" / "manifest.json"
        anchors = json.loads(manifest.read_text()).get("time_anchors_hr")
        if anchors is None:
            raise SystemExit(f"{path} has normalized times and no anchors to rebuild them")
        t0, t1 = (float(x) for x in anchors)
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / path.name
        shutil.copy(path, tmp)
        with h5py.File(tmp, "r+") as f:
            del f["times"]
            f["times"] = t0 + times * (t1 - t0)
        path = tmp
    return eh.movie.load_hdf5(str(path))


def score(obs, movie) -> dict:
    """Closure and amplitude chi-squared of a movie against an observation.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
        The observation.
    movie : ehtim.movie.Movie
        Reconstruction or truth.

    Returns
    -------
    dict
        ``{dtype: chi2}``; a failed metric is reported as NaN rather than raising.
    """
    out = {}
    for dt in DTYPES:
        try:
            out[dt] = float(obs.chisq(movie, dtype=dt, pol="I", ttype="direct"))
        except Exception:
            out[dt] = float("nan")
    return out


def main() -> None:
    """Score each run directory and print a table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories containing recon_pol.hdf5")
    ap.add_argument("-d", "--data", required=True, help="uvfits the runs were fitted to")
    ap.add_argument("--truth", default=None, help="ground-truth hdf5, for the floor")
    args = ap.parse_args()

    import ehtim as eh

    obs = eh.obsdata.load_uvfits(args.data)

    rows = []
    if args.truth:
        rows.append(("TRUTH (floor)", score(obs, eh.movie.load_hdf5(args.truth))))
    for r in args.runs:
        run = Path(r)
        recon = run / "recon_pol.hdf5"
        if not recon.exists():
            continue
        rows.append((run.name, score(obs, _movie_on_obs_clock(recon, run))))

    width = max(len(n) for n, _ in rows) if rows else 10
    print(f"{'run':{width}s} " + " ".join(f"{d:>12s}" for d in DTYPES))
    print("-" * (width + 13 * len(DTYPES)))
    for name, s in rows:
        print(f"{name:{width}s} " + " ".join(f"{s[d]:12.3f}" for d in DTYPES))


if __name__ == "__main__":
    main()
