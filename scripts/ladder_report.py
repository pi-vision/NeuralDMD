#!/usr/bin/env python
"""Score a ladder run the way the collaboration does, and print one summary row.

Three families of numbers per run:

* gain-invariant visibility chi-squared (closure phase, log closure amplitude,
  and the polarimetric ratio m-breve), each next to the ground truth's own score
  on the same data -- the floor the data permit;
* ehteval's video nxcorr (I, polarized magnitude, polarized vector, EVPA), per
  frame with alignment, against the beam-blur threshold that defines "good";
* the fraction of frames passing that threshold, per quantity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EHTEVAL = Path("/scratch/ondemand33/rdahale/ndmd/ehteval")
LADDER = Path("/scratch/ondemand33/rdahale/ndmd/validation_ladder/besttime")


def recon_on_ut_clock(run: Path) -> Path:
    """Return a UT-clock copy of the run's reconstruction movie.

    Parameters
    ----------
    run : Path
        Run directory holding ``recon_pol.hdf5`` and ``data/manifest.json``.

    Returns
    -------
    Path
        ``recon_ut.hdf5`` inside the run directory (created if absent).
    """
    out = run / "recon_ut.hdf5"
    if out.exists():
        return out
    sys.path.insert(0, str(Path(__file__).parent))
    from score_with_ehtim import _movie_on_obs_clock

    mov = _movie_on_obs_clock(run / "recon_pol.hdf5", run)
    mov.save_hdf5(str(out))
    return out


def chi2_row(obs, movie) -> dict:
    """Gain-invariant chi-squared of a movie against an observation.

    Parameters
    ----------
    obs : ehtim.obsdata.Obsdata
        The observation.
    movie : ehtim.movie.Movie
        Reconstruction or truth, on the observation's clock.

    Returns
    -------
    dict
        ``cphase`` and ``logcamp`` for Stokes I, and ``mbreve`` (median over
        scans, JCMT flagged) for polarization.
    """
    out = {}
    for dt in ("cphase", "logcamp"):
        try:
            out[dt] = float(obs.chisq(movie, dtype=dt, pol="I", ttype="direct"))
        except Exception:
            out[dt] = float("nan")

    obs2 = obs.copy()
    obs2.add_scans()
    vals = []
    for o in obs2.split_obs(scan_gather=False):
        op = o.flag_sites(["JC"])
        if len(op.data) < 3:
            continue
        bad = np.isnan(op.data["vis"]) + np.isnan(op.data["qvis"]) + np.isnan(op.data["uvis"])
        op.data = op.data[~bad]
        if len(op.data) < 3:
            continue
        im = movie.get_image(op.data[0]["time"])
        if im.vvec is None or len(im.vvec) == 0:
            im.add_v(np.zeros((im.ydim, im.xdim)))
        try:
            vals.append(op.polchisq(im, dtype="m", ttype="direct"))
        except Exception:
            pass
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    out["mbreve"] = float(np.median(a)) if a.size else float("nan")
    return out


def nxcorr_rows(uvfits: Path, truth: Path, recon: Path, workdir: Path) -> dict:
    """Run ehteval's nxcorr and summarize its per-frame CSV.

    Parameters
    ----------
    uvfits, truth, recon : Path
        Observation, ground-truth movie, and UT-clock reconstruction.
    workdir : Path
        Where ehteval writes its CSV and plots.

    Returns
    -------
    dict
        Per quantity (I, Pmag, Pvec, X): median nxcorr, median threshold, and
        pass rate over frames, for both static and dynamic modes.
    """
    import pandas as pd

    prefix = workdir / "nx"
    cmd = [
        sys.executable,
        str(EHTEVAL / "src" / "nxcorr.py"),
        "-d",
        str(uvfits),
        "--truthmv",
        str(truth),
        "--input",
        str(recon),
        "-o",
        str(prefix),
        "-n",
        "8",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    out = {}
    for mode in ("static", "dynamic"):
        csv = Path(f"{prefix}_{mode}_nxcorr.csv")
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for q in ("I", "Pmag", "Pvec", "X"):
            col, thr = f"nxcorr_{q}", f"nxcorr_{q}_thres"
            if col not in df:
                continue
            v, t = df[col].to_numpy(float), df[thr].to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(t)
            out[f"{mode}_{q}"] = float(np.median(v[ok])) if ok.any() else float("nan")
            out[f"{mode}_{q}_thr"] = float(np.median(t[ok])) if ok.any() else float("nan")
            out[f"{mode}_{q}_pass"] = float(np.mean(v[ok] > t[ok])) if ok.any() else float("nan")
    return out


def main() -> None:
    """Score each run directory and print the summary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="ladder run directories (runs/ladder_F_<model>)")
    ap.add_argument("--band", default="LO")
    ap.add_argument("--json-out", default=None, help="also dump all numbers to this JSON")
    args = ap.parse_args()

    import ehtim as eh

    rows = {}
    for r in args.runs:
        run = Path(r)
        model = run.name.replace("ladder_F_", "")
        uvfits = LADDER / "data" / "netcal+tavg60s" / f"{model}_{args.band}_onsky.uvfits"
        truth = LADDER / "groundtruth" / f"{model}_{args.band}_onsky_truth.hdf5"
        if not (run / "recon_pol.hdf5").exists() or not uvfits.exists() or not truth.exists():
            print(f"[skip] {model}: missing recon, uvfits, or truth")
            continue

        obs = eh.obsdata.load_uvfits(str(uvfits))
        tr = eh.movie.load_hdf5(str(truth))
        tr.reset_interp(bounds_error=False)
        rec = eh.movie.load_hdf5(str(recon_on_ut_clock(run)))
        rec.reset_interp(bounds_error=False)

        row = {"chi2": chi2_row(obs, rec), "chi2_floor": chi2_row(obs, tr)}
        with tempfile.TemporaryDirectory() as td:
            row["nxcorr"] = nxcorr_rows(uvfits, truth, run / "recon_ut.hdf5", Path(td))
        rows[model] = row
        print(f"[done] {model}", flush=True)

    print("\nchi2 (recon | truth floor):")
    print(f"{'model':16s} {'cphase':>15s} {'logcamp':>15s} {'mbreve':>15s}")
    for m, r in rows.items():
        c, f = r["chi2"], r["chi2_floor"]
        print(
            f"{m:16s} "
            f"{c['cphase']:7.2f}|{f['cphase']:5.2f}  "
            f"{c['logcamp']:7.2f}|{f['logcamp']:5.2f}  "
            f"{c['mbreve']:7.2f}|{f['mbreve']:5.2f}"
        )

    for mode in ("static", "dynamic"):
        print(f"\nnxcorr {mode} (value|threshold, pass fraction):")
        print(f"{'model':16s}" + "".join(f"{q:>22s}" for q in ("I", "Pmag", "Pvec", "X")))
        for m, r in rows.items():
            nx = r["nxcorr"]
            cells = []
            for q in ("I", "Pmag", "Pvec", "X"):
                v = nx.get(f"{mode}_{q}", float("nan"))
                t = nx.get(f"{mode}_{q}_thr", float("nan"))
                p = nx.get(f"{mode}_{q}_pass", float("nan"))
                cells.append(f"{v:5.2f}|{t:4.2f} p={p:4.0%}")
            print(f"{m:16s}" + "".join(f"{c:>22s}" for c in cells))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
