#!/usr/bin/env python
"""Collect metrics and mode tables from a set of runs into one comparison table.

Reads ``m2_metrics.json`` (and ``mode_table.json`` when present) from each run
directory and prints a ranked summary. Used to decide the coupling default from
the E1 sweep rather than by eye.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The hotspot orbit is 80 min, so every truth in the parametric suite puts power
# at integer multiples of this. Recovered frequencies are scored against it.
FUNDAMENTAL_RAD_PER_HR = 2 * np.pi * 60 / 80.0


def load_run(path: Path) -> dict | None:
    """Read one run directory.

    Parameters
    ----------
    path : Path
        Run directory holding ``m2_metrics.json``.

    Returns
    -------
    dict or None
        Flattened metrics, or ``None`` when the run has no metrics file.
    """
    mfile = path / "m2_metrics.json"
    if not mfile.exists():
        return None
    m = json.loads(mfile.read_text())
    cfg = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    chi2 = m.get("final_chi2", {}) or {}
    floor = m.get("chi2_truth_floor", {}) or {}
    nrmse = m.get("nrmse", {}) or {}
    b2 = m.get("beta2_dynamics", {}) or {}
    row = {
        "run": path.name,
        "couple": cfg.get("couple", "none"),
        "n_shared": cfg.get("n_shared_modes"),
        "r": cfg.get("r"),
        "chi2_max": max((float(v) for v in chi2.values()), default=float("nan")),
        # the truth itself does not reach 1, so excess over its floor is the
        # honest measure of how much of the fit is unexplained
        "chi2_over_floor": (
            max(float(v) for v in chi2.values()) - max(float(v) for v in floor.values())
            if chi2 and floor
            else float("nan")
        ),
        "nrmse_I": float(nrmse.get("I", "nan")),
        "nrmse_Q": float(nrmse.get("Q", "nan")),
        "nrmse_U": float(nrmse.get("U", "nan")),
        "evpa": float(m.get("evpa_error_deg", "nan")),
        "b2_amp": float(b2.get("amp_ratio", "nan")),
        "b2_phase_corr": float(b2.get("phase_corr", "nan")),
    }

    tfile = path / "mode_table.json"
    if tfile.exists():
        t = json.loads(tfile.read_text())
        freqs = np.asarray(t.get("freq_rad_per_hr", []), dtype=float)
        if freqs.size:
            # distance of each recovered frequency to the nearest harmonic
            harm = np.round(freqs / FUNDAMENTAL_RAD_PER_HR)
            resid = np.abs(freqs - harm * FUNDAMENTAL_RAD_PER_HR)
            on = (harm >= 1) & (resid < 0.15 * FUNDAMENTAL_RAD_PER_HR)
            row["n_on_harmonic"] = int(on.sum())
            row["harmonics"] = sorted({int(h) for h in harm[on]})
        # the split only says something when private modes exist; with
        # n_shared == r everything is shared by construction
        fields = t.get("fields", {})
        informative = int(t.get("n_shared", 0)) < int(cfg.get("r") or 0)
        for name in ("intensity", "frac"):
            if name in fields and informative:
                row[f"shared_{name}"] = float(fields[name]["shared_power"])
    return row


def main() -> None:
    """Print a comparison table over the given run directories."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--sort", default="nrmse_Q", help="column to sort by")
    args = ap.parse_args()

    rows = [r for r in (load_run(Path(p)) for p in args.runs) if r is not None]
    if not rows:
        print("no runs with metrics found")
        return
    rows.sort(key=lambda r: (np.isnan(r.get(args.sort, np.nan)), r.get(args.sort, np.nan)))

    cols = [
        "run",
        "couple",
        "n_shared",
        "r",
        "chi2_max",
        "chi2_over_floor",
        "nrmse_I",
        "nrmse_Q",
        "nrmse_U",
        "evpa",
        "b2_amp",
        "b2_phase_corr",
        "n_on_harmonic",
        "shared_intensity",
        "shared_frac",
    ]
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    """Render one cell."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.3f}"
    if isinstance(v, list):
        return ",".join(str(x) for x in v) or "-"
    return str(v)


if __name__ == "__main__":
    main()
