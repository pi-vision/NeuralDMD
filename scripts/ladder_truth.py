#!/usr/bin/env python
"""Write ``truth_pol.npz`` into an obs_dir from a validation-ladder movie.

Thin CLI over :mod:`neuraldmd.data.ladder`; useful for re-scoring an existing run
against a ladder truth without retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuraldmd.data.ladder import write_truth_npz


def main() -> None:
    """Convert one ladder movie and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("movie", help="ladder .hdf5 ground-truth movie")
    ap.add_argument("--obs-dir", required=True, help="obs_dir to write truth_pol.npz into")
    ap.add_argument("--npix", type=int, default=50)
    args = ap.parse_args()

    obs_dir = Path(args.obs_dir)
    times = np.load(obs_dir / "times.npy")
    anchors = None
    manifest = obs_dir / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        if meta.get("time_anchors_hr"):
            anchors = tuple(float(x) for x in meta["time_anchors_hr"])

    info = write_truth_npz(args.movie, obs_dir, args.npix, times, anchors_hr=anchors)
    print(
        f"wrote {obs_dir / 'truth_pol.npz'}: {len(times)} frames at {args.npix}px, "
        f"fov {info['fov_uas']:.1f} uas, flux {info['flux_mean']:.3f} "
        f"+- {info['flux_std']:.3f} Jy"
    )


if __name__ == "__main__":
    main()
