#!/usr/bin/env python
"""Render the current best checkpoint of a (possibly still-training) run.

Runs save ``models/polarized_model.eqx`` continuously as the best-so-far, so this
rebuilds the sky model from ``config.json``, loads that checkpoint, and renders
aligned truth-vs-reconstruction panels -- letting us judge morphology and EVPA
long before a 12000-epoch run finishes. Gains do not affect the sky images, so a
zero-gain skeleton is enough to deserialize and reconstruct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vis_compare import best_shift, evpa_ticks  # noqa: E402


def build_skeleton(cfg: dict):
    """Rebuild the model skeleton (with gains, if the run fit them) from a config.

    Parameters
    ----------
    cfg : dict
        Parsed ``config.json``.

    Returns
    -------
    PolarizedNeuralDMD
        A freshly-initialized model with the same pytree structure as the
        checkpoint, ready for ``eqx.tree_deserialise_leaves``.
    """
    from neuraldmd.calibration import StationGains
    from neuraldmd.polarized import PolarizedNeuralDMD, with_gains

    pol_kwargs = {}
    for src, dst in (
        ("pol_hidden", "hidden_size"),
        ("pol_layers", "num_layers"),
        ("pol_frequencies", "num_frequencies"),
    ):
        if cfg.get(src) is not None:
            pol_kwargs[dst] = cfg[src]

    model = PolarizedNeuralDMD(
        ("I", "Q", "U"),
        r=cfg["r"],
        key=jax.random.PRNGKey(cfg.get("seed", 0)),
        outshift=cfg.get("outshift", 2.0),
        scaling_ml=cfg.get("scaling_ml", 1.0),
        r_pol=cfg.get("r_pol"),
        pol_param=cfg.get("pol_param", "fractional"),
        pol_model_kwargs=pol_kwargs or None,
        couple=cfg.get("couple", "none"),
        n_shared=cfg.get("n_shared_modes"),
        share_trunk=cfg.get("share_trunk", False),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 4),
        num_frequencies=cfg.get("frequencies", 2),
        theta_max=cfg.get("theta_max", 1.0),
        dyn_cap=cfg.get("dyn_cap"),
    )
    if cfg.get("fit_gains"):
        bounds = cfg.get("gain_amp_bounds_resolved")
        amp_bounds = tuple(tuple(b) for b in bounds) if bounds else tuple(cfg["gain_amp_bounds"])
        n_st = len(amp_bounds)
        gains = StationGains(
            n_stations=n_st,
            n_times=int(cfg["num_frames"]),
            n_hands=cfg.get("gain_hands", 2),
            use_phase=cfg.get("gain_phase", False),
            amp_bounds=amp_bounds,
            ref_station=cfg.get("gain_ref_station", 0),
        )
        model = with_gains(model, gains)
    return model


def load_recon(run: Path):
    """Reconstruct I/Q/U cubes from a run's current best checkpoint.

    Parameters
    ----------
    run : Path
        Run directory.

    Returns
    -------
    dict
        ``I``/``Q``/``U`` cubes ``(T, npix, npix)``.
    """
    import neuraldmd.evaluation as ev

    cfg = json.loads((run / "config.json").read_text())
    # n_times for uvfits mode comes from the obs_dir, not config
    if cfg.get("uvfits") is not None:
        cfg = dict(cfg)
        cfg["num_frames"] = len(np.load(run / "data" / "times.npy"))

    skeleton = build_skeleton(cfg)
    model = eqx.tree_deserialise_leaves(str(run / "models" / "polarized_model.eqx"), skeleton)

    truth = np.load(run / "data" / "truth_pol.npz")
    fmax = {"I": float(np.abs(truth["I"]).max())}
    fmin = {"I": 0.0}
    return ev.reconstruct_polarized_cubes(model, cfg["npix"], truth["times"], fmax, fmin)


def main() -> None:
    """Render a peek figure for each run directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    for r in args.runs:
        run = Path(r)
        try:
            recon = load_recon(run)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {run.name}: {exc}")
            continue
        truth = np.load(run / "data" / "truth_pol.npz")
        t = {k: truth[k].astype(float) for k in ("I", "Q", "U")}
        r_c = {k: np.asarray(recon[k], float) for k in ("I", "Q", "U")}
        dy, dx = best_shift(r_c["I"].mean(0), t["I"].mean(0))
        r_c = {k: np.roll(np.roll(v, dy, 0), dx, 1) for k, v in r_c.items()}

        frames = (10, 27, 44) if r_c["I"].shape[0] < 70 else (20, 50, 80)
        ncol = 2 + len(frames)
        fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))
        for row, (src, lbl) in enumerate([(t, "truth"), (r_c, "recon")]):
            mI = src["I"].mean(0)
            axes[row, 0].imshow(mI, cmap="inferno", origin="lower")
            axes[row, 0].set_ylabel(lbl, fontsize=12)
            axes[row, 0].set_title("mean I")
            axes[row, 1].imshow(mI, cmap="inferno", origin="lower")
            evpa_ticks(axes[row, 1], mI, src["Q"].mean(0), src["U"].mean(0))
            axes[row, 1].set_title("mean I + EVPA")
            for j, fr in enumerate(frames):
                axes[row, 2 + j].imshow(src["I"][fr], cmap="inferno", origin="lower")
                axes[row, 2 + j].set_title(f"frame {fr}")
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{run.name}  (best checkpoint, shift {dx * 4},{dy * 4} uas)")
        fig.tight_layout()
        out = Path(args.outdir) / f"peek_{run.name}.png"
        fig.savefig(out, dpi=110)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
