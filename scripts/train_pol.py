#!/usr/bin/env python
"""Milestone M2: train a polarized (I, Q, U) NeuralDMD on a synthetic dataset.

Generates a polarized m-ring + hot-spot dataset (or reuses one), trains
``PolarizedNeuralDMD`` with :func:`neuraldmd.training.train_polarized_model`
(simultaneous I/Q/U, circular basis by default), then evaluates against the
ground-truth cubes and checks the M2 gates. Writes metrics + a P/EVPA summary
figure. Run under the ``ndmd`` env; on GPU via ``slurm/train_pol.slurm``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--npix", type=int, default=50)
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--fov-uas", type=float, default=200.0)
    ap.add_argument("--frac-pol", type=float, default=0.2, help="ring linear pol fraction")
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--r", type=int, default=8, help="Number of complex DMD modes per Stokes")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--basis", default="circular", choices=["stokes", "circular"])
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument(
        "--frequencies",
        type=int,
        default=2,
        help="Stokes-I positional-encoding frequencies; the finest representable "
        "feature is ~fov/2^freq, so a thin ring / sharp background edge needs >2",
    )
    ap.add_argument(
        "--theta-max",
        type=float,
        default=1.0,
        help="cap on temporal mode frequency (x t_scale=200 rad over the window); set "
        "below the scan Nyquist to prevent inter-scan flux ringing",
    )
    ap.add_argument(
        "--truth-model",
        default="mring_hs",
        choices=["mring_hs", "mring_hs_pol", "varbeta2"],
        help="synthetic truth: 'mring_hs' (static spiral EVPA + unpolarized orbiting "
        "hot spot; dynamic I), 'mring_hs_pol' (weakly-polarized ring + a POLARIZED "
        "orbiting hot spot; dynamic I AND pol), or 'varbeta2' (rotating EVPA on a "
        "static ring; dynamic pol only). Use --direction CCW with mring_hs to get "
        "the direction-bias variant",
    )
    ap.add_argument(
        "--direction",
        default="CW",
        choices=["CW", "CCW"],
        help="hot-spot orbital sense for 'mring_hs'/'mring_hs_pol' (CCW is the "
        "direction-bias variant, ehteval's mring+hsCCW)",
    )
    # ── station gains (M3) ── off by default: M2 data are gain-free by construction
    ap.add_argument(
        "--fit-gains",
        action="store_true",
        help="solve per-station complex gains (amp + phase) alongside the sky. RIME "
        "is applied to the MODEL visibilities: V_pq <- g_p V_pq conj(g_q). Requires a "
        "dataset carrying bl_station_ids",
    )
    ap.add_argument(
        "--gain-hands",
        type=int,
        default=2,
        choices=[1, 2],
        help="2 = separate R and L gains (per-hand, the physical case); 1 ties them",
    )
    ap.add_argument(
        "--gain-amp-bounds",
        type=float,
        nargs=2,
        default=(0.5, 2.0),
        metavar=("LO", "HI"),
        help="hard sigmoid bounds on gain amplitude, applied to EVERY station; must "
        "strictly bracket 1. Ignored when --gain-bounds-physical is set",
    )
    ap.add_argument(
        "--gain-bounds-physical",
        action="store_true",
        help="bound each station by ITS OWN calibration quality (EHT_GAIN_PRIORS: ALMA "
        "+-3%%, LMT +-15%%, ...) instead of one global box. A global box makes every "
        "station equally free, leaving the overall gain scale -- which is degenerate "
        "with total source flux -- unconstrained; well-calibrated stations are what pin "
        "it. Requires station names in the data",
    )
    ap.add_argument(
        "--gain-phase",
        action="store_true",
        help="also solve gain PHASES (complex gains). Phases are referenced to "
        "--gain-ref-station, since a global phase offset is degenerate with source position",
    )
    ap.add_argument("--gain-ref-station", type=int, default=0, help="phase reference station")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-data", action="store_true", help="Reuse an existing data/ obs_dir")
    # external observation mode: fit a provided uvfits (e.g. the on-sky synthetic
    # mring+hsCW) instead of self-generating one; the ground truth is rebuilt on
    # the observation's own clock for evaluation
    ap.add_argument("--uvfits", default=None, help="external uvfits to fit (skips generation)")
    ap.add_argument(
        "--truth-hdf5",
        default=None,
        help="validation-ladder ground-truth movie to score against (uvfits mode). "
        "Without it, uvfits mode rebuilds a synthetic parametric truth, which is "
        "meaningless for a ladder dataset",
    )
    ap.add_argument("--syserr", type=float, default=0.0, help="fractional syserr added to sigma")
    ap.add_argument(
        "--truth-frames", type=int, default=200, help="truth-movie frames (uvfits mode)"
    )
    # Stokes-I disk-template pretraining (image prior)
    ap.add_argument("--no-pretrain", action="store_true", help="Skip the Stokes-I pretrain")
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--pretrain-lr", type=float, default=1e-4)
    ap.add_argument(
        "--pretrain-radius",
        type=float,
        default=1.0,
        help="disk-template radius as a multiple of the gyration radius",
    )
    # optimizer / LR schedule (exponential annealing when decay-rate < 1)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "adamax"])
    ap.add_argument("--lr-decay-rate", type=float, default=1.0, help="<1 enables exp decay")
    ap.add_argument("--lr-decay-steps", type=int, default=2000)
    # fractional-pol controls
    ap.add_argument("--scaling-ml", type=float, default=1.0, help="Cap on linear pol fraction")
    ap.add_argument("--outshift", type=float, default=2.0, help="m_l sigmoid bias (small init)")
    ap.add_argument(
        "--pol-param",
        default="fractional",
        choices=["fractional", "direct", "iscaled", "expm", "expm_full"],
        help="pol parameterization: 'fractional' (m_l,EVPA; winding blocks m>=2), "
        "'direct' (free Q,U; leaks haze), 'iscaled' (Q=I*tanh q; no haze/winding, "
        "P<=sqrt2 I), 'expm' (I*tanh(p)*(q,u,v)/p on our I; exact P<=I, V-capable), "
        "or 'expm_full' (full resolve matrix-exp I=e^s cosh p; PSD everywhere, "
        "needs --no-pretrain -- recommended)",
    )
    ap.add_argument(
        "--p-weight",
        type=float,
        default=0.0,
        help="soft P<=I penalty weight (sum relu(sqrt(Q^2+U^2)-I)^2); "
        "recommended >0 with --pol-param direct to suppress off-source pol",
    )
    # polarization regularization / curriculum (anti-overfit levers)
    ap.add_argument(
        "--r-pol",
        type=int,
        default=None,
        help="DMD modes for the pol fields (default = --r; use < r to starve capacity)",
    )
    # ── shared temporal bank / spatial trunk across Stokes ──
    ap.add_argument(
        "--couple",
        default="none",
        choices=["none", "pol", "all"],
        help="share one temporal spectrum: 'pol' ties the polarization fields to "
        "each other (Q,U are one spin-2 field), 'all' adds Stokes I so mode k "
        "means the same frequency in every Stokes and per-mode polarimetry is "
        "defined. Default 'none' = every field fits its own spectrum",
    )
    ap.add_argument(
        "--n-shared-modes",
        type=int,
        default=None,
        help="how many of the r modes come from the shared bank (default: all). "
        "Leaving some private lets polarization carry a periodicity Stokes I "
        "does not have; the shared-vs-private power split in mode_table.json "
        "then measures whether it does",
    )
    ap.add_argument(
        "--share-trunk",
        action="store_true",
        help="also share the spatial trunk, so pol fields are thin heads on the "
        "bank owner's features (encodes: pol structure lives where I structure is)",
    )
    ap.add_argument(
        "--pol-frequencies",
        type=int,
        default=None,
        help="positional-encoding frequencies for the pol fields (spatial band-limit)",
    )
    ap.add_argument(
        "--pol-hidden",
        type=int,
        default=None,
        help="hidden width of the pol spatial MLPs (default = --hidden-size)",
    )
    ap.add_argument(
        "--pol-layers",
        type=int,
        default=None,
        help="depth of the pol spatial MLPs (default = --num-layers)",
    )
    ap.add_argument(
        "--pol-warmup-epochs",
        type=int,
        default=0,
        help="ramp the pol learning rate 0->1 over N epochs (I converges first)",
    )
    ap.add_argument(
        "--freeze-i-after",
        type=int,
        default=None,
        help="hard-freeze Stokes I from this epoch on (fit pol on a fixed I)",
    )
    # two-stage curriculum: RR/LL-only I fit (pol frozen), then pol on frozen I
    ap.add_argument(
        "--i-only-epochs",
        type=int,
        default=0,
        help="stage-A epochs fitting only RR,LL with pol frozen (0 = single joint stage)",
    )
    ap.add_argument(
        "--pol-lr",
        type=float,
        default=None,
        help="stage-B (pol) initial learning rate (default = --lr)",
    )
    # total-flux (lightcurve) anchor: no zero-spacing baseline measures this
    ap.add_argument(
        "--flux", type=float, default=None, help="known total flux [Jy] (anchor off=None)"
    )
    ap.add_argument("--flux-weight", type=float, default=1.0, help="total-flux anchor weight")
    ap.add_argument(
        "--flux-curve",
        choices=("measured",),
        default=None,
        help="anchor the total flux SOFTLY to the per-frame light curve measured "
        "from intra-site baselines, instead of the scalar --flux. Use whenever the "
        "source varies (GRMHD, a flaring Sgr A*): one number would fight the real "
        "variability, while the measured curve is a data product, not a tuned knob",
    )
    ap.add_argument(
        "--fix-flux",
        choices=("measured", "given"),
        default=None,
        help="FIX the total flux structurally instead of nudging it with --flux-weight: "
        "the networks supply only the shape and the total is supplied per frame, so the "
        "flux degree of freedom -- degenerate with the global gain amplitude -- does not "
        "exist. 'measured' takes the lightcurve from intra-site baselines (works on any "
        "array with a co-located pair, no truth needed); 'given' pins it to --flux. A "
        "soft anchor cannot substitute: swept to --flux-weight 1000, a global gain scale "
        "survived and the fitted gains still lost to a do-nothing baseline",
    )
    ap.add_argument(
        "--compact-weight",
        type=float,
        default=0.0,
        help="compactness prior weight: penalize I flux by squared radius (kills "
        "off-source haze the data cannot constrain)",
    )
    ap.add_argument(
        "--compact-pol-weight",
        type=float,
        default=0.0,
        help="polarized compactness weight: penalize P=sqrt(Q^2+U^2) by squared "
        "radius (kills off-ring pol haze from direct Q,U; radius-gated)",
    )
    ap.add_argument(
        "--pol-support-weight",
        type=float,
        default=0.0,
        help="polarized support weight: penalize P where I is faint, "
        "mean(P*exp(-I/tau)) (confines pol to the bright ring incl. dark center; "
        "I-gated, preferred over --compact-pol-weight for direct Q,U)",
    )
    ap.add_argument(
        "--pol-support-tau",
        type=float,
        default=0.05,
        help="support gate scale as a fraction of peak I (smaller = harder gate)",
    )
    ap.add_argument(
        "--dyn-compact-weight",
        type=float,
        default=0.0,
        help="compactness prior on the Stokes-I DYNAMIC modes only (sum |W|*r^2): "
        "confines time-varying structure to small radius while leaving the static "
        "ring untouched -- unlike --compact-weight, which penalizes total I and "
        "squeezes the ring along with the off-source dynamic haze",
    )
    ap.add_argument(
        "--pol-l1-weight",
        type=float,
        default=0.0,
        help="L1 prior on total polarized flux, mean(sum(P)) -- the classic RML "
        "sparsity regularizer on P. The on-ring m=2 swirl and the off-ring haze "
        "are near-degenerate in chi2 but not in flux (the ring buys the same chi2 "
        "with several times less), so this selects the ring",
    )
    # evaluation: also report metrics after restoring both cubes to this beam
    ap.add_argument("--blur-uas", type=float, default=15.0, help="metric beam FWHM [uas]")
    ap.add_argument(
        "--dynamic-quiver",
        action="store_true",
        help="draw EVPA ticks on the comparison GIF's Dynamic panel (off by default)",
    )
    # per-product early stop: stop once ALL products <= this; <1 so images sharpen
    ap.add_argument("--early-stop-chi2", type=float, default=0.8)
    return ap.parse_args()


def main():
    args = parse_args()
    from neuraldmd import evaluation as ev
    from neuraldmd.data.generation import generate_polarized_dataset
    from neuraldmd.data.loader import PolarizedDMDDataLoader
    from neuraldmd.data.movies import save_movie_hdf5, to_ehtim_movie
    from neuraldmd.data.observations import ObsProducts
    from neuraldmd.polarized import PolarizedNeuralDMD
    from neuraldmd.pretraining import pretrain_log_intensity, pretrain_stokes_i
    from neuraldmd.training import train_polarized_model

    print("jax devices:", jax.devices(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # persist the exact configuration: without it a run's recipe is unrecoverable
    # once the launching shell is gone, and comparing runs becomes archaeology
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    data_dir = out / "data"
    stokes = ("I", "Q", "U")

    if args.reuse_data and (data_dir / "manifest.json").exists():
        print(f"Reusing dataset at {data_dir}", flush=True)
        op = ObsProducts.from_obs_dir(data_dir)
    elif args.uvfits:
        from neuraldmd.data.generation import save_truth_npz
        from neuraldmd.data.movies import make_mring_hs_pol_movie
        from neuraldmd.data.observations import load_uvfits_to_products

        print(f"Loading external uvfits {args.uvfits} ...", flush=True)
        op = load_uvfits_to_products(
            args.uvfits,
            npix=args.npix,
            fov_uas=args.fov_uas,
            stokes=stokes,
            basis=args.basis,
            syserr=args.syserr,
        )
        op.to_obs_dir(data_dir)
        t0, t1 = op.time_anchors_hr
        if args.truth_hdf5:
            from neuraldmd.data.ladder import write_truth_npz

            print(f"Ladder truth {args.truth_hdf5} on {t0:.3f}..{t1:.3f} UT ...", flush=True)
            info = write_truth_npz(
                args.truth_hdf5, data_dir, args.npix, op.times, anchors_hr=(t0, t1)
            )
            print(
                f"  fov {info['fov_uas']:.1f} uas, flux {info['flux_mean']:.3f} "
                f"+- {info['flux_std']:.3f} Jy",
                flush=True,
            )
        else:
            # rebuild the canonical truth on the observation's own clock: uniform
            # frames spanning first->last scan; training uses the actual scan times
            print(f"Rebuilding truth movie on {t0:.3f}..{t1:.3f} UT ...", flush=True)
            movie = make_mring_hs_pol_movie(
                npix=args.npix,
                fov_uas=args.fov_uas,
                num_frames=args.truth_frames,
                tstart_hr=t0,
                tstop_hr=t1,
                linpol_frac=args.frac_pol,
            )
            save_truth_npz(movie, data_dir, args.npix, args.fov_uas)
    else:
        print("Generating polarized dataset ...", flush=True)
        op = generate_polarized_dataset(
            data_dir,
            npix=args.npix,
            fov_uas=args.fov_uas,
            num_frames=args.num_frames,
            linpol_frac=args.frac_pol,
            stokes=stokes,
            basis=args.basis,
            seed=args.seed,
            truth_model=args.truth_model,
            direction=args.direction,
        )
    print(f"Dataset keys={op.stokes}  A={op.A.shape}", flush=True)

    truth = np.load(data_dir / "truth_pol.npz")
    truth_cubes = {s: truth[s] for s in stokes}
    frame_max = {
        "I": float(truth["I"].max()),
        "Q": float(np.abs(truth["Q"]).max()),
        "U": float(np.abs(truth["U"]).max()),
    }
    frame_min = {s: 0.0 for s in stokes}

    loader = PolarizedDMDDataLoader(
        op, npix=args.npix, batch_size=args.batch_size, epochs=args.epochs, fov_x=np.pi, fov_y=np.pi
    )
    # spatial band-limit for the pol fields: smaller/lower-frequency nets than I
    pol_kwargs = {
        k: v
        for k, v in {
            "num_frequencies": args.pol_frequencies,
            "hidden_size": args.pol_hidden,
            "num_layers": args.pol_layers,
        }.items()
        if v is not None
    }
    # Total flux: supply it rather than fit it. It is degenerate with the global gain
    # amplitude, and a soft anchor loses that argument -- swept to flux_weight=1000 and
    # the gains still scored worse than assuming no gains at all. Measured from
    # intra-site baselines (co-located dishes see the source unresolved, so |V| is the
    # total flux), which is a data product, so this works on any array that has such a
    # pair -- no truth knowledge.
    fix_flux = None
    if args.fix_flux:
        from neuraldmd.data.lightcurve import measure_lightcurve

        lc = measure_lightcurve(op)
        fix_flux = lc if args.fix_flux == "measured" else float(args.flux)
        src = (
            f"measured from intra-site baselines: {lc.mean():.4f} +- {lc.std():.4f} Jy"
            if args.fix_flux == "measured"
            else f"held at --flux {args.flux} Jy (measured would have been {lc.mean():.4f})"
        )
        print(f"Total flux FIXED, {src}", flush=True)

    # Soft anchor on a MEASURED light curve. A real source varies -- the ladder's
    # GRMHD truth swings 2.08-3.22 Jy over one track -- so anchoring every frame to
    # one number would fight the variability it is meant to constrain.
    flux_target = args.flux
    if args.flux_curve == "measured":
        from neuraldmd.data.lightcurve import measure_lightcurve

        curve = measure_lightcurve(op)
        flux_target = curve
        print(
            f"Flux anchor: measured curve, {curve.mean():.4f} Jy mean, "
            f"range [{curve.min():.4f}, {curve.max():.4f}]",
            flush=True,
        )

    model = PolarizedNeuralDMD(
        stokes,
        r=args.r,
        key=jax.random.PRNGKey(args.seed),
        outshift=args.outshift,
        scaling_ml=args.scaling_ml,
        fix_flux=fix_flux,
        r_pol=args.r_pol,
        pol_param=args.pol_param,
        pol_model_kwargs=pol_kwargs or None,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_frequencies=args.frequencies,
        theta_max=args.theta_max,
        couple=args.couple,
        n_shared=args.n_shared_modes,
        share_trunk=args.share_trunk,
    )

    if args.fit_gains:
        # Attach the gain table to the sky model: it becomes part of the same pytree,
        # so the optimizer solves calibration and image together. Needs station ids
        # (any dataset from load_uvfits_to_products has them).
        from neuraldmd.calibration import StationGains, eht_amp_bounds
        from neuraldmd.polarized import with_gains

        if op.bl_station_ids is None:
            raise SystemExit("--fit-gains needs a dataset with bl_station_ids")
        n_st = len(op.stations) if op.stations else int(op.bl_station_ids.max()) + 1
        if args.gain_bounds_physical:
            # Per-station bounds from each station's real calibration quality. A single
            # global box leaves every station equally free, so the overall gain scale --
            # degenerate with total flux -- floats: measured, a 1.153 scale with the
            # source flux collapsing to match.
            if not op.stations:
                raise SystemExit("--gain-bounds-physical needs station names in the data")
            amp_bounds = eht_amp_bounds(op.stations)
            bounds_desc = "physical per-station: " + ", ".join(
                f"{s}[{lo:.2f},{hi:.2f}]"
                for s, (lo, hi) in zip(op.stations, amp_bounds, strict=True)
            )
        else:
            amp_bounds = tuple(args.gain_amp_bounds)
            bounds_desc = f"global {amp_bounds}"
        gains = StationGains(
            n_stations=n_st,
            n_times=int(op.A.shape[0]),
            n_hands=args.gain_hands,
            use_phase=args.gain_phase,
            amp_bounds=amp_bounds,
            ref_station=args.gain_ref_station,
        )
        model = with_gains(model, gains)
        # Persist the RESOLVED per-station bounds. config.json was written from
        # vars(args) before this point, so it records the CLI default, not what was
        # used. The bounds are static fields -- tree_deserialise_leaves will not
        # restore them -- so anything reloading this checkpoint rebuilds StationGains
        # from config and would decode amp_raw through the WRONG bounds, reporting
        # wrong amplitudes with no error.
        cfg_path = out / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["gain_amp_bounds_resolved"] = [list(b) for b in gains.amp_bounds]
        cfg_path.write_text(json.dumps(cfg, indent=2, default=str))
        print(
            f"Fitting station gains: {n_st} stations x {op.A.shape[0]} times, "
            f"{args.gain_hands} hand(s){' + phase' if args.gain_phase else ' (amplitude only)'}, "
            f"amp bounds {bounds_desc}",
            flush=True,
        )

    if not args.no_pretrain:
        # expm_full parameterizes s = log I, so the disk template must be fit in
        # log space (e^s ~ disk); the linear pretrain would leave e^0=1 background.
        if args.pol_param == "expm_full":
            print(f"Pretraining LOG-I disk template ({args.pretrain_steps} steps) ...", flush=True)
            model, _ = pretrain_log_intensity(
                model,
                truth_cubes["I"],
                num_steps=args.pretrain_steps,
                lr=max(args.pretrain_lr, 1e-3),
                radius_scale=args.pretrain_radius,
                key=jax.random.PRNGKey(args.seed + 2),
            )
        else:
            print(f"Pretraining Stokes-I disk template ({args.pretrain_steps} steps) ...")
            model, _ = pretrain_stokes_i(
                model,
                truth_cubes["I"],
                num_steps=args.pretrain_steps,
                lr=args.pretrain_lr,
                radius_scale=args.pretrain_radius,
                key=jax.random.PRNGKey(args.seed + 2),
            )

    common = dict(
        models_dir=str(out / "models"),
        frame_max=frame_max,
        frame_min=frame_min,
        basis=args.basis,
        optimizer_name=args.optimizer,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_steps=args.lr_decay_steps,
        flux_target=flux_target,
        flux_weight=args.flux_weight,
        compact_weight=args.compact_weight,
        compact_pol_weight=args.compact_pol_weight,
        pol_support_weight=args.pol_support_weight,
        pol_support_tau=args.pol_support_tau,
        pol_l1_weight=args.pol_l1_weight,
        dyn_compact_weight=args.dyn_compact_weight,
        p_le_i_weight=args.p_weight,
        early_stop_chi2=args.early_stop_chi2,
        print_every=200,
    )
    total_epochs = 0
    if args.i_only_epochs > 0:
        # stage A: fit Stokes I alone on the parallel hands; pol frozen at its
        # near-unpolarized init (no pol residual can corrupt I, and vice versa)
        if args.basis != "circular":
            raise SystemExit("--i-only-epochs requires --basis circular")
        print(f"Stage A: I-only (RR,LL), pol frozen, {args.i_only_epochs} epochs ...", flush=True)
        model, hist_a = train_polarized_model(
            model,
            loader,
            num_epochs=args.i_only_epochs,
            key=jax.random.PRNGKey(args.seed + 1),
            initial_lr=args.lr,
            products=("RR", "LL"),
            freeze_pol=True,
            **common,
        )
        total_epochs += len(hist_a["total"])
        ev.plot_training_history(
            hist_a, str(out / "loss_history_stageA.png"), title="Stage A: I-only"
        )
        print(f"Stage B: pol on frozen I (all products), {args.epochs} epochs ...", flush=True)
        model, hist = train_polarized_model(
            model,
            loader,
            num_epochs=args.epochs,
            key=jax.random.PRNGKey(args.seed + 3),
            initial_lr=args.pol_lr if args.pol_lr is not None else args.lr,
            products=op.stokes,
            freeze_intensity=True,
            **common,
        )
    else:
        extra = {"products": op.stokes} if args.basis == "circular" else {}
        print(
            f"Training ({args.basis} basis, {args.optimizer}, {args.epochs} epochs) ...", flush=True
        )
        model, hist = train_polarized_model(
            model,
            loader,
            num_epochs=args.epochs,
            key=jax.random.PRNGKey(args.seed + 1),
            initial_lr=args.lr,
            pol_warmup_epochs=args.pol_warmup_epochs,
            freeze_i_after=args.freeze_i_after,
            **common,
            **extra,
        )
    total_epochs += len(hist["total"])
    ev.plot_training_history(hist, str(out / "loss_history.png"), title="Training")

    recon = ev.reconstruct_polarized_cubes(model, args.npix, truth["times"], frame_max, frame_min)

    # SELF-CHECK: `final_chi2` below is read out of the training history, i.e. it
    # describes the model as the training loop saw it. Recompute chi2 directly from
    # the EXPORTED cube through the same operator -- that cube is what every metric
    # (NRMSE/EVPA/beta2) and every downstream audit actually reads. The two must
    # agree; if they diverge, the reported chi2 is describing something other than
    # the reconstruction, and the gate/early-stop are being driven by a fiction.
    chi2_from_cube: dict[str, float] = {}
    if args.basis == "circular" and set(stokes) == {"I", "Q", "U"}:
        t_op = int(op.A.shape[0])
        i_c = recon["I"][:t_op].reshape(t_op, -1).astype(np.complex64)
        p_c = (recon["Q"][:t_op] + 1j * recon["U"][:t_op]).reshape(t_op, -1)
        vis_cube = {
            "RR": np.einsum("tmp,tp->tm", op.A, i_c),  # RR = I + V, V = 0
            "LL": np.einsum("tmp,tp->tm", op.A, i_c),  # LL = I - V
            "RL": np.einsum("tmp,tp->tm", op.A, p_c),  # RL = Q + iU
            "LR": np.einsum("tmp,tp->tm", op.A, np.conj(p_c)),  # LR = Q - iU
        }
        for k in op.stokes:
            if k not in vis_cube:
                continue
            d2 = np.abs(vis_cube[k] - op.targets[k]) ** 2
            denom = 2.0 * float(op.masks[k].sum())
            chi2_from_cube[k] = float((d2 * op.masks[k] / op.sigmas[k] ** 2).sum() / denom)
        # the truth through the same operator: the achievable floor. NB it is ~0.4,
        # not 1.0, because generation inflates sigma via add_fractional_noise without
        # adding matching noise -- so a chi2 gate centred on 1.0 can never pass.
        t_i = truth_cubes["I"][:t_op].reshape(t_op, -1).astype(np.complex64)
        t_p = (truth_cubes["Q"][:t_op] + 1j * truth_cubes["U"][:t_op]).reshape(t_op, -1)
        vis_truth = {
            "RR": np.einsum("tmp,tp->tm", op.A, t_i),
            "LL": np.einsum("tmp,tp->tm", op.A, t_i),
            "RL": np.einsum("tmp,tp->tm", op.A, t_p),
            "LR": np.einsum("tmp,tp->tm", op.A, np.conj(t_p)),
        }
        chi2_truth_floor = {}
        for k in op.stokes:
            if k not in vis_truth:
                continue
            d2 = np.abs(vis_truth[k] - op.targets[k]) ** 2
            denom = 2.0 * float(op.masks[k].sum())
            chi2_truth_floor[k] = float((d2 * op.masks[k] / op.sigmas[k] ** 2).sum() / denom)
    else:
        chi2_truth_floor = {}

    nrmse = ev.polarized_nrmse(recon, truth_cubes)
    evpa_err = ev.evpa_error_deg(recon, truth_cubes)
    # global EVPA-swirl metric (EHT standard, Palumbo et al.): m=2 azimuthal mode of the
    # polarization field -- amplitude ratio recovered + phase (orientation) error
    beta2_amp_ratio, beta2_phase_err = ev.beta2_error(recon, truth_cubes, args.fov_uas)
    # record both absolute |beta2| values, not just their ratio: if the truth movie
    # carries no m=2 swirl (|beta2|~0) the ratio is a meaningless 0/0 and can read
    # large while nothing was recovered
    beta2_truth_abs = abs(
        ev.beta2_coefficient(truth_cubes["Q"], truth_cubes["U"], truth_cubes["I"], args.fov_uas)
    )
    # beta2 DYNAMICS: does the recon track a *rotating* swirl frame by frame? The
    # time-averaged beta2 above cancels a rotating EVPA (it reports ~0.01 for a
    # perfect varbeta2 fit), so a dynamic-pol truth must be scored per frame.
    # Recovered station gains. Reported per station/hand so M3's amplitude RMSE is
    # measurable, and so a NEGATIVE CONTROL is visible: fitting gain-free data must
    # leave these at ~1. NB a global amplitude is degenerate with source flux (the
    # --flux anchor breaks it) and a global phase with source position (broken by
    # referencing to --gain-ref-station).
    gain_report = None
    if args.fit_gains and getattr(model, "gains", None) is not None:
        _amp = np.asarray(model.gains.amplitudes())  # (n_st, n_t, n_hands)
        _ph = np.degrees(np.asarray(model.gains.phases()))
        gain_report = {
            "n_stations": int(_amp.shape[0]),
            "n_hands": int(_amp.shape[2]),
            "phase_solved": bool(args.gain_phase),
            "amp_mean_per_station": [[float(x) for x in row] for row in _amp.mean(axis=1)],
            "amp_median": float(np.median(_amp)),
            "amp_min": float(_amp.min()),
            "amp_max": float(_amp.max()),
            "amp_rms_dev_from_1": float(np.sqrt(np.mean((_amp - 1.0) ** 2))),
            "phase_rms_deg": float(np.sqrt(np.mean(_ph**2))),
            "stations": list(op.stations) if op.stations else None,
        }
        print(f"[gains] recovered: {json.dumps(gain_report)}", flush=True)

    beta2_dyn = ev.beta2_dynamics_error(recon, truth_cubes, args.fov_uas)
    beta2_recon_abs = abs(ev.beta2_coefficient(recon["Q"], recon["U"], recon["I"], args.fov_uas))
    # beam-restored metrics: the data only constrain structure to ~the array
    # resolution, so also compare after blurring both cubes to a common beam
    recon_b = ev.blur_polarized_cubes(recon, args.blur_uas, args.fov_uas)
    truth_b = ev.blur_polarized_cubes(truth_cubes, args.blur_uas, args.fov_uas)
    nrmse_b = ev.polarized_nrmse(recon_b, truth_b)
    evpa_err_b = ev.evpa_error_deg(recon_b, truth_b)
    ev.plot_polarized_summary(
        recon, truth_cubes, str(out / "pol_summary.png"), fov_uas=args.fov_uas
    )

    # spatial DMD modes + eigenvalues of the I and pol-fraction fields
    try:
        import matplotlib.pyplot as plt

        xy_grid = ev.pixel_grid_coords(args.npix, args.npix)  # jax accepts numpy
        pol_field_name = "Q" if args.pol_param == "direct" else "mfrac"
        # Under coupling a field borrows its spectrum, so its own temporal net is
        # (partly) untrained -- read the effective Omega the model actually uses.
        shared_kw = model._shared_state(jnp.asarray(xy_grid))
        for name, attr in (("I", "intensity"), (pol_field_name, "frac")):
            sub = getattr(model, attr)
            w0, w, om, b0, b = sub(xy_grid, **shared_kw[attr])
            w_s, om_s, _ = ev.sort_modes_by_lambda(w, om, b)
            ev.plot_modes(
                np.asarray(w_s),
                args.npix,
                args.npix,
                str(out / f"modes_{name}_abs.png"),
                title=f"{name} mode",
                part="abs",
            )
            ev.plot_modes(
                np.asarray(w_s),
                args.npix,
                args.npix,
                str(out / f"modes_{name}_real.png"),
                title=f"{name} mode",
                part="real",
            )
            ev.plot_unit_circle(np.asarray(om_s), str(out / f"omega_{name}.png"))
        plt.close("all")
    except Exception as exc:
        print(f"[warn] mode plots failed: {exc}", flush=True)

    # per-mode table on the shared index: recovered frequencies, per-Stokes
    # amplitudes and phase lags, and how much power sits on shared vs private modes
    if args.couple != "none":
        try:
            window_hr = None
            anchors = getattr(op, "time_anchors_hr", None)
            if anchors is not None:
                window_hr = float(anchors[1]) - float(anchors[0])
            table = ev.mode_table(model, args.npix, truth["times"], window_hr=window_hr)
            (out / "mode_table.json").write_text(json.dumps(table, indent=2))
            for name, entry in table["fields"].items():
                print(
                    f"[modes] {name}: shared power {entry['shared_power']:.3f}, "
                    f"private {entry['private_power']:.3f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[warn] mode table failed: {exc}", flush=True)

    # export the reconstruction as an ehtim HDF5 movie (I, Q, U) for video / scoring
    try:
        recon_movie = to_ehtim_movie(
            recon["I"].astype(np.float64),
            truth["times"],
            fov_uas=args.fov_uas,
            qframes=recon["Q"].astype(np.float64),
            uframes=recon["U"].astype(np.float64),
        )
        save_movie_hdf5(recon_movie, str(out / "recon_pol.hdf5"))
    except Exception as exc:
        print(f"[warn] reconstruction hdf5 export failed: {exc}", flush=True)

    # animated GIFs: truth-vs-recon total/dynamic/static comparison + singles
    try:
        ev.make_polarized_comparison_gif(
            recon,
            truth_cubes,
            str(out / "pol_lp.gif"),
            fov_uas=args.fov_uas,
            times=truth["times"],
            dynamic_quiver=args.dynamic_quiver,
        )
        ev.make_polarized_gif(recon, str(out / "recon_pol.gif"), fov_uas=args.fov_uas)
        ev.make_polarized_gif(truth_cubes, str(out / "truth_pol.gif"), fov_uas=args.fov_uas)
    except Exception as exc:
        print(f"[warn] gif export failed: {exc}", flush=True)

    # report the chi2 of the epoch whose model was checkpointed and evaluated
    # (training restores the best model by worst-product chi2, not the last)
    per_epoch_max = np.max(np.stack([np.asarray(v) for v in hist["chi2"].values()]), axis=0)
    best_ep = int(np.argmin(per_epoch_max))
    final_chi2 = {k: float(v[best_ep]) for k, v in hist["chi2"].items()}
    # surface any disagreement between the history's chi2 and the exported cube's:
    # they describe the same model, so a gap means the reported number is a fiction
    if chi2_from_cube:
        worst = max(
            abs(final_chi2[k] - chi2_from_cube[k]) / max(chi2_from_cube[k], 1e-9)
            for k in chi2_from_cube
            if k in final_chi2
        )
        if worst > 0.15:
            _rep = {k: round(final_chi2[k], 3) for k in chi2_from_cube}
            _cub = {k: round(v, 3) for k, v in chi2_from_cube.items()}
            _flr = {k: round(v, 3) for k, v in chi2_truth_floor.items()}
            print(
                "[WARN] reported chi2 disagrees with the exported cube's chi2 "
                f"(worst rel. gap {worst:.1%}):\n"
                f"       reported (history) : {_rep}\n"
                f"       from exported cube : {_cub}\n"
                f"       truth floor        : {_flr}",
                flush=True,
            )
    metrics = {
        "basis": args.basis,
        "epochs_run": total_epochs,
        "best_epoch": best_ep + 1,
        "final_chi2": final_chi2,
        # independent recomputation from the exported cube + the achievable floor
        "chi2_from_cube": chi2_from_cube,
        "chi2_truth_floor": chi2_truth_floor,
        "nrmse": nrmse,
        "evpa_error_deg": evpa_err,
        "beta2_amp_ratio": beta2_amp_ratio,
        "beta2_phase_err_deg": beta2_phase_err,
        "blur_uas": args.blur_uas,
        "nrmse_blurred": nrmse_b,
        "evpa_error_deg_blurred": evpa_err_b,
        "beta2_truth_abs": float(beta2_truth_abs),
        "gains": gain_report,
        "beta2_dynamics": beta2_dyn,
        "beta2_recon_abs": float(beta2_recon_abs),
        "gate": {
            # [0.4, 2]: the achievable floor here is ~0.42, not 1.0 -- generation
            # inflates sigma via add_fractional_noise (which widens the error bars
            # without adding matching noise), so even the TRUTH scores ~0.42. A band
            # centred on 1.0 could never pass.
            "chi2_in_0.4_2": all(0.4 <= v <= 2.0 for v in final_chi2.values()),
            "nrmse_QU_le_0.15": (nrmse["Q"] <= 0.15 and nrmse["U"] <= 0.15),
            "evpa_le_10deg": bool(evpa_err <= 10.0),
            # global-swirl gate: recover >=70% of the m=2 amplitude with <=20 deg
            # orientation error (the EHT-style "structure recovered" bar).
            # Guarded on the truth actually carrying a swirl -- on a truth with
            # |beta2|~0 the ratio is 0/0 and reads large for a pure-noise recon.
            "beta2_recovered": bool(
                beta2_truth_abs >= 0.05 and beta2_amp_ratio >= 0.7 and abs(beta2_phase_err) <= 20.0
            ),
            "nrmse_QU_blurred_le_0.15": (nrmse_b["Q"] <= 0.15 and nrmse_b["U"] <= 0.15),
            "evpa_blurred_le_10deg": bool(evpa_err_b <= 10.0),
        },
    }
    (out / "m2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
