"""Generate NeuralDMD training data from a movie observed with the EHT 2017 array.

Given a movie (ehtim hdf5 format) and a telescope array file, this script
simulates an interferometric observation frame-by-frame with ehtim and saves
everything NeuralDMD's Fourier-domain loader needs: per-frame forward
operators, complex visibilities, amplitudes, closure phases, and diagnostics.

See README.md in this directory for the exact format of every output file.

Requires ehtim (https://github.com/achael/eht-imaging); it does NOT require
JAX, so it can run in a separate environment from training.
"""

from __future__ import annotations

import csv
import json
import pickle
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import ehtim as eh
import numpy as np
from astropy import units as u
from ehtim.imaging.imager_utils import chisqdata
from skimage.transform import resize


@dataclass
class Config:
    # ------------------------------------------------------------------
    # User-facing experiment choices
    # ------------------------------------------------------------------
    movie_name: str = "mring+hs"
    array_name: str = "EHT2017"
    array_suffix: str = "cp"  # tags the output dir, e.g. "cp", "vis", or ""

    fractional_noise: float = 0.04
    scale_factor: int = 4  # image downsampling factor for the model grid

    # If True, visibilities are generated from the resized image grid.
    # If False, they are generated from the original full-resolution movie,
    # while A is still built on the resized/prior grid.
    use_resized_for_observation: bool = True

    include_cp: bool = True
    include_amp: bool = True
    add_noise: bool = True

    # Calibration / corruption flags passed to ehtim observe_same.
    ampcal: bool = True
    phasecal: bool = True
    rlgaincal: bool = True
    stabilize_scan_phase: bool = True
    stabilize_scan_amp: bool = True

    # Observation settings
    tint: float = 5.0  # integration time per visibility [s]
    bw: float = 2.0e9  # bandwidth [Hz]
    ttype: str = "direct"  # ehtim Fourier transform type

    # Paths
    movie_dir: Path = Path(".")
    movie_file: str | None = None  # defaults to f"{movie_name}.hdf5"
    array_dir: Path = Path(__file__).parent / "arrays"  # EHT2017.txt lives next to this file
    output_root: Path = Path("./data")

    # Output values
    save_gt_movies: bool = True
    save_obs_uvfits: bool = True

    @property
    def movie_path(self) -> Path:
        if self.movie_file is not None:
            return Path(self.movie_dir) / self.movie_file
        return Path(self.movie_dir) / f"{self.movie_name}.hdf5"

    @property
    def array_path(self) -> Path:
        return Path(self.array_dir) / f"{self.array_name}.txt"

    @property
    def array_output_name(self) -> str:
        return f"{self.array_name}_{self.array_suffix}" if self.array_suffix else self.array_name

    @property
    def noise_tag(self) -> str:
        return "nonoise" if not self.add_noise else f"f{self.fractional_noise:g}"

    @property
    def cal_tag(self) -> str:
        tags = []
        if not self.ampcal:
            tags.append("ampgain")
        if not self.phasecal:
            tags.append("phasegain")
        if not self.rlgaincal:
            tags.append("rlgain")
        return "_".join(tags)

    @property
    def grid_tag(self) -> str:
        return "resizedobs" if self.use_resized_for_observation else "fullobs"

    @property
    def dataset_tag(self) -> str:
        parts = [self.movie_name, self.noise_tag, self.grid_tag]
        if self.cal_tag:
            parts.append(self.cal_tag)
        return "_".join(parts)

    @property
    def outdir(self) -> Path:
        return Path(self.output_root) / self.array_output_name / self.dataset_tag


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def to_jsonable(d):
    out = {}
    for k, v in d.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def imvec(im) -> np.ndarray:
    return np.asarray(im._imdict["I"])


def imarr(im) -> np.ndarray:
    return imvec(im).reshape(im.ydim, im.xdim)


def make_image_like(arr2d: np.ndarray, ref_im, psize: float | None = None):
    return eh.image.Image(
        np.asarray(arr2d, dtype=np.float64),
        psize=ref_im.psize if psize is None else psize,
        ra=ref_im.ra,
        dec=ref_im.dec,
        rf=ref_im.rf,
        mjd=ref_im.mjd,
        source=getattr(ref_im, "source", "SgrA"),
    )


def align_image_to_obs(im, obs_frame):
    # Rebuild rather than mutate the original ehtim image object
    out = make_image_like(imarr(im), im, psize=im.psize)
    out.ra = obs_frame.ra
    out.dec = obs_frame.dec
    out.mjd = obs_frame.mjd
    return out


def pad(x, shape, value=0, dtype=None):
    x = np.asarray(x)
    y = np.full(shape, value, dtype=dtype or x.dtype)
    slices = tuple(slice(0, s) for s in x.shape)
    y[slices] = x
    return y


# ----------------------------------------------------------------------
# Image/movie I/O
# ----------------------------------------------------------------------


def resize_images(images, scale_factor: int):
    npix_old = images[0].xdim
    npix = npix_old // scale_factor
    new_psize = images[0].psize * scale_factor

    resized = []
    # Preserve total flux: pixel values scale with pixel area
    flux_scaling = (npix_old**2) / (npix**2)

    for im in images:
        arr = imarr(im)
        arr_small = resize(arr, (npix, npix), anti_aliasing=True)
        arr_small = np.clip(arr_small, 0, None) * flux_scaling
        resized.append(make_image_like(arr_small, im, psize=new_psize))

    return resized


def save_movie(images, movie, path: Path):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([imarr(im).astype(np.float32) for im in images])

    with h5py.File(path, "w") as f:
        f.create_dataset("times", data=np.asarray(movie.times))
        f.create_dataset("I", data=frames)
        f.attrs["xdim"] = images[0].xdim
        f.attrs["ydim"] = images[0].ydim
        f.attrs["psize"] = images[0].psize
        f.attrs["ra"] = images[0].ra
        f.attrs["dec"] = images[0].dec
        f.attrs["rf"] = images[0].rf
        f.attrs["mjd"] = images[0].mjd


# ----------------------------------------------------------------------
# Observation generation
# ----------------------------------------------------------------------
def make_obs_schedule(image_ref, movie, array, npix: int, cfg: Config):
    t_frames = np.asarray(movie.times) * u.h
    tstart_hr = float(movie.times[0])
    tstop_hr = float(movie.times[-1])
    t_gather = (t_frames[-1] - t_frames[0]).to("s").value / (len(t_frames) - 1)
    tadv = float(np.floor(t_gather - cfg.tint))

    obs = array.obsdata(
        tint=cfg.tint,
        tadv=tadv,
        tstart=tstart_hr,
        tstop=tstop_hr,
        ra=image_ref.ra,
        dec=image_ref.dec,
        rf=image_ref.rf,
        mjd=image_ref.mjd,
        bw=cfg.bw,
        timetype="UTC",
        polrep="stokes",
    )

    obs_frames = obs.split_obs(t_gather=t_gather)
    prior = eh.image.make_square(obs, npix, image_ref.fovx())
    return obs, obs_frames, prior, image_ref.fovx(), t_gather, tadv


def observe_frame(image, obs_frame, cfg: Config):
    image = align_image_to_obs(image, obs_frame)

    obs = image.observe_same(
        obs_frame,
        ttype=cfg.ttype,
        ampcal=cfg.ampcal,
        phasecal=cfg.phasecal,
        rlgaincal=cfg.rlgaincal,
        stabilize_scan_phase=cfg.stabilize_scan_phase,
        stabilize_scan_amp=cfg.stabilize_scan_amp,
    )

    if cfg.add_noise:
        obs = obs.add_fractional_noise(cfg.fractional_noise)

    return obs


# ----------------------------------------------------------------------
# Visibility / amplitude / closure-phase data
# ----------------------------------------------------------------------
def vis_data(obs, prior):
    target, sigma, A = chisqdata(obs, prior, mask=[], pol="I", dtype="vis")
    return target.astype(np.complex64), sigma.astype(np.float32), A.astype(np.complex64)


def amp_data(obs, prior):
    target, sigma, _ = chisqdata(obs, prior, mask=[], pol="I", dtype="amp")
    return target.astype(np.float32), sigma.astype(np.float32)


def closure_data(vis, amp_sigma, rows, eps: float = 1e-12):
    """
    Build all closure phases from the visibility rows of one frame.

    Returns
    -------
    cp_target : (N_tri,) float32
    cp_sigma  : (N_tri,) float32
    tri_idx   : (N_tri, 3, 2) int32
        tri_idx[:, :, 0] gives baseline row indices.
        tri_idx[:, :, 1] gives signs, where -1 means conjugate that row.
    """
    assert len(vis) == len(rows)

    bl_lookup = {}
    for idx, row in enumerate(rows):
        s1, s2 = str(row[2]), str(row[3])
        bl_lookup[(s1, s2)] = (idx, +1)
        bl_lookup[(s2, s1)] = (idx, -1)

    stations = sorted({str(r[2]) for r in rows} | {str(r[3]) for r in rows})
    cp_list, sig_list, tri_list = [], [], []

    for s1, s2, s3 in combinations(stations, 3):
        try:
            i1, sign1 = bl_lookup[(s1, s2)]
            i2, sign2 = bl_lookup[(s2, s3)]
            i3, sign3 = bl_lookup[(s3, s1)]
        except KeyError:
            continue

        V1 = vis[i1] if sign1 > 0 else np.conj(vis[i1])
        V2 = vis[i2] if sign2 > 0 else np.conj(vis[i2])
        V3 = vis[i3] if sign3 > 0 else np.conj(vis[i3])

        cp = np.angle(V1 * V2 * V3)
        sig = np.sqrt(
            (amp_sigma[i1] / max(abs(V1), eps)) ** 2
            + (amp_sigma[i2] / max(abs(V2), eps)) ** 2
            + (amp_sigma[i3] / max(abs(V3), eps)) ** 2
        )

        cp_list.append(cp)
        sig_list.append(sig)
        tri_list.append([(i1, sign1), (i2, sign2), (i3, sign3)])

    return (
        np.asarray(cp_list, dtype=np.float32),
        np.asarray(sig_list, dtype=np.float32),
        np.asarray(tri_list, dtype=np.int32),
    )


def baseline_station_ids(obs, max_vis: int, station_to_id: dict[str, int]):
    out = np.full((max_vis, 2), -1, dtype=np.int32)
    for i, row in enumerate(obs.data[:max_vis]):
        out[i, 0] = station_to_id[str(row[2])]
        out[i, 1] = station_to_id[str(row[3])]
    return out


def compute_cp_chi2(vis_pred, cp_target, cp_sigma, cp_mask, cp_tri):
    idxs = cp_tri[:, :, 0]
    signs = cp_tri[:, :, 1]

    valid = cp_mask > 0
    idxs = idxs[valid]
    signs = signs[valid]
    cp_target = cp_target[valid]
    cp_sigma = cp_sigma[valid]

    if len(cp_target) == 0:
        return np.nan

    bl1, bl2, bl3 = idxs.T
    s1, s2, s3 = signs.T

    V1 = vis_pred[bl1].copy()
    V2 = vis_pred[bl2].copy()
    V3 = vis_pred[bl3].copy()

    V1[s1 < 0] = np.conj(V1[s1 < 0])
    V2[s2 < 0] = np.conj(V2[s2 < 0])
    V3[s3 < 0] = np.conj(V3[s3 < 0])

    cp_pred = np.angle(V1 * V2 * V3)
    return np.mean(2 * (1 - np.cos(cp_target - cp_pred)) / cp_sigma**2)


def compute_frame_diagnostics(r, A_pad=None, target_pad=None, sigma_pad=None, mask=None):
    """Chi-squared of the (noisy) data against the ground-truth frame itself.

    vis_chi2 is reduced per real degree of freedom (each complex visibility
    carries 2, hence the 2n denominator). With thermal-only noise these hover
    around 1; with an inflated error budget (fractional systematic noise added
    to sigma) they sit below 1.
    """
    A = r["A"] if A_pad is None else A_pad
    target = r["target"] if target_pad is None else target_pad
    sigma = r["sigma"] if sigma_pad is None else sigma_pad

    im = r["model_imvec"]
    if mask is None:
        mask = np.ones(len(r["target"]), dtype=np.float32)
        n = len(r["target"])
    else:
        n = int(np.sum(mask))

    vis_pred = A @ im
    residual = target - vis_pred

    valid_abs_res_sum = float(np.sum(np.abs(residual[:n])))
    vis_chi2 = float(np.sum(np.abs(residual) ** 2 * mask / sigma**2) / (2 * max(n, 1)))

    out = {
        "frame": int(r["frame"]),
        "num_vis": int(n),
        "num_cp": int(len(r.get("cp_target", []))),
        "sum_abs_vis_residual": valid_abs_res_sum,
        "vis_chi2": vis_chi2,
        "sigma_min": float(np.min(r["sigma"])),
        "sigma_max": float(np.max(r["sigma"])),
        "target_abs_min": float(np.min(np.abs(r["target"]))),
        "target_abs_max": float(np.max(np.abs(r["target"]))),
        "uv_max_glambda": float(np.max(np.sqrt(r["u"] ** 2 + r["v"] ** 2)) / 1e9),
    }

    if "amp_target" in r:
        amp_pred = np.abs(vis_pred[:n])
        amp_target = r["amp_target"][:n]
        amp_sigma = r["amp_sigma"][:n]
        out["amp_chi2"] = float(np.sum((amp_target - amp_pred) ** 2 / amp_sigma**2) / max(n, 1))
    else:
        out["amp_chi2"] = np.nan

    if "cp_target" in r:
        cp_mask = np.ones(len(r["cp_target"]), dtype=np.float32)
        out["cp_chi2"] = float(
            compute_cp_chi2(vis_pred, r["cp_target"], r["cp_sigma"], cp_mask, r["cp_tri"])
        )
    else:
        out["cp_chi2"] = np.nan

    return out


def save_diagnostics_table(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Dataset generation and saving
# ----------------------------------------------------------------------
def save_records(records, station_to_id, cfg: Config, npix: int):
    outdir = cfg.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    max_vis = max(len(r["target"]) for r in records)
    max_tri = max((len(r.get("cp_target", [])) for r in records), default=0)
    n_pix = npix * npix

    As, targets, sigmas, masks, num_vis_list = [], [], [], [], []
    amp_targets, amp_sigmas = [], []
    cp_targets, cp_sigmas, cp_masks, cp_tris = [], [], [], []
    uv_coords, bl_station_ids = [], []

    vis_chi2s, amp_chi2s, cp_chi2s = [], [], []
    diagnostic_rows = []

    for r in records:
        n = len(r["target"])
        num_vis_list.append(n)

        A = pad(r["A"], (max_vis, n_pix), 0, np.complex64)
        target = pad(r["target"], (max_vis,), 0, np.complex64)
        sigma = pad(r["sigma"], (max_vis,), 1e6, np.float32)
        mask = pad(np.ones(n, dtype=np.float32), (max_vis,), 0, np.float32)

        As.append(A)
        targets.append(target)
        sigmas.append(sigma)
        masks.append(mask)

        frame_diag = compute_frame_diagnostics(r, A, target, sigma, mask)
        diagnostic_rows.append(frame_diag)
        vis_chi2s.append(frame_diag["vis_chi2"])

        uv = np.full((max_vis, 2), np.nan, dtype=np.float32)
        uv[:n, 0] = r["u"] / 1e9
        uv[:n, 1] = r["v"] / 1e9
        uv_coords.append(uv)

        bl_station_ids.append(baseline_station_ids(r["obs"], max_vis, station_to_id))

        if cfg.include_amp:
            amp_t = pad(r["amp_target"], (max_vis,), 0, np.float32)
            amp_s = pad(r["amp_sigma"], (max_vis,), 1e6, np.float32)
            amp_targets.append(amp_t)
            amp_sigmas.append(amp_s)
            amp_chi2s.append(frame_diag["amp_chi2"])

        if cfg.include_cp:
            m = len(r["cp_target"])
            cp_t = pad(r["cp_target"], (max_tri,), 0, np.float32)
            cp_s = pad(r["cp_sigma"], (max_tri,), 1e6, np.float32)
            cp_m = pad(np.ones(m, dtype=np.float32), (max_tri,), 0, np.float32)
            cp_i = pad(r["cp_tri"], (max_tri, 3, 2), -1, np.int32)

            cp_targets.append(cp_t)
            cp_sigmas.append(cp_s)
            cp_masks.append(cp_m)
            cp_tris.append(cp_i)
            cp_chi2s.append(frame_diag["cp_chi2"])

    np.save(outdir / "As.npy", np.asarray(As))
    np.save(outdir / "targets.npy", np.asarray(targets))
    np.save(outdir / "sigmas.npy", np.asarray(sigmas))
    np.save(outdir / "masks.npy", np.asarray(masks))
    np.save(outdir / "num_vis_list.npy", np.asarray(num_vis_list, dtype=np.int32))
    np.save(outdir / "uv_coords.npy", np.asarray(uv_coords))
    np.save(outdir / "bl_station_ids.npy", np.asarray(bl_station_ids, dtype=np.int32))

    if cfg.include_amp:
        np.save(outdir / "amp_targets.npy", np.asarray(amp_targets))
        np.save(outdir / "amp_sigmas.npy", np.asarray(amp_sigmas))

    if cfg.include_cp:
        np.save(outdir / "cp_targets.npy", np.asarray(cp_targets))
        np.save(outdir / "cp_sigmas.npy", np.asarray(cp_sigmas))
        np.save(outdir / "cp_masks.npy", np.asarray(cp_masks))
        np.save(outdir / "cp_tris.npy", np.asarray(cp_tris))

    with open(outdir / "stations.pkl", "wb") as f:
        pickle.dump(station_to_id, f)

    diagnostics = {
        "max_vis": int(max_vis),
        "max_tri": int(max_tri),
        "mean_vis_chi2": float(np.nanmean(vis_chi2s)),
        "mean_amp_chi2": float(np.nanmean(amp_chi2s)) if amp_chi2s else None,
        "mean_cp_chi2": float(np.nanmean(cp_chi2s)) if cp_chi2s else None,
    }

    save_diagnostics_table(diagnostic_rows, outdir / "frame_diagnostics.csv")

    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(json.dumps(diagnostics, indent=2))


def generate(cfg: Config):
    from tqdm import tqdm

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    with open(cfg.outdir / "config.json", "w") as f:
        json.dump(to_jsonable(asdict(cfg)), f, indent=2)

    print("Movie:", cfg.movie_path)
    print("Array:", cfg.array_path)
    print("Output:", cfg.outdir)

    array = eh.array.load_txt(str(cfg.array_path))
    movie = eh.movie.load_hdf5(str(cfg.movie_path))

    images_full = movie.im_list()
    images_resized = resize_images(images_full, cfg.scale_factor)

    npix_full = images_full[0].xdim
    npix = images_resized[0].xdim

    if cfg.save_gt_movies:
        save_movie(images_full, movie, cfg.outdir / "gt_video.hdf5")
        save_movie(images_resized, movie, cfg.outdir / "gt_video_resized.hdf5")

    obs0, obs_frames, prior, fov, t_gather, tadv = make_obs_schedule(
        image_ref=images_resized[0],
        movie=movie,
        array=array,
        npix=npix,
        cfg=cfg,
    )

    station_to_id = {str(site): i for i, site in enumerate(obs0.tarr["site"])}

    images_obs = images_resized if cfg.use_resized_for_observation else images_full
    images_model = images_resized

    if len(obs_frames) > len(images_obs):
        raise ValueError(
            f"More obs frames ({len(obs_frames)}) than movie frames ({len(images_obs)}). "
            "Check movie.times, t_gather, tstart, and tstop."
        )

    records = []
    all_obs = []
    for i, obs_frame in tqdm(list(enumerate(obs_frames)), desc="Generating frames"):
        obs = observe_frame(images_obs[i], obs_frame, cfg)
        target, sigma, A = vis_data(obs, prior)

        model_imvec = imvec(align_image_to_obs(images_model[i], obs_frame)).astype(np.float32)

        if A.shape[1] != model_imvec.size:
            raise ValueError(
                f"A has {A.shape[1]} pixels but model image has {model_imvec.size}. "
                "This usually means prior npix and image npix do not match."
            )

        r = {
            "frame": i,
            "obs": obs,
            "A": A,
            "target": target,
            "sigma": sigma,
            "model_imvec": model_imvec,
            "u": obs.data["u"][: len(target)].astype(np.float32),
            "v": obs.data["v"][: len(target)].astype(np.float32),
        }

        if cfg.include_amp or cfg.include_cp:
            amp_target, amp_sigma = amp_data(obs, prior)
            r["amp_target"] = amp_target
            r["amp_sigma"] = amp_sigma

        if cfg.include_cp:
            cp_target, cp_sigma, cp_tri = closure_data(
                target, r["amp_sigma"], obs.data[: len(target)]
            )
            r["cp_target"] = cp_target
            r["cp_sigma"] = cp_sigma
            r["cp_tri"] = cp_tri

        records.append(r)
        all_obs.append(obs)

    save_records(records, station_to_id, cfg, npix=npix)

    if cfg.save_obs_uvfits:
        final_obs = eh.obsdata.merge_obs(all_obs)
        final_obs.save_uvfits(str(cfg.outdir / "obs.uvfits"))

    print("Saved dataset to:", cfg.outdir)
    print("npix_full:", npix_full, "npix:", npix, "fov:", fov)
    print("t_gather:", t_gather, "tadv:", tadv)
    return cfg.outdir


def generate_polarized_dataset(
    out_dir,
    *,
    npix: int = 50,
    fov_uas: float = 200.0,
    num_frames: int = 64,
    tstart_hr: float = 9.0,
    tstop_hr: float = 15.0,
    tint: float = 30.0,
    bw: float = 2.0e9,
    linpol_frac: float = 0.2,
    fractional_noise: float = 0.04,
    ampcal: bool = True,
    phasecal: bool = True,
    array_name: str = "EHT2017",
    array_dir=None,
    stokes: tuple[str, ...] = ("I", "Q", "U"),
    basis: str = "stokes",
    ttype: str = "direct",
    seed: int = 42,
    save_truth: bool = True,
    **mring_kwargs,
):
    """Generate the canonical polarized ``mring+hsCW`` dataset and write a v2 obs_dir.

    Pipeline: build the polarized m-ring + hot-spot movie
    (:func:`neuraldmd.data.movies.make_mring_hs_pol_movie`, an ehtim thick m-ring
    with radial-EVPA polarization and an orbiting hot spot) at the model grid,
    observe it with the EHT array (Stokes polrep, thermal + optional fractional
    noise), save ``obs.uvfits``, ingest via :func:`load_uvfits_to_products` in the
    requested ``basis``, and write the obs_dir. Requires ehtim.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output directory (obs_dir + ``obs.uvfits`` + ``truth_pol.npz``).
    npix, fov_uas, num_frames, tstart_hr, tstop_hr, tint, bw
        Image grid, field of view, sampling, integration time [s], bandwidth [Hz].
    linpol_frac : float
        Fractional linear polarization of the ring (spiral EVPA).
    fractional_noise : float
        Fractional systematic noise added in quadrature (0 disables).
    ampcal, phasecal : bool
        If False, ehtim injects amplitude/phase gain errors (for later cal work).
    array_name, array_dir : str, path
        Telescope array (defaults to the packaged EHT2017).
    stokes, basis : passed to :func:`load_uvfits_to_products`.
    ttype : str
        ehtim Fourier type; ``"direct"`` gives the dense operator the loader needs.
    seed : int
        Observation noise seed.
    save_truth : bool
        Save ``truth_pol.npz`` (I, Q, U cubes + normalized frame times).
    **mring_kwargs
        m-ring / hot-spot / polarization overrides for
        :func:`make_mring_hs_pol_movie` (``diameter_uas``, ``alpha_uas``,
        ``beta1_abs``, ``period_min``, ``direction``, ``circpol_frac``, ...).

    Returns
    -------
    ObsProducts
        The ingested dataset (also written to ``out_dir``).
    """
    import ehtim as eh

    from .movies import make_mring_hs_pol_movie
    from .observations import load_uvfits_to_products

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    array_dir = Path(array_dir) if array_dir is not None else Path(__file__).parent / "arrays"
    array = eh.array.load_txt(str(array_dir / f"{array_name}.txt"))

    movie = make_mring_hs_pol_movie(
        npix=npix,
        fov_uas=fov_uas,
        num_frames=num_frames,
        tstart_hr=tstart_hr,
        tstop_hr=tstop_hr,
        linpol_frac=linpol_frac,
        **mring_kwargs,
    )

    tadv = float((tstop_hr - tstart_hr) * 3600.0 / max(num_frames - 1, 1))
    obs = movie.observe(
        array,
        tint,
        tadv,
        tstart_hr,
        tstop_hr,
        bw,
        polrep_obs="stokes",
        ttype=ttype,
        add_th_noise=True,
        ampcal=ampcal,
        phasecal=phasecal,
        seed=seed,
        verbose=False,
    )
    if fractional_noise:
        obs = obs.add_fractional_noise(fractional_noise)

    uvfits_path = out_dir / "obs.uvfits"
    obs.save_uvfits(str(uvfits_path))

    op = load_uvfits_to_products(
        uvfits_path,
        npix=npix,
        fov_uas=fov_uas,
        stokes=tuple(stokes),
        basis=basis,
        # anchor the training clock on the movie window so the model's
        # normalized times and the truth cubes' times are the same axis
        time_anchors=(tstart_hr, tstop_hr),
    )
    op.to_obs_dir(out_dir)

    if save_truth:
        save_truth_npz(movie, out_dir, npix, fov_uas)
    return op


def save_truth_npz(movie, out_dir, npix: int, fov_uas: float) -> None:
    """Save an ehtim movie's Stokes cubes as ``truth_pol.npz`` for evaluation.

    Writes ``I``/``Q``/``U`` cubes of shape ``(T, npix, npix)`` plus frame times
    normalized to [0, 1] over the movie's own span -- the same normalization the
    training clock uses when the dataset is generated with matching
    ``time_anchors``.

    Parameters
    ----------
    movie : ehtim.movie.Movie
        Ground-truth movie (frames carry ``imvec``/``qvec``/``uvec``).
    out_dir : str or pathlib.Path
        Destination directory for ``truth_pol.npz``.
    npix : int
        Image grid side length of the movie frames.
    fov_uas : float
        Field of view [micro-arcsec] recorded alongside the cubes.

    Returns
    -------
    None
        Writes ``out_dir / "truth_pol.npz"``.
    """
    out_dir = Path(out_dir)
    ims = movie.im_list()

    def _cube(attr):
        return np.stack([np.asarray(getattr(im, attr)).reshape(npix, npix) for im in ims]).astype(
            np.float32
        )

    mtimes = np.array([float(im.time) for im in ims])
    span = float(mtimes.max() - mtimes.min())
    tnorm = (mtimes - mtimes.min()) / (span if span > 0 else 1.0)
    np.savez(
        out_dir / "truth_pol.npz",
        I=_cube("imvec"),
        Q=_cube("qvec"),
        U=_cube("uvec"),
        times=tnorm.astype(np.float32),
        npix=npix,
        fov_uas=fov_uas,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie", required=True, help="Path to the movie hdf5 (ehtim format)")
    parser.add_argument(
        "--movie-name",
        default=None,
        help="Tag used in the output dir name (default: movie file stem)",
    )
    parser.add_argument("--out", default="./data", help="Output root directory")
    parser.add_argument(
        "--array", default="EHT2017", help="Array name (expects <array>.txt in --array-dir)"
    )
    parser.add_argument(
        "--array-dir",
        default=str(Path(__file__).parent / "arrays"),
        help="Directory containing the array .txt file",
    )
    parser.add_argument("--noise", type=float, default=0.04, help="Fractional systematic noise")
    parser.add_argument(
        "--scale-factor", type=int, default=5, help="Downsampling factor for the model grid"
    )
    args = parser.parse_args()

    movie_path = Path(args.movie)
    cfg = Config(
        movie_name=args.movie_name or movie_path.stem,
        movie_dir=movie_path.parent,
        movie_file=movie_path.name,
        array_name=args.array,
        array_dir=Path(args.array_dir),
        output_root=Path(args.out),
        fractional_noise=args.noise,
        scale_factor=args.scale_factor,
    )
    generate(cfg)
