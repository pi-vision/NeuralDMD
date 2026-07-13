# EHT 2017 data: implementation details

This directory documents how NeuralDMD's Fourier-domain (black-hole imaging)
datasets are generated and what every file in a dataset directory means. The
Fourier tutorial ([tutorial/Fourier/](../tutorial/Fourier/)) consumes datasets
in exactly this format; if you want to run NeuralDMD on your own array or
movie, this is the format to produce.

Contents:

- `EHT2017.txt` — the April 2017 Event Horizon Telescope array (8 stations:
  ALMA, APEX, JCMT, LMT, PV, SMA, SMT, SPT) with positions, SEFDs, and
  polarimetric leakage terms, in ehtim's array text format.
- `data_generation.py` — config-driven script that turns a movie (ehtim hdf5)
  into a NeuralDMD training dataset by simulating an interferometric
  observation frame by frame.

## How a dataset is generated

`data_generation.py` does, per movie frame:

1. **Resize** the movie to the model grid: the full-resolution movie (e.g.
   200×200) is downsampled by `scale_factor` (e.g. 4 → 50×50) with
   anti-aliasing, clipped at zero, and rescaled by `(npix_old/npix)^2` so that
   total flux (Jy) is preserved.
2. **Schedule** an observation with `array.obsdata(...)`: integration time
   `tint` (default 5 s), bandwidth `bw` (default 2 GHz), scan cadence `tadv`
   derived from the movie's frame spacing (`t_gather` = time between frames),
   at the movie's `ra/dec/rf/mjd`. The scan list is split into per-frame
   observations with `split_obs(t_gather)`.
3. **Observe** frame *i* with `image.observe_same(obs_frame, ...)`, which adds
   thermal noise from the station SEFDs. `add_fractional_noise(f)` then
   inflates the error budget with a systematic term (`f = 0.04` by default),
   as is standard in EHT analyses.
4. **Extract data products** with `ehtim.imaging.imager_utils.chisqdata`:
   complex visibilities (`dtype="vis"`) and visibility amplitudes
   (`dtype="amp"`), together with the forward operator **A** — the discrete
   Fourier matrix mapping the flattened model image to the sampled (u,v)
   points of that frame.
5. **Build closure phases** from every station triangle available in the
   frame: for triangle (s1,s2,s3) the bispectrum phase
   `angle(V_12 V_23 V_31)` is recorded, where a baseline stored as (s2,s1) is
   conjugated (sign −1, see `cp_tris.npy` below). Closure-phase uncertainties
   use the standard first-order propagation `sqrt(sum (sigma_i/|V_i|)^2)`.
6. **Pad** everything to fixed shapes so frames can be stacked into arrays
   (the number of visibilities per frame varies as stations rise and set),
   and save diagnostics.

Because the EHT observes each instant with only a handful of baselines
(≤ ~42 visibilities per frame here, from up to 8 stations), each frame is
extremely underconstrained on its own — this is the regime NeuralDMD is
designed for: the dynamics are constrained jointly across all frames through
the shared spatial modes and global temporal spectrum.

## Dataset directory layout

A dataset lives at `<output_root>/<ARRAY>_<suffix>/<movie>_<noise>_<grid>/`,
e.g. `data/EHT2017_cp/mring+hs_f0.04_resizedobs/`. Shapes below use:

- `T` — number of frames
- `M` — `max_vis`, the maximum number of visibilities in any frame
- `K` — `max_tri`, the maximum number of closure triangles in any frame
- `P` — number of model pixels (`npix²`, e.g. 2500 for a 50×50 grid)

| File | Shape / dtype | Meaning |
|---|---|---|
| `As.npy` | `(T, M, P)` complex64 | Per-frame forward operators. `A[t] @ image_vec` = model visibilities of frame `t`. Rows beyond `num_vis_list[t]` are zero-padded. |
| `targets.npy` | `(T, M)` complex64 | Observed complex visibilities (padded with 0). |
| `sigmas.npy` | `(T, M)` float32 | Thermal + systematic std dev per visibility. **Padded with 1e6** so padded rows are harmless even if a mask is forgotten. |
| `masks.npy` | `(T, M)` float32 | 1 for real measurements, 0 for padding. |
| `num_vis_list.npy` | `(T,)` int32 | Number of valid visibilities per frame. |
| `amp_targets.npy` | `(T, M)` float32 | Visibility amplitudes (debiased by ehtim; padded with 0). |
| `amp_sigmas.npy` | `(T, M)` float32 | Amplitude uncertainties (padded with 1e6). |
| `cp_targets.npy` | `(T, K)` float32 | Closure phases in **radians** (padded with 0). |
| `cp_sigmas.npy` | `(T, K)` float32 | Closure-phase uncertainties in radians (padded with 1e6). |
| `cp_masks.npy` | `(T, K)` float32 | 1 for real triangles, 0 for padding. |
| `cp_tris.npy` | `(T, K, 3, 2)` int32 | Triangle bookkeeping. `[..., 0]` = visibility **row index** into the `M` axis; `[..., 1]` = sign, −1 meaning that visibility enters the bispectrum conjugated. Padded with −1. |
| `bl_station_ids.npy` | `(T, M, 2)` int32 | Global station ids (see `stations.pkl`) of the two stations forming each baseline; −1 for padding. Needed only for gain modeling. |
| `uv_coords.npy` | `(T, M, 2)` float32 | (u, v) in **Gigalambda** (NaN for padding). For plotting coverage; the physics is already inside `As.npy`. |
| `stations.pkl` | dict | `{station_name: global_id}` mapping used by `bl_station_ids.npy`. |
| `gt_video.hdf5` | `(T, H_full, W_full)` | Ground-truth movie at full resolution (`I`, `times` datasets + geometry attrs). |
| `gt_video_resized.hdf5` | `(T, npix, npix)` | Ground truth on the model grid — what a perfect reconstruction would look like. |
| `obs.uvfits` | uvfits | All frames merged into a single ehtim observation, for use with other imaging software. |
| `config.json` | json | The exact `Config` used to generate the dataset. |
| `diagnostics.json` / `frame_diagnostics.csv` | json / csv | Sanity χ² of the noisy data against the ground truth itself, per dataset and per frame — the effective noise floor for training (≈ 0.6 with 4% fractional noise; see χ² definitions below). |

### Conventions worth knowing

- **Padding**: variable-length per-frame data are padded to fixed `M`/`K`.
  Always multiply by the mask *and* divide by `sum(mask)` when averaging;
  σ-padding of 1e6 is a second line of defense, not the primary mechanism.
- **Units**: images are in Jy/pixel on the model grid; visibilities in Jy;
  closure phases in radians; uv in Gλ (only in `uv_coords.npy`).
- **The forward operator absorbs all geometry.** The network's pixel
  coordinates are *normalized* (the tutorial uses a nominal fov of π in
  arbitrary units); the mapping to physical angular scale lives entirely in
  `As.npy`. Changing the network's coordinate convention does not change the
  data model.
- **χ² definitions** used in training/evaluation (per batch, mask-averaged),
  all *reduced per real degree of freedom* so that ≈ 1 means the model fits
  the data at the noise level:
  - complex vis: `Σ mask·|V_pred − V_obs|²/σ² / (2·Σ mask)` — the factor 2
    because each complex visibility carries two degrees of freedom (Re and
    Im, each with variance σ²);
  - amplitude: `Σ mask·(|V_pred| − amp_obs)²/σ_amp² / Σ mask`
  - closure phase (phasor form, robust to wrapping):
    `Σ mask·|e^{iψ_pred} − e^{iψ_obs}|²/σ_cp² / Σ mask`, where
    `|e^{iψ_pred}| = 1` is enforced by normalizing the predicted bispectrum.

  Because `add_fractional_noise` inflates σ *without* adding matching noise
  to the data, the ground truth itself scores below 1 on these datasets
  (≈ 0.6 at 4% fractional noise) — treat that value, reported in
  `diagnostics.json`, as the effective noise floor. Training stops early once
  `chi2_vis` reaches ~1 (see `train_model`'s `early_stop_chi2`); fitting far
  below the ground-truth level would mean fitting noise.

## Regenerating or customizing

From this directory (needs ehtim, astropy, scikit-image; **not** JAX):

```bash
python data_generation.py \
    --movie /path/to/movie.hdf5 \
    --out ./data \
    --noise 0.04 --scale-factor 4
```

or from Python (what the tutorial's first notebook does):

```python
from data_generation import Config, generate
cfg = Config(movie_name="mring+hs", movie_dir="...", movie_file="mring+hs.hdf5",
             output_root="./data", fractional_noise=0.04, scale_factor=4)
generate(cfg)
```

To use a different array, drop another ehtim-format `<name>.txt` in this
directory and pass `array_name=<name>`. To study calibration errors, the
`ampcal`/`phasecal` flags of `Config` are forwarded to ehtim's
`observe_same` (the tutorial keeps everything calibrated; gain handling in
the model is deliberately out of scope for the tutorial).

The movie must be an ehtim-format hdf5 (`eh.movie.load_hdf5` must read it),
with `ra/dec/rf/mjd` set to something the array can actually observe — for
Sgr A* use `ra=17.761`, `dec=-29.008`, `rf=227.07e9`, `mjd=57854`.
