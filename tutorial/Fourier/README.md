# Fourier-domain tutorial: black-hole imaging of an orbiting hot spot

Reconstruct a dynamic black-hole scene — an m-ring with a hot spot orbiting it
every 80 minutes — from sparse EHT 2017 visibilities. Each movie frame is seen
by at most 42 visibilities; NeuralDMD recovers the full spatiotemporal movie
by sharing spatial modes and a global temporal spectrum across all frames.

## Notebooks (run in order)

| # | Notebook | What it does | Environment | Runtime |
|---|---|---|---|---|
| 1 | [01_generate_data.ipynb](01_generate_data.ipynb) | Synthesize the m-ring + hot spot movie and observe it with the EHT 2017 array | ehtim (no JAX) | ~15–30 min |
| 2 | [02_pretrain_disk_template.ipynb](02_pretrain_disk_template.ipynb) | Initialize the spatial modes with a Zernike disk template | JAX (no ehtim) | ~2 min GPU |
| 3 | [03_train.ipynb](03_train.ipynb) | Train on the complex visibilities | JAX | ~30–45 min GPU |
| 4 | [04_evaluate.ipynb](04_evaluate.ipynb) | Modes, spectrum, reconstruction, χ², forecasting | JAX | ~1 min |

**Setup**: notebooks 02–04 import the `neuraldmd` package — install it once
into your JAX environment from the repo root:

```bash
pip install -e .
```

Data generation (ehtim) and training (JAX) have conflicting dependency
habits, so the tutorial is split so each notebook needs only one of the two —
see `requirements.txt` for both dependency sets. All outputs land in `./data`,
`./models`, and `./plots` (gitignored).

## Code map

The core library lives in the [`neuraldmd` package](../../neuraldmd/):

- `neuraldmd.model` — NeuralDMD architecture (spatial ResidualMLP + temporal
  latent nets, gauge fixing)
- `neuraldmd.training` — visibility-χ² loss, jitted training loop, early
  stopping at the noise level
- `neuraldmd.loader` — `DMDDataLoader`, batches the per-frame observation
  products
- `neuraldmd.zernike` — complex Zernike basis on a disk (masked-QR orthonormal)
- `neuraldmd.pretraining` — radius-of-gyration sizing + Zernike alignment
  pretraining
- `neuraldmd.evaluation` — mode plots, unit-circle spectrum, GIF/MP4 export,
  full-dataset χ²

Local to this tutorial:

- `make_movie.py` — m-ring + orbiting hot spot synthesizer (needs ehtim,
  notebook 01 only)

Dataset formats and observation-generation details live in
[`eht2017/`](../../eht2017/).

## Using your own data

Produce a dataset directory in the documented format (for any array/movie,
[`eht2017/data_generation.py`](../../eht2017/data_generation.py) does it from
an ehtim-format movie), point `obs_dir` at it in notebooks 02–04, and adjust
`r` (number of modes), `num_frequencies`, and the disk radius.
