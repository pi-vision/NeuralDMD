# NeuralDMD

NeuralDMD fuses classic dynamic mode decomposition (DMD) with neural implicit fields to reconstruct full-resolution spatiotemporal data from sparse pixel samples or incomplete Fourier (visibility) measurements.

Key features
----------------------
- Reconstruct images, videos, or volumes from highly undersampled measurements (< 1 % pixels or sparse visibilities)

- Provide interpretable spatial modes and temporal spectrum, with a static/dynamic decomposition and forecasting for free.

- Train on CPU or GPU through JAX (CUDA 12 supported)

Requirements
----------------------
See requirements.txt for the full list. The Fourier tutorial uses two dependency sets (ehtim for data generation, JAX for training) — see tutorial/Fourier/requirements.txt.

Installation

```# clone
git clone git@github.com:as2c/NeuralDMD.git
cd NeuralDMD

# (optional) virtual environment
python -m venv .neuraldmd_env
source .neuraldmd_env/bin/activate              # Windows: .venv\Scripts\activate

# core dependencies
pip install -r requirements.txt

# GPU acceleration
pip install "jax[cuda12]"
```

Repository layout

```
NeuralDMD/
 ├─ tutorial/
 │   ├─ pixel/            # sparse-pixel experiment (Apr 1–7 2025 weather data)
 │   └─ Fourier/          # sparse-visibility experiment: EHT 2017 imaging of
 │                        # an orbiting hot spot (data → pretrain → train → evaluate)
 ├─ eht2017/              # EHT 2017 observation pipeline + data-format reference
 └─ requirements.txt
```

Quick start

Pixel-domain example
----------------------
```
cd tutorial/pixel
python train_model.py    # train on 10 % random pixels
after training:
python test_model.py     # plot modes/spectrum and save GIF/MP4
```

Fourier-domain example (black-hole imaging)
----------------------
Run the notebooks in `tutorial/Fourier/` in order:

1. `01_generate_data.ipynb` — synthesize an m-ring + orbiting hot spot movie and observe it with the EHT 2017 array (needs ehtim, no JAX)
2. `02_pretrain_disk_template.ipynb` — initialize the spatial modes with a Zernike disk template (needs JAX)
3. `03_train.ipynb` — train on the sparse complex visibilities
4. `04_evaluate.ipynb` — modes, temporal spectrum, reconstruction, χ², forecasting

See `tutorial/Fourier/README.md` for environments and runtimes, and `eht2017/README.md` for the dataset format and observation-generation details.

Custom data workflow
----------------------
- Pixel domain: convert your sequence to NumPy `.npy` or NetCDF, place it under `tutorial/<new_expt>/data/`, adjust parameters in `train_model.py` (rank, learning rate, mask), and run.
- Fourier domain: produce a dataset directory in the format documented in `eht2017/README.md` (for ehtim-format movies, `eht2017/data_generation.py` does this for any array), then point the Fourier notebooks' `obs_dir` at it.

Citation
----------------------
```
@misc{saraertoosi2025neuraldynamicmodescomputational,
  title        = {Neural Dynamic Modes: Computational Imaging of Dynamical Systems from Sparse Observations},
  author       = {Ali SaraerToosi and Renbo Tu and Kamyar Azizzadenesheli and Aviad Levis},
  year         = {2025},
  eprint       = {2507.03094},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2507.03094}
}
```
