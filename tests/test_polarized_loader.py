"""PolarizedDMDDataLoader batching + end-to-end polarized training through the
epoch scan and driver (pure numpy/jax fixture; no ehtim)."""

from __future__ import annotations

import jax
import numpy as np

from neuraldmd.data.loader import PolarizedDMDDataLoader
from neuraldmd.data.observations import ObsProducts
from neuraldmd.polarized import PolarizedNeuralDMD
from neuraldmd.training import train_polarized_model

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)


def _fittable_obs(stokes=("I", "Q", "U"), npix=8, t=6, m=5, seed=0):
    """ObsProducts whose targets are exact visibilities of a fixed truth image."""
    rng = np.random.default_rng(seed)
    p = npix * npix
    a = (rng.normal(size=(t, m, p)) + 1j * rng.normal(size=(t, m, p))).astype(np.complex64)
    truth = {s: rng.normal(size=(p, t)).astype(np.float32) for s in stokes}
    targets = {
        s: np.einsum("tvp,pt->tv", a, truth[s].astype(np.complex64)).astype(np.complex64)
        for s in stokes
    }
    sig = {s: np.ones((t, m), np.float32) for s in stokes}
    msk = {s: np.ones((t, m), np.float32) for s in stokes}
    return ObsProducts(a, stokes, targets, sig, msk)


def test_epoch_batch_shapes():
    """get_epoch_data returns correctly batched per-key dicts and coordinates."""
    op = _fittable_obs(npix=8, t=6, m=5)
    loader = PolarizedDMDDataLoader(op, npix=8, batch_size=2, epochs=3)
    coords, a_b, tgt, sig, msk, times = loader.get_epoch_data(0)
    assert coords.shape == (64, 2)
    assert a_b.shape == (3, 2, 5, 64)  # (n_batches, batch_size, M, P)
    assert set(tgt) == {"I", "Q", "U"}
    for s in ("I", "Q", "U"):
        assert tgt[s].shape == sig[s].shape == msk[s].shape == (3, 2, 5)
    assert times.shape == (3, 2)


def test_loader_serves_circular_product_keys():
    """The loader is key-agnostic: product-keyed data round-trips too."""
    op = _fittable_obs(stokes=("RR", "LL", "RL", "LR"), npix=8, t=4, m=5)
    loader = PolarizedDMDDataLoader(op, npix=8, batch_size=2, epochs=2)
    _, _, tgt, _, _, _ = loader.get_epoch_data(0)
    assert set(tgt) == {"RR", "LL", "RL", "LR"}


def test_from_obs_dir_roundtrip(tmp_path):
    """from_obs_dir reads a written v2 obs_dir and yields the same keys."""
    op = _fittable_obs(npix=8, t=4, m=5)
    op.to_obs_dir(tmp_path / "obs")
    loader = PolarizedDMDDataLoader.from_obs_dir(tmp_path / "obs", npix=8, batch_size=2, epochs=2)
    assert loader.keys == ("I", "Q", "U")
    assert loader.num_frames == 4


def test_end_to_end_minitrain_reduces_loss(tmp_path):
    """train_polarized_model over the loader reduces total loss and every χ²."""
    op = _fittable_obs(npix=8, t=6, m=5)
    loader = PolarizedDMDDataLoader(
        op, npix=8, batch_size=2, epochs=15, times=np.linspace(0.0, 1.0, 6)
    )
    model = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    fmax = {s: 1.0 for s in op.stokes}
    fmin = {s: 0.0 for s in op.stokes}
    model, hist = train_polarized_model(
        model, loader, num_epochs=15, key=jax.random.PRNGKey(1), models_dir=str(tmp_path),
        frame_max=fmax, frame_min=fmin, initial_lr=3e-3,
        neg_weight=0.0, w_sparse_weight=0.0, b_sparse_weight=0.0, print_every=1000,
    )
    assert hist["total"][-1] < hist["total"][0]
    for s in op.stokes:
        assert hist["chi2"][s][-1] < hist["chi2"][s][0]
    assert (tmp_path / "polarized_model.eqx").exists()
