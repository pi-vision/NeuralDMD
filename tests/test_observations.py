"""ObsProducts: legacy-v1 load, v2 round-trip, validation."""

from __future__ import annotations

import numpy as np
import pytest

from neuraldmd.data.observations import ObsProducts


def test_load_legacy_v1_as_stokes_i(tiny_obs):
    """A legacy (unsuffixed) obs_dir loads as a Stokes-I v1 dataset."""
    op = ObsProducts.from_obs_dir(tiny_obs.data_dir)
    assert op.stokes == ("I",)
    assert op.version == 1
    assert op.n_frames == tiny_obs.T_obs
    assert op.n_pixels == tiny_obs.P
    np.testing.assert_array_equal(op.targets["I"], tiny_obs.targets)
    np.testing.assert_array_equal(op.masks["I"], tiny_obs.masks)


def _synthetic_iqu(T=4, M=6, P=25, seed=0):
    """Build a small random (I, Q, U) ObsProducts for round-trip tests.

    Parameters
    ----------
    T, M, P : int
        Frame, visibility, and pixel counts.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ObsProducts
        A self-consistent random three-Stokes dataset.
    """
    rng = np.random.default_rng(seed)
    A = (rng.normal(size=(T, M, P)) + 1j * rng.normal(size=(T, M, P))).astype(np.complex64)
    stokes = ("I", "Q", "U")
    targets = {
        s: (rng.normal(size=(T, M)) + 1j * rng.normal(size=(T, M))).astype(np.complex64)
        for s in stokes
    }
    sigmas = {s: np.abs(rng.normal(size=(T, M))).astype(np.float32) + 0.1 for s in stokes}
    masks = {s: (rng.random((T, M)) > 0.2).astype(np.float32) for s in stokes}
    return ObsProducts(A, stokes, targets, sigmas, masks)


def test_v2_roundtrip(tmp_path):
    """Writing then reloading a v2 obs_dir recovers A and per-Stokes arrays."""
    op = _synthetic_iqu()
    op.to_obs_dir(tmp_path / "obs")
    assert (tmp_path / "obs" / "manifest.json").exists()
    assert (tmp_path / "obs" / "targets_Q.npy").exists()

    back = ObsProducts.from_obs_dir(tmp_path / "obs")
    assert back.stokes == ("I", "Q", "U")
    assert back.version == 2
    np.testing.assert_array_equal(back.A, op.A)  # A stored once, shared
    for s in op.stokes:
        np.testing.assert_array_equal(back.targets[s], op.targets[s])
        np.testing.assert_array_equal(back.sigmas[s], op.sigmas[s])
        np.testing.assert_array_equal(back.masks[s], op.masks[s])


def test_masks_are_per_stokes(tmp_path):
    """Per-Stokes masks are independent and survive a round-trip."""
    # a station flagged for Q but not I -> different masks survive round-trip
    op = _synthetic_iqu(seed=1)
    op.masks["Q"][:, 0] = 0.0
    op.masks["I"][:, 0] = 1.0
    op.to_obs_dir(tmp_path / "o")
    back = ObsProducts.from_obs_dir(tmp_path / "o")
    assert np.all(back.masks["Q"][:, 0] == 0.0)
    assert np.all(back.masks["I"][:, 0] == 1.0)


def test_validate_catches_shape_mismatch():
    """A per-Stokes array whose shape disagrees with A raises ValueError."""
    A = np.zeros((3, 5, 9), np.complex64)
    good = {"I": np.zeros((3, 5), np.complex64)}
    with pytest.raises(ValueError, match="shape"):
        ObsProducts(A, ("I",), {"I": np.zeros((3, 4), np.complex64)}, good, good)


def test_validate_catches_missing_stokes():
    """Declaring a Stokes with no matching target array raises ValueError."""
    A = np.zeros((3, 5, 9), np.complex64)
    t = {"I": np.zeros((3, 5), np.complex64)}
    with pytest.raises(ValueError, match="!="):
        ObsProducts(A, ("I", "Q"), t, t, t)  # targets missing Q
