"""Characterize DMDDataLoader: T+1->T trim, padding, grid, batch shapes, seed."""

from __future__ import annotations

import numpy as np
from _impl import DMDDataLoader


def _loader(tiny_obs, batch_size=2, epochs=1, seed=42, time_fraction=1.0):
    return DMDDataLoader(
        data=tiny_obs.movie,
        batch_size=batch_size,
        epochs=epochs,
        data_dir=tiny_obs.data_dir,
        times=tiny_obs.times,
        fov_x=tiny_obs.fov,
        fov_y=tiny_obs.fov,
        time_fraction=time_fraction,
        seed=seed,
    )


def test_trims_movie_to_observed_frames(tiny_obs):
    ld = _loader(tiny_obs)
    assert ld.num_frames == tiny_obs.T_obs  # 7 -> 6
    assert ld.data.shape[0] == tiny_obs.T_obs
    assert ld.times.shape[0] == tiny_obs.T_obs


def test_pixel_grid_matches_reference(tiny_obs):
    ld = _loader(tiny_obs)
    np.testing.assert_allclose(ld.pixel_coords, tiny_obs.loader_grid(), rtol=1e-6)


def test_epoch_batch_shapes(tiny_obs):
    ld = _loader(tiny_obs, batch_size=2)  # 6 frames / 2 -> 3 batches
    out = ld.get_epoch_data(0)
    assert len(out) == 13
    (frames, coords, A, tgt, sig, msk, times, amp, amps, cpt, cps, cpm, tri) = out
    B, bs, P = 3, 2, tiny_obs.P
    M, K = tiny_obs.M, tiny_obs.K
    assert frames.shape == (B, bs, P)
    assert coords.shape == (P, 2)
    assert A.shape == (B, bs, M, P)
    assert tgt.shape == sig.shape == msk.shape == (B, bs, M)
    assert times.shape == (B, bs)
    assert cpt.shape == cps.shape == cpm.shape == (B, bs, K)
    assert tri.shape == (B, bs, K, 3, 2)


def test_padding_conventions(tiny_obs):
    ld = _loader(tiny_obs)
    n = tiny_obs.n_real
    assert np.all(ld.masks_full[:, n:] == 0.0)
    assert np.all(ld.masks_full[:, :n] == 1.0)
    assert np.all(ld.sigmas_full[:, n:] >= 1e6)
    assert np.all(ld.triangles[..., :] != -1)  # our tiny tris are all valid


def test_seed_determinism(tiny_obs):
    a = _loader(tiny_obs, seed=123).precomputed_time_indices
    b = _loader(tiny_obs, seed=123).precomputed_time_indices
    c = _loader(tiny_obs, seed=999).precomputed_time_indices
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])
