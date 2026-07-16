"""ObsProducts must survive the obs_dir round-trip bit-for-bit.

Training scores chi2 against the in-memory ``op`` returned by generation, while every
audit, ``--reuse-data`` rerun, and downstream analysis reads the obs_dir that same call
wrote. If the two differ, the reported chi2 describes an operator nobody else ever sees
-- which is exactly the ~8x reported-vs-actual chi2 gap observed on the generate path
(run 29309: reported RR 9.07, its own exported cube scores 72.68).

numpy-only: builds ObsProducts directly, so it runs in the core suite without ehtim.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuraldmd.data.observations import ObsProducts

P = ("RR", "LL", "RL", "LR")


def _fake_op(t=5, m=7, npix=4, seed=0):
    """A padded, masked ObsProducts shaped exactly like the real polarized ones."""
    rng = np.random.default_rng(seed)
    n_pix = npix * npix
    A = (rng.normal(size=(t, m, n_pix)) + 1j * rng.normal(size=(t, m, n_pix))).astype(np.complex64)
    targets, sigmas, masks = {}, {}, {}
    for k in P:
        targets[k] = (rng.normal(size=(t, m)) + 1j * rng.normal(size=(t, m))).astype(np.complex64)
        sigmas[k] = (np.abs(rng.normal(size=(t, m))) + 0.05).astype(np.float32)
        mk = (rng.random((t, m)) > 0.25).astype(np.float32)
        mk[:, -1] = 0.0  # a padded column, as the real loader produces
        sigmas[k][:, -1] = 1e6
        masks[k] = mk
    return ObsProducts(
        A=A,
        stokes=P,
        targets=targets,
        sigmas=sigmas,
        masks=masks,
        bl_station_ids=rng.integers(-1, 3, size=(t, m, 2)).astype(np.int32),
        stations=("ALMA", "PV", "LMT"),
        times=np.linspace(0.0, 0.984, t).astype(np.float32),
        time_anchors_hr=(9.0, 15.0),
    )


def test_obs_dir_roundtrip_is_bit_exact(tmp_path):
    """Every array survives to_obs_dir -> from_obs_dir unchanged."""
    op = _fake_op()
    op.to_obs_dir(tmp_path)
    back = ObsProducts.from_obs_dir(tmp_path)

    assert np.array_equal(op.A, back.A), "A changed across the obs_dir round-trip"
    assert tuple(back.stokes) == tuple(op.stokes)
    for k in P:
        assert np.array_equal(op.targets[k], back.targets[k]), f"targets[{k}] changed"
        assert np.array_equal(op.sigmas[k], back.sigmas[k]), f"sigmas[{k}] changed"
        assert np.array_equal(op.masks[k], back.masks[k]), f"masks[{k}] changed"


def test_obs_dir_roundtrip_preserves_times(tmp_path):
    """times must survive: the model's clock (t_scale=200) amplifies any drift.

    A 1.6% time error is ~3 rad of temporal-mode phase, which silently scrambles the
    reconstruction while leaving the images superficially ring-like.
    """
    op = _fake_op()
    op.to_obs_dir(tmp_path)
    back = ObsProducts.from_obs_dir(tmp_path)
    assert back.times is not None, (
        "times.npy was not round-tripped (loader falls back to linspace!)"
    )
    assert np.array_equal(op.times, back.times)


def test_obs_dir_roundtrip_preserves_chi2_of_a_fixed_image(tmp_path):
    """The SAME image must score the SAME chi2 through the in-memory op and the saved one.

    This is the invariant that actually matters: training scores against the in-memory
    op and every audit scores against the obs_dir. If they disagree, the reported chi2
    is meaningless to everyone downstream.
    """
    op = _fake_op()
    op.to_obs_dir(tmp_path)
    back = ObsProducts.from_obs_dir(tmp_path)
    rng = np.random.default_rng(1)
    t, _, n_pix = op.A.shape
    img = rng.normal(size=(t, n_pix)).astype(np.complex64)  # arbitrary fixed "reconstruction"

    def chi2(o):
        out = {}
        for k in P:
            vis = np.einsum("tmp,tp->tm", o.A, img)
            d2 = np.abs(vis - o.targets[k]) ** 2
            out[k] = float((d2 * o.masks[k] / o.sigmas[k] ** 2).sum() / (2.0 * o.masks[k].sum()))
        return out

    a, b = chi2(op), chi2(back)
    for k in P:
        assert a[k] == pytest.approx(b[k], rel=1e-6), (
            f"chi2[{k}] differs across the round-trip: {a[k]} (in-memory) vs {b[k]} (on disk)"
        )
