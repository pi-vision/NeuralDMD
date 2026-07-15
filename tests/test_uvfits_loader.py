"""UVFITS -> ObsProducts loader, on a real polarized EHT dataset.

The load-bearing check: the loader's Stokes-I A / targets / sigmas equal a direct
ehtim ``chisqdata`` call (which is exactly what ``data/generation.py`` uses), so
the multi-Stokes path is a faithful superset of the existing I-only pipeline.
Marked ``ehtim`` and skipped unless the test UVFITS is present.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ehtim")

from neuraldmd.data.observations import (  # noqa: E402  (after importorskip by design)
    ObsProducts,
    load_uvfits_to_products,
)

UVFITS = Path(
    os.environ.get(
        "NEURALDMD_TEST_UVFITS",
        "/scratch/ondemand33/rdahale/ndmd/mring+hsCW_LO_onsky.uvfits",
    )
)
NPIX = 32
FOV_UAS = 100.0

pytestmark = [
    pytest.mark.ehtim,
    pytest.mark.skipif(not UVFITS.exists(), reason=f"test uvfits not found: {UVFITS}"),
    pytest.mark.filterwarnings("ignore"),
]


@pytest.fixture(scope="module")
def op():
    """Load the test UVFITS into an ``ObsProducts`` once, shared across tests."""
    return load_uvfits_to_products(UVFITS, npix=NPIX, fov_uas=FOV_UAS, stokes=("I", "Q", "U"))


def test_shapes_and_metadata(op):
    """Array shapes, pixel count, and station metadata are self-consistent."""
    T, M, P = op.A.shape
    assert op.stokes == ("I", "Q", "U")
    assert P == NPIX * NPIX
    assert op.bl_station_ids.shape == (T, M, 2)
    assert op.stations is not None and len(op.stations) >= 4
    for s in op.stokes:
        assert op.targets[s].shape == (T, M)
        assert op.sigmas[s].shape == (T, M)
        assert op.masks[s].shape == (T, M)
    # station ids referenced by baselines are valid indices into `stations`
    valid = op.bl_station_ids[op.bl_station_ids >= 0]
    assert valid.max() < len(op.stations)


def test_i_channel_matches_chisqdata_pipeline(op):
    """A / targets / sigmas for Stokes I == a direct chisqdata call, row-for-row
    (== data/generation.py's vis_data), on several snapshots. rtol 1e-6."""
    import ehtim as eh
    from ehtim.imaging.imager_utils import chisqdata

    obs = eh.obsdata.load_uvfits(str(UVFITS)).switch_polrep("stokes")
    frames = obs.split_obs()
    prior = eh.image.make_square(obs, NPIX, FOV_UAS * eh.RADPERUAS)
    for i in (0, len(frames) // 2, len(frames) - 1):
        t, s, A = chisqdata(frames[i], prior, mask=[], pol="I", dtype="vis")
        m = len(t)
        np.testing.assert_allclose(op.A[i, :m], A, rtol=1e-6)
        np.testing.assert_allclose(op.targets["I"][i, :m], t, rtol=1e-6)
        np.testing.assert_allclose(op.sigmas["I"][i, :m], s, rtol=1e-6)
        assert np.all(op.masks["I"][i, :m] == 1.0)  # all returned baselines valid
        assert np.all(op.A[i, m:] == 0)  # padded rows are zero


def test_qu_carry_signal(op):
    """This is a polarized dataset -- Q and U must be nonzero where observed."""
    for s in ("Q", "U"):
        vals = op.targets[s][op.masks[s] > 0]
        assert vals.size > 0
        assert np.mean(np.abs(vals)) > 0


def test_padding_is_masked_and_sigma_inflated(op):
    """Padded (mask==0) entries carry zero target and 1e6 sigma everywhere."""
    for s in op.stokes:
        pad = op.masks[s] == 0
        if pad.any():
            np.testing.assert_array_equal(op.targets[s][pad], 0)
            np.testing.assert_array_equal(op.sigmas[s][pad], 1e6)


def test_v_supported_via_direct_extract():
    """Stokes V is read straight from the data table (chisqdata cannot form it)."""
    opv = load_uvfits_to_products(UVFITS, npix=16, fov_uas=FOV_UAS, stokes=("I", "V"))
    assert opv.stokes == ("I", "V")
    vals = opv.targets["V"][opv.masks["V"] > 0]
    assert vals.size > 0


def test_obs_dir_roundtrip_preserves_metadata(op, tmp_path):
    """Writing then reloading a v2 obs_dir preserves A, per-Stokes arrays, and
    the station metadata (baseline ids + station names)."""
    op.to_obs_dir(tmp_path / "o")
    assert (tmp_path / "o" / "bl_station_ids.npy").exists()
    back = ObsProducts.from_obs_dir(tmp_path / "o")
    assert back.stokes == op.stokes
    assert back.stations == op.stations
    np.testing.assert_array_equal(back.bl_station_ids, op.bl_station_ids)
    np.testing.assert_array_equal(back.A, op.A)
    for s in op.stokes:
        np.testing.assert_array_equal(back.targets[s], op.targets[s])
        np.testing.assert_array_equal(back.masks[s], op.masks[s])
