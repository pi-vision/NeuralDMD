"""End-to-end polarized dataset generation (movie -> observe -> obs_dir), both
bases. Marked ehtim; small grid/frames keep it quick."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ehtim")

from neuraldmd.data.generation import generate_polarized_dataset  # noqa: E402
from neuraldmd.data.loader import PolarizedDMDDataLoader  # noqa: E402
from neuraldmd.data.observations import ObsProducts  # noqa: E402

pytestmark = [pytest.mark.ehtim, pytest.mark.filterwarnings("ignore")]

GEN_KW = dict(npix=16, num_frames=8, tstart_hr=9.0, tstop_hr=10.0, tint=60.0, fractional_noise=0.0)


def test_generate_stokes_dataset(tmp_path):
    """Stokes-basis generation yields a valid I,Q,U obs_dir with Q/U signal that
    the polarized loader can consume."""
    op = generate_polarized_dataset(tmp_path, stokes=("I", "Q", "U"), basis="stokes", **GEN_KW)
    assert op.stokes == ("I", "Q", "U")
    assert (tmp_path / "obs.uvfits").exists()
    assert (tmp_path / "manifest.json").exists()
    _, _, p = op.A.shape
    assert p == 256  # npix**2

    for s in ("Q", "U"):
        vals = op.targets[s][op.masks[s] > 0]
        assert vals.size > 0 and np.mean(np.abs(vals)) > 0

    # round-trips through disk and feeds the loader
    back = ObsProducts.from_obs_dir(tmp_path)
    assert back.stokes == ("I", "Q", "U")
    loader = PolarizedDMDDataLoader(op, npix=16, batch_size=2, epochs=2)
    _, a_b, tgt, _, _, _ = loader.get_epoch_data(0)
    assert set(tgt) == {"I", "Q", "U"}
    assert a_b.shape[-1] == 256


def test_generate_circular_dataset(tmp_path):
    """Circular-basis generation yields product-keyed (RR/LL/RL/LR) data."""
    op = generate_polarized_dataset(tmp_path, basis="circular", **GEN_KW)
    assert op.stokes == ("RR", "LL", "RL", "LR")
    for prod in ("RR", "LL", "RL", "LR"):
        assert op.targets[prod].shape == (op.A.shape[0], op.A.shape[1])
    # parallel hands carry the bulk of the flux; cross hands the polarization
    rr = op.targets["RR"][op.masks["RR"] > 0]
    rl = op.targets["RL"][op.masks["RL"] > 0]
    assert np.mean(np.abs(rr)) > np.mean(np.abs(rl))
