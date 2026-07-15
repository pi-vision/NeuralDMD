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


def test_dataset_times_and_truth_noise_floor(tmp_path):
    """The generated dataset stores movie-anchored frame times (a moving source
    trained on frame-index times learns a warped clock), and the ground truth
    pushed through the saved operator reaches the noise floor on EVERY product
    -- the single end-to-end guard for A, orientation, sigmas, masks, and the
    time axis at once."""
    out = tmp_path / "ds"
    op = generate_polarized_dataset(
        out,
        npix=24,
        fov_uas=200.0,
        num_frames=12,
        tstart_hr=9.0,
        tstop_hr=15.0,
        fractional_noise=0.04,
        basis="circular",
        seed=3,
    )
    assert op.times is not None
    # movie-anchored: strictly increasing, inside [0, 1], NOT the index grid
    assert np.all(np.diff(op.times) > 0)
    assert op.times.min() >= 0.0 and op.times.max() <= 1.0
    assert not np.allclose(op.times, np.linspace(0, 1, len(op.times)))

    tr = np.load(out / "truth_pol.npz")
    t_truth = tr["times"]

    def cube_at(c, tq):
        idx = np.interp(tq, t_truth, np.arange(len(t_truth)))
        lo = np.clip(np.floor(idx).astype(int), 0, len(t_truth) - 1)
        hi = np.clip(lo + 1, 0, len(t_truth) - 1)
        w = (idx - lo)[:, None, None]
        return (1 - w) * c[lo] + w * c[hi]

    T = op.A.shape[0]
    cubes = {s: cube_at(tr[s], op.times).reshape(T, -1) for s in ("I", "Q", "U")}
    vis = {
        s: np.einsum("tvp,tp->tv", op.A, cubes[s].astype(np.complex128)) for s in ("I", "Q", "U")
    }
    model = {
        "RR": vis["I"],
        "LL": vis["I"],
        "RL": vis["Q"] + 1j * vis["U"],
        "LR": vis["Q"] - 1j * vis["U"],
    }
    for p in ("RR", "LL", "RL", "LR"):
        m = op.masks[p] > 0
        chi2 = float(
            (np.abs(model[p] - op.targets[p])[m] ** 2 / op.sigmas[p][m] ** 2).sum() / (2 * m.sum())
        )
        # thermal-only data against 4%-syserr-inflated sigma: floor well below 1;
        # any time-axis / convention regression blows this up by orders of magnitude
        assert chi2 < 1.0, f"truth chi2_{p} = {chi2:.3f} (noise floor regression)"
