"""Polarized reconstruction metrics: NRMSE, EVPA error, and cube reconstruction."""

from __future__ import annotations

import jax
import numpy as np

from neuraldmd.evaluation import (
    blur_polarized_cubes,
    evpa_error_deg,
    polarized_nrmse,
    reconstruct_polarized_cubes,
)
from neuraldmd.polarized import PolarizedNeuralDMD

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)


def test_nrmse_zero_for_identical():
    """NRMSE of a cube against itself is exactly 0."""
    t = {"I": np.ones((3, 8, 8)), "Q": np.full((3, 8, 8), -0.3)}
    out = polarized_nrmse(t, t)
    assert out["I"] == 0.0 and out["Q"] == 0.0


def test_nrmse_known_value():
    """A uniform offset gives ||diff|| / ||truth|| = offset / value."""
    truth = {"I": np.full((2, 4, 4), 2.0)}
    recon = {"I": np.full((2, 4, 4), 3.0)}
    np.testing.assert_allclose(polarized_nrmse(recon, truth)["I"], 0.5)


def test_evpa_error_zero_and_45deg():
    """EVPA error is 0 for an exact match and 45 deg for a Q<->U swap."""
    truth = {"I": np.ones((1, 8, 8)), "Q": np.ones((1, 8, 8)), "U": np.zeros((1, 8, 8))}
    assert evpa_error_deg(truth, truth) == 0.0
    swapped = {"I": np.ones((1, 8, 8)), "Q": np.zeros((1, 8, 8)), "U": np.ones((1, 8, 8))}
    np.testing.assert_allclose(evpa_error_deg(swapped, truth), 45.0, atol=1e-6)


def test_reconstruct_cubes_shape():
    """reconstruct_polarized_cubes returns (T, npix, npix) per Stokes."""
    model = PolarizedNeuralDMD(("I", "Q", "U"), r=3, key=jax.random.PRNGKey(0), **MODEL_KW)
    cubes = reconstruct_polarized_cubes(
        model, 8, np.linspace(0, 1, 4), {s: 1.0 for s in "IQU"}, {s: 0.0 for s in "IQU"}
    )
    assert set(cubes) == {"I", "Q", "U"}
    for s in cubes:
        assert cubes[s].shape == (4, 8, 8)


def test_blur_polarized_cubes_identity_and_beam():
    """fwhm<=0 is the identity; a positive beam conserves flux and lowers the
    peak of a point source by the expected Gaussian-beam factor."""
    t, n = 2, 33
    cube = np.zeros((t, n, n), np.float64)
    cube[:, n // 2, n // 2] = 1.0  # unit point source
    cubes = {"I": cube, "Q": 0.5 * cube}

    same = blur_polarized_cubes(cubes, 0.0, fov_uas=200.0)
    assert same["I"] is cube  # untouched

    fwhm, fov = 15.0, 200.0
    out = blur_polarized_cubes(cubes, fwhm, fov_uas=fov)
    # flux conserved per frame
    np.testing.assert_allclose(out["I"].sum(axis=(1, 2)), 1.0, rtol=1e-6)
    # peak of a delta blurred by a Gaussian: 1 / (2*pi*sigma_pix^2)
    sigma_pix = (fwhm / (fov / n)) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    expected_peak = 1.0 / (2.0 * np.pi * sigma_pix**2)
    np.testing.assert_allclose(out["I"][:, n // 2, n // 2], expected_peak, rtol=0.02)
    # linearity: Q = 0.5 * I everywhere
    np.testing.assert_allclose(out["Q"], 0.5 * out["I"], rtol=1e-12)
