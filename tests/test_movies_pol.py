"""Polarized m-ring + hot-spot movie: Stokes (I, Q, U) truth conventions.

The truth is exactly recoverable by construction: a uniform fractional
polarization ``m`` and a smooth EVPA field ``chi`` give ``Q = m*I*cos(2*chi)``,
``U = m*I*sin(2*chi)``, so ``sqrt(Q^2+U^2)/I = m`` and ``0.5*atan2(U,Q) = chi``.
These tie the synthesizer to ``physics.stokes`` (evpa, polarized intensity).
"""

from __future__ import annotations

import numpy as np
import pytest

from neuraldmd.data.movies import make_polarized_frames, polarization_maps, to_ehtim_movie
from neuraldmd.physics.stokes import evpa, linear_polarized_intensity


def test_shapes_and_v_absent():
    """I, Q, U share shape (T, npix, npix); no V channel is produced."""
    out = make_polarized_frames(num_frames=3, npix=32, frac_pol=0.3)
    assert len(out) == 4  # I, Q, U, times -- V is identically zero, not returned
    intensity, q, u, times = out
    assert intensity.shape == q.shape == u.shape == (3, 32, 32)
    assert times.shape == (3,)


def test_pol_decomposition_roundtrips():
    """(P, EVPA) from ``physics.stokes`` reconstruct Q, U exactly."""
    _, q, u, _ = make_polarized_frames(num_frames=2, npix=40, frac_pol=0.3, evpa_offset_deg=15.0)
    p = linear_polarized_intensity(q, u)
    chi = evpa(q, u)
    np.testing.assert_allclose(p * np.cos(2 * chi), q, atol=1e-9)
    np.testing.assert_allclose(p * np.sin(2 * chi), u, atol=1e-9)


def test_fractional_pol_is_uniform():
    """sqrt(Q^2+U^2)/I equals the requested fractional polarization where I>0."""
    intensity, q, u, _ = make_polarized_frames(num_frames=2, npix=48, frac_pol=0.3)
    mask = intensity > 1e-6 * intensity.max()
    m = linear_polarized_intensity(q, u)[mask] / intensity[mask]
    np.testing.assert_allclose(m, 0.3, atol=1e-6)


def test_evpa_matches_input_field():
    """Recovered EVPA equals the input chi field (compared via 2*chi, no pi-wrap)."""
    npix = 40
    kw = dict(npix=npix, fov_uas=200.0, frac_pol=0.3, evpa_winding=1, evpa_offset_deg=10.0)
    intensity, q, u, _ = make_polarized_frames(num_frames=1, **kw)
    _, chi_map = polarization_maps(**kw)
    chi_rec = evpa(q[0], u[0])
    mask = intensity[0] > 1e-6 * intensity[0].max()
    np.testing.assert_allclose(np.cos(2 * chi_rec)[mask], np.cos(2 * chi_map)[mask], atol=1e-6)
    np.testing.assert_allclose(np.sin(2 * chi_rec)[mask], np.sin(2 * chi_map)[mask], atol=1e-6)


def test_polarized_intensity_not_exceed_i():
    """Physical: P = sqrt(Q^2+U^2) <= I everywhere (since m <= 1)."""
    intensity, q, u, _ = make_polarized_frames(num_frames=2, npix=32, frac_pol=0.3)
    assert np.all(linear_polarized_intensity(q, u) <= intensity + 1e-9)


@pytest.mark.ehtim
@pytest.mark.filterwarnings("ignore")
def test_ehtim_movie_carries_qu():
    """to_ehtim_movie stores Q, U on each frame (identity orientation, verified)."""
    intensity, q, u, times = make_polarized_frames(num_frames=3, npix=24, frac_pol=0.3)
    movie = to_ehtim_movie(intensity, times, fov_uas=200.0, qframes=q, uframes=u)
    ims = movie.im_list()
    assert len(ims) == 3
    for k, im in enumerate(ims):
        assert np.any(im.qvec != 0)
        np.testing.assert_allclose(im.qvec, q[k].flatten(), atol=1e-9)
        np.testing.assert_allclose(im.uvec, u[k].flatten(), atol=1e-9)
