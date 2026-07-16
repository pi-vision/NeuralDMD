"""Time-resolved beta2: a rotating EVPA must not be scored by the time average.

``beta2_coefficient`` averages the Stokes cubes over time before projecting onto the
m=2 mode. For a STATIC swirl that is correct. For a ROTATING one (the
``mring-varbeta2`` model, whose beta2 phase turns 2*pi every 2.67 hr) the average
cancels: measured on the real truth movies,

    mring+hsCW     |beta2(t)| 0.1801   time-averaged 0.1801   phase swing     0 deg
    mring-varbeta2 |beta2(t)| 0.2000   time-averaged 0.0137   phase swing  1620 deg

so the time-averaged metric would score a PERFECT varbeta2 reconstruction at ~8% and
we would wrongly conclude the method cannot do polarization dynamics. These tests pin
that distinction with analytic fields.
"""

from __future__ import annotations

import numpy as np

from neuraldmd.evaluation import beta2_coefficient, beta2_dynamics_error, beta2_series

FOV, NPIX = 200.0, 40


def _ring_with_evpa(theta_rot, npix=NPIX, fov=FOV, linpol=0.2):
    """One frame: a ring whose radial EVPA field is rotated by ``theta_rot``."""
    yy, xx = np.mgrid[0:npix, 0:npix]
    c = (npix - 1) / 2
    rho = np.hypot(xx - c, yy - c) * (fov / npix)
    phi = np.arctan2(yy - c, xx - c)
    i = np.exp(-(((rho - 26.0) / 6.0) ** 2))  # thick ring inside the 10-34 uas annulus
    chi = phi + theta_rot
    q = i * linpol * np.cos(2 * chi)
    u = -i * linpol * np.sin(2 * chi)
    return i, q, u


def _cube(rotations, n=32):
    """A movie whose EVPA turns through ``rotations`` full turns over the window."""
    out = [_ring_with_evpa(2 * np.pi * rotations * k / n) for k in range(n)]
    i = np.stack([o[0] for o in out])
    q = np.stack([o[1] for o in out])
    u = np.stack([o[2] for o in out])
    return {"I": i, "Q": q, "U": u}


def test_static_swirl_series_matches_the_time_average():
    """With no rotation the per-frame and time-averaged beta2 must agree."""
    c = _cube(rotations=0.0)
    series = beta2_series(c["Q"], c["U"], c["I"], FOV)
    avg = beta2_coefficient(c["Q"], c["U"], c["I"], FOV)
    assert np.allclose(np.abs(series), np.abs(series[0]), rtol=1e-6)  # constant in time
    assert np.isclose(np.abs(series).mean(), abs(avg), rtol=1e-6)


def test_time_average_cancels_a_rotating_swirl_but_the_series_does_not():
    """THE trap: |beta2| survives per-frame and collapses in the time average."""
    c = _cube(rotations=2.0)  # two full turns, as varbeta2 does over its window
    series = beta2_series(c["Q"], c["U"], c["I"], FOV)
    avg = abs(beta2_coefficient(c["Q"], c["U"], c["I"], FOV))

    per_frame = np.abs(series).mean()
    assert per_frame > 0.1, "per-frame |beta2| should be intact for a rotating swirl"
    assert avg < 0.1 * per_frame, (
        f"time-averaged beta2 should cancel a rotating swirl, got {avg:.4f} "
        f"vs per-frame {per_frame:.4f}"
    )


def test_perfect_reconstruction_of_a_rotating_swirl_scores_perfect():
    """recon == truth must give amp 1, phase RMSE 0, phase correlation 1.

    And the time-averaged metric must be seen to fail on the same (perfect) input --
    that is the number that would have misled us.
    """
    c = _cube(rotations=2.0)
    d = beta2_dynamics_error(c, c, FOV)
    assert np.isclose(d["amp_ratio"], 1.0, rtol=1e-6)
    assert d["phase_rmse_deg"] < 1e-6
    assert np.isclose(d["phase_corr"], 1.0, rtol=1e-6)
    assert d["truth_phase_swing_deg"] > 300.0  # the truth really does rotate
    # the old metric on a PERFECT fit: 0/0 -> unreliable, and nowhere near 1
    assert d["amp_ratio_timeavg"] < 0.5 or not np.isfinite(d["amp_ratio_timeavg"])


def test_counter_rotating_reconstruction_is_caught_by_phase_correlation():
    """A swirl rotating the WRONG way has the right |beta2| but must not pass.

    Amplitude alone cannot tell these apart -- only the phase track can, which is
    why ``phase_corr`` is the metric that answers "does the pol follow the truth".
    """
    truth = _cube(rotations=2.0)
    wrong = _cube(rotations=-2.0)  # same amplitude, opposite rotation
    d = beta2_dynamics_error(wrong, truth, FOV)
    assert np.isclose(d["amp_ratio"], 1.0, rtol=1e-3), "amplitude is blind to the sense"
    assert d["phase_corr"] < -0.5, f"counter-rotation must anti-correlate, got {d['phase_corr']}"
    assert d["phase_rmse_deg"] > 30.0


def test_static_truth_reports_no_rotation_to_track():
    """Against a static truth the tracking test is vacuous and must say so."""
    c = _cube(rotations=0.0)
    d = beta2_dynamics_error(c, c, FOV)
    assert d["truth_phase_swing_deg"] < 1.0
    assert np.isnan(d["phase_corr"]), "a static track has no rotation to correlate"
