"""The exported reconstruction must be the field training actually optimized.

`stokes_fields` does NOT return the same field eagerly and under jit: eagerly it picks
up a near-uniform positive offset (measured ~3e-3 per pixel = ~7.3 Jy over a 50x50 grid;
eager flux/frame 9.95 vs jitted 2.63 against a truth of 2.7). Training runs jitted, so
the eager export was carrying spurious diffuse flux into every cube -- and that offset
is exactly the "off-source haze" the polarization priors were built to fight.

`reconstruct_polarized_cubes` must therefore evaluate under jit, like training does.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neuraldmd.evaluation import pixel_grid_coords, reconstruct_polarized_cubes
from neuraldmd.polarized import PolarizedNeuralDMD

MODEL_KW = dict(
    hidden_size=32,
    num_layers=2,
    num_frequencies=2,
    temporal_latent_dim=16,
    temporal_hidden=32,
    temporal_layers=2,
)
S = ("I", "Q", "U")


def _model_and_grid(npix=8, t=5, seed=0):
    model = PolarizedNeuralDMD(S, r=3, key=jax.random.PRNGKey(seed), **MODEL_KW)
    xy = jnp.asarray(pixel_grid_coords(npix, npix))
    times = jnp.asarray(np.linspace(0.0, 0.98, t, dtype=np.float32))
    fmax = {s: 1.0 for s in S}
    fmin = {s: 0.0 for s in S}
    return model, xy, times, fmax, fmin, npix, t


def test_exported_cube_matches_the_jitted_field_training_uses():
    """reconstruct_polarized_cubes == the jitted stokes_fields, exactly.

    Training scores and differentiates the JITTED field. If the export used the eager
    field instead, every NRMSE/EVPA/beta2 would describe a different image than the one
    that was fit -- which is what produced the phantom off-source haze.
    """
    model, xy, times, fmax, fmin, npix, t = _model_and_grid()

    sf = eqx.filter_jit(lambda m, x, ti: m.stokes_fields(x, ti, fmax, fmin))
    jitted = sf(model, xy, times)[0]
    cube = reconstruct_polarized_cubes(model, npix, times, fmax, fmin)

    for s in S:
        want = np.asarray(jitted[s]).T.reshape(t, npix, npix)
        got = np.asarray(cube[s])
        assert np.array_equal(got, want), (
            f"exported {s} cube is not the jitted field training optimized; "
            f"max|diff| = {np.abs(got - want).max():.3e}"
        )


def test_exported_cube_total_flux_matches_the_jitted_field():
    """Total flux per frame must match the jitted field.

    This is the invariant that actually bit: the eager path's error is a near-uniform
    offset, which is nearly invisible per-pixel but integrates into several Jy of fake
    diffuse emission -- so compare the integral, not just the max pixel difference.
    """
    model, xy, times, fmax, fmin, npix, t = _model_and_grid(seed=1)

    sf = eqx.filter_jit(lambda m, x, ti: m.stokes_fields(x, ti, fmax, fmin))
    jitted = sf(model, xy, times)[0]
    cube = reconstruct_polarized_cubes(model, npix, times, fmax, fmin)

    for s in S:
        want = float(np.asarray(jitted[s]).sum(0).mean())
        got = float(np.asarray(cube[s]).sum(axis=(1, 2)).mean())
        assert got == np.float32(want), f"{s} flux/frame {got} != jitted field's {want}"
