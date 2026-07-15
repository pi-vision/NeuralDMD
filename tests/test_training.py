"""Characterize training: PlateauScheduler semantics + a tiny end-to-end fit
that drives chi2_vis down (the truth reconstructs the fixture visibilities).
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import optax
import pytest
from _impl import DMDDataLoader, NeuralDMD, PlateauScheduler, train_epoch_jit


def test_scheduler_is_noop_at_factor_one():
    # factor=1.0 -> new_lr == lr -> the guard (new_lr < lr) never fires
    s = PlateauScheduler(1e-3, factor=1.0, patience=2)
    for loss in [5.0, 6.0, 7.0, 8.0, 9.0]:  # never improves
        lr = s.step(loss)
    assert lr == 1e-3


def test_scheduler_reduces_on_plateau():
    s = PlateauScheduler(1e-3, factor=0.5, patience=2, min_lr=1e-8)
    lrs = [s.step(loss) for loss in [5.0, 6.0, 7.0, 8.0]]  # 2 non-improving -> halve
    assert lrs[-1] == pytest.approx(5e-4)


@pytest.mark.slow
def test_mini_train_reduces_chi2(tiny_obs):
    n_epochs = 30
    model = NeuralDMD(r=6, key=jax.random.PRNGKey(0), num_frequencies=2)
    loader = DMDDataLoader(
        data=tiny_obs.movie,
        batch_size=2,
        epochs=n_epochs,
        data_dir=tiny_obs.data_dir,
        times=tiny_obs.times,
        fov_x=tiny_obs.fov,
        fov_y=tiny_obs.fov,
        time_fraction=1.0,
        seed=0,
    )
    fmax = float(tiny_obs.movie.max())
    fmin = float(tiny_obs.movie.min())
    opt = optax.adamw(3e-3, weight_decay=1e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    key = jax.random.PRNGKey(1)

    chi2 = []
    for e in range(n_epochs):
        data = loader.get_epoch_data(e)
        model, opt_state, (loss, rec, c_vis, c_amp, c_cp) = train_epoch_jit(
            model, opt_state, data, opt, key, fmax, fmin
        )
        chi2.append(float(c_vis))

    assert np.isfinite(chi2).all()
    assert chi2[-1] < chi2[0]  # training reduces the data misfit
    assert min(chi2) < 0.5 * chi2[0]  # meaningful, not marginal, improvement
