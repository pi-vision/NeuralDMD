"""The tutorial re-export shims must expose the exact package objects."""

from __future__ import annotations

import sys
from pathlib import Path

import neuraldmd.data.loader
import neuraldmd.encoding
import neuraldmd.evaluation
import neuraldmd.losses
import neuraldmd.model
import neuraldmd.networks
import neuraldmd.pretraining
import neuraldmd.training
import neuraldmd.zernike

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "tutorial" / "Fourier", _REPO / "eht2017"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def test_neural_dmd_shim_parity():
    import neural_dmd

    assert neural_dmd.NeuralDMD is neuraldmd.model.NeuralDMD
    assert neural_dmd.TemporalOmegaMLP is neuraldmd.model.TemporalOmegaMLP
    assert neural_dmd.SinusoidalEncoding is neuraldmd.encoding.SinusoidalEncoding
    assert neural_dmd.ResidualMLP is neuraldmd.networks.ResidualMLP
    assert neural_dmd.loss_fn is neuraldmd.losses.loss_fn
    assert neural_dmd.train_model is neuraldmd.training.train_model
    assert neural_dmd.PlateauScheduler is neuraldmd.training.PlateauScheduler


def test_loader_shim_parity():
    import dmd_data_loader

    assert dmd_data_loader.DMDDataLoader is neuraldmd.data.loader.DMDDataLoader


def test_helper_shim_parity():
    import pretraining
    import util_funcs
    import zernike_bank

    assert zernike_bank.build_zernike_targets is neuraldmd.zernike.build_zernike_targets
    assert pretraining.pretrain_model is neuraldmd.pretraining.pretrain_model
    assert util_funcs.evaluate_chi2 is neuraldmd.evaluation.evaluate_chi2
