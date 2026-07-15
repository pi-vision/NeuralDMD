"""Implementation indirection: monolith (default) vs packaged code.

The identical characterization suite runs against either the current monolith
(``tutorial/Fourier/*.py``) or the packaged ``neuraldmd.*`` code, selected by
``NEURALDMD_IMPL=monolith|package``. During the Phase-2 refactor this lets us
prove the package is behavior-identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("NEURALDMD_IMPL", "monolith")

if IMPL == "package":
    from neuraldmd.data.loader import DMDDataLoader
    from neuraldmd.evaluation import (
        calc_psnr,
        evaluate_chi2,
        load_hdf5,
        pixel_grid_coords,
        sort_modes_by_lambda,
    )
    from neuraldmd.losses import calculate_closure_phases, loss_fn, sparsity_loss
    from neuraldmd.model import (
        NeuralDMD,
        ResBlock,
        ResidualMLP,
        SinusoidalEncoding,
        TemporalBMLP,
        TemporalOmegaMLP,
        zero_init_linear,
    )
    from neuraldmd.pretraining import (
        _best_complex_scale_residual,
        pretrain_loss_fn,
        pretrain_model,
        radius_of_gyration,
        zernike_alignment_loss,
    )
    from neuraldmd.training import (
        PlateauScheduler,
        train_epoch_jit,
        train_model,
        train_step,
    )
    from neuraldmd.zernike import (
        build_zernike_targets,
        make_xy_grid,
        pick_mode_set,
        zernike_complex_basis,
    )
else:
    for _p in (_REPO / "tutorial" / "Fourier", _REPO / "eht2017"):
        _sp = str(_p)
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
    from dmd_data_loader import DMDDataLoader
    from neural_dmd import (
        NeuralDMD,
        PlateauScheduler,
        ResBlock,
        ResidualMLP,
        SinusoidalEncoding,
        TemporalBMLP,
        TemporalOmegaMLP,
        calculate_closure_phases,
        loss_fn,
        sparsity_loss,
        train_epoch_jit,
        train_model,
        train_step,
        zero_init_linear,
    )
    from pretraining import (
        _best_complex_scale_residual,
        pretrain_loss_fn,
        pretrain_model,
        radius_of_gyration,
        zernike_alignment_loss,
    )
    from util_funcs import (
        calc_psnr,
        evaluate_chi2,
        load_hdf5,
        pixel_grid_coords,
        sort_modes_by_lambda,
    )
    from zernike_bank import (
        build_zernike_targets,
        make_xy_grid,
        pick_mode_set,
        zernike_complex_basis,
    )

__all__ = [
    "IMPL",
    "DMDDataLoader",
    "NeuralDMD",
    "PlateauScheduler",
    "ResBlock",
    "ResidualMLP",
    "SinusoidalEncoding",
    "TemporalBMLP",
    "TemporalOmegaMLP",
    "_best_complex_scale_residual",
    "build_zernike_targets",
    "calc_psnr",
    "calculate_closure_phases",
    "evaluate_chi2",
    "load_hdf5",
    "loss_fn",
    "make_xy_grid",
    "pick_mode_set",
    "pixel_grid_coords",
    "pretrain_loss_fn",
    "pretrain_model",
    "radius_of_gyration",
    "sort_modes_by_lambda",
    "sparsity_loss",
    "train_epoch_jit",
    "train_model",
    "train_step",
    "zernike_alignment_loss",
    "zernike_complex_basis",
    "zero_init_linear",
]
