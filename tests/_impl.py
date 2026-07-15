"""Implementation indirection: monolith (default) vs packaged code.

The identical characterization suite runs against either the current monolith
(``tutorial/Fourier/neural_dmd.py`` + ``dmd_data_loader.py``) or the packaged
``neuraldmd.*`` code, selected by ``NEURALDMD_IMPL=monolith|package``. During
the Phase-2 refactor this lets us prove the package is behavior-identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("NEURALDMD_IMPL", "monolith")

if IMPL == "package":
    from neuraldmd.data.loader import DMDDataLoader
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
    from neuraldmd.training import (
        PlateauScheduler,
        train_epoch_jit,
        train_model,
        train_step,
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

__all__ = [
    "DMDDataLoader",
    "NeuralDMD",
    "PlateauScheduler",
    "ResBlock",
    "ResidualMLP",
    "SinusoidalEncoding",
    "TemporalBMLP",
    "TemporalOmegaMLP",
    "calculate_closure_phases",
    "loss_fn",
    "sparsity_loss",
    "train_epoch_jit",
    "train_model",
    "train_step",
    "zero_init_linear",
    "IMPL",
]
