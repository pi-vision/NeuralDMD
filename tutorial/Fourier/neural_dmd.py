"""Re-export shim: the Fourier NeuralDMD code now lives in the `neuraldmd` package.

Kept so existing imports (``from neural_dmd import ...``) and the tutorial
notebooks keep working during the transition; import from ``neuraldmd.*`` in
new code.
"""

from neuraldmd.encoding import SinusoidalEncoding
from neuraldmd.losses import calculate_closure_phases, loss_fn, sparsity_loss
from neuraldmd.model import NeuralDMD, TemporalBMLP, TemporalOmegaMLP
from neuraldmd.networks import ResBlock, ResidualMLP, zero_init_linear
from neuraldmd.training import (
    PlateauScheduler,
    plot_losses,
    train_epoch_jit,
    train_model,
    train_step,
)

__all__ = [
    "NeuralDMD",
    "PlateauScheduler",
    "ResBlock",
    "ResidualMLP",
    "SinusoidalEncoding",
    "TemporalBMLP",
    "TemporalOmegaMLP",
    "calculate_closure_phases",
    "loss_fn",
    "plot_losses",
    "sparsity_loss",
    "train_epoch_jit",
    "train_model",
    "train_step",
    "zero_init_linear",
]
