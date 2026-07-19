"""NeuralDMD: computational imaging of dynamical systems from sparse observations.

Paper: https://arxiv.org/abs/2507.03094
"""

from .model import NeuralDMD
from .training import train_model, loss_fn
from .loader import DMDDataLoader
from . import zernike, pretraining, evaluation

__version__ = "1.0.0"
__all__ = [
    "NeuralDMD",
    "train_model",
    "loss_fn",
    "DMDDataLoader",
    "zernike",
    "pretraining",
    "evaluation",
    "__version__",
]
