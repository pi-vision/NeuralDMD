"""Polarized NeuralDMD: an independent :class:`NeuralDMD` per Stokes parameter.

The polarized reconstruction is the natural generalization of the scalar model:
each Stokes parameter (I, Q, U, and optionally V) gets its own coordinate and
temporal networks. With ``stokes=("I",)`` the container is a thin wrapper whose
Stokes-I sub-model is identical to a standalone :class:`NeuralDMD` -- so no
Stokes-I behavior changes (asserted by the parity test in
``tests/test_polarized.py``).
"""

from __future__ import annotations

import equinox as eqx
import jax

from .config import StokesConfig
from .model import NeuralDMD


class PolarizedNeuralDMD(eqx.Module):
    """A dict of per-Stokes :class:`NeuralDMD` models over a shared coordinate grid.

    Each Stokes parameter has independent spatial and temporal networks; the
    models are held in a plain dict, which equinox treats as a pytree, so the
    container composes with ``eqx.partition`` / ``optax`` exactly like a single
    model.

    Attributes
    ----------
    models : dict of str -> NeuralDMD
        One model per Stokes parameter, keyed by name (dynamic pytree leaves).
    stokes : tuple of str
        Stokes parameters, in order (static metadata).
    """

    models: dict[str, NeuralDMD]
    stokes: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        stokes: tuple[str, ...] | StokesConfig,
        r: int,
        *,
        key: jax.Array,
        **model_kwargs,
    ):
        """Build one :class:`NeuralDMD` per Stokes from independent split keys.

        Parameters
        ----------
        stokes : tuple of str or StokesConfig
            Stokes parameters to model; validated via :class:`StokesConfig`.
        r : int
            Number of complex DMD modes per Stokes (forwarded to each NeuralDMD).
        key : jax.Array
            PRNG key, split into one independent subkey per Stokes (so an I-only
            container matches ``NeuralDMD(key=jax.random.split(key, 1)[0])``).
        **model_kwargs
            Forwarded verbatim to every :class:`NeuralDMD` (e.g. ``hidden_size``,
            ``num_layers``, ``num_frequencies``, ``t_scale``).
        """
        cfg = stokes if isinstance(stokes, StokesConfig) else StokesConfig(tuple(stokes))
        self.stokes = cfg.stokes
        keys = jax.random.split(key, len(self.stokes))
        self.models = {
            s: NeuralDMD(r=r, key=keys[i], **model_kwargs) for i, s in enumerate(self.stokes)
        }

    def __call__(self, xy: jax.Array) -> dict[str, tuple]:
        """Evaluate every sub-model's spatial/temporal outputs at ``xy``.

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates.

        Returns
        -------
        dict of str -> tuple
            ``{stokes: (W0, W, Omega, b0, b)}`` -- each sub-model's raw outputs.
        """
        return {s: m(xy) for s, m in self.models.items()}

    def reconstruct(
        self,
        xy: jax.Array,
        times: jax.Array,
        frame_max: float = 1.0,
        frame_min: float = 0.0,
    ) -> dict[str, tuple]:
        """Reconstruct each Stokes movie on ``xy`` at ``times``.

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates.
        times : jax.Array
            ``(T,)`` normalized times.
        frame_max, frame_min : float
            Output scaling applied to every Stokes, matching
            :meth:`NeuralDMD.reconstruct`. The signedness of Q/U/V (which have no
            ``frame_min`` offset) is handled by the loss/training layer, not here.

        Returns
        -------
        dict of str -> tuple
            ``{stokes: (intensities, static, dynamic)}``, each ``(P, T)``.
        """
        return {
            s: m.reconstruct(xy, times, frame_max, frame_min) for s, m in self.models.items()
        }
