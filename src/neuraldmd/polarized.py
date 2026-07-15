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
import jax.numpy as jnp

from .config import StokesConfig
from .model import NeuralDMD, physical_intensities


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

    def stokes_fields(self, xy, times, frame_max: dict, frame_min: dict):
        """Physical per-Stokes image cubes and per-network modes (loss/eval interface).

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates.
        times : jax.Array
            ``(T,)`` normalized frame times.
        frame_max, frame_min : dict of str -> float
            Per-Stokes output scaling.

        Returns
        -------
        images : dict of str -> jax.Array
            ``{stokes: (P, T)}`` physical Stokes images (Q, U, V signed).
        modes : list of tuple
            One ``(W0, W, b0, b)`` per sub-network, for the sparsity penalty.
        """
        images, modes = {}, []
        for s in self.stokes:
            img, mode = physical_intensities(self.models[s], xy, times, frame_max[s], frame_min[s])
            images[s] = img
            modes.append(mode)
        return images, modes

    @property
    def i_submodel(self) -> NeuralDMD:
        """The Stokes-I :class:`NeuralDMD` (target of disk-template pretraining)."""
        return self.models["I"]

    def replace_i_submodel(self, new_i: NeuralDMD) -> PolarizedNeuralDMD:
        """Return a copy with the Stokes-I sub-model replaced (e.g. by a pretrained one)."""
        return eqx.tree_at(lambda m: m.models["I"], self, new_i)


class FractionalPolNeuralDMD(eqx.Module):
    """Polarization parameterized as a *fraction of I* (KINE-style, tied to I).

    Rather than free per-Stokes fields, the linear polarization is
    ``Q = m_l * I * cos(2 xi)``, ``U = m_l * I * sin(2 xi)`` (and ``V = m_c * I``),
    where the fractional magnitude and EVPA are bounded fields of a NeuralDMD raw
    output::

        m_l           = sigmoid(raw - outshift) * scaling_ml    in [0, scaling_ml]
        (cos2xi, sin2xi) = (c_raw, s_raw) / ||(c_raw, s_raw)||   unit EVPA direction
        m_c           = (sigmoid(raw - outshift) - 0.5) * 2      in [-1, 1]   (V only)

    The EVPA direction is unit-normalized, so ``P = sqrt(Q^2+U^2) = m_l*|I|``
    exactly and the physical bound ``P <= scaling_ml*|I| <= I`` holds strictly
    (unlike separate-sigmoid EVPA components, which allow ``P`` up to ``sqrt(2) m_l I``).
    This structurally ties polarization to I (pol only where I is) and -- because
    ``outshift`` makes
    ``m_l ~ sigmoid(-outshift) ~ 0`` at init -- starts the source *unpolarized*
    and grows polarization during training. The loss/eval remain on absolute
    Stokes (or products): pol is a *multiply* by I, never a divide.

    Attributes
    ----------
    intensity : NeuralDMD
        Stokes-I field (physical scaling via frame_max/frame_min; disk-pretrainable).
    frac, cos2xi, sin2xi : NeuralDMD
        Raw fields for ``m_l`` and the EVPA components.
    circ : NeuralDMD or None
        Raw field for ``m_c`` (present iff V is modeled).
    stokes : tuple of str
        Stokes produced (static). outshift, scaling_ml : bound/init controls (static).
    """

    intensity: NeuralDMD
    frac: NeuralDMD
    cos2xi: NeuralDMD
    sin2xi: NeuralDMD
    circ: NeuralDMD | None
    stokes: tuple[str, ...] = eqx.field(static=True)
    outshift: float = eqx.field(static=True)
    scaling_ml: float = eqx.field(static=True)

    def __init__(
        self,
        stokes: tuple[str, ...] | StokesConfig,
        r: int,
        *,
        key: jax.Array,
        outshift: float = 10.0,
        scaling_ml: float = 1.0,
        **model_kwargs,
    ):
        """Build the I field and the fractional-pol fields from independent split keys.

        Parameters
        ----------
        stokes : tuple of str or StokesConfig
            Must contain I; Q/U enable linear pol, V enables circular pol.
        r : int
            Complex DMD modes per field.
        key : jax.Array
            PRNG key (split per field).
        outshift : float
            Sigmoid bias for ``m_l``/``m_c`` -> unpolarized initialization.
        scaling_ml : float
            Cap on the linear polarization fraction (``<= 1``).
        **model_kwargs
            Forwarded to every :class:`NeuralDMD`.
        """
        cfg = stokes if isinstance(stokes, StokesConfig) else StokesConfig(tuple(stokes))
        self.stokes = cfg.stokes
        self.outshift = float(outshift)
        self.scaling_ml = float(scaling_ml)
        want_v = "V" in self.stokes
        keys = jax.random.split(key, 5 if want_v else 4)
        self.intensity = NeuralDMD(r=r, key=keys[0], **model_kwargs)
        self.frac = NeuralDMD(r=r, key=keys[1], **model_kwargs)
        self.cos2xi = NeuralDMD(r=r, key=keys[2], **model_kwargs)
        self.sin2xi = NeuralDMD(r=r, key=keys[3], **model_kwargs)
        self.circ = NeuralDMD(r=r, key=keys[4], **model_kwargs) if want_v else None

    def stokes_fields(self, xy, times, frame_max: dict, frame_min: dict):
        """Physical per-Stokes image cubes and per-network modes (loss/eval interface).

        Returns
        -------
        images : dict of str -> jax.Array
            ``{stokes: (P, T)}`` with ``Q=m_l*I*cos2xi``, ``U=m_l*I*sin2xi``,
            ``V=m_c*I``.
        modes : list of tuple
            ``(W0, W, b0, b)`` per sub-network, for the sparsity penalty.
        """
        i_img, i_modes = physical_intensities(
            self.intensity, xy, times, frame_max["I"], frame_min["I"]
        )
        ml_raw, ml_modes = physical_intensities(self.frac, xy, times, 1.0, 0.0)
        c_raw, c_modes = physical_intensities(self.cos2xi, xy, times, 1.0, 0.0)
        s_raw, s_modes = physical_intensities(self.sin2xi, xy, times, 1.0, 0.0)
        # m_l in [0, scaling_ml]; outshift -> m_l ~ 0 at init (unpolarized)
        m_l = jax.nn.sigmoid(ml_raw - self.outshift) * self.scaling_ml
        # EVPA direction as a UNIT vector (cos2xi^2 + sin2xi^2 = 1), so that
        # P = sqrt(Q^2+U^2) = m_l*|I| exactly -> P <= scaling_ml*|I| <= I (strict).
        norm = jnp.sqrt(c_raw**2 + s_raw**2 + 1e-8)
        cos2xi = c_raw / norm
        sin2xi = s_raw / norm

        images = {"I": i_img}
        modes = [i_modes, ml_modes, c_modes, s_modes]
        if "Q" in self.stokes:
            images["Q"] = m_l * i_img * cos2xi
        if "U" in self.stokes:
            images["U"] = m_l * i_img * sin2xi
        if self.circ is not None:
            v_raw, v_modes = physical_intensities(self.circ, xy, times, 1.0, 0.0)
            m_c = (jax.nn.sigmoid(v_raw - self.outshift) - 0.5) * 2.0
            images["V"] = m_c * i_img
            modes.append(v_modes)
        return images, modes

    @property
    def i_submodel(self) -> NeuralDMD:
        """The Stokes-I :class:`NeuralDMD` (target of disk-template pretraining)."""
        return self.intensity

    def replace_i_submodel(self, new_i: NeuralDMD) -> FractionalPolNeuralDMD:
        """Return a copy with the Stokes-I field replaced (e.g. by a pretrained one)."""
        return eqx.tree_at(lambda m: m.intensity, self, new_i)
