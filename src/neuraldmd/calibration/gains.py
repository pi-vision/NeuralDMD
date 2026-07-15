"""Per-station complex antenna gains for visibility-domain calibration.

A station-based gain corrupts every visibility on a baseline ``(i, j)`` at time
``t`` as ``V_ij -> V_ij * g_i(t) * conj(g_j(t))``.  This module holds the gains
as trainable :mod:`equinox` parameters so they can be fit jointly with the image
model.  Amplitudes are stored in log-space (guaranteeing positivity) and hard-
clipped to per-array bounds so a runaway gain cannot absorb source structure.
Phases are optional and referenced to a fixed station to remove the global
phase degeneracy.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class StationGains(eqx.Module):
    """Trainable per-station, per-time complex gains.

    Parameters
    ----------
    n_stations : int
        Number of physical stations; gains are indexed ``0 .. n_stations - 1``
        to match ``bl_station_ids`` (padded baseline slots use id ``-1`` and are
        assigned unit gain).
    n_times : int
        Number of time frames (gains vary per frame).
    use_phase : bool, optional
        If ``True``, fit phase gains as well; otherwise gains are real
        amplitudes only. Default ``False``.
    amp_bounds : tuple of float, optional
        ``(lo, hi)`` hard clip applied to the gain amplitude. Default
        ``(0.9, 1.1)``.
    ref_station : int, optional
        Station whose phase is pinned to zero (removes the global-phase
        degeneracy). Ignored when ``use_phase`` is ``False``. Default ``0``.

    Attributes
    ----------
    log_amp : jax.Array
        ``(n_stations, n_times)`` log-amplitudes; gain amplitude is
        ``exp(log_amp)``, initialised to 1 (``log_amp = 0``).
    phase : jax.Array or None
        ``(n_stations, n_times)`` phases in radians, initialised to 0, or
        ``None`` when ``use_phase`` is ``False``.
    """

    log_amp: jax.Array
    phase: jax.Array | None
    n_stations: int = eqx.field(static=True)
    amp_bounds: tuple[float, float] = eqx.field(static=True)
    ref_station: int = eqx.field(static=True)

    def __init__(
        self,
        n_stations: int,
        n_times: int,
        *,
        use_phase: bool = False,
        amp_bounds: tuple[float, float] = (0.9, 1.1),
        ref_station: int = 0,
    ):
        self.n_stations = int(n_stations)
        self.amp_bounds = (float(amp_bounds[0]), float(amp_bounds[1]))
        self.ref_station = int(ref_station)
        self.log_amp = jnp.zeros((n_stations, n_times))
        self.phase = jnp.zeros((n_stations, n_times)) if use_phase else None

    def amplitudes(self) -> jax.Array:
        """Return the per-station gain amplitudes, hard-clipped to ``amp_bounds``.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times)`` real gain amplitudes in ``amp_bounds``.
        """
        lo, hi = self.amp_bounds
        return jnp.clip(jnp.exp(self.log_amp), lo, hi)

    def phases(self) -> jax.Array:
        """Return the reference-subtracted, wrapped per-station gain phases.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times)`` phases in ``[-pi, pi]`` with the reference
            station pinned to zero. All-zero when ``use_phase`` is ``False``.
        """
        if self.phase is None:
            return jnp.zeros_like(self.log_amp)
        ph = self.phase - self.phase[self.ref_station]
        return jnp.arctan2(jnp.sin(ph), jnp.cos(ph))

    def station_gains(self) -> jax.Array:
        """Return the complex per-station gains ``amp * exp(i * phase)``.

        The complex width matches the parameter dtype (``float32 -> complex64``,
        ``float64 -> complex128``), so an ``x64`` context yields full precision.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times)`` complex gains.
        """
        amp = self.amplitudes()
        if self.phase is None:
            return jax.lax.complex(amp, jnp.zeros_like(amp))
        ph = self.phases()
        return jax.lax.complex(amp * jnp.cos(ph), amp * jnp.sin(ph))

    def apply(
        self,
        vis: jax.Array,
        bl_station_ids: jax.Array,
        *,
        inverse: bool = False,
    ) -> jax.Array:
        """Apply (or invert) the station gains on a set of model visibilities.

        Each visibility ``V_ij(t)`` is multiplied by ``g_i(t) * conj(g_j(t))``
        (or divided by it when ``inverse`` is ``True``). Padded baseline slots
        (station id ``-1``) are assigned unit gain and pass through unchanged.

        Parameters
        ----------
        vis : jax.Array
            ``(T, M)`` complex model visibilities (``T`` times, ``M`` baselines).
        bl_station_ids : jax.Array
            ``(T, M, 2)`` integer station ids per baseline; ``-1`` marks padding.
        inverse : bool, optional
            If ``True``, divide by the gain factor instead of multiplying (the
            exact inverse of a forward apply). Default ``False``.

        Returns
        -------
        jax.Array
            ``(T, M)`` complex visibilities with gains applied.
        """
        g = self.station_gains()
        if inverse:
            g = 1.0 / g
        # append a unit-gain row so padded id -1 indexes it (numpy negative index)
        g_pad = jnp.concatenate([g, jnp.ones((1, g.shape[1]), g.dtype)], axis=0)
        ids = jnp.asarray(bl_station_ids)
        t_idx = jnp.broadcast_to(jnp.arange(vis.shape[0])[:, None], ids.shape[:2])
        gi = g_pad[ids[..., 0], t_idx]
        gj = g_pad[ids[..., 1], t_idx]
        return vis * gi * jnp.conj(gj)
