"""Per-station complex antenna gains for visibility-domain calibration.

A station-based gain corrupts every visibility on a baseline ``(i, j)`` at time
``t`` as ``V_ij -> V_ij * g_i(t) * conj(g_j(t))``. For dual-polarization feeds
each station carries one gain per hand: a parallel-hand product ``RR`` takes
``g_R,i * conj(g_R,j)`` while a cross-hand ``RL`` takes ``g_R,i * conj(g_L,j)``
(see :data:`PRODUCT_HANDS`); fitting a hand-blind gain on cross-hand data would
silently assert ``g_R = g_L``. This module holds the gains as trainable
:mod:`equinox` parameters so they can be fit jointly with the image model.
Amplitudes are sigmoid-bounded inside per-array limits -- the gradient never
vanishes, unlike a hard clip, so a gain that reaches a bound can still
re-enter. Phases are optional and referenced to a fixed station to remove the
global phase degeneracy.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

#: per-product hand indices ``(unconjugated station i, conjugated station j)``
#: with R = 0 and L = 1: ``V_p(i,j) -> g_{h_i},i * conj(g_{h_j},j) * V_p(i,j)``
PRODUCT_HANDS: dict[str, tuple[int, int]] = {
    "RR": (0, 0),
    "LL": (1, 1),
    "RL": (0, 1),
    "LR": (1, 0),
}


class StationGains(eqx.Module):
    """Trainable per-station, per-time (optionally per-hand) complex gains.

    Parameters
    ----------
    n_stations : int
        Number of physical stations; gains are indexed ``0 .. n_stations - 1``
        to match ``bl_station_ids`` (padded baseline slots use id ``-1`` and are
        assigned unit gain).
    n_times : int
        Number of time frames (gains vary per frame).
    n_hands : int, optional
        ``1`` (default) ties the polarization hands (``g_R = g_L``, adequate
        for Stokes-I / parallel-hand fits); ``2`` gives each hand its own gain
        (required for cross-hand products on real data).
    use_phase : bool, optional
        If ``True``, fit phase gains as well; otherwise gains are real
        amplitudes only. Default ``False``.
    amp_bounds : tuple of float, optional
        ``(lo, hi)`` bounds on the gain amplitude, enforced smoothly via
        ``amp = lo + (hi - lo) * sigmoid(amp_raw)``. Must strictly bracket 1 so
        the initialization is exactly unit gain. Default ``(0.9, 1.1)``.
    ref_station : int, optional
        Station whose phase is pinned to zero (removes the global-phase
        degeneracy). Ignored when ``use_phase`` is ``False``. Default ``0``.

    Attributes
    ----------
    amp_raw : jax.Array
        ``(n_stations, n_times, n_hands)`` unconstrained amplitude parameters,
        initialized so that the amplitude is exactly 1.
    phase : jax.Array or None
        ``(n_stations, n_times, n_hands)`` phases in radians, initialized to 0,
        or ``None`` when ``use_phase`` is ``False``.

    Raises
    ------
    ValueError
        If ``amp_bounds`` does not strictly bracket 1, or ``n_hands`` is not
        1 or 2.
    """

    amp_raw: jax.Array
    phase: jax.Array | None
    n_stations: int = eqx.field(static=True)
    n_hands: int = eqx.field(static=True)
    amp_bounds: tuple[float, float] = eqx.field(static=True)
    ref_station: int = eqx.field(static=True)

    def __init__(
        self,
        n_stations: int,
        n_times: int,
        *,
        n_hands: int = 1,
        use_phase: bool = False,
        amp_bounds: tuple[float, float] = (0.9, 1.1),
        ref_station: int = 0,
    ):
        lo, hi = float(amp_bounds[0]), float(amp_bounds[1])
        if not (lo < 1.0 < hi):
            raise ValueError(f"amp_bounds must strictly bracket 1, got {(lo, hi)}")
        if n_hands not in (1, 2):
            raise ValueError(f"n_hands must be 1 or 2, got {n_hands}")
        self.n_stations = int(n_stations)
        self.n_hands = int(n_hands)
        self.amp_bounds = (lo, hi)
        self.ref_station = int(ref_station)
        # sigmoid(raw0) = (1 - lo) / (hi - lo)  =>  amplitude(raw0) = 1 exactly
        raw0 = float(np.log((1.0 - lo) / (hi - 1.0)))
        self.amp_raw = jnp.full((n_stations, n_times, n_hands), raw0)
        self.phase = jnp.zeros((n_stations, n_times, n_hands)) if use_phase else None

    def amplitudes(self) -> jax.Array:
        """Return the per-station gain amplitudes, sigmoid-bounded in ``amp_bounds``.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times, n_hands)`` real amplitudes strictly inside
            ``(amp_bounds[0], amp_bounds[1])``, with nonzero gradient everywhere.
        """
        lo, hi = self.amp_bounds
        return lo + (hi - lo) * jax.nn.sigmoid(self.amp_raw)

    def phases(self) -> jax.Array:
        """Return the reference-subtracted, wrapped per-station gain phases.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times, n_hands)`` phases in ``[-pi, pi]`` with the
            reference station pinned to zero. All-zero when ``use_phase`` is
            ``False``.
        """
        if self.phase is None:
            return jnp.zeros_like(self.amp_raw)
        ph = self.phase - self.phase[self.ref_station]
        return jnp.arctan2(jnp.sin(ph), jnp.cos(ph))

    def station_gains(self) -> jax.Array:
        """Return the complex per-station gains ``amp * exp(i * phase)``.

        The complex width matches the parameter dtype (``float32 -> complex64``,
        ``float64 -> complex128``), so an ``x64`` context yields full precision.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times, n_hands)`` complex gains.
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
        hands: tuple[int, int] = (0, 0),
        time_indices: jax.Array | None = None,
        inverse: bool = False,
    ) -> jax.Array:
        """Apply (or invert) the station gains on a set of model visibilities.

        Each visibility ``V_ij(t)`` is multiplied by
        ``g_{hands[0]},i(t) * conj(g_{hands[1]},j(t))`` (or divided by it when
        ``inverse`` is ``True``). Padded baseline slots (station id ``-1``) are
        assigned unit gain and pass through unchanged.

        Parameters
        ----------
        vis : jax.Array
            ``(T, M)`` complex model visibilities (``T`` times, ``M`` baselines).
        bl_station_ids : jax.Array
            ``(T, M, 2)`` integer station ids per baseline; ``-1`` marks padding.
        hands : tuple of int, optional
            Hand index (R=0, L=1) for the unconjugated and conjugated station,
            e.g. ``PRODUCT_HANDS["RL"] == (0, 1)``. Indices are clamped to the
            available ``n_hands``, so a single-hand model ties R and L.
            Default ``(0, 0)``.
        time_indices : jax.Array or None, optional
            ``(T,)`` integer frame indices into the gain table for each row of
            ``vis``. Required whenever ``vis`` is a time **minibatch** rather
            than the full time axis; the default ``None`` assumes row ``t`` of
            ``vis`` is frame ``t``.
        inverse : bool, optional
            If ``True``, divide by the gain factor instead of multiplying (the
            exact inverse of a forward apply). Default ``False``.

        Returns
        -------
        jax.Array
            ``(T, M)`` complex visibilities with gains applied.
        """
        g = self.station_gains()  # (S, T, H)
        h_i = min(int(hands[0]), self.n_hands - 1)
        h_j = min(int(hands[1]), self.n_hands - 1)
        g_i_tab, g_j_tab = g[:, :, h_i], g[:, :, h_j]  # (S, T) each
        if inverse:
            g_i_tab, g_j_tab = 1.0 / g_i_tab, 1.0 / g_j_tab
        # append a unit-gain row so padded id -1 indexes it (numpy negative index)
        ones = jnp.ones((1, g_i_tab.shape[1]), g_i_tab.dtype)
        g_i_pad = jnp.concatenate([g_i_tab, ones], axis=0)
        g_j_pad = jnp.concatenate([g_j_tab, ones], axis=0)
        ids = jnp.asarray(bl_station_ids)
        if time_indices is None:
            t_idx = jnp.broadcast_to(jnp.arange(vis.shape[0])[:, None], ids.shape[:2])
        else:
            t_idx = jnp.broadcast_to(jnp.asarray(time_indices)[:, None], ids.shape[:2])
        gi = g_i_pad[ids[..., 0], t_idx]
        gj = g_j_pad[ids[..., 1], t_idx]
        return vis * gi * jnp.conj(gj)
