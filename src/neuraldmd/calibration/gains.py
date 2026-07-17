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

from collections.abc import Sequence

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

#: Per-station gain-amplitude bounds reflecting real EHT calibration quality.
#:
#: A single global bound makes every station equally free, which leaves the overall
#: gain scale unconstrained: it is degenerate with total source flux, so the fit
#: slides along that direction (measured: a 1.153 global scale with the source flux
#: collapsing to compensate). Well-calibrated stations pin the scale, and that only
#: works if their bounds actually say they are well calibrated.
#:
#: Bounds are set per station from that station's known calibration quality, and are
#: validated against the gains actually observed on sky: the April-11 Sgr A* caltable
#: (``ehteval/caltable_april11``) gives, across both hands and all times,
#:
#:     ALMA 0.989-1.006   SMA 0.995-1.004   JCMT 0.960-1.014   APEX 0.958-1.030
#:     LMT  0.976-1.046   SMT 0.963-1.046   SPT  0.943-1.029
#:
#: (1.55% rms deviation from unity overall). Because the bound is a hard sigmoid clip,
#: a bound that excludes the truth makes the fit measure the clip rather than the gain
#: -- so every entry here contains the observed range with margin, and errs loose.
#: ``tests/test_gains.py`` asserts exactly that, per station.
#:
#: PV and SMAR do not appear in that caltable; their 10% is a moderate default.
EHT_GAIN_PRIORS: dict[str, tuple[float, float]] = {
    "ALMA": (0.97, 1.03),  # best-calibrated; observed 0.989-1.006
    "SMA": (0.97, 1.03),  # observed 0.995-1.004
    "APEX": (0.94, 1.06),  # observed down to 0.958 -- needs more than +-3%
    "JCMT": (0.94, 1.06),  # observed down to 0.960 -- needs more than +-3%
    "SPT": (0.94, 1.06),  # observed 0.943-1.029
    "SMT": (0.90, 1.10),  # observed 0.963-1.046
    "LMT": (0.85, 1.15),  # the known-worst EHT station; observed 0.976-1.046
    "PV": (0.90, 1.10),  # not in the caltable; default
    "SMAR": (0.90, 1.10),  # SMA reference antenna; pinned to 1.000 in the caltable
}


def eht_amp_bounds(
    stations: Sequence[str], default: tuple[float, float] = (0.90, 1.10)
) -> tuple[tuple[float, float], ...]:
    """Look up per-station amplitude bounds for an array, by station name.

    Parameters
    ----------
    stations : sequence of str
        Station names, in the order the gain table is indexed.
    default : tuple of float, optional
        Bounds for any station absent from :data:`EHT_GAIN_PRIORS`.

    Returns
    -------
    tuple of tuple of float
        ``(lo, hi)`` per station, ready to pass as ``amp_bounds``.
    """
    return tuple(EHT_GAIN_PRIORS.get(str(s).upper(), default) for s in stations)


def _normalize_amp_bounds(
    amp_bounds: tuple[float, float] | Sequence[tuple[float, float]], n_stations: int
) -> tuple[tuple[float, float], ...]:
    """Coerce scalar or per-station bounds to one ``(lo, hi)`` per station."""
    flat_pair = len(amp_bounds) == 2 and all(
        isinstance(b, (int, float, np.floating)) for b in amp_bounds
    )
    if flat_pair:
        pairs = [(float(amp_bounds[0]), float(amp_bounds[1]))] * n_stations
    else:
        if len(amp_bounds) != n_stations:
            raise ValueError(
                f"amp_bounds has {len(amp_bounds)} entries for {n_stations} stations; "
                "pass a single (lo, hi) or one pair per station"
            )
        pairs = [(float(b[0]), float(b[1])) for b in amp_bounds]
    for i, (lo, hi) in enumerate(pairs):
        if not (lo < 1.0 < hi):
            raise ValueError(
                f"amp_bounds must strictly bracket 1 so the fit starts at unit gain; "
                f"station {i} has {(lo, hi)}"
            )
    return tuple(pairs)


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
        initialized so that the amplitude is exactly 1 for every station,
        whatever its individual bounds.
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
    amp_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    ref_station: int = eqx.field(static=True)

    def __init__(
        self,
        n_stations: int,
        n_times: int,
        *,
        n_hands: int = 1,
        use_phase: bool = False,
        amp_bounds: tuple[float, float] | Sequence[tuple[float, float]] = (0.9, 1.1),
        ref_station: int = 0,
    ):
        if n_hands not in (1, 2):
            raise ValueError(f"n_hands must be 1 or 2, got {n_hands}")
        bounds = _normalize_amp_bounds(amp_bounds, int(n_stations))
        self.n_stations = int(n_stations)
        self.n_hands = int(n_hands)
        self.amp_bounds = bounds
        self.ref_station = int(ref_station)
        # sigmoid(raw0) = (1 - lo) / (hi - lo)  =>  amplitude(raw0) = 1 exactly,
        # per station, so a tight-bound station still starts at unit gain.
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        raw0 = np.log((1.0 - lo) / (hi - 1.0))
        self.amp_raw = jnp.asarray(
            np.broadcast_to(raw0[:, None, None], (n_stations, n_times, n_hands)).copy(),
            dtype=jnp.float32,
        )
        self.phase = jnp.zeros((n_stations, n_times, n_hands)) if use_phase else None

    def amplitudes(self) -> jax.Array:
        """Return the per-station gain amplitudes, sigmoid-bounded per station.

        Returns
        -------
        jax.Array
            ``(n_stations, n_times, n_hands)`` real amplitudes strictly inside each
            station's own ``(lo, hi)``, with nonzero gradient everywhere.
        """
        lo = jnp.asarray([b[0] for b in self.amp_bounds]).reshape(-1, 1, 1)
        hi = jnp.asarray([b[1] for b in self.amp_bounds]).reshape(-1, 1, 1)
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
