"""Measure the source light curve (total flux per frame) from the data itself.

Total flux is not a free parameter worth fitting: it is degenerate with the overall
station-gain amplitude, so a fit that solves for both slides along that degeneracy
(measured: a 1.153 global gain scale with the source flux collapsing 2.70 -> 1.92 Jy to
compensate). A soft flux penalty does not fix it -- even at ``flux_weight=1000`` the
recovered gains lost to a do-nothing baseline. The flux has to be *supplied*, not fitted.

It is also directly measurable. An interferometer's visibility at zero baseline is the
total flux, and **intra-site** baselines -- pairs of dishes at the same site, so short
that any plausible source is completely unresolved -- sample essentially that:

    ALMA-APEX   (Llano de Chajnantor)
    SMA-JCMT    (Maunakea)      [and SMA-SMAR, the SMA reference antenna]

so ``|V|`` on those baselines is the total flux. This is how the EHT sets absolute flux,
and it is why those baselines matter far beyond their (negligible) resolving power.

The measurement inherits the gain errors of the two stations involved, which is exactly
why the intra-site pairs are the *well-calibrated* ones: ALMA and APEX are each good to
~3%, so their product is good to ~4%. That is the accuracy of the resulting flux scale,
and it is far better than leaving the flux free.
"""

from __future__ import annotations

import numpy as np

#: Groups of EHT stations sharing a site. A baseline within a group is intra-site:
#: its length is metres rather than thousands of km, so the source is unresolved and
#: the visibility amplitude is the total flux.
CO_LOCATED_SITES: tuple[frozenset[str], ...] = (
    frozenset({"ALMA", "APEX"}),
    frozenset({"SMA", "JCMT", "SMAR"}),
)

#: Real EHT uvfits label stations by two-letter code, not by name. Without this,
#: an array that plainly has intra-site baselines looks like it has none and the
#: flux scale silently falls back to being fitted.
STATION_ALIASES: dict[str, str] = {
    "AA": "ALMA",
    "AP": "APEX",
    "JC": "JCMT",
    "SM": "SMA",
    "SR": "SMAR",
    "SMAP": "SMA",
    "AZ": "SMT",
    "LM": "LMT",
    "SP": "SPT",
    "PV": "PV",
    "GL": "GLT",
    "KT": "KP",
}


def canonical_station(name: str) -> str:
    """Map a station label to its canonical name.

    Parameters
    ----------
    name : str
        Station label, either a full name or a two-letter EHT code.

    Returns
    -------
    str
        Upper-case canonical name, unchanged if already canonical.
    """
    up = str(name).upper()
    return STATION_ALIASES.get(up, up)


def _intra_site_mask(stations: tuple[str, ...], bl_station_ids: np.ndarray) -> np.ndarray:
    """``(T, M)`` bool: True where a baseline joins two dishes at the same site.

    Parameters
    ----------
    stations : tuple of str
        Station labels in gain-table order.
    bl_station_ids : np.ndarray
        ``(T, M, 2)`` station indices per baseline.

    Returns
    -------
    np.ndarray
        ``(T, M)`` boolean mask.
    """
    upper = [canonical_station(s) for s in stations]
    site_of: dict[int, int] = {}
    for idx, name in enumerate(upper):
        for g, group in enumerate(CO_LOCATED_SITES):
            if name in group:
                site_of[idx] = g
    i, j = bl_station_ids[..., 0], bl_station_ids[..., 1]
    out = np.zeros(i.shape, dtype=bool)
    for a, ga in site_of.items():
        for b, gb in site_of.items():
            if a != b and ga == gb:
                out |= (i == a) & (j == b)
    return out


def measure_lightcurve(op) -> np.ndarray:
    """Total flux per frame, measured from intra-site baselines.

    Parameters
    ----------
    op : ObsProducts
        Observation products; needs ``bl_station_ids`` and ``stations``.

    Returns
    -------
    np.ndarray
        ``(T,)`` total flux [Jy] per frame. Frames with no valid intra-site sample are
        filled by interpolation from the frames that have one (and by the global median
        if there are none at all in a run of frames at the edges).

    Raises
    ------
    ValueError
        If the array has no intra-site baselines at all, or the products contain
        neither a Stokes ``I`` nor both circular parallel hands. Callers should treat
        this as "this dataset cannot supply its own flux scale" rather than guessing.
    """
    if op.bl_station_ids is None or not op.stations:
        raise ValueError("measure_lightcurve needs bl_station_ids and station names")

    if "I" in op.stokes:
        vis_i = op.targets["I"]
        mask = op.masks["I"] > 0
    elif "RR" in op.stokes and "LL" in op.stokes:
        # I = (RR + LL) / 2 exactly, for circular feeds
        vis_i = 0.5 * (op.targets["RR"] + op.targets["LL"])
        mask = (op.masks["RR"] > 0) & (op.masks["LL"] > 0)
    else:
        raise ValueError(f"need Stokes I, or both RR and LL to form it; got {op.stokes}")

    intra = _intra_site_mask(op.stations, op.bl_station_ids) & mask
    if not intra.any():
        sites = ", ".join("+".join(sorted(g)) for g in CO_LOCATED_SITES)
        raise ValueError(
            f"no intra-site baselines in this array (stations: {list(op.stations)}); "
            f"the flux scale cannot be measured from the data. Known co-located "
            f"pairs: {sites}"
        )

    amp = np.abs(vis_i)
    n_t = amp.shape[0]
    lc = np.full(n_t, np.nan)
    for t in range(n_t):
        sel = intra[t]
        if sel.any():
            lc[t] = float(np.mean(amp[t][sel]))

    good = np.isfinite(lc)
    if not good.all():
        # frames without an intra-site sample: interpolate across them rather than
        # dropping to zero, which would tell the model the source vanished
        idx = np.arange(n_t)
        lc = np.interp(idx, idx[good], lc[good])
    return lc
