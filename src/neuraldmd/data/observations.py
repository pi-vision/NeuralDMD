"""Observation products: the on-disk obs_dir contract, polarization-aware.

The image->visibility operator ``A`` (T, M, P) is shared across Stokes -- the
DFT geometry does not depend on polarization -- so it is stored once. Only the
visibility ``targets``/``sigmas``/``masks`` differ per Stokes. Masks are
per-Stokes because some stations observe only one hand (e.g. JCMT in EHT), so a
given baseline may be present for I but flagged for Q/U.

obs_dir layout:
  v2 (this module):   As.npy, targets_<S>.npy, sigmas_<S>.npy, masks_<S>.npy,
                      optional bl_station_ids.npy, manifest.json {version, stokes,
                      stations}
  v1 (legacy):        As.npy, targets.npy, sigmas.npy, masks.npy  (Stokes I only)

The ``ObsProducts`` container is pure numpy. ``load_uvfits_to_products`` (bottom
of this module) is the one UVFITS ingestion path; it imports ehtim lazily, so it
only needs the ``[data]`` extra -- importing this module and using
``ObsProducts`` never touches ehtim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObsProducts:
    """Shared image->visibility operator plus per-Stokes products for a dataset.

    Attributes
    ----------
    A : numpy.ndarray
        Complex ``(T, M, P)`` image->visibility operator, shared across Stokes.
    stokes : tuple of str
        Stokes parameters present, e.g. ``("I", "Q", "U")``.
    targets, sigmas, masks : dict of str -> numpy.ndarray
        Per-Stokes visibilities, 1-sigma errors, and 0/1 validity masks, each
        of shape ``(T, M)``. Padded visibility slots have sigma 1e6 and mask 0.
    version : int
        obs_dir schema version (1 = legacy Stokes-I, 2 = per-Stokes).
    bl_station_ids : numpy.ndarray or None
        ``(T, M, 2)`` integer station ids per baseline (-1 for padded slots).
    stations : tuple of str or None
        Station names indexed by the ids in ``bl_station_ids``.
    """

    A: np.ndarray  # (T, M, P) complex64 image->visibility operator
    stokes: tuple[str, ...]
    targets: dict[str, np.ndarray]  # Stokes -> (T, M) complex
    sigmas: dict[str, np.ndarray]  # Stokes -> (T, M) float
    masks: dict[str, np.ndarray]  # Stokes -> (T, M) float {0,1}
    version: int = 2
    # Optional station metadata (populated by load_uvfits_to_products; needed by
    # the Phase-6 gains / Phase-7 RIME calibration). None for hand-built datasets.
    bl_station_ids: np.ndarray | None = None  # (T, M, 2) int; padded rows are -1
    stations: tuple[str, ...] | None = None  # station names, indexed by id

    def __post_init__(self):
        """Coerce tuple fields and validate the arrays on construction."""
        self.stokes = tuple(self.stokes)
        if self.stations is not None:
            self.stations = tuple(self.stations)
        self.validate()

    def validate(self) -> None:
        """Check ``A``'s rank and that every per-Stokes array is ``(T, M)``.

        Raises
        ------
        ValueError
            If ``A`` is not 3-D, the ``targets`` keys disagree with ``stokes``,
            any ``targets``/``sigmas``/``masks`` entry is missing or mis-shaped,
            or ``bl_station_ids`` (when present) is not ``(T, M, 2)``.
        """
        if self.A.ndim != 3:
            raise ValueError(f"A must be (T, M, P), got shape {self.A.shape}")
        T, M, _ = self.A.shape
        if tuple(self.targets) != self.stokes:
            raise ValueError(f"targets keys {tuple(self.targets)} != stokes {self.stokes}")
        for name, d in (("targets", self.targets), ("sigmas", self.sigmas), ("masks", self.masks)):
            for s in self.stokes:
                if s not in d:
                    raise ValueError(f"missing {name} for Stokes {s!r}")
                if d[s].shape != (T, M):
                    raise ValueError(f"{name}[{s!r}] shape {d[s].shape} != {(T, M)}")
        if self.bl_station_ids is not None and self.bl_station_ids.shape != (T, M, 2):
            raise ValueError(f"bl_station_ids shape {self.bl_station_ids.shape} != {(T, M, 2)}")

    @property
    def n_frames(self) -> int:
        """Number of time frames ``T``."""
        return self.A.shape[0]

    @property
    def n_pixels(self) -> int:
        """Number of image pixels ``P`` (``npix * npix``)."""
        return self.A.shape[2]

    @classmethod
    def from_obs_dir(cls, obs_dir: str | Path) -> ObsProducts:
        """Load an ``ObsProducts`` from an on-disk obs_dir.

        Auto-detects the layout: a ``manifest.json`` selects v2 (per-Stokes
        products, optional station metadata); its absence falls back to legacy
        v1 (Stokes-I only, unsuffixed files).

        Parameters
        ----------
        obs_dir : str or pathlib.Path
            Directory written by :meth:`to_obs_dir` (v2) or the legacy pipeline.

        Returns
        -------
        ObsProducts
            The loaded dataset, with ``version`` set to 1 or 2 accordingly.
        """
        obs_dir = Path(obs_dir)
        A = np.load(obs_dir / "As.npy")
        manifest = obs_dir / "manifest.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text())
            stokes = tuple(meta["stokes"])
            version = int(meta.get("version", 2))

            def per(kind: str) -> dict[str, np.ndarray]:
                """Load the per-Stokes ``.npy`` dict for one product kind."""
                return {s: np.load(obs_dir / f"{kind}_{s}.npy") for s in stokes}

            bl_path = obs_dir / "bl_station_ids.npy"
            bl = np.load(bl_path) if bl_path.exists() else None
            stations = meta.get("stations")
            return cls(
                A,
                stokes,
                per("targets"),
                per("sigmas"),
                per("masks"),
                version=version,
                bl_station_ids=bl,
                stations=tuple(stations) if stations is not None else None,
            )

        # legacy v1: Stokes I only, unsuffixed files
        return cls(
            A,
            ("I",),
            {"I": np.load(obs_dir / "targets.npy")},
            {"I": np.load(obs_dir / "sigmas.npy")},
            {"I": np.load(obs_dir / "masks.npy")},
            version=1,
        )

    def to_obs_dir(self, obs_dir: str | Path) -> None:
        """Write this dataset to disk as a v2 obs_dir.

        Writes the shared ``As.npy``, per-Stokes ``targets``/``sigmas``/``masks``
        arrays, a ``manifest.json`` (version, Stokes, and station names if set),
        and ``bl_station_ids.npy`` when station metadata is present.

        Parameters
        ----------
        obs_dir : str or pathlib.Path
            Destination directory; created (with parents) if it does not exist.
        """
        obs_dir = Path(obs_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)
        np.save(obs_dir / "As.npy", self.A)
        for s in self.stokes:
            np.save(obs_dir / f"targets_{s}.npy", self.targets[s])
            np.save(obs_dir / f"sigmas_{s}.npy", self.sigmas[s])
            np.save(obs_dir / f"masks_{s}.npy", self.masks[s])
        if self.bl_station_ids is not None:
            np.save(obs_dir / "bl_station_ids.npy", self.bl_station_ids)
        manifest: dict[str, object] = {"version": 2, "stokes": list(self.stokes)}
        if self.stations is not None:
            manifest["stations"] = list(self.stations)
        (obs_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# Stokes -> (visibility column, sigma column) in ehtim's stokes-polrep data table.
_STOKES_COLS: dict[str, tuple[str, str]] = {
    "I": ("vis", "sigma"),
    "Q": ("qvis", "qsigma"),
    "U": ("uvis", "usigma"),
    "V": ("vvis", "vsigma"),
}


def load_uvfits_to_products(
    path: str | Path,
    npix: int,
    fov_uas: float,
    *,
    stokes: tuple[str, ...] = ("I", "Q", "U"),
    tavg: float = 0.0,
    syserr: float = 0.0,
    flag_sites: tuple[str, ...] = (),
    t_gather: float | None = None,
) -> ObsProducts:
    """Load a UVFITS file into an :class:`ObsProducts`.

    Pipeline (mirrors NeuralDMD's own ``data/generation.py``):
    ``load_uvfits -> switch_polrep('stokes') -> flag -> avg_coherent ->
    add_fractional_noise -> split_obs``, then per snapshot build the shared
    image->visibility operator ``A`` once with ehtim ``chisqdata``
    (``pol='I'``; ``A`` is polarization-independent -- verified) and read each
    Stokes' visibilities/sigmas straight from the (row-aligned) obs data table.
    Per-Stokes masks come from finiteness, so single-hand stations (e.g. JCMT,
    which lacks one circular feed) are masked out for Q/U while kept for I.

    ehtim is imported lazily here; the rest of this module is pure numpy.

    Parameters
    ----------
    path : str or Path
        UVFITS file.
    npix : int
        Model grid side length; ``A`` maps ``npix*npix`` pixels -> visibilities.
    fov_uas : float
        Field of view in microarcseconds (defines the pixel grid, ``psize``).
    stokes : tuple of str
        Stokes to extract, default ``("I", "Q", "U")``. ``V`` is supported
        (read directly; ehtim ``chisqdata`` cannot form a Stokes-V vis term).
    tavg : float
        Coherent-averaging time [s]; 0 disables (no averaging).
    syserr : float
        Fractional systematic noise added in quadrature; 0 disables.
    flag_sites : tuple of str
        Station codes to drop before splitting.
    t_gather : float or None
        Snapshot length [h] for ``split_obs``; None uses ehtim's default
        (one snapshot per integration/scan).

    Returns
    -------
    ObsProducts
        Container with the shared operator ``A`` of shape ``(T, M, P)``,
        per-Stokes ``targets``/``sigmas``/``masks`` of shape ``(T, M)`` each,
        and station metadata (``bl_station_ids`` ``(T, M, 2)`` and ``stations``).
        ``M`` is padded to the largest snapshot; padded rows have mask 0.

    Raises
    ------
    ValueError
        If any requested Stokes is not one of I/Q/U/V.
    """
    import ehtim as eh
    from ehtim.imaging.imager_utils import chisqdata

    unknown = set(stokes) - _STOKES_COLS.keys()
    if unknown:
        raise ValueError(f"unknown Stokes requested: {sorted(unknown)}")

    obs = eh.obsdata.load_uvfits(str(path))
    obs = obs.switch_polrep("stokes")
    if flag_sites:
        obs = obs.flag_sites(list(flag_sites))
    if tavg > 0:
        obs = obs.avg_coherent(float(tavg))
    if syserr > 0:
        obs = obs.add_fractional_noise(float(syserr))

    frames = obs.split_obs(t_gather=t_gather) if t_gather is not None else obs.split_obs()
    prior = eh.image.make_square(obs, npix, fov_uas * eh.RADPERUAS)
    site_to_id = {str(site): i for i, site in enumerate(obs.tarr["site"])}

    # Per-frame extraction (ragged in the visibility axis; padded below).
    A_frames: list[np.ndarray] = []
    bl_frames: list[np.ndarray] = []
    t_frames: dict[str, list[np.ndarray]] = {s: [] for s in stokes}
    s_frames: dict[str, list[np.ndarray]] = {s: [] for s in stokes}
    m_frames: dict[str, list[np.ndarray]] = {s: [] for s in stokes}

    for f in frames:
        # Shared operator; chisqdata's row order matches f.data row order (verified).
        _, _, A = chisqdata(f, prior, mask=[], pol="I", dtype="vis")
        A_frames.append(np.asarray(A, dtype=np.complex64))
        d = f.data
        bl_frames.append(
            np.array([[site_to_id[str(r[2])], site_to_id[str(r[3])]] for r in d], dtype=np.int32)
        )
        for s in stokes:
            vcol, scol = _STOKES_COLS[s]
            t = np.asarray(d[vcol], dtype=np.complex64)
            sig = np.asarray(d[scol], dtype=np.float32)
            valid = np.isfinite(t.real) & np.isfinite(t.imag) & np.isfinite(sig) & (sig > 0)
            t_frames[s].append(np.where(valid, t, 0).astype(np.complex64))
            s_frames[s].append(np.where(valid, sig, 1e6).astype(np.float32))
            m_frames[s].append(valid.astype(np.float32))

    # Pad + stack to (T, M_max, ...); padding matches data/generation.py (sigma 1e6).
    n_t = len(frames)
    m_max = max((a.shape[0] for a in A_frames), default=0)
    n_pix = npix * npix
    A = np.zeros((n_t, m_max, n_pix), dtype=np.complex64)
    bl = np.full((n_t, m_max, 2), -1, dtype=np.int32)
    targets = {s: np.zeros((n_t, m_max), dtype=np.complex64) for s in stokes}
    sigmas = {s: np.full((n_t, m_max), 1e6, dtype=np.float32) for s in stokes}
    masks = {s: np.zeros((n_t, m_max), dtype=np.float32) for s in stokes}
    for i in range(n_t):
        m = A_frames[i].shape[0]
        A[i, :m] = A_frames[i]
        bl[i, :m] = bl_frames[i]
        for s in stokes:
            targets[s][i, :m] = t_frames[s][i]
            sigmas[s][i, :m] = s_frames[s][i]
            masks[s][i, :m] = m_frames[s][i]

    return ObsProducts(
        A,
        stokes,
        targets,
        sigmas,
        masks,
        bl_station_ids=bl,
        stations=tuple(str(site) for site in obs.tarr["site"]),
    )
