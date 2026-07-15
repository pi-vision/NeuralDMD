"""Observation products: the on-disk obs_dir contract, polarization-aware.

The image->visibility operator ``A`` (T, M, P) is shared across Stokes -- the
DFT geometry does not depend on polarization -- so it is stored once. Only the
visibility ``targets``/``sigmas``/``masks`` differ per Stokes. Masks are
per-Stokes because some stations observe only one hand (e.g. JCMT in EHT), so a
given baseline may be present for I but flagged for Q/U.

obs_dir layout:
  v2 (this module):   As.npy, targets_<S>.npy, sigmas_<S>.npy, masks_<S>.npy,
                      manifest.json {version, stokes}
  v1 (legacy):        As.npy, targets.npy, sigmas.npy, masks.npy  (Stokes I only)

Pure numpy -- no ehtim (the uvfits loader that *produces* these lives in the
``[data]`` extra).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObsProducts:
    """Shared A-matrix + per-Stokes visibility products for one dataset."""

    A: np.ndarray  # (T, M, P) complex64 image->visibility operator
    stokes: tuple[str, ...]
    targets: dict[str, np.ndarray]  # Stokes -> (T, M) complex
    sigmas: dict[str, np.ndarray]  # Stokes -> (T, M) float
    masks: dict[str, np.ndarray]  # Stokes -> (T, M) float {0,1}
    version: int = 2

    def __post_init__(self):
        self.stokes = tuple(self.stokes)
        self.validate()

    def validate(self) -> None:
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

    @property
    def n_frames(self) -> int:
        return self.A.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.A.shape[2]

    @classmethod
    def from_obs_dir(cls, obs_dir: str | Path) -> ObsProducts:
        """Load an obs_dir; auto-detects v2 (manifest) vs legacy v1 (Stokes-I)."""
        obs_dir = Path(obs_dir)
        A = np.load(obs_dir / "As.npy")
        manifest = obs_dir / "manifest.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text())
            stokes = tuple(meta["stokes"])
            version = int(meta.get("version", 2))

            def per(kind: str) -> dict[str, np.ndarray]:
                return {s: np.load(obs_dir / f"{kind}_{s}.npy") for s in stokes}

            return cls(A, stokes, per("targets"), per("sigmas"), per("masks"), version=version)

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
        """Write this dataset as a v2 obs_dir (shared A + per-Stokes products)."""
        obs_dir = Path(obs_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)
        np.save(obs_dir / "As.npy", self.A)
        for s in self.stokes:
            np.save(obs_dir / f"targets_{s}.npy", self.targets[s])
            np.save(obs_dir / f"sigmas_{s}.npy", self.sigmas[s])
            np.save(obs_dir / f"masks_{s}.npy", self.masks[s])
        (obs_dir / "manifest.json").write_text(
            json.dumps({"version": 2, "stokes": list(self.stokes)}, indent=2)
        )
