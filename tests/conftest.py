"""Shared test fixtures.

``tiny_obs`` builds a complete, self-consistent NeuralDMD observation directory
(all 11 ``.npy`` products the loader reads) plus the ground-truth movie, using
only numpy — no ehtim — so the core suite is fast and dependency-light. The
forward operator is an exact DFT on the loader's own pixel grid, and every
target is derived from the truth movie, so the truth reconstructs the data with
chi2 ~ 0 (useful for the mini-train test).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


@dataclass
class TinyObs:
    """A tiny synthetic dataset + the raw arrays behind it."""

    data_dir: str
    movie: np.ndarray  # (num_frames, H, W) physical truth
    times: np.ndarray  # (num_frames,)
    H: int
    W: int
    P: int
    num_frames: int
    T_obs: int
    M: int
    n_real: int
    K: int
    fov: float
    # raw observation products (first T_obs frames)
    A: np.ndarray
    targets: np.ndarray
    sigmas: np.ndarray
    masks: np.ndarray
    amp_targets: np.ndarray
    amp_sigmas: np.ndarray
    cp_targets: np.ndarray
    cp_sigmas: np.ndarray
    cp_masks: np.ndarray
    tris: np.ndarray
    num_vis: np.ndarray

    def loader_grid(self) -> np.ndarray:
        """Pixel coords exactly as ``DMDDataLoader._pixel_to_physical`` builds them."""
        idx = np.arange(self.P)
        gx = ((idx % self.W) - self.W / 2.0) * (self.fov / self.W)
        gy = ((idx // self.W) - self.H / 2.0) * (self.fov / self.H)
        return np.stack([gx, gy], axis=-1).astype(np.float32)


@pytest.fixture(scope="session")
def tiny_obs(tmp_path_factory) -> TinyObs:
    rng = np.random.default_rng(0)
    H = W = 16
    P = H * W
    num_frames = 7  # movie length ...
    T_obs = 6  # ... obs has one fewer -> exercises the loader's trim
    n_real = 10
    M = 12  # 10 real + 2 padded visibilities
    K = 4  # closure triangles
    fov = float(np.pi)

    # loader pixel grid (must match DMDDataLoader._pixel_to_physical)
    idx = np.arange(P)
    gx = ((idx % W) - W / 2.0) * (fov / W)
    gy = ((idx // W) - H / 2.0) * (fov / H)

    # truth movie: two counter-orbiting Gaussians, real & non-negative
    ts = np.linspace(0.0, 1.0, num_frames)
    XX, YY = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    movie = np.zeros((num_frames, H, W), np.float32)
    for phase in (0.0, np.pi):
        for i, tt in enumerate(ts):
            cx = 0.4 * np.cos(2 * np.pi * tt + phase)
            cy = 0.4 * np.sin(2 * np.pi * tt + phase)
            movie[i] += np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * 0.15**2))
    # normalize each frame to unit total flux
    movie /= movie.reshape(num_frames, -1).sum(axis=1)[:, None, None]
    Iflat = movie.reshape(num_frames, P).astype(np.float32)

    # exact-DFT forward operator on the loader grid (fixed uv across frames is
    # fine for arithmetic tests; real EHT uv rotates per frame)
    u = rng.uniform(-1.5, 1.5, n_real)
    v = rng.uniform(-1.5, 1.5, n_real)
    A_real = np.exp(-2j * np.pi * (u[:, None] * gx[None, :] + v[:, None] * gy[None, :])).astype(
        np.complex64
    )  # (n_real, P)
    A = np.zeros((T_obs, M, P), np.complex64)
    A[:, :n_real, :] = A_real[None]  # padded rows are zero

    # targets from the truth (first T_obs frames) -> truth gives chi2 ~ 0
    targets = np.zeros((T_obs, M), np.complex64)
    for i in range(T_obs):
        targets[i, :n_real] = A_real @ Iflat[i]

    sigmas = np.full((T_obs, M), 1e6, np.float32)
    sigmas[:, :n_real] = 0.02
    masks = np.zeros((T_obs, M), np.float32)
    masks[:, :n_real] = 1.0
    num_vis = np.full((T_obs,), n_real, np.int32)
    amp_targets = np.abs(targets).astype(np.float32)
    amp_sigmas = sigmas.copy()

    # closure triangles over the real visibilities (all signs +1)
    base = [(0, 1, 2), (1, 2, 3), (3, 4, 5), (0, 4, 6)]
    tris = np.full((T_obs, K, 3, 2), -1, np.int32)
    cp_targets = np.zeros((T_obs, K), np.float32)
    cp_sigmas = np.full((T_obs, K), 1e6, np.float32)
    cp_masks = np.zeros((T_obs, K), np.float32)
    for i in range(T_obs):
        for j, (a, b, c) in enumerate(base):
            for leg, bl in enumerate((a, b, c)):
                tris[i, j, leg, 0] = bl
                tris[i, j, leg, 1] = 1
            bispectrum = targets[i, a] * targets[i, b] * targets[i, c]
            cp_targets[i, j] = np.angle(bispectrum)
            cp_sigmas[i, j] = 0.05
            cp_masks[i, j] = 1.0

    d = Path(tmp_path_factory.mktemp("obs_dir"))
    files = {
        "As.npy": A,
        "targets.npy": targets,
        "sigmas.npy": sigmas,
        "masks.npy": masks,
        "num_vis_list.npy": num_vis,
        "amp_targets.npy": amp_targets,
        "amp_sigmas.npy": amp_sigmas,
        "cp_targets.npy": cp_targets,
        "cp_sigmas.npy": cp_sigmas,
        "cp_masks.npy": cp_masks,
        "cp_tris.npy": tris,
    }
    for name, arr in files.items():
        np.save(d / name, arr)

    return TinyObs(
        data_dir=str(d),
        movie=movie,
        times=ts.astype(np.float32),
        H=H,
        W=W,
        P=P,
        num_frames=num_frames,
        T_obs=T_obs,
        M=M,
        n_real=n_real,
        K=K,
        fov=fov,
        A=A,
        targets=targets,
        sigmas=sigmas,
        masks=masks,
        amp_targets=amp_targets,
        amp_sigmas=amp_sigmas,
        cp_targets=cp_targets,
        cp_sigmas=cp_sigmas,
        cp_masks=cp_masks,
        tris=tris,
        num_vis=num_vis,
    )
