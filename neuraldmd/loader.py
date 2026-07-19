"""Loader for NeuralDMD visibility datasets.

Consumes a dataset directory produced by eht2017/data_generation.py (see
eht2017/README.md for the exact file formats) and serves per-epoch batches of
time frames: each batch bundles the frame's forward operator, complex
visibilities, amplitudes, and closure phases, all padded to fixed shapes with
accompanying masks.
"""

import os
import numpy as np


class DMDDataLoader:
    def __init__(
        self,
        data,  # ground-truth image cube (num_frames, H, W); diagnostics only
        batch_size,  # number of time frames per batch
        epochs,  # total number of epochs (time indices are precomputed)
        data_dir,  # dataset directory with the .npy observation products
        times=None,  # (num_frames,) times, typically normalized to [0, 1]
        fov_x=np.pi,  # network coordinate extents (arbitrary units; the
        fov_y=np.pi,  # physical scale lives in the forward operators)
        time_fraction=1.0,  # fraction of frames sampled per epoch
        shuffle=True,
        seed=42,
    ):
        self.data_dir = data_dir
        self.times = np.asarray(times) if times is not None else None

        self.data = np.asarray(data)
        self.num_frames, self.height, self.width = self.data.shape
        self.num_samples = self.height * self.width

        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)
        self.pixel_size_x = self.fov_x / self.width
        self.pixel_size_y = self.fov_y / self.height

        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Pixel coordinates for the coordinate network, (P, 2)
        self.indices = np.arange(self.num_samples, dtype=np.int64)
        self.pixel_coords = self._pixel_to_physical(self.indices)

        # Observation products (see eht2017/README.md)
        load = lambda name: np.load(os.path.join(data_dir, name))
        self.As_full = load("As.npy")  # (T, M, P) complex64
        self.targets_full = load("targets.npy")  # (T, M) complex64
        self.sigmas_full = load("sigmas.npy")  # (T, M)
        self.masks_full = load("masks.npy")  # (T, M)
        self.num_vis_list = load("num_vis_list.npy")  # (T,)

        self.amp_targets = load("amp_targets.npy")  # (T, M)
        self.amp_sigmas = load("amp_sigmas.npy")  # (T, M)

        self.cp_targets_full = load("cp_targets.npy")  # (T, K)
        self.cp_sigmas_full = load("cp_sigmas.npy")  # (T, K)
        self.cp_masks_full = load("cp_masks.npy")  # (T, K)
        self.triangles = load("cp_tris.npy")  # (T, K, 3, 2)

        assert self.As_full.shape[2] == self.num_samples, (
            f"forward operators expect {self.As_full.shape[2]} pixels but the "
            f"movie grid has {self.num_samples}"
        )

        # The obs scheduler can produce slightly fewer observed frames than
        # the movie has (a trailing scan can fall outside the window); the
        # first T_obs frames correspond one-to-one by construction.
        T_obs = self.As_full.shape[0]
        if T_obs < self.num_frames:
            print(
                f"[DMDDataLoader] dataset has {T_obs} observed frames; "
                f"trimming movie from {self.num_frames} frames to match"
            )
            self.data = self.data[:T_obs]
            self.num_frames = T_obs
            if self.times is not None:
                self.times = self.times[:T_obs]
        elif T_obs > self.num_frames:
            raise ValueError(
                f"dataset has {T_obs} frames but the movie has only "
                f"{self.num_frames}"
            )

        # Time sampling
        self.time_fraction = float(time_fraction)
        self.num_time_samples = int(self.time_fraction * self.num_frames)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self._precompute_time_indices()

    def _pixel_to_physical(self, pixel_indices: np.ndarray) -> np.ndarray:
        """Flat pixel indices -> centered network coordinates, (P, 2)."""
        x_coords = pixel_indices % self.width
        y_coords = pixel_indices // self.width
        theta_x = (x_coords - self.width / 2.0) * self.pixel_size_x
        theta_y = (y_coords - self.height / 2.0) * self.pixel_size_y
        return np.stack([theta_x, theta_y], axis=-1).astype(np.float32)

    def _precompute_time_indices(self):
        """Draw the random subset of frames used in each epoch up front."""
        self.precomputed_time_indices = []
        frame_indices = np.arange(self.num_frames, dtype=np.int64)
        for _ in range(self.epochs):
            t_indices = self.rng.choice(
                frame_indices, size=self.num_time_samples, replace=False
            )
            self.precomputed_time_indices.append(t_indices)

    def get_epoch_data(self, epoch: int):
        """Batched arrays for one epoch.

        Returns a tuple matching neural_dmd.train_epoch_jit:
        (frame_batches (B, T_b, P), pixel_coords (P, 2),
         As (B, T_b, M, P), targets, sigmas, masks (B, T_b, M),
         times (B, T_b), amps, amp_sigmas (B, T_b, M),
         cp_targets, cp_sigmas, cp_masks (B, T_b, K), triangles (B, T_b, K, 3, 2))
        """
        time_indices = self.precomputed_time_indices[epoch % self.epochs]
        if self.shuffle:
            time_indices = self.rng.permutation(time_indices)

        # trim to a multiple of batch_size
        total_trim = len(time_indices) - (len(time_indices) % self.batch_size)
        time_indices = time_indices[:total_trim]
        num_batches = total_trim // self.batch_size

        times = (
            self.times[time_indices]
            if self.times is not None
            else time_indices.astype(np.float32)
        )

        def batched(arr):
            sel = arr[time_indices, ...]
            return sel.reshape(num_batches, self.batch_size, *sel.shape[1:])

        frame_batches = batched(self.data.reshape(self.num_frames, -1))

        return (
            frame_batches,
            self.pixel_coords,
            batched(self.As_full),
            batched(self.targets_full),
            batched(self.sigmas_full),
            batched(self.masks_full),
            times.reshape(num_batches, self.batch_size),
            batched(self.amp_targets),
            batched(self.amp_sigmas),
            batched(self.cp_targets_full),
            batched(self.cp_sigmas_full),
            batched(self.cp_masks_full),
            batched(self.triangles),
        )
