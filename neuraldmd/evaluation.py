"""Evaluation and visualization helpers (modes, spectrum, movies, chi-squared)."""

import os

import h5py
import imageio.v3 as iio
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_hdf5(dir_, file):
    with h5py.File(os.path.join(dir_, file), "r") as f:
        frames = f["I"][:]
        times = f["times"][:]
    return frames, times


def observed_frame_count(obs_dir):
    """Number of frames actually present in a dataset directory.

    The observation scheduler can yield slightly fewer frames than the movie
    has (a trailing scan can fall outside the observation window); the first
    T_obs movie frames correspond to them one-to-one. DMDDataLoader trims to
    this count during training, so evaluation should too.
    """
    return int(np.load(os.path.join(obs_dir, "num_vis_list.npy")).shape[0])


def pixel_grid_coords(height, width, fov_x=np.pi, fov_y=np.pi):
    """Full-grid network coordinates, identical to DMDDataLoader.pixel_coords."""
    idx = np.arange(height * width, dtype=np.int64)
    x = idx % width
    y = idx // width
    theta_x = (x - width / 2.0) * (fov_x / width)
    theta_y = (y - height / 2.0) * (fov_y / height)
    return np.stack([theta_x, theta_y], axis=-1).astype(np.float32)


def sort_modes_by_lambda(W, Omega, b):
    """Order dynamic modes by |exp(Omega)| (slowest-decaying first)."""
    order = jnp.argsort(jnp.abs(jnp.exp(Omega)))[::-1]
    return W[:, order], Omega[order], b[order]


# -------------------------
# Plots
# -------------------------
def plot_modes(W, height, width, file_dir=None, title="Mode", part="real"):
    """Grid of spatial mode images (columns of W)."""
    take = {"real": np.real, "imag": np.imag, "abs": np.abs}[part]
    r = W.shape[1]
    cols = min(6, r)
    rows = r // cols + (r % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(r):
        mode_i = np.asarray(take(W[:, i])).reshape(height, width)
        vmax = np.abs(mode_i).max()
        vmin = 0.0 if part == "abs" else -vmax
        cmap = "inferno" if part == "abs" else "RdBu_r"
        im = axes[i].imshow(mode_i, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        axes[i].set_title(f"{title} {i}", fontsize=10)
        axes[i].axis("off")
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    for ax in axes[r:]:
        ax.axis("off")
    fig.tight_layout()
    if file_dir:
        fig.savefig(file_dir, dpi=150, bbox_inches="tight")
    return fig


def plot_unit_circle(Omega, file_path=None, r_max=1.4, include_conjugates=True):
    """Eigenvalues Lambda = exp(Omega) relative to the unit circle.

    |Lambda| < 1: decaying; |Lambda| = 1: purely oscillatory; the static mode
    sits at Lambda = 1.
    """
    Omega = np.asarray(Omega)
    if include_conjugates:
        Omega = np.concatenate([Omega, np.conj(Omega)])
    Lam = np.exp(Omega)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(
        patches.Circle((0, 0), radius=r_max + 1, facecolor="lightcoral",
                       alpha=0.15, edgecolor=None, zorder=0)
    )
    ax.add_patch(
        patches.Circle((0, 0), radius=1.0, facecolor="lightskyblue",
                       alpha=0.25, edgecolor=None, zorder=1)
    )
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="green", lw=1.0, zorder=2)
    ax.scatter([1, *Lam.real], [0, *Lam.imag], s=40, zorder=3)

    ax.text(-0.5, 0.05, "decaying", ha="center", color="navy", fontsize=12)
    ax.text(1.22, -0.12, "growing", ha="center", color="darkred", fontsize=12)
    ax.set_aspect("equal", "box")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, linestyle=":", zorder=0)
    fig.tight_layout()
    if file_path:
        fig.savefig(file_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frames(frame_list, titles=None, suptitle=None, file_path=None, vmax=None):
    """A row of movie frames with a shared color scale."""
    n = len(frame_list)
    vmax = vmax or max(np.max(f) for f in frame_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes = np.atleast_1d(axes)
    for i, (ax, f) in enumerate(zip(axes, frame_list)):
        ax.imshow(f, cmap="afmhot", vmin=0, vmax=vmax)
        if titles:
            ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    if file_path:
        fig.savefig(file_path, dpi=150, bbox_inches="tight")
    return fig


def calc_psnr(frame1, frame2, max_pixel_value=1.0):
    mse = np.mean((np.asarray(frame1) - np.asarray(frame2)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(max_pixel_value**2 / mse)


# -------------------------
# Movies
# -------------------------
def _to_uint8(frames, vmin=None, vmax=None, cmap="afmhot"):
    import matplotlib.cm as cm

    frames = np.asarray(frames)
    vmin = frames.min() if vmin is None else vmin
    vmax = frames.max() if vmax is None else vmax
    normed = np.clip((frames - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cm.get_cmap(cmap)(normed)
    return (255 * rgba[..., :3]).astype(np.uint8)


def _upscale(frames_u8, factor, resample="lanczos"):
    """Enlarge (T, H, W, 3) uint8 frames by an integer factor.

    Model grids are small (a 50x50 image is 50 pixels wide on screen), so
    videos are enlarged for display. This is purely cosmetic: it does not add
    information, and smooth resampling can make the reconstruction look like
    it resolved more than the baselines support — use resample="nearest" to
    keep the true pixel grid visible.
    """
    factor = int(factor or 1)
    if factor <= 1:
        return frames_u8

    if resample == "nearest":
        return frames_u8.repeat(factor, axis=1).repeat(factor, axis=2)

    from PIL import Image  # ships with imageio

    filt = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
    }[resample]
    T, H, W = frames_u8.shape[:3]
    out = np.empty((T, H * factor, W * factor, 3), dtype=np.uint8)
    for i, frame in enumerate(frames_u8):
        out[i] = np.asarray(
            Image.fromarray(frame).resize((W * factor, H * factor), filt)
        )
    return out


def make_gif(
    frames,
    path,
    fps=20,
    vmin=None,
    vmax=None,
    cmap="afmhot",
    upscale=4,
    resample="lanczos",
):
    """Save a (T, H, W) array as a colormapped GIF, enlarged upscale-fold."""
    frames_u8 = _upscale(_to_uint8(frames, vmin, vmax, cmap), upscale, resample)
    iio.imwrite(path, frames_u8, duration=int(1000 / fps), loop=0)
    print(f"Saved {path}  ({frames_u8.shape[2]}x{frames_u8.shape[1]}, {len(frames_u8)} frames)")


def write_mp4(
    frames,
    path,
    fps=20,
    vmin=None,
    vmax=None,
    cmap="afmhot",
    upscale=4,
    resample="lanczos",
):
    """Save a (T, H, W) array as an mp4 (requires the imageio-ffmpeg plugin)."""
    frames_u8 = _upscale(_to_uint8(frames, vmin, vmax, cmap), upscale, resample)
    iio.imwrite(path, frames_u8, fps=fps, codec="libx264")
    print(f"Saved {path}  ({frames_u8.shape[2]}x{frames_u8.shape[1]}, {len(frames_u8)} frames)")


def make_comparison_gif(
    truth, recon, path, fps=20, cmap="afmhot", upscale=4, resample="lanczos"
):
    """Side-by-side (truth | reconstruction) GIF with a shared color scale."""
    truth, recon = np.asarray(truth), np.asarray(recon)
    vmin = min(truth.min(), recon.min())
    vmax = max(truth.max(), recon.max())
    pad = np.full((truth.shape[0], truth.shape[1], 2), vmin)
    combined = np.concatenate([truth, pad, recon], axis=2)
    make_gif(
        combined,
        path,
        fps=fps,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        upscale=upscale,
        resample=resample,
    )


# -------------------------
# Chi-squared evaluation on the full dataset
# -------------------------
def evaluate_chi2(intensities, obs_dir, chunk=50):
    """Data-space goodness of fit of a reconstructed movie.

    intensities: (P, T) reconstruction in physical units (Jy/pixel)
    obs_dir: dataset directory with the observation products

    Returns a dict with mask-averaged chi2_vis, chi2_amp, chi2_cp. chi2_vis is
    reduced per real degree of freedom (a complex visibility carries 2), so
    all three sit at ~1 for a model that fits the data at the noise level.
    """
    load = lambda name: np.load(os.path.join(obs_dir, name))
    As = load("As.npy")
    targets = load("targets.npy")
    sigmas = load("sigmas.npy")
    masks = load("masks.npy")
    amp_targets = load("amp_targets.npy")
    amp_sigmas = load("amp_sigmas.npy")
    cp_targets = load("cp_targets.npy")
    cp_sigmas = load("cp_sigmas.npy")
    cp_masks = load("cp_masks.npy")
    triangles = load("cp_tris.npy")

    T = As.shape[0]
    I = np.asarray(intensities)
    if I.shape[1] < T:
        raise ValueError(
            f"reconstruction has {I.shape[1]} frames but the dataset has {T}; "
            "evaluate it on the dataset's time samples"
        )
    if I.shape[1] > T:
        # extra frames are normal when the movie outruns the observation
        # window (see observed_frame_count); the leading frames line up
        print(
            f"[evaluate_chi2] using the first {T} of {I.shape[1]} "
            "reconstructed frames (dataset has fewer observed frames)"
        )
        I = I[:, :T]

    chi2_vis_num = chi2_amp_num = cp_num = 0.0
    for t0 in range(0, T, chunk):
        sl = slice(t0, min(t0 + chunk, T))
        vis_pred = np.einsum("tvp,pt->tv", As[sl], I[:, sl].astype(np.complex64))

        diff = np.abs(vis_pred - targets[sl])
        chi2_vis_num += np.sum(diff**2 * masks[sl] / sigmas[sl] ** 2)
        chi2_amp_num += np.sum(
            (np.abs(vis_pred) - amp_targets[sl]) ** 2 * masks[sl] / amp_sigmas[sl] ** 2
        )

        idxs = triangles[sl][..., 0]
        signs = triangles[sl][..., 1]
        V = [np.take_along_axis(vis_pred, idxs[..., k], axis=-1) for k in range(3)]
        V = [
            np.where(signs[..., k] < 0, np.conj(V[k]), V[k]) for k in range(3)
        ]
        phasor = V[0] * V[1] * V[2]
        phasor = phasor / (np.abs(phasor) + 1e-12)
        res = np.abs(phasor - np.exp(1j * cp_targets[sl])) ** 2
        cp_num += np.sum(res * cp_masks[sl] / cp_sigmas[sl] ** 2)

    return {
        "chi2_vis": float(chi2_vis_num / (2.0 * masks.sum())),
        "chi2_amp": float(chi2_amp_num / masks.sum()),
        "chi2_cp": float(cp_num / cp_masks.sum()),
    }
