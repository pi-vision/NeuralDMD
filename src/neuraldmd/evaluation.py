"""Evaluation and visualization helpers for the Fourier tutorial."""

import os
import warnings

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
        im = axes[i].imshow(
            mode_i, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax), interpolation="bicubic"
        )
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
        patches.Circle(
            (0, 0), radius=r_max + 1, facecolor="lightcoral", alpha=0.15, edgecolor=None, zorder=0
        )
    )
    ax.add_patch(
        patches.Circle(
            (0, 0), radius=1.0, facecolor="lightskyblue", alpha=0.25, edgecolor=None, zorder=1
        )
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
    for i, (ax, f) in enumerate(zip(axes, frame_list, strict=False)):
        ax.imshow(f, cmap="afmhot", vmin=0, vmax=vmax, interpolation="bicubic")
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


def make_gif(frames, path, fps=20, vmin=None, vmax=None, cmap="afmhot"):
    """Save a (T, H, W) array as a colormapped GIF."""
    frames_u8 = _to_uint8(frames, vmin, vmax, cmap)
    iio.imwrite(path, frames_u8, duration=int(1000 / fps), loop=0)
    print(f"Saved {path}")


def write_mp4(frames, path, fps=20, vmin=None, vmax=None, cmap="afmhot"):
    """Save a (T, H, W) array as an mp4 (requires the imageio-ffmpeg plugin)."""
    frames_u8 = _to_uint8(frames, vmin, vmax, cmap)
    iio.imwrite(path, frames_u8, fps=fps, codec="libx264")
    print(f"Saved {path}")


def make_comparison_gif(truth, recon, path, fps=20, cmap="afmhot"):
    """Side-by-side (truth | reconstruction) GIF with a shared color scale."""
    truth, recon = np.asarray(truth), np.asarray(recon)
    vmin = min(truth.min(), recon.min())
    vmax = max(truth.max(), recon.max())
    pad = np.full((truth.shape[0], truth.shape[1], 2), vmin)
    combined = np.concatenate([truth, pad, recon], axis=2)
    make_gif(combined, path, fps=fps, vmin=vmin, vmax=vmax, cmap=cmap)


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

    def load(name):
        return np.load(os.path.join(obs_dir, name))

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
    # The observation scheduler can drop a trailing scan, so a full-movie
    # reconstruction may carry one more frame than there are observed frames
    # (DMDDataLoader applies the same trim). Match by using the first T frames.
    if I.shape[1] < T:
        raise ValueError(f"reconstruction has {I.shape[1]} frames, fewer than {T} observed")
    if I.shape[1] > T:
        warnings.warn(
            f"reconstruction has {I.shape[1]} frames; using the first {T} to match "
            f"the {T} observed frames.",
            stacklevel=2,
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
        V = [np.where(signs[..., k] < 0, np.conj(V[k]), V[k]) for k in range(3)]
        phasor = V[0] * V[1] * V[2]
        phasor = phasor / (np.abs(phasor) + 1e-12)
        res = np.abs(phasor - np.exp(1j * cp_targets[sl])) ** 2
        cp_num += np.sum(res * cp_masks[sl] / cp_sigmas[sl] ** 2)

    return {
        "chi2_vis": float(chi2_vis_num / (2.0 * masks.sum())),
        "chi2_amp": float(chi2_amp_num / masks.sum()),
        "chi2_cp": float(cp_num / cp_masks.sum()),
    }


# ---------------------------------------------------------------------------
# Polarized reconstruction metrics (Milestone M2)
# ---------------------------------------------------------------------------


def reconstruct_polarized_cubes(model, npix, times, frame_max, frame_min, fov_x=np.pi, fov_y=np.pi):
    """Reconstruct per-Stokes image cubes from a ``PolarizedNeuralDMD``.

    Each sub-model is evaluated with its own physical scaling.

    Parameters
    ----------
    model : PolarizedNeuralDMD
    npix : int
        Image grid side length.
    times : array-like
        ``(T,)`` normalized frame times.
    frame_max, frame_min : dict of str -> float
        Per-Stokes output scaling (same dicts used for training).
    fov_x, fov_y : float
        Network coordinate extents (must match the loader).

    Returns
    -------
    dict of str -> numpy.ndarray
        ``{stokes: (T, npix, npix)}`` reconstructed cubes.
    """
    xy = jnp.asarray(pixel_grid_coords(npix, npix, fov_x, fov_y))
    times = jnp.asarray(np.asarray(times, dtype=np.float32))
    images, _ = model.stokes_fields(xy, times, frame_max, frame_min)
    return {s: np.asarray(images[s]).T.reshape(len(times), npix, npix) for s in images}


def polarized_nrmse(recon, truth):
    """Per-Stokes normalized RMSE ``||recon - truth|| / ||truth||`` (Frobenius)."""
    out = {}
    for s in recon:
        r, t = np.asarray(recon[s]), np.asarray(truth[s])
        denom = np.linalg.norm(t)
        out[s] = float(np.linalg.norm(r - t) / denom) if denom > 0 else float("nan")
    return out


def evpa_error_deg(recon, truth, frac_thresh: float = 0.5):
    """Median EVPA error [degrees] where the truth polarized intensity is bright.

    Compares ``0.5*atan2(U, Q)`` of the reconstruction and the truth over pixels
    with ``P_truth > frac_thresh * P_truth.max()``, wrapping the angular
    difference into ``(-90, 90]`` degrees before taking the median absolute value.
    """
    from .physics.stokes import evpa, linear_polarized_intensity

    tq, tu = np.asarray(truth["Q"]), np.asarray(truth["U"])
    rq, ru = np.asarray(recon["Q"]), np.asarray(recon["U"])
    p = linear_polarized_intensity(tq, tu)
    mask = p > frac_thresh * p.max()
    if not mask.any():
        return float("nan")
    diff = evpa(rq, ru)[mask] - evpa(tq, tu)[mask]
    diff = (diff + np.pi / 2) % np.pi - np.pi / 2  # wrap to (-pi/2, pi/2]
    return float(np.degrees(np.median(np.abs(diff))))


def _evpa_quiver(ax, intensity, q, u, fov_uas, *, cmap_bg="afmhot", vmax=None, skip=None):
    """Overlay EVPA ticks on a Stokes-I background (EHT dynamics-plot convention).

    Ticks point along ``(-sin chi, cos chi)`` with ``chi = 0.5*angle(Q + iU)``,
    have length proportional to the polarized intensity, and are colored by the
    fractional polarization; the intensity map uses ``afmhot`` with RA increasing
    to the left. Ticks are masked where I or P is below 10% of its peak.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    intensity, q, u : numpy.ndarray
        ``(H, W)`` Stokes I, Q, U maps for a single frame.
    fov_uas : float
        Field of view [micro-arcsec] setting the tick geometry.
    cmap_bg : str, optional
        Background colormap for Stokes I. Default ``"afmhot"``.
    vmax : float or None, optional
        Upper limit of the intensity color scale (``None`` autoscales).
    skip : int or None, optional
        Draw one tick every ``skip`` pixels (default ``~W/20``).

    Returns
    -------
    tuple of (matplotlib.quiver.Quiver, matplotlib.image.AxesImage)
        The tick and background-image artists.
    """
    from matplotlib.colors import Normalize

    intensity = np.asarray(intensity)
    q, u = np.asarray(q), np.asarray(u)
    _, nx = intensity.shape
    lims = [fov_uas / 2, -fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    bg = ax.imshow(
        intensity,
        cmap=cmap_bg,
        origin="upper",
        vmin=0.0,
        vmax=vmax,
        extent=lims,
        interpolation="bicubic",
    )

    px = fov_uas / nx
    yy, xx = np.mgrid[slice(-fov_uas / 2, fov_uas / 2, px), slice(-fov_uas / 2, fov_uas / 2, px)]
    amp = np.sqrt(q**2 + u**2)
    scal = float(amp.max() * 0.5) or 1.0
    angle = np.angle(q + 1j * u)
    vx = -np.sin(angle / 2) * amp / scal
    vy = np.cos(angle / 2) * amp / scal
    with np.errstate(divide="ignore", invalid="ignore"):
        mfrac = amp / np.abs(intensity)

    imax = float(np.abs(intensity).max()) or 1.0
    qumax = float(amp.max()) or 1.0
    mask = (np.abs(intensity) < 0.1 * imax) | (amp < 0.1 * qumax)
    vx = np.ma.masked_where(mask, vx)
    vy = np.ma.masked_where(mask, vy)
    mfrac = np.ma.masked_where(mask, mfrac)

    if skip is None:
        skip = max(1, nx // 20)
    quiv = ax.quiver(
        -xx[::skip, ::skip],
        -yy[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        mfrac[::skip, ::skip],
        cmap="rainbow",
        norm=Normalize(vmin=0.0, vmax=0.5),
        headlength=0,
        headwidth=1,
        pivot="mid",
        scale=16,
        width=0.01,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return quiv, bg


def plot_polarized_summary(recon, truth, path, frame=None, fov_uas=200.0):
    """Save a truth-vs-reconstruction figure: time-mean (or one frame) I/Q/U maps
    plus a Stokes-I image overlaid with EVPA ticks (EHT dynamics-plot style)."""

    def pick(cube):
        cube = np.asarray(cube)
        return cube.mean(0) if frame is None else cube[frame]

    lims = [fov_uas / 2, -fov_uas / 2, -fov_uas / 2, fov_uas / 2]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, s in enumerate(["I", "Q", "U"]):
        cmap = "afmhot" if s == "I" else "coolwarm"
        for row, (label, cube) in enumerate([("truth", truth), ("recon", recon)]):
            axes[row, col].imshow(
                pick(cube[s]), cmap=cmap, origin="upper", extent=lims, interpolation="bicubic"
            )
            axes[row, col].set_title(f"{label} {s}")
            axes[row, col].axis("off")

    for row, (label, cube) in enumerate([("truth", truth), ("recon", recon)]):
        ax = axes[row, 3]
        _evpa_quiver(ax, pick(cube["I"]), pick(cube["Q"]), pick(cube["U"]), fov_uas)
        ax.set_title(f"{label} I + EVPA")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_polarized_gif(cubes, path, fps=10, cmap="afmhot", fov_uas=200.0):
    """Animate a polarized reconstruction (EHT dynamics-plot style): per-frame
    Stokes I with EVPA ticks colored by fractional polarization, length
    proportional to polarized intensity.

    Parameters
    ----------
    cubes : dict of str -> numpy.ndarray
        ``{"I": (T, H, W), "Q": ..., "U": ...}`` reconstruction (or truth).
    path : str
        Output GIF path.
    fps : int, optional
        Frames per second. Default 10.
    cmap : str, optional
        Background colormap for Stokes I. Default ``"afmhot"``.
    fov_uas : float, optional
        Field of view [micro-arcsec] for the tick geometry. Default 200.
    """
    intensity = np.asarray(cubes["I"])
    q = np.asarray(cubes["Q"]) if "Q" in cubes else None
    u = np.asarray(cubes["U"]) if "U" in cubes else None
    n_t = intensity.shape[0]
    vmax = float(intensity.max()) or 1.0

    frames = []
    for t in range(n_t):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        if q is not None and u is not None:
            _evpa_quiver(ax, intensity[t], q[t], u[t], fov_uas, cmap_bg=cmap, vmax=vmax)
        else:
            lims = [fov_uas / 2, -fov_uas / 2, -fov_uas / 2, fov_uas / 2]
            ax.imshow(
                intensity[t],
                cmap=cmap,
                origin="upper",
                vmin=0.0,
                vmax=vmax,
                extent=lims,
                interpolation="bicubic",
            )
        ax.axis("off")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    iio.imwrite(path, np.stack(frames), duration=int(1000 / fps), loop=0)
    print(f"Saved {path}")
