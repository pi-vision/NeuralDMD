"""Complex Zernike basis used to initialize NeuralDMD's spatial modes.

Zernike polynomials Z_n^m form a natural orthogonal basis on a disk. We use
the complex-valued variant R_n^{|m|}(rho) * exp(i m phi), masked to a disk of
the source's approximate size, and orthonormalized on the pixel grid with a
masked QR. Pretraining aligns the spatial network's modes with a low-order
subset of this bank (see pretraining.py).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import gammaln


def make_xy_grid(H: int, W: int, fov_x: float, fov_y: float):
    """Pixel-center coordinates (P, 2) in the same units the network uses."""
    xs = jnp.linspace(-fov_x / 2, fov_x / 2, W)
    ys = jnp.linspace(-fov_y / 2, fov_y / 2, H)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")
    return jnp.stack([X.ravel(), Y.ravel()], axis=1).astype(jnp.float32)


def _radial_R(n: int, mabs: int, rho):
    """Zernike radial polynomial R_n^{|m|}."""
    kmax = (n - mabs) // 2
    s = jnp.arange(kmax + 1)

    def lf(x):
        return gammaln(x + 1.0)

    logc = lf(n - s) - (lf(s) + lf((n + mabs) // 2 - s) + lf((n - mabs) // 2 - s))
    c = ((-1.0) ** s) * jnp.exp(logc)
    return jnp.sum(c * rho[..., None] ** (n - 2 * s), axis=-1)


def zernike_complex_basis(xy, radius: float, max_n: int, do_masked_qr: bool = True):
    """All Zernike modes with n <= max_n on a disk of the given radius.

    Returns
    -------
    Q : (P, K) complex64 — mode images as columns (orthonormalized on the
        masked grid if do_masked_qr)
    nm_list : list of (n, m) per column
    mask : (P,) float32 — 1 inside the disk
    """
    x, y = xy[:, 0], xy[:, 1]
    rho = jnp.sqrt(x * x + y * y) / float(radius)
    theta = jnp.arctan2(y, x)
    mask = (rho <= 1.0).astype(jnp.float32)
    rho = jnp.clip(rho, 0.0, 1.0)

    nm_list = []
    cols = []
    for n in range(max_n + 1):
        for m in range(-n, n + 1, 2):
            mabs = abs(m)
            if (n - mabs) % 2:  # R_n^m only defined for even n - |m|
                continue
            R = _radial_R(n, mabs, rho)
            Z = (R * jnp.exp(1j * m * theta)) * mask
            cols.append(Z.astype(jnp.complex64))
            nm_list.append((n, m))

    if not cols:
        return jnp.zeros((xy.shape[0], 0), jnp.complex64), [], mask

    B = jnp.stack(cols, axis=1)  # (P, K)
    if do_masked_qr:
        Wm = jnp.sqrt(mask)[:, None] * B
        Q, _ = jnp.linalg.qr(Wm, mode="reduced")
        Q = Q.astype(jnp.complex64)
    else:
        Q = B
    return Q, nm_list, mask


def pick_mode_set(Q, nm_list, r: int, prefer_ms=(0, 1, 2, 3)):
    """Select r columns, favoring low azimuthal order |m|, then low n.

    Low-|m| modes capture rings and dipole/quadrupole asymmetries — the right
    vocabulary for ring + orbiting hot spot morphologies.
    """
    order = []
    for k, (n, m) in enumerate(nm_list):
        try:
            m_rank = prefer_ms.index(abs(m))
        except ValueError:
            m_rank = len(prefer_ms) + abs(m)
        order.append((m_rank, n, abs(m), k))
    order.sort()
    idx = [t[-1] for t in order[:r]]
    W_target = Q[:, idx]
    picked = [nm_list[i] for i in idx]
    return W_target, picked


def build_zernike_targets(
    H, W, radius, fov_x, fov_y, r, max_n=8, prefer_ms=(0, 1, 2, 3), do_masked_qr=True
):
    """Zernike target images for pretraining.

    Returns
    -------
    Z_targets : (P, r) complex64 — chosen Zernike columns
    picked : list[(n, m)]
    mask : (P,) float32 — 1 inside the disk
    xy : (P, 2) float32 — grid coordinates (matches the training loader's)
    """
    xy = make_xy_grid(H, W, fov_x, fov_y)
    Q, nm_list, mask = zernike_complex_basis(
        xy, radius=radius, max_n=max_n, do_masked_qr=do_masked_qr
    )
    Z_targets, picked = pick_mode_set(Q, nm_list, r=r, prefer_ms=prefer_ms)
    return Z_targets.astype(jnp.complex64), picked, mask.astype(jnp.float32), xy


def plot_mode_bank(cols, labels, H, W, which="real", cols_per_row=6, savepath=None):
    """Plot the real/imag/abs part of a (P, K) stack of mode images."""
    take = {"real": np.real, "imag": np.imag, "abs": np.abs}[which]
    K = cols.shape[1]
    rows = (K + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(2.2 * cols_per_row, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for k in range(K):
        img = np.asarray(take(cols[:, k])).reshape(H, W)
        cmap = "viridis" if which == "abs" else "RdBu"
        axes[k].imshow(img, cmap=cmap)
        axes[k].set_title(labels[k], fontsize=9)
        axes[k].axis("off")
    for ax in axes[K:]:
        ax.axis("off")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
