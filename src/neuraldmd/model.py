"""The NeuralDMD model: spatial mode network + temporal spectrum/amplitude nets.

A movie is represented as

    I(x, t) = W0(x) b0 + 2 Re[ sum_k W_k(x) b_k exp(Omega_k * t_scale * t) ]

with the static mode ``W0`` and the ``r`` complex spatial modes ``W_k`` produced
by a coordinate ``ResidualMLP``, and the continuous spectrum
``Omega_k = alpha_k + i theta_k`` and amplitudes ``b_k`` produced by two small
latent-vector networks. The image is real, so ``r`` counts complex modes
directly (the conjugate twins are implicit in the ``2 Re[...]``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .encoding import SinusoidalEncoding
from .networks import ResidualMLP, zero_init_linear


class TemporalOmegaMLP(eqx.Module):
    """Maps a learned latent vector to the continuous spectrum Omega.

    ``alphas = -sigmoid(raw)`` -> decay rates in ``[-1, 0]`` (no growing modes);
    ``thetas = theta_min + sigmoid(raw) * (theta_max - theta_min)``.
    """

    latent: jax.Array
    core: ResidualMLP
    head: eqx.nn.Linear
    r_half: int = eqx.field(static=True)
    theta_min: float = eqx.field(static=True)
    theta_max: float = eqx.field(static=True)

    def __init__(
        self,
        r_half: int,
        latent_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        theta_min: float = 0.0,
        theta_max: float = 1.0,
        key=None,
        res_scale: float = 0.1,
    ):
        self.r_half = r_half
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        k_lat, k_core, k_head = jax.random.split(key, 3)
        self.latent = jax.random.normal(k_lat, (latent_dim,))
        self.core = ResidualMLP(
            in_dim=latent_dim,
            width=hidden,
            depth=depth,
            out_dim=hidden,
            scale=res_scale,
            key=k_core,
        )
        self.head = eqx.nn.Linear(hidden, 2 * r_half, key=k_head)

    def __call__(self):
        out = self.head(self.core(self.latent))  # (2 * r_half,)
        raw_alpha = out[: self.r_half]
        raw_theta = out[self.r_half :]

        alphas = -jax.nn.sigmoid(raw_alpha)  # alpha <= 0: no growing modes
        sig = jax.nn.sigmoid(raw_theta)
        thetas = self.theta_min + sig * (self.theta_max - self.theta_min)
        return alphas, thetas


class TemporalBMLP(eqx.Module):
    """Maps a learned latent vector to the mode amplitudes (b0, b).

    ``b0 = softplus(raw)`` (positive static amplitude, ~1 at init);
    ``b = softplus(raw_r) * init_mag * exp(i * pi * tanh(raw_phi))``. The head is
    zero-initialized so training starts from small, well-scaled dynamic modes.
    """

    latent: jax.Array
    core: ResidualMLP
    head: eqx.nn.Linear
    r_half: int = eqx.field(static=True)
    init_mag: float = eqx.field(static=True)

    def __init__(
        self,
        r_half: int,
        latent_dim: int = 16,
        hidden: int = 64,
        depth: int = 2,
        key=None,
        res_scale: float = 0.1,
        init_mag: float = 0.1,
    ):
        self.r_half = r_half
        self.init_mag = float(init_mag)

        k_lat, k_core, k_head = jax.random.split(key, 3)
        self.latent = jax.random.normal(k_lat, (latent_dim,))
        self.core = ResidualMLP(
            in_dim=latent_dim,
            width=hidden,
            depth=depth,
            out_dim=hidden,
            scale=res_scale,
            key=k_core,
        )
        self.head = zero_init_linear(hidden, 1 + 2 * r_half, key=k_head)

    def __call__(self):
        out = self.head(self.core(self.latent))

        b0 = jax.nn.softplus(out[0:1])
        raw = out[1:].reshape(self.r_half, 2)
        raw_r, raw_phi = raw[:, 0], raw[:, 1]
        r = jax.nn.softplus(raw_r) * self.init_mag
        phi = jnp.pi * jnp.tanh(raw_phi)
        b_half = r * jnp.exp(1j * phi)
        return b0, b_half


class NeuralDMD(eqx.Module):
    """Coordinate-network DMD model; see the module docstring for the form."""

    mlp: ResidualMLP  # spatial network
    encoding: SinusoidalEncoding
    temporal_omega: TemporalOmegaMLP
    temporal_b: TemporalBMLP
    output_size: int = eqx.field(static=True)
    r: int = eqx.field(static=True)
    num_frequencies: int = eqx.field(static=True)
    t_scale: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        r: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        key=None,
        num_frequencies: int = 2,
        temporal_latent_dim: int = 32,
        temporal_hidden: int = 128,
        temporal_layers: int = 2,
        theta_min: float = 0.0,
        theta_max: float = 1.0,
        t_scale: float = 200.0,
    ):
        self.r = r
        self.num_frequencies = num_frequencies
        self.t_scale = float(t_scale)
        self.output_size = 2 * self.r + 1  # W0 plus Re/Im of r complex modes

        keys = jax.random.split(key, 3)
        self.encoding = SinusoidalEncoding(num_frequencies=num_frequencies)
        enc_dim = 2 * (2 * num_frequencies + 1)
        self.mlp = ResidualMLP(
            in_dim=enc_dim,
            width=hidden_size,
            depth=num_layers,
            out_dim=self.output_size,
            key=keys[0],
        )
        self.temporal_omega = TemporalOmegaMLP(
            r_half=self.r,
            latent_dim=temporal_latent_dim,
            hidden=temporal_hidden,
            depth=temporal_layers,
            theta_min=theta_min,
            theta_max=theta_max,
            key=keys[1],
        )
        self.temporal_b = TemporalBMLP(
            r_half=self.r,
            latent_dim=temporal_latent_dim,
            hidden=temporal_hidden,
            depth=temporal_layers,
            key=keys[2],
        )
        self.eps = 1e-10

    def spatial_forward(self, xy: jax.Array):
        """Spatial modes at one coordinate: W0 (1,) real, W (r,) complex."""
        encoded = self.encoding(xy)
        output = self.mlp(encoded)

        W0 = jnp.expand_dims(output[0], axis=0)
        w_part = output[1 : 1 + 2 * self.r].reshape(self.r, 2)
        W = w_part[:, 0] + 1j * w_part[:, 1]
        return W0, W

    def _gauge_fix(self, W0: jax.Array, W: jax.Array):
        """Normalize each mode to unit RMS over the pixel batch.

        The scale freedom between W_k and b_k is a gauge; fixing it here (with
        stop-gradient on the norms) keeps the b amplitudes interpretable and the
        optimization well-conditioned.
        """
        eps = self.eps

        mode_norm = jnp.sqrt(jnp.mean(jnp.abs(W) ** 2, axis=0) + eps)  # (r,)
        Wn = W / jax.lax.stop_gradient(mode_norm)[None, :]

        w0_norm = jnp.sqrt(jnp.mean(jnp.abs(W0[:, 0]) ** 2) + eps)
        W0n = W0 / jax.lax.stop_gradient(w0_norm)
        return W0n, Wn

    def __call__(self, xy: jax.Array):
        """xy: (P, 2) -> (W0 (P,1), W (P,r), Omega (r,), b0 (1,), b (r,))."""
        W0, W = jax.vmap(self.spatial_forward)(xy)
        W0, W = self._gauge_fix(W0, W)

        alphas, thetas = self.temporal_omega()
        Omega = alphas + 1j * thetas
        b0, b = self.temporal_b()
        return W0, W, Omega, b0, b

    def reconstruct(
        self, xy: jax.Array, times: jax.Array, frame_max: float = 1.0, frame_min: float = 0.0
    ):
        """Evaluate the movie on coordinates xy at (normalized) times.

        Returns (intensities, static, dynamic), each (P, T), in the physical
        units defined by frame_max/frame_min.
        """
        W0, W, Omega, b0, b = self(xy)
        lambda_exp = jnp.exp(Omega[:, None] * times[None, :] * self.t_scale)
        I_stat = W0[:, 0:1] * b0[0]
        I_dyn = 2 * jnp.real(jnp.einsum("pr,rt,r->pt", W, lambda_exp, b))
        scale = frame_max - frame_min
        return (
            (I_stat + I_dyn) * scale + frame_min,
            I_stat * scale + frame_min,
            I_dyn * scale,
        )


def physical_intensities(model: NeuralDMD, xy, time_indices, frame_max, frame_min):
    """Physical-unit intensities ``(P, T)`` and modes ``(W0, W, b0, b)`` for one model.

    The same reconstruction the loss uses, factored out so the scalar and both
    polarized models share one code path.

    Returns
    -------
    intensities : jax.Array
        ``(P, T)`` = ``(W0 b0 + 2 Re[sum_k W_k b_k e^{Omega_k t}]) * scale + min``.
    modes : tuple
        ``(W0, W, b0, b)`` for the sparsity penalties.
    """
    W0, W, Omega, b0, b = model(xy)
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * model.t_scale)
    i_stat = W0[:, 0:1] * b0[0]
    i_dyn = 2 * jnp.real(jnp.einsum("pr,rt,r->pt", W, lambda_exp, b))
    intensities = (i_stat + i_dyn) * (frame_max - frame_min) + frame_min
    return intensities, (W0, W, b0, b)
