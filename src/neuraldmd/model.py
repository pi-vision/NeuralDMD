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
        """Build the spectrum network.

        Parameters
        ----------
        r_half : int
            Number of complex modes.
        latent_dim, hidden, depth : int
            Latent width, hidden width, and number of residual blocks.
        theta_min, theta_max : float
            Bounds on the mode frequency (before ``t_scale``).
        key : jax.Array
            PRNG key.
        res_scale : float
            Damping on each residual branch.
        """
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
        """Decode the latent into decay rates and frequencies.

        Returns
        -------
        alphas : jax.Array
            ``(r_half,)`` decay rates in ``[-1, 0]``.
        thetas : jax.Array
            ``(r_half,)`` frequencies in ``[theta_min, theta_max]``.
        """
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
        """Build the amplitude network.

        Parameters
        ----------
        r_half : int
            Number of complex modes.
        latent_dim, hidden, depth : int
            Latent width, hidden width, and number of residual blocks.
        key : jax.Array
            PRNG key.
        res_scale : float
            Damping on each residual branch.
        init_mag : float
            Scale of the dynamic amplitudes at initialization.
        """
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
        """Decode the latent into the static and dynamic mode amplitudes.

        Returns
        -------
        b0 : jax.Array
            ``(1,)`` positive static amplitude.
        b_half : jax.Array
            ``(r_half,)`` complex dynamic amplitudes.
        """
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
    dyn_cap: float | None = eqx.field(static=True)

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
        dyn_cap: float | None = None,
    ):
        """Build the spatial network and the two temporal networks.

        Parameters
        ----------
        r : int
            Number of complex dynamic modes.
        hidden_size, num_layers : int
            Width and depth of the spatial network.
        key : jax.Array
            PRNG key, split across the three sub-networks.
        num_frequencies : int
            Octaves of sinusoidal coordinate encoding; low values band-limit the
            image spatially.
        temporal_latent_dim, temporal_hidden, temporal_layers : int
            Sizes of the spectrum and amplitude networks.
        theta_min, theta_max : float
            Bounds on the mode frequency (before ``t_scale``).
        t_scale : float
            Multiplier on ``Omega * t``, so normalized times span a useful range
            of oscillation rates.
        """
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
        self.dyn_cap = None if dyn_cap is None else float(dyn_cap)

    def spatial_features(self, xy: jax.Array) -> jax.Array:
        """Trunk activations at one coordinate, before the mode head.

        Parameters
        ----------
        xy : jax.Array
            ``(2,)`` coordinate.

        Returns
        -------
        jax.Array
            ``(hidden_size,)`` pre-head activations.
        """
        return self.mlp.features(self.encoding(xy))

    def _spatial_from_features(self, feats: jax.Array):
        """Split the mode head's output into ``W0`` and the complex ``W``.

        Parameters
        ----------
        feats : jax.Array
            ``(hidden_size,)`` trunk activations.

        Returns
        -------
        W0 : jax.Array
            ``(1,)`` real static mode.
        W : jax.Array
            ``(r,)`` complex dynamic modes.
        """
        output = self.mlp.out_head(feats)

        W0 = jnp.expand_dims(output[0], axis=0)
        w_part = output[1 : 1 + 2 * self.r].reshape(self.r, 2)
        W = w_part[:, 0] + 1j * w_part[:, 1]
        return W0, W

    def spatial_forward(self, xy: jax.Array):
        """Spatial modes at one coordinate: W0 (1,) real, W (r,) complex.

        Parameters
        ----------
        xy : jax.Array
            ``(2,)`` coordinate.

        Returns
        -------
        W0 : jax.Array
            ``(1,)`` real static mode.
        W : jax.Array
            ``(r,)`` complex dynamic modes.
        """
        return self._spatial_from_features(self.spatial_features(xy))

    def _gauge_fix(self, W0: jax.Array, W: jax.Array):
        """Normalize each mode to unit RMS over the pixel batch.

        The scale freedom between W_k and b_k is a gauge; fixing it here (with
        stop-gradient on the norms) keeps the b amplitudes interpretable and the
        optimization well-conditioned.

        Parameters
        ----------
        W0 : jax.Array
            ``(P, 1)`` static mode.
        W : jax.Array
            ``(P, r)`` complex dynamic modes.

        Returns
        -------
        W0n, Wn
            The same arrays, each mode scaled to unit RMS over the pixels.
        """
        eps = self.eps

        mode_norm = jnp.sqrt(jnp.mean(jnp.abs(W) ** 2, axis=0) + eps)  # (r,)
        Wn = W / jax.lax.stop_gradient(mode_norm)[None, :]

        w0_norm = jnp.sqrt(jnp.mean(jnp.abs(W0[:, 0]) ** 2) + eps)
        W0n = W0 / jax.lax.stop_gradient(w0_norm)
        return W0n, Wn

    def __call__(
        self,
        xy: jax.Array,
        *,
        omega: jax.Array | None = None,
        spatial_features: jax.Array | None = None,
        b_mask: jax.Array | None = None,
    ):
        """Evaluate the model's modes, optionally from externally supplied pieces.

        The three keyword arguments let a container drive this field from shared
        state -- a spectrum shared across Stokes, a shared spatial trunk, or a
        per-mode on/off mask. All default to ``None``, which reproduces the
        field's own standalone behaviour exactly.

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates. Ignored when ``spatial_features`` is given.
        omega : jax.Array or None
            ``(r,)`` complex spectrum to use instead of this field's own.
        spatial_features : jax.Array or None
            ``(P, hidden_size)`` trunk activations to use instead of this field's
            own trunk. The field's head is still applied.
        b_mask : jax.Array or None
            ``(>=r,)`` multiplier on the dynamic amplitudes; a zero entry removes
            that mode's contribution and its gradient path.

        Returns
        -------
        W0, W, Omega, b0, b
            ``(P, 1)`` real static modes, ``(P, r)`` complex dynamic modes,
            ``(r,)`` complex spectrum, ``(1,)`` static amplitude, and ``(r,)``
            complex dynamic amplitudes.
        """
        if spatial_features is None:
            W0, W = jax.vmap(self.spatial_forward)(xy)
        else:
            W0, W = jax.vmap(self._spatial_from_features)(spatial_features)
        W0, W = self._gauge_fix(W0, W)

        if omega is None:
            alphas, thetas = self.temporal_omega()
            Omega = alphas + 1j * thetas
        else:
            Omega = omega
        b0, b = self.temporal_b()
        if b_mask is not None:
            b = b * b_mask[: self.r]
        if self.dyn_cap is not None:
            # Hard cap on dynamic power relative to the static mode. Bounding the
            # spectrum (theta_max) limits only WHERE variability sits: the fit
            # answers a frequency bound by inflating amplitudes instead. This
            # bounds how MUCH there is. Modes are unit-RMS gauge-fixed, so
            # sqrt(sum |b_j|^2) / b_0 is the dynamic-to-static amplitude ratio.
            power = jnp.sqrt(jnp.sum(jnp.abs(b) ** 2) + self.eps)
            limit = self.dyn_cap * jnp.abs(b0[0])
            b = b * jnp.minimum(1.0, limit / power)
        return W0, W, Omega, b0, b

    def reconstruct(
        self, xy: jax.Array, times: jax.Array, frame_max: float = 1.0, frame_min: float = 0.0
    ):
        """Evaluate the movie on coordinates xy at (normalized) times.

        Parameters
        ----------
        xy : jax.Array
            ``(P, 2)`` pixel coordinates.
        times : jax.Array
            ``(T,)`` normalized frame times.
        frame_max, frame_min : float
            Output scaling to physical units.

        Returns
        -------
        intensities, static, dynamic
            Each ``(P, T)``, in the physical units set by frame_max/frame_min.
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


def physical_intensities(
    model: NeuralDMD,
    xy,
    time_indices,
    frame_max,
    frame_min,
    *,
    omega=None,
    spatial_features=None,
    b_mask=None,
):
    """Physical-unit intensities ``(P, T)`` and modes ``(W0, W, b0, b)`` for one model.

    The same reconstruction the loss uses, factored out so the scalar and both
    polarized models share one code path.

    Parameters
    ----------
    model : NeuralDMD
        Field to evaluate.
    xy : jax.Array
        ``(P, 2)`` pixel coordinates.
    time_indices : jax.Array
        ``(T,)`` normalized frame times.
    frame_max, frame_min : float
        Output scaling to physical units.
    omega, spatial_features, b_mask : jax.Array or None
        Shared spectrum, shared trunk activations, and per-mode mask; passed
        through to :meth:`NeuralDMD.__call__`.

    Returns
    -------
    intensities : jax.Array
        ``(P, T)`` = ``(W0 b0 + 2 Re[sum_k W_k b_k e^{Omega_k t}]) * scale + min``.
    modes : tuple
        ``(W0, W, b0, b)`` for the sparsity penalties, with ``b`` already masked.
    """
    W0, W, Omega, b0, b = model(xy, omega=omega, spatial_features=spatial_features, b_mask=b_mask)
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * model.t_scale)
    i_stat = W0[:, 0:1] * b0[0]
    i_dyn = 2 * jnp.real(jnp.einsum("pr,rt,r->pt", W, lambda_exp, b))
    intensities = (i_stat + i_dyn) * (frame_max - frame_min) + frame_min
    return intensities, (W0, W, b0, b)
