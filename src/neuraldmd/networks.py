"""Coordinate-network building blocks: a pre-LayerNorm residual MLP."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class ResBlock(eqx.Module):
    """Pre-LN residual block: ``x + scale * lin2(silu(lin1(LayerNorm(x))))``."""

    ln: eqx.nn.LayerNorm
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    scale: float = eqx.field(static=True)

    def __init__(self, width: int, scale: float = 0.1, key=None):
        k1, k2 = jax.random.split(key, 2)
        self.ln = eqx.nn.LayerNorm(width)
        self.lin1 = eqx.nn.Linear(width, width, key=k1)
        self.lin2 = eqx.nn.Linear(width, width, key=k2)
        self.scale = scale

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.ln(x)
        h = jax.nn.silu(self.lin1(h))
        h = self.lin2(h)
        return x + self.scale * h


class ResidualMLP(eqx.Module):
    """Linear-in -> ``depth`` residual blocks -> linear-out."""

    in_proj: eqx.nn.Linear
    blocks: tuple
    out_head: eqx.nn.Linear

    def __init__(
        self, in_dim: int, width: int, depth: int, out_dim: int, scale: float = 0.1, key=None
    ):
        k_in, k_out, *ks = jax.random.split(key, depth + 2)
        self.in_proj = eqx.nn.Linear(in_dim, width, key=k_in)
        self.blocks = tuple(ResBlock(width, scale=scale, key=ks[i]) for i in range(depth))
        self.out_head = eqx.nn.Linear(width, out_dim, key=k_out)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_head(h)


def zero_init_linear(in_dim: int, out_dim: int, key) -> eqx.nn.Linear:
    """A ``Linear`` layer with weight and bias initialized to zero."""
    lin = eqx.nn.Linear(in_dim, out_dim, key=key)
    return eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        lin,
        (jnp.zeros_like(lin.weight), jnp.zeros_like(lin.bias)),
    )
