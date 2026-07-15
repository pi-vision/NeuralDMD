"""Fixed sinusoidal positional encoding for coordinate networks."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class SinusoidalEncoding(eqx.Module):
    """NeRF-style positional encoding of a 2D coordinate.

    Maps ``(x, y)`` to ``[x, sin(2^k x), cos(2^k x), ..., y, sin(2^k y),
    cos(2^k y), ...]`` for ``k = 0 .. num_frequencies-1``, i.e. an output of
    length ``2 * (2 * num_frequencies + 1)``.
    """

    frequencies: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, num_frequencies: int = 10):
        self.frequencies = tuple(float(2**k) for k in range(num_frequencies))

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        x, y = xy[0], xy[1]
        encoding_x = [x]
        encoding_y = [y]
        for freq in self.frequencies:
            encoding_x.append(jnp.sin(freq * x))
            encoding_x.append(jnp.cos(freq * x))
            encoding_y.append(jnp.sin(freq * y))
            encoding_y.append(jnp.cos(freq * y))
        return jnp.array(encoding_x + encoding_y)
