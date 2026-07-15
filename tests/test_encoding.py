"""Characterize SinusoidalEncoding: exact positional-encoding values + shape."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from _impl import SinusoidalEncoding


def test_encoding_shape():
    for f in (0, 1, 2, 5, 10):
        enc = SinusoidalEncoding(num_frequencies=f)
        out = np.asarray(enc(jnp.array([0.1, 0.2])))
        assert out.shape == (2 * (2 * f + 1),)


def test_encoding_exact_values():
    enc = SinusoidalEncoding(num_frequencies=3)
    x, y = 0.3, -0.7
    out = np.asarray(enc(jnp.array([x, y])))

    ex = [x]
    ey = [y]
    for k in range(3):
        f = float(2**k)
        ex += [np.sin(f * x), np.cos(f * x)]
        ey += [np.sin(f * y), np.cos(f * y)]
    expected = np.array(ex + ey, dtype=np.float32)

    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_encoding_frequencies_are_powers_of_two():
    enc = SinusoidalEncoding()  # default 10
    assert enc.frequencies == tuple(float(2**k) for k in range(10))


def test_encoding_zero_input_is_x_then_ones():
    # at (0, 0): x/y = 0, sin(0)=0, cos(0)=1 -> [0,0,1,0,1,...] for each axis
    enc = SinusoidalEncoding(num_frequencies=2)
    out = np.asarray(enc(jnp.array([0.0, 0.0])))
    expected = np.array([0.0, 0.0, 1.0, 0.0, 1.0] * 2, dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)
