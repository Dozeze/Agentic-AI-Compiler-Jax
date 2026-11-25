"""Row-tiled Pallas softmax over the last axis (G-11 prototype). Same contract as workloads/softmax.py.

Each program instance owns a (block_rows, width) tile, so one row is reduced entirely in registers:
max, exp and sum stay in one pass over on-chip data instead of separate XLA reductions. Triton needs
power-of-two block dimensions, so the wrapper pads the width with -inf (exp(-inf) = 0 leaves every
row's softmax unchanged) and the rows with zeros, then slices the padding away.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

TILE_ELEMENTS = 8192    # elements per program instance; a tuning knob, not part of the contract


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def _kernel(x_ref, o_ref):
    x = x_ref[...]
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)


@functools.partial(jax.jit, static_argnames=("tile_elements",))
def softmax(x, tile_elements=TILE_ELEMENTS):
    shape = x.shape
    width = shape[-1]
    x2 = x.reshape(-1, width)
    rows = x2.shape[0]
    padded_width = _next_power_of_two(width)
    block_rows = max(1, min(_next_power_of_two(rows), tile_elements // padded_width))
    padded_rows = -(-rows // block_rows) * block_rows
    if padded_width != width:
        x2 = jnp.pad(x2, ((0, 0), (0, padded_width - width)), constant_values=-jnp.inf)
    if padded_rows != rows:
        x2 = jnp.pad(x2, ((0, padded_rows - rows), (0, 0)))
    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
        grid=(padded_rows // block_rows,),
        in_specs=[pl.BlockSpec(block_shape=(block_rows, padded_width), index_map=lambda i: (i, 0))],
        out_specs=pl.BlockSpec(block_shape=(block_rows, padded_width), index_map=lambda i: (i, 0)),
        interpret=jax.default_backend() == "cpu",
    )(x2)
    return out[:rows, :width].reshape(shape)
