"""Row-tiled Pallas affine layer norm (G-11 prototype). Same contract as workloads/layer_norm.py.

One program instance normalizes a (block_rows, width) tile: mean, two-pass population variance,
normalization and the affine step all run on the same on-chip tile. The width is padded with zeros
to a power of two (Triton block rule); the kernel divides by the true width and masks the padded
columns out of the variance, so the result equals the unpadded computation.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

EPSILON = 1e-5
TILE_ELEMENTS = 8192    # elements per program instance; a tuning knob, not part of the contract


def _next_power_of_two(n):
    return 1 << (n - 1).bit_length()


def _kernel(x_ref, scale_ref, bias_ref, o_ref, *, width):
    x = x_ref[...]
    mean = jnp.sum(x, axis=-1, keepdims=True) / width
    valid = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1) < width
    centered = jnp.where(valid, x - mean, 0.0)
    variance = jnp.sum(centered * centered, axis=-1, keepdims=True) / width
    o_ref[...] = centered / jnp.sqrt(variance + EPSILON) * scale_ref[...] + bias_ref[...]


@functools.partial(jax.jit, static_argnames=("tile_elements",))
def layer_norm(x, scale, bias, tile_elements=TILE_ELEMENTS):
    if x.dtype != jnp.float32 or scale.dtype != jnp.float32 or bias.dtype != jnp.float32:
        raise ValueError("layer_norm supports float32 inputs only")
    if x.ndim != 2 or scale.shape != (x.shape[-1],) or bias.shape != scale.shape:
        raise ValueError("expected x[rows,width], scale[width], bias[width]")
    rows, width = x.shape
    padded_width = _next_power_of_two(width)
    block_rows = max(1, min(_next_power_of_two(rows), tile_elements // padded_width))
    padded_rows = -(-rows // block_rows) * block_rows
    pad = padded_width - width
    x2 = jnp.pad(x, ((0, padded_rows - rows), (0, pad)))
    scale2 = jnp.pad(scale, (0, pad)).reshape(1, padded_width)
    bias2 = jnp.pad(bias, (0, pad)).reshape(1, padded_width)
    row_tile = pl.BlockSpec(block_shape=(block_rows, padded_width), index_map=lambda i: (i, 0))
    vector = pl.BlockSpec(block_shape=(1, padded_width), index_map=lambda i: (0, 0))
    out = pl.pallas_call(
        functools.partial(_kernel, width=width),
        out_shape=jax.ShapeDtypeStruct(x2.shape, x2.dtype),
        grid=(padded_rows // block_rows,),
        in_specs=[row_tile, vector, vector],
        out_specs=row_tile,
        interpret=jax.default_backend() == "cpu",
    )(x2, scale2, bias2)
    return out[:rows, :width]
