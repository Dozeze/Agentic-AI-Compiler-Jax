"""Affine layer normalization of each row, one row at a time (population variance, epsilon 1e-5)."""
import jax.numpy as jnp

EPS = 1e-5


def layer_norm_loop(x, scale, bias):
    out = []
    for r in range(x.shape[0]):
        row = x[r]
        mean = jnp.mean(row)
        var = jnp.mean((row - mean) ** 2)
        out.append((row - mean) / jnp.sqrt(var + EPS) * scale + bias)
    return jnp.stack(out, axis=0)
