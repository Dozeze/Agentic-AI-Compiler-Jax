import jax.numpy as jnp

EPS = 1e-5


def layer_norm_loop(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + EPS) * scale + bias
