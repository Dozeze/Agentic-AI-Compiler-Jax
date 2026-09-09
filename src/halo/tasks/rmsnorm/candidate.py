import jax.numpy as jnp

EPS = 1e-6


def candidate(x, w):
    scale = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + EPS)
    return x / scale * w
