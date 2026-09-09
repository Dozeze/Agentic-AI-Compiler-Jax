import jax.numpy as jnp


def candidate(x):
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    weights = jnp.exp(shifted)
    return weights / jnp.sum(weights, axis=-1, keepdims=True)
