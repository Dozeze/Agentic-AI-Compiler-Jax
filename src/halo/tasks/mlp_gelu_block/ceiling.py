import jax
import jax.numpy as jnp


def candidate(x, w1, b1, w2, b2):
    return jax.nn.gelu(x @ w1 + b1, approximate=True) @ w2 + b2
