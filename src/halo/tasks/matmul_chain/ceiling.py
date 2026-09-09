import jax.numpy as jnp


def candidate(a, b, c):
    return jnp.matmul(a, jnp.matmul(b, c))
