import jax.numpy as jnp


def candidate(a, b, c):
    return jnp.matmul(jnp.matmul(a, b), c)
