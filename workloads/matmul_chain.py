"""Product of three float32 matrices, evaluated left to right."""
import jax.numpy as jnp


def matmul_chain(a, b, c):
    return jnp.matmul(jnp.matmul(a, b), c)
