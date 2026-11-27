import jax.numpy as jnp


def batched_matmul_loop(a, b):
    return jnp.matmul(a, b)
