import jax.numpy as jnp


def matmul_chain(a, b, c):
    return jnp.matmul(a, jnp.matmul(b, c))      # reassociation: (big x small) @ (small x small)
