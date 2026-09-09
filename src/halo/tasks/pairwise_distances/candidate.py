import jax.numpy as jnp


def candidate(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)
