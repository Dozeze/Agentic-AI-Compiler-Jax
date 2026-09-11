import jax.numpy as jnp

K = 8


def candidate(x):
    order = jnp.argsort(-x, axis=-1)
    return jnp.take_along_axis(x, order[:, :K], axis=-1)
