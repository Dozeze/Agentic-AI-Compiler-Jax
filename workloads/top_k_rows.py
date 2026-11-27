"""The 8 largest values of each row, sorted from largest to smallest."""
import jax.numpy as jnp

K = 8


def top_k_rows(x):
    order = jnp.argsort(-x, axis=-1)
    return jnp.take_along_axis(x, order[:, :K], axis=-1)
