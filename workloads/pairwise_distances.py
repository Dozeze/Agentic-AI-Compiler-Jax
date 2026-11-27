"""Squared euclidean distance between every pair of rows of a and b."""
import jax.numpy as jnp


def pairwise_distances(a, b):
    diff = a[:, None, :] - b[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)
