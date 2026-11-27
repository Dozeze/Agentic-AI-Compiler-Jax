import jax.numpy as jnp


def pairwise_distances(a, b):
    a2 = jnp.sum(a * a, axis=-1)[:, None]       # ||a||^2 + ||b||^2 - 2 a.b: no (n, m, d) tensor
    b2 = jnp.sum(b * b, axis=-1)[None, :]
    return a2 + b2 - 2.0 * jnp.matmul(a, b.T)
