import jax.numpy as jnp


def candidate(a, b):
    a2 = jnp.sum(a * a, axis=-1)[:, None]
    b2 = jnp.sum(b * b, axis=-1)[None, :]
    return a2 + b2 - 2.0 * jnp.matmul(a, b.T)
