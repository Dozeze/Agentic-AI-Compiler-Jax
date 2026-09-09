import jax.numpy as jnp


def candidate(q, k, v):
    dim = q.shape[-1]
    scale = 1.0 / jnp.sqrt(jnp.float32(dim))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)
