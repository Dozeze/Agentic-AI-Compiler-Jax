import jax.numpy as jnp


def candidate(q, k, v):
    group = q.shape[1] // k.shape[1]
    k = jnp.repeat(k, group, axis=1)
    v = jnp.repeat(v, group, axis=1)
    scale = 1.0 / jnp.sqrt(jnp.float32(q.shape[-1]))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)
