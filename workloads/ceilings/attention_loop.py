import jax.numpy as jnp


def attention_loop(q, k, v):
    scale = 1.0 / jnp.sqrt(jnp.float32(q.shape[-1]))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)
