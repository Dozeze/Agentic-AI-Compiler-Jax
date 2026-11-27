import jax.numpy as jnp


def gqa_decode(q, k, v):
    b, hq, t, d = q.shape
    hkv = k.shape[1]
    qg = q.reshape(b, hkv, (hq // hkv) * t, d)        # group query heads instead of repeating k and v
    scale = 1.0 / jnp.sqrt(jnp.float32(d))
    scores = jnp.einsum("bgqd,bgkd->bgqk", qg, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.einsum("bgqk,bgkd->bgqd", weights, v).reshape(b, hq, t, d)
