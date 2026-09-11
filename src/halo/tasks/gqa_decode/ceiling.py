import jax.numpy as jnp


def candidate(q, k, v):
    b, hq, t, d = q.shape
    hkv = k.shape[1]
    # Fold the query heads that share a kv head into the row axis: one batched
    # matmul over (b, hkv) with (group * t) rows, and k and v read once.
    qg = q.reshape(b, hkv, (hq // hkv) * t, d)
    scale = 1.0 / jnp.sqrt(jnp.float32(d))
    scores = jnp.einsum("bgqd,bgkd->bgqk", qg, k) * scale
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = jnp.einsum("bgqk,bgkd->bgqd", weights, v)
    return out.reshape(b, hq, t, d)
