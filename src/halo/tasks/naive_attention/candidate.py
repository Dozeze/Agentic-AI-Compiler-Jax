import jax.numpy as jnp


def candidate(q, k, v):
    batch, heads, seq, dim = q.shape
    scale = 1.0 / jnp.sqrt(jnp.float32(dim))
    heads_out = []
    for h in range(heads):
        qh = q[:, h, :, :]
        kh = k[:, h, :, :]
        vh = v[:, h, :, :]
        scores = jnp.matmul(qh, jnp.transpose(kh, (0, 2, 1))) * scale
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        weights = jnp.exp(scores)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        heads_out.append(jnp.matmul(weights, vh))
    return jnp.stack(heads_out, axis=1)
