import jax.numpy as jnp


def candidate(x, w):
    heads = w.shape[0]
    out = []
    for h in range(heads):
        out.append(jnp.matmul(x, w[h]))
    return jnp.stack(out, axis=0)
