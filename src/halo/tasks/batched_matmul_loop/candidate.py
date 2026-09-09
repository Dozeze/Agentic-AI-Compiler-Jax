import jax.numpy as jnp


def candidate(a, b):
    batch = a.shape[0]
    out = []
    for i in range(batch):
        out.append(jnp.matmul(a[i], b[i]))
    return jnp.stack(out, axis=0)
