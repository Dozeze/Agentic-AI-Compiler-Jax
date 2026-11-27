"""A batch of independent matrix products, one product at a time."""
import jax.numpy as jnp


def batched_matmul_loop(a, b):
    out = []
    for i in range(a.shape[0]):
        out.append(jnp.matmul(a[i], b[i]))
    return jnp.stack(out, axis=0)
