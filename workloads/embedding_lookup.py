"""Token embedding lookup: out[b, t] = table[ids[b, t]]."""
import jax
import jax.numpy as jnp


def embedding_lookup(ids, table):
    one_hot = jax.nn.one_hot(ids, table.shape[0], dtype=table.dtype)
    return jnp.matmul(one_hot, table)
