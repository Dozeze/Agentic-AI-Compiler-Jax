import jax
import jax.numpy as jnp


def candidate(ids, table):
    vocab = table.shape[0]
    one_hot = jax.nn.one_hot(ids, vocab, dtype=table.dtype)
    return one_hot @ table
