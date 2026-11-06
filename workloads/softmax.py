"""Row-wise softmax over the last axis. float32 in, float32 out."""
import jax.numpy as jnp


def softmax(x):
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    exps = jnp.exp(shifted)
    return exps / jnp.sum(exps, axis=-1, keepdims=True)
