import jax.numpy as jnp


def candidate(logits, labels):
    m = jnp.max(logits, axis=-1, keepdims=True)
    lse = (m + jnp.log(jnp.sum(jnp.exp(logits - m), axis=-1, keepdims=True)))[:, 0]
    picked = jnp.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    return lse - picked
