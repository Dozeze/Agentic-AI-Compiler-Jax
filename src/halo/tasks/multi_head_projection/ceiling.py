import jax.numpy as jnp


def candidate(x, w):
    return jnp.einsum("nd,hdk->hnk", x, w)
