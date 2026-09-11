import jax
import jax.numpy as jnp


def candidate(x, w_in, w_rec, bias, h0):
    def step(h, x_t):
        h = jnp.tanh(x_t @ w_in + h @ w_rec + bias)
        return h, h

    _, hs = jax.lax.scan(step, h0, x)
    return hs
