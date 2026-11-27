import jax
import jax.numpy as jnp


def rnn_layer(x, w_in, w_rec, bias, h0):
    xw = x @ w_in + bias                              # hoist the input projection out of the recurrence
    def step(h, xw_t):
        h = jnp.tanh(xw_t + h @ w_rec)
        return h, h
    _, hs = jax.lax.scan(step, h0, xw)
    return hs
