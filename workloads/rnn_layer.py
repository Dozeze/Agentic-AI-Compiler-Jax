"""Elman RNN over a sequence: h_t = tanh(x_t @ w_in + h_(t-1) @ w_rec + bias); returns every h_t."""
import jax
import jax.numpy as jnp


def rnn_layer(x, w_in, w_rec, bias, h0):
    def step(h, x_t):
        h = jnp.tanh(x_t @ w_in + h @ w_rec + bias)
        return h, h
    _, hs = jax.lax.scan(step, h0, x)
    return hs
