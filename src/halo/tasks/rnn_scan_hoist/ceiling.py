import jax
import jax.numpy as jnp


def candidate(x, w_in, w_rec, bias, h0):
    # The input projection does not depend on the recurrence: one large matmul
    # outside the loop instead of a small one per step.
    xw = x @ w_in + bias

    def step(h, xw_t):
        h = jnp.tanh(xw_t + h @ w_rec)
        return h, h

    _, hs = jax.lax.scan(step, h0, xw)
    return hs
