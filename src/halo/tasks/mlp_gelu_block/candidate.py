import jax.numpy as jnp

COEFF = 0.044715
SQRT_2_OVER_PI = 0.7978845608028654


def candidate(x, w1, b1, w2, b2):
    u = x @ w1 + b1
    g = 0.5 * u * (1.0 + jnp.tanh(SQRT_2_OVER_PI * (u + COEFF * u ** 3)))
    return g @ w2 + b2
