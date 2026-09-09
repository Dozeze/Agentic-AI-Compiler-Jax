import jax.numpy as jnp

COEFF = 0.044715
SQRT_2_OVER_PI = 0.7978845608028654


def candidate(x):
    inner = SQRT_2_OVER_PI * (x + COEFF * x ** 3)
    return 0.5 * x * (1.0 + jnp.tanh(inner))
