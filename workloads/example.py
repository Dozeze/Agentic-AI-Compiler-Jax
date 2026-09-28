"""Small starter workload for the generic source-file workflow."""

import jax.numpy as jnp


# Inputs used by the trusted reference evaluation.
TEST_ARGS = (
    jnp.arange(32, dtype=jnp.float32),
    jnp.ones(32, dtype=jnp.float32),
)


def bad_func(x, y):
    """Simple baseline function for the first agent experiment."""
    return x + y
