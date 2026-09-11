import jax

K = 8


def candidate(x):
    values, _ = jax.lax.top_k(x, K)
    return values
