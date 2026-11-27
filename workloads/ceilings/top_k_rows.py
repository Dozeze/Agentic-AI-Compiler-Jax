import jax

K = 8


def top_k_rows(x):
    values, _ = jax.lax.top_k(x, K)                   # partial selection instead of a full sort
    return values
