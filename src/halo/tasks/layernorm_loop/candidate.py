import jax.numpy as jnp

EPS = 1e-5


def candidate(x, gamma, beta):
    rows = x.shape[0]
    out = []
    for r in range(rows):
        row = x[r]
        mean = jnp.mean(row)
        var = jnp.mean((row - mean) ** 2)
        out.append((row - mean) / jnp.sqrt(var + EPS) * gamma + beta)
    return jnp.stack(out, axis=0)
