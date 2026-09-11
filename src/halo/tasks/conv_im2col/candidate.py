import jax.numpy as jnp


def candidate(x, w):
    n, h, wd, c = x.shape
    x_pad = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    patches = jnp.stack(
        [x_pad[:, di:di + h, dj:dj + wd, :] for di in range(3) for dj in range(3)],
        axis=3,
    )  # (n, h, w, 9, c)
    cols = patches.reshape(n * h * wd, 9 * c)
    return (cols @ w.reshape(9 * c, -1)).reshape(n, h, wd, -1)
