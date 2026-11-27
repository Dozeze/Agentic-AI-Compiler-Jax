import jax
import jax.numpy as jnp


def layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + 1e-5) * scale + bias   # control
