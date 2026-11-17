"""Last-axis affine layer normalization; float32 inputs and population variance."""
import jax.numpy as jnp


def layer_norm(x, scale, bias):
    if x.dtype != jnp.float32 or scale.dtype != jnp.float32 or bias.dtype != jnp.float32:
        raise ValueError("layer_norm supports float32 inputs only")
    if x.ndim != 2 or scale.shape != (x.shape[-1],) or bias.shape != scale.shape:
        raise ValueError("expected x[rows,width], scale[width], bias[width]")
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / jnp.sqrt(variance + 1e-5)
    return normalized * scale + bias
