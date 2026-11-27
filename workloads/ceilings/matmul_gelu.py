import jax
import jax.numpy as jnp


def matmul_gelu(x, weight, bias):
    projected = jnp.dot(x, weight, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return jax.nn.gelu(projected + bias, approximate=False)                # control
