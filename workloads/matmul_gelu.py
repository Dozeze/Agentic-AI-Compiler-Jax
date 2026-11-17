"""Float32 dense projection, column bias, then exact erfc GELU."""
import jax
import jax.numpy as jnp


def matmul_gelu(x, weight, bias):
    if x.dtype != jnp.float32 or weight.dtype != jnp.float32 or bias.dtype != jnp.float32:
        raise ValueError("matmul_gelu supports float32 inputs only")
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]
            or bias.shape != (weight.shape[-1],)):
        raise ValueError("expected x[M,K], weight[K,N], bias[N]")
    projected = jnp.matmul(x, weight, precision=jax.lax.Precision.HIGHEST,
                           preferred_element_type=jnp.float32)
    return jax.nn.gelu(projected + bias, approximate=False)
