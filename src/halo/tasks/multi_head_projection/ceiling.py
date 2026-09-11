import jax.numpy as jnp


# Found by the tool-using agent (runs/ablation/multi_head_projection-agentic-*,
# 2026-09-11), replacing the hand-written einsum "nd,hdk->hnk" that measured
# 1.04x. Broadcasting x over the head axis lowers to the same single dot but
# without the output transpose the einsum's index order costs: 1.26x over the
# seed, 1.19x over the einsum, on Apple M5.
def candidate(x, w):
    return jnp.matmul(x[None, :, :], w)
