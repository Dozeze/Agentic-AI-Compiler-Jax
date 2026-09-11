# Streaming attention on CPU: measured, and not a win

The project brief's headline example of what a compiler cannot do is the
streaming-softmax reformulation behind FlashAttention. Before adding it to the
benchmark suite as an improvable task, its ceiling was measured. On Apple M5,
CPU backend, JAX 0.11.1, single head, D=64, key block 256:

| T | score matrix | naive | streaming | ratio |
|---|---|---|---|---|
| 1024 | 4 MiB | 0.75 ms | 1.32 ms | 0.57x |
| 2048 | 16 MiB | 2.30 ms | 4.16 ms | 0.55x |
| 4096 | 64 MiB | 11.15 ms | 13.32 ms | 0.84x |
| 8192 | 256 MiB | 49.20 ms | 49.27 ms | 1.00x |

The streaming version is slower at every size and only reaches parity once the
score matrix is 256 MiB. XLA lowers the naive form to one blocked matmul, a fused
softmax and a second matmul into tuned dense kernels; a `scan` over key blocks
pays per-iteration overhead and cannot use the same blocking. The reformulation
wins on GPU because it avoids round-trips to HBM for the score matrix, and a CPU
with a large cache hierarchy does not present that cost until the matrix is huge.

Consequence for the suite: this is **not** an improvable task on CPU, and adding
it as one would have scored a correct "no change" as a failure. It is the first
ceiling to re-measure once the CUDA machines are up; on that hardware it should
flip to the largest headroom in the suite, and become the task the project is
actually about.

Both implementations agree with each other to 3e-07.

## Naive

```python
import jax.numpy as jnp


def candidate(q, k, v):
    scale = 1.0 / jnp.sqrt(jnp.float32(q.shape[-1]))
    s = jnp.matmul(q, k.T) * scale
    s = s - jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s)
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    return jnp.matmul(p, v)
```

## Streaming (FlashAttention forward pass, pure JAX)

```python
import jax
import jax.numpy as jnp

BLOCK = 256


def candidate(q, k, v):
    seq, dim = q.shape
    scale = 1.0 / jnp.sqrt(jnp.float32(dim))
    k_blocks = k.reshape(seq // BLOCK, BLOCK, dim)
    v_blocks = v.reshape(seq // BLOCK, BLOCK, dim)

    def step(carry, kv):
        m, l, acc = carry
        k_b, v_b = kv
        s = jnp.matmul(q, k_b.T) * scale
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))
        p = jnp.exp(s - m_new[:, None])
        alpha = jnp.exp(m - m_new)
        l = alpha * l + jnp.sum(p, axis=-1)
        acc = alpha[:, None] * acc + jnp.matmul(p, v_b)
        return (m_new, l, acc), None

    init = (jnp.full((seq,), -jnp.inf, jnp.float32),
            jnp.zeros((seq,), jnp.float32),
            jnp.zeros((seq, dim), jnp.float32))
    (m, l, acc), _ = jax.lax.scan(step, init, (k_blocks, v_blocks))
    return acc / l[:, None]
```
