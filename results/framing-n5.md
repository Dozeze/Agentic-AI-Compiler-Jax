## Effect of context level

| context | improvable: accepted | mean % of ceiling | proposed a change | controls: false positives | cost |
|---|---|---|---|---|---|
| full | 16/25 | 64% | 18/25 | 0/25 | $0.0170 |
| algorithmic | 24/25 | 95% | 25/25 | 0/25 | $0.0217 |

## Per task, mean % of ceiling reached

| task | ceiling | full | algorithmic |
|---|---|---|---|
| `batched_matmul_loop` | 2.63x | 99% | 95% |
| `layernorm_loop` | 4.88x | 100% | 100% |
| `matmul_chain` | 11.31x | 20% | 100% |
| `naive_attention` | 1.65x | 80% | 100% |
| `pairwise_distances` | 4.58x | 20% | 79% |

## Controls, times a proposal was wrongly accepted

| task | full | algorithmic |
|---|---|---|
| `gelu` | 0/5 | 0/5 |
| `matmul` | 0/5 | 0/5 |
| `multi_head_projection` | 0/5 | 0/5 |
| `rmsnorm` | 0/5 | 0/5 |
| `softmax` | 0/5 | 0/5 |
