## Effect of context level

| context | improvable: accepted | mean % of ceiling | proposed a change | controls: false positives | cost |
|---|---|---|---|---|---|
| code | 15/15 | 99% | 15/15 | 1/15 | $0.0089 |
| timing | 11/15 | 72% | 11/15 | 1/15 | $0.0087 |
| hlo | 11/15 | 72% | 12/15 | 1/15 | $0.0093 |
| full | 12/15 | 79% | 12/15 | 0/15 | $0.0099 |

## Per task, mean % of ceiling reached

| task | ceiling | code | timing | hlo | full |
|---|---|---|---|---|---|
| `batched_matmul_loop` | 2.63x | 97% | 96% | 97% | 96% |
| `layernorm_loop` | 4.88x | 97% | 99% | 98% | 98% |
| `matmul_chain` | 11.31x | 100% | 67% | 100% | 100% |
| `naive_attention` | 1.65x | 100% | 100% | 67% | 100% |
| `pairwise_distances` | 4.58x | 100% | 0% | 0% | 0% |

## Controls, times a proposal was wrongly accepted

| task | code | timing | hlo | full |
|---|---|---|---|---|
| `gelu` | 0/3 | 0/3 | 0/3 | 0/3 |
| `matmul` | 0/3 | 0/3 | 0/3 | 0/3 |
| `multi_head_projection` | 1/3 | 1/3 | 1/3 | 0/3 |
| `rmsnorm` | 0/3 | 0/3 | 0/3 | 0/3 |
| `softmax` | 0/3 | 0/3 | 0/3 | 0/3 |
