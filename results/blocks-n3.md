# The block tier: single-shot vs agentic, flash-lite vs flash, n=3

**Question.** The op-level tasks each have one well-known rewrite and both agents
find it. Do the *blocks of an AI model* — where the seed's problem is a real
anti-pattern from real code — separate the agents, and is the ceiling of the loop
the loop or the model?

**Setup.** Seven block tasks (`embedding_onehot` 189x, `top_k_argsort` 16.9x,
`gqa_decode` 6.3x, `conv_im2col` 2.3x, `cross_entropy_onehot` 2.3x, `rnn_scan_hoist`
1.45x, `mlp_gelu_block` control), ceilings measured before the tasks were added.
`algorithmic` framing in both arms; at most 3 measurements per cell (`steps=3` /
`max_evaluations=3`, `patience=3`); Apple M5 CPU, JAX 0.11.1; 3 replicates.
Two models, same everything else. Rows in `blocks-flash-lite-n3.jsonl` and
`blocks-flash-n3.jsonl`.

## Headline

| | flash-lite single-shot | flash-lite agentic | **flash** single-shot | **flash** agentic |
|---|---|---|---|---|
| improvable tasks accepted | 6/18 | 7/18 | **18/18** | 17/18 |
| mean % of ceiling | 33% | 33% | **99%** | 89% |
| false positives on control | 0/3 | 0/3 | 0/3 | 0/3 |
| measurements per cell | 3.0 | 0.6 | 2.7 | **1.1** |
| runs ended by the model's `finish` | – | 15/21 | – | 20/21 |
| cost (21 cells) | $0.035 | $0.063 | $0.808 | $0.319 |

**The model was the ceiling, not the loop.** On the same tasks, the same harness
and the same budget, `gemini-2.5-flash` reaches 99% of the available speedup where
`flash-lite` reaches 33%, with zero false positives from either. The block tier is
the first place the suite is hard enough to tell the two models apart; on the ops
tier they were indistinguishable (99–100%).

## Per task, mean % of ceiling

| task | ceiling | lite single-shot | lite agentic | flash single-shot | flash agentic |
|---|---|---|---|---|---|
| `embedding_onehot` | 189x | 33% | 33% | 100% | 99% |
| `top_k_argsort` | 16.9x | 66% | 0% | 99% | 98% |
| `gqa_decode` | 6.3x | 33% | 32% | 98% | 44% |
| `conv_im2col` | 2.3x | 0% | 31% | 95% | 92% |
| `cross_entropy_onehot` | 2.3x | 67% | 100% | 100% | 99% |
| `rnn_scan_hoist` | 1.45x | 0% | 0% | 100% | 100% |

## What separates the models

flash-lite's failures on blocks were **API errors, not idea errors**. Its rewrites
named the right move — a gather instead of a one-hot matmul, `lax.top_k`, a grouped
einsum, `conv_general_dilated` — and then failed to trace: `conv_general_dilated()`
with a `dilations` keyword, einsum labels sized 8 against 4, `take_along_axis` with
3-d indices against a 2-d array, `jax.numpy.lib`, `jax.lax.stride_tricks`. Every one
was caught by the harness and reported back; flash-lite then made the same class of
error again. flash made the same rewrites and they traced.

Its wrong-but-plausible rewrites were also caught: hoisting the bias *outside* the
tanh in the RNN (`tanh(x @ w + h @ r) + b`, worst error 0.35), an unstable
cross-entropy, a top-k via `argpartition` that measured 0.47x. None was accepted.

## What separates the agents

At block level with flash, single-shot and agentic capture the same speedup
(99% vs 89%; the gap is `gqa_decode`, below) and the agentic loop does it with
**1.1 measurements per cell against 2.7, at 40% of the cost**, because it stops. On
the control it did not evaluate at all in two runs of three: it read the seed and
called `finish` — *"the non-linear GELU separating the two matmuls leaves no
algebraic identity to exploit"* — the first zero-measurement correct rejection the
system has produced.

flash used the tools the way they were meant: `check` before `evaluate` (4 times),
`lookup` to confirm a signature (7), `inspect` twice. flash-lite in the previous
pass, given `lookup`, went browsing for a faster library function and
`cross_entropy` fell from 67% to 4% of ceiling while `conv` rose from 0% to 95%;
the protocol now frames `lookup` as verification, and *"a library function lowers
to the same XLA operations as the code you write."* A tool changes what a model
searches for, not only what it can do.

`gqa_decode` under flash agentic is the interesting miss: one run replaced
`jnp.repeat` with `jnp.take` (0.54x — the copy is still there), read that, and
finished; another found a per-kv-head Python loop at 3.5x, then tried to fold it
into a reshape and measured 0.71x, and finished on the 3.5x. The ceiling folds the
group into the row axis of one batched matmul (6.3x). Single-shot found the fold
3 times out of 3. n=3 is not enough to say this is a pattern.

## Caveats

n=3; the flash-lite numbers are the third pass over these cells, after two harness
fixes (malformed-call handling, trace failures not spending evaluations) and one
prompt change (the `lookup` framing) that the earlier passes motivated — they are a
pilot, not a clean measurement. The flash numbers are a single pass with no
changes in between. The final numbers for the report should be a fresh n=5 with
flash on the full 17-task suite. flash's cost is dominated by thinking tokens
(single-shot: 3 thinking-heavy calls per cell); the agentic loop's shorter runs are
where its cost advantage comes from.

---

## Report, flash-lite

## Effect of condition

| condition | improvable: accepted | mean % of ceiling | proposed a change | controls: false positives | evaluations/cell | cost |
|---|---|---|---|---|---|---|
| algorithmic | 6/18 | 33% | 14/18 | 0/3 | 3.0 | $0.0350 |
| agentic/algorithmic | 7/18 | 33% | 16/18 | 0/3 | 0.6 | $0.0634 |

## Per task, mean % of ceiling reached

| task | ceiling | algorithmic | agentic/algorithmic |
|---|---|---|---|
| `conv_im2col` (block) | 2.33x | 0% | 31% |
| `cross_entropy_onehot` (block) | 2.31x | 67% | 100% |
| `embedding_onehot` (block) | 189.04x | 33% | 33% |
| `gqa_decode` (block) | 6.30x | 33% | 32% |
| `rnn_scan_hoist` (block) | 1.45x | 0% | 0% |
| `top_k_argsort` (block) | 16.91x | 66% | 0% |

## Controls, times a proposal was wrongly accepted

| task | algorithmic | agentic/algorithmic |
|---|---|---|
| `mlp_gelu_block` | 0/3 | 0/3 |

## How runs ended

| condition | finished | patience | steps completed | stopped calling tools |
|---|---|---|---|---|
| algorithmic | 0 | 0 | 21 | 0 |
| agentic/algorithmic | 15 | 5 | 0 | 1 |

---

## Report, flash

## Effect of condition

| condition | improvable: accepted | mean % of ceiling | proposed a change | controls: false positives | evaluations/cell | cost |
|---|---|---|---|---|---|---|
| algorithmic | 18/18 | 99% | 18/18 | 0/3 | 2.7 | $0.8082 |
| agentic/algorithmic | 17/18 | 89% | 18/18 | 0/3 | 1.1 | $0.3189 |

## Per task, mean % of ceiling reached

| task | ceiling | algorithmic | agentic/algorithmic |
|---|---|---|---|
| `conv_im2col` (block) | 2.33x | 95% | 92% |
| `cross_entropy_onehot` (block) | 2.31x | 100% | 99% |
| `embedding_onehot` (block) | 189.04x | 100% | 99% |
| `gqa_decode` (block) | 6.30x | 98% | 44% |
| `rnn_scan_hoist` (block) | 1.45x | 100% | 100% |
| `top_k_argsort` (block) | 16.91x | 99% | 98% |

## Controls, times a proposal was wrongly accepted

| task | algorithmic | agentic/algorithmic |
|---|---|---|
| `mlp_gelu_block` | 0/3 | 0/3 |

## How runs ended

| condition | budget | finished | steps completed |
|---|---|---|---|
| algorithmic | 0 | 0 | 21 |
| agentic/algorithmic | 1 | 20 | 0 |
