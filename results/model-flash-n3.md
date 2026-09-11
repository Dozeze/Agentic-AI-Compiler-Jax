# The model tier: `tiny_gpt_loss` with gemini-2.5-flash, n=3

**Question.** A whole model forward pass and loss with three anti-patterns composed
in one function — does the agent find all of them, and does the agentic loop's
step-by-step verification matter when a single rewrite could do everything?

**Setup.** `tiny_gpt_loss` (2-layer GPT, GQA, tied head, per-token cross-entropy;
ceiling 1.93x, decomposing as gather 1.51x, direct cross-entropy 1.16x, batched
attention ~1.06x). `gemini-2.5-flash`, `algorithmic` framing, 4 measurements per
cell (`steps=4` / `max_evaluations=4`, `patience=3`), 3 replicates each arm.
Rows: `model-flash-n3.jsonl`; transcripts under `runs/flash/ablation/tiny_gpt_loss-*`.

## Result

| | single-shot | agentic |
|---|---|---|
| accepted | 3/3 | 3/3 |
| speedup over the seed | 1.94x, 1.82x, 1.79x | 1.85x, 1.95x, 1.82x |
| mean % of ceiling | 91% | 93% |
| measurements per cell | 3.0 (+1 trace failure) | 2.7 |
| cost per cell | $0.11 | $0.05 |

Both arms reach the ceiling. flash does **the full model rewrite in one proposal**
when asked for one — the single-shot arm's accepted attempts each bundle all three
fixes — so the model tier did not need the loop to *find* the speedup. What the
loop changed is how it got there and what it cost:

- **Agentic r0** took the three fixes as three verified steps: gather 1.47x, then
  batched attention x1.10 on top, then direct cross-entropy x1.09 on top — each
  accepted against the previous best, each with its own interval — and finished.
- **Agentic r1** tried to bundle attention and a library cross-entropy in one step
  and failed to trace; retried with attention alone (accepted); tried the direct
  cross-entropy and **failed correctness** on the `labels_are_ids` case; fixed the
  logsumexp and passed. A wrong loss caught and repaired inside one run, for $0.07.
- **Agentic r2** did everything in one evaluation (1.82x) and finished.
- **Single-shot** spent its remaining steps after the big rewrite on fusions —
  `jax.nn.softmax` variants (0.90x), a fused QKV projection (1.046x, lower bound
  1.031: rejected as noise) — and on one run returned the code unchanged, at
  $0.03 a step.

## The ceiling was raised again

The hand-written ceiling used a grouped 5-d einsum for the attention
(`bgrqd,bgkd->bgrqk`), measured 1.83x. The single-shot arm's best rewrite keeps
`jnp.repeat` on K/V and does plain `@` over `(b, heads, t, hd)`: **1.058x
[1.051, 1.064] over the einsum ceiling** (the grouped layout costs transposes that
the repeat does not, at this size). The agentic arm's best measures the same,
1.057x. `ceiling.py` is now the agent's attention and the task's headroom is 1.93x.
This is the second task whose ceiling the system raised (`multi_head_projection`
was the first).

## What it says for the report

On the model tier with a capable model, the agentic loop's advantage is
*verification and cost*, not discovery: it delivers the same rewrite as a chain of
individually measured, individually accepted steps, catches its own wrong loss on
the way, and spends half as much. The harness's gates did their job in both arms:
one trace failure, one correctness failure, three noise-floor rejections, zero
false accepts. n=3.
