# Single-shot vs agentic, n=3

**Question.** Does letting the model drive the loop through tools — deciding what to
evaluate next, whether to build on an earlier attempt, and when to stop — change what
it finds, and what it costs?

**Setup.** `gemini-2.5-flash-lite` via Vertex, `algorithmic` framing in both arms,
Apple M5 CPU, JAX 0.11.1. Every cell may take at most **3 measurements**: the
single-shot agent gets `steps=3` (three propose–measure rounds, each judged against
the current best); the agentic agent gets `max_evaluations=3`, `patience=3`, 24 tool
calls, and stops when it calls `finish`. 10 tasks x 2 conditions x 3 replicates = 60
cells, all completed. Total cost $0.068.

Rows: `agentic-vs-single-shot-n3.jsonl`. Artifacts: `runs/ablation/<task>-<condition>-r<k>/`,
each agentic run with its full `transcript.json`.

## Headline

| | single-shot | agentic |
|---|---|---|
| improvable tasks accepted | 17/18 | **18/18** |
| mean fraction of ceiling | 85% | **100%** |
| false positives on controls | 0/12 | 0/12 |
| measurements per cell | 3.0 | **1.3** |
| tool calls per cell | – | 2.4 |
| runs ended by the model's own `finish` | – | 28/30 |
| cost | $0.0363 | $0.0314 |

Same tasks, same model, same framing: the agentic loop captures more of the available
speedup with **less than half the measurements**, because it stops when it is done. On
the four controls it spent at most one evaluation, read the rejection, and finished
in 10/12 runs (twice it declined to evaluate at all); the single-shot arm measured
the noise floor three times per cell by construction.

## The agent beat the human ceiling

`multi_head_projection` entered the suite as a *deceptive null*: a Python loop over
eight heads, structurally the seed of `naive_attention` (1.65x), but the hand-written
einsum `"nd,hdk->hnk"` measured only 1.04x, so the task was classified as having no
headroom and the single-shot arm's 1.04–1.05x results were scored as false positives.

The agentic arm accepted it 3/3 at **1.26–1.29x**, with `jnp.matmul(x[None, :, :], w)`.
Re-measured directly: 1.263x [1.250, 1.280] over the seed and **1.190x [1.176, 1.202]
over the einsum ceiling** — same HLO op counts, one `dot`, but the einsum's `hnk`
index order costs an output transpose that broadcasting the left operand does not.
The ceiling is now the agent's program (`tasks/multi_head_projection/ceiling.py`),
the task is classified `headroom` at 1.26x, and the single-shot arm scores 13% of it.

A ceiling is the best implementation *known*, not a bound. This is the first time the
system produced one. The earlier result files (`ablation-pilot-n3.md`, `framing-n5.md`)
were scored with this task as a control and are left as written; re-scoring their rows
would move the single-shot 1.04x accepts from "false positive" to "13% of ceiling".

## What the transcripts show

- **Tool use is thin.** Across the 30 agentic runs: 42 `evaluate`, 28 `finish`,
  2 `inspect`, 0 `check`. The typical run is `evaluate` -> `finish`. Flash-lite acts
  on the compiler feedback in the prompt rather than pulling more. Whether a stronger
  model pulls the raw HLO, or checks before evaluating, is experiment E5.
- **One call per turn was necessary.** Before the guard, flash-lite emitted six
  `evaluate`s and a `finish` in one response on `softmax` — a plan guessed without a
  single result. 22 further calls were discarded across the sweep (recorded in each
  transcript's `discarded` field); none executed.
- **Duplicates were a real budget sink.** In the first sweep one `pairwise_distances`
  run evaluated the same trivial refactor three times and got three noise-floor
  rejections. The tool layer now returns the earlier verdict for an AST-identical
  program without measuring; that run went to 4.58x on the re-run.
- **Infrastructure faults found and fixed by the sweep:** Vertex 429 under burst
  (now retried with backoff), an empty model turn echoed back as a 400 (now dropped),
  and the seed's HLO being unavailable to `inspect` in a sweep because the cached
  baseline's artifacts were not copied.

## Caveats

n=3 and a cheap model. Every improvable task here has a single well-known rewrite,
which the single-shot agent also finds; the agentic loop's advantage on this suite is
stopping, not searching. The block- and model-level tasks (multiple wins in one
function) are where the two should separate on captured speedup, not just cost.

---

## Effect of condition

| condition | improvable: accepted | mean % of ceiling | proposed a change | controls: false positives | evaluations/cell | cost |
|---|---|---|---|---|---|---|
| algorithmic | 17/18 | 85% | 18/18 | 0/12 | 3.0 | $0.0363 |
| agentic/algorithmic | 18/18 | 100% | 18/18 | 0/12 | 1.3 | $0.0314 |

## Per task, mean % of ceiling reached

| task | ceiling | algorithmic | agentic/algorithmic |
|---|---|---|---|
| `batched_matmul_loop` | 2.63x | 98% | 98% |
| `layernorm_loop` | 4.88x | 100% | 100% |
| `matmul_chain` | 11.31x | 100% | 100% |
| `multi_head_projection` | 1.26x | 13% | 100% |
| `naive_attention` | 1.65x | 97% | 100% |
| `pairwise_distances` | 4.58x | 99% | 100% |

## Controls, times a proposal was wrongly accepted

| task | algorithmic | agentic/algorithmic |
|---|---|---|
| `gelu` | 0/3 | 0/3 |
| `matmul` | 0/3 | 0/3 |
| `rmsnorm` | 0/3 | 0/3 |
| `softmax` | 0/3 | 0/3 |

## How runs ended

| condition | budget | finished | steps completed |
|---|---|---|---|
| algorithmic | 0 | 0 | 30 |
| agentic/algorithmic | 2 | 28 | 0 |

