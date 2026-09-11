# HALO — HLO-Aware Lowering Optimizer

An agentic optimization loop around JAX/XLA. Given a JAX function, HALO measures it,
shows a language model the code plus the compiler's own feedback, takes the model's
rewrite, measures it again, and accepts it only if it is both **correct** and
**measurably faster**.

KTH DD2430 Group 19 · Project 1 (Ericsson, *Agentic AI Compiler*).

Two kinds of agent drive the same loop. The **single-shot** agent answers once per
step and the controller measures it. The **agentic** agent drives the loop itself
through tools — evaluate, check, inspect, finish — deciding what to try next, whether
to build on an earlier attempt, and when to stop. In both cases the harness, not the
model, decides what is accepted.

## Status

Verified end to end on Apple M5 (CPU backend), `gemini-2.5-flash-lite` via Vertex AI:

| task | outcome | speedup (95% CI) | cost |
|---|---|---|---|
| `naive_attention` | accepted | **1.726x** [1.691, 1.761] | $0.000508 |
| `softmax` | rejected (performance) | 1.018x [0.996, 1.042] | $0.000276 |

The softmax result is the expected one and is not a failure. XLA already fuses the
standard formulation; the model correctly identified the kernel as memory-bound and
proposed no change, and the harness declined to call a 1.8% difference a speedup.

## The benchmark suite

Seventeen tasks in two tiers — single **operations** and the **blocks** an AI model is
built from — with real headroom or as controls, so an ablation can measure both
whether the agent finds speedups *and* whether it invents them. Headroom is the
measured speedup of the best known implementation (`ceiling.py`) over the seed, on
Apple M5 CPU / JAX 0.11.1 — re-measure with `halo ceilings`. A task is only added
after its ceiling is measured; below ~1.3x it becomes a control.

| task | tier | class | headroom | the seed's problem |
|---|---|---|---|---|
| `matmul_chain` | op | headroom | 11.3x | `(a@b)@c` materialises 1024x1024; `a@(b@c)` needs 16x16 |
| `layernorm_loop` | op | headroom | 4.9x | Python loop over rows |
| `pairwise_distances` | op | headroom | 4.6x | materialises an (N, M, D) difference tensor |
| `batched_matmul_loop` | op | headroom | 2.6x | Python loop over the batch |
| `naive_attention` | op | headroom | 1.65x | Python loop over heads |
| `multi_head_projection` | op | headroom | 1.26x | Python loop over heads; the ceiling is the agent's own find |
| `rmsnorm` | op | null | 1.01x | already vectorised |
| `softmax` | op | null | 1.01x | XLA already fuses it |
| `gelu` | op | null | 1.00x | pure elementwise chain |
| `matmul` | op | null | 0.99x | a single dot into a tuned kernel |
| `embedding_onehot` | block | headroom | 189x | `one_hot(ids) @ table`: a (1024, 8192) matmul for a gather |
| `top_k_argsort` | block | headroom | 16.9x | a full sort of 16k columns to take 8 |
| `gqa_decode` | block | headroom | 6.3x | `jnp.repeat` of the KV cache to match the query heads |
| `conv_im2col` | block | headroom | 2.3x | explicit (N, H, W, 9, C) patches, then a matmul |
| `cross_entropy_onehot` | block | headroom | 2.3x | `one_hot(labels) * log_softmax` over the vocabulary |
| `rnn_scan_hoist` | block | headroom | 1.45x | the input projection recomputed inside `lax.scan` |
| `mlp_gelu_block` | block | null | 1.00x | matmul, GELU, matmul: XLA already fuses it |

Two block tasks were reformulated by their ceilings. Grouped-query attention over a
256-token *prefill* measured only 1.10x — repeating K/V costs O(T·D) against O(T²·D)
of scores, invisible at that length — so the task is the *decode* step (one query
against a 2048-entry cache), where the repeat is the same order as the work and the
ceiling is 6.3x. And `mlp_gelu_block` is a control on purpose: a block that looks
like it should have room but XLA already handles.

`multi_head_projection` is the one worth understanding. It is a Python loop over
eight heads — structurally the same shape as the `naive_attention` seed — and the
hand-written einsum ceiling measured only 1.04x, so it was classified as a control:
a task that *looks* improvable but is not. Then the tool-using agent found
`jnp.matmul(x[None], w)` at **1.26x over the seed and 1.19x over the einsum**, three
runs out of three (same HLO op counts; the einsum's index order costs an output
transpose). The ceiling is now the agent's program. A ceiling is the best
implementation *known*, not a bound, and the suite is scored against it knowing that.

Classification lives in `tasks/ceilings.py`, never in `task.py`, because `task.py`
goes into the prompt.

One task is deliberately absent. The streaming-softmax attention behind
FlashAttention — the brief's headline example of what a compiler cannot do — was
measured before being added and is **slower than the naive form on this CPU at every
size**, reaching parity only at a 256 MiB score matrix
(`results/streaming-attention-cpu.md`). It is a GPU optimisation; on CPU it would
have been a control mislabelled as improvable. It is the first ceiling to re-measure
on the CUDA machines.

## The ablation

The experiment the suite exists for: hold the model, the tasks and the acceptance
policy fixed, vary only what the agent is told, and see what changes.

```bash
uv run halo ablate --replicates 5
```

Four levels vary the information, each a superset of the last: `code` (the function
and its semantics only), `timing` (adds runtimes), `hlo` (adds what XLA produced),
`full` (adds the jaxpr and StableHLO levels and the buffer report). A fifth,
`algorithmic`, shows **exactly what `full` shows** and changes only how the job is
framed, so comparing those two isolates the prompt from the data.

Improvable tasks are scored as the fraction of their known ceiling reached; controls
by how often something was wrongly accepted. Rows land in JSONL and sweeps are
resumable.

### What it found

The first pilot (`results/ablation-pilot-n3.md`) showed compiler feedback *hurting*:
on `pairwise_distances` the agent proposed a rewrite 3/3 times with no compiler
feedback and 0/9 times with it. Its own words explain why — it named the right answer
and then dismissed it:

> "The current implementation is already very close to optimal... XLA has fused them
> effectively. Further optimization would likely involve a different approach, such as
> using matrix multiplication properties. However, given the current HLO and
> performance, significant gains are unlikely."

Using matrix multiplication properties is exactly the 4.58x rewrite. Shown that XLA
had fused its code well, the model reasoned like the compiler and stopped looking for
the algorithmic change the compiler also cannot find.

The prompt was at fault, and it was ours. It told the agent that few fusions meant
"expect a smaller win", and every hint it offered was a help-the-compiler move. The
`algorithmic` framing separates the two questions instead — *was this algorithm
lowered well* (the compiler's job, which fusion counts answer) versus *is there a
different algorithm* (the agent's job, which nothing in the compiler's output answers)
— and lists the moves no rule-based compiler can make: algebraic identities,
reassociation, streaming reformulation, exploiting structure.

Same data, same tasks, same model. The 10 op-level tasks x 5 replicates (`results/framing-n5.md`):

| context | improvable: accepted | % of ceiling | proposed a change | false positives |
|---|---|---|---|---|
| full | 16/25 | 64% | 18/25 | 0/25 |
| **algorithmic** | **24/25** | **95%** | **25/25** | **0/25** |

| task | ceiling | full | algorithmic |
|---|---|---|---|
| `matmul_chain` | 11.31x | 20% | **100%** |
| `pairwise_distances` | 4.58x | 20% | **79%** |
| `naive_attention` | 1.65x | 80% | **100%** |
| `layernorm_loop` | 4.88x | 100% | 100% |
| `batched_matmul_loop` | 2.63x | 99% | 95% |

The gain is concentrated exactly where the thesis predicts: the two tasks needing an
*algebraic* change, not a mechanical one. On `matmul_chain` the agent under `full`
returned the code unchanged 4 times out of 5; under `algorithmic` it found the
reassociation 5 times out of 5.

Two things this did not cost. **False positives stayed at zero** — the more assertive
framing did not make the agent start rewriting the five already-optimal controls. And
it is more *consistent*: under `full` whether the agent acts at all sits near a
decision boundary and flips between replicates, which is why `matmul_chain` scored
100% in the n=3 pilot and 20% here.

It did produce two candidates that failed to compile out of 50 (`a @ b` where
`a @ b.T` was meant; `jnp.rsqrt`, which does not exist). The harness caught both, as
it caught a correctness failure under `full`. That is the safety net doing its job,
and it is why the acceptance gate is worth its complexity.

### Single-shot vs agentic

Same model, same `algorithmic` framing, at most three measurements per cell; the 10 op-level tasks
x 3 replicates (`results/agentic-vs-single-shot-n3.md`):

```bash
uv run halo ablate --conditions algorithmic,agentic/algorithmic --replicates 3 --steps 3
```

| | single-shot | agentic |
|---|---|---|
| improvable tasks accepted | 17/18 | **18/18** |
| mean % of ceiling | 85% | **100%** |
| false positives on controls | 0/12 | 0/12 |
| measurements per cell | 3.0 | **1.3** |
| runs ended by the model's own `finish` | – | 28/30 |

The agentic loop captures more of the available speedup with less than half the
measurements, because it stops when it is done: on the controls it evaluates once,
reads the rejection, and finishes. And on `multi_head_projection` it beat the
hand-written ceiling — the einsum measured 1.04x over the seed; the agent's
`jnp.matmul(x[None], w)` measures 1.26x, three runs out of three, and is now the
ceiling. On this suite the loop's advantage is stopping, not searching: every task has
one well-known rewrite. The block- and model-level tasks are where searching should
start to matter.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is fetched automatically.

```bash
uv sync
```

For the LLM agent, authenticate to Google Cloud once. The education credits live in a
GCP billing account, and only the Vertex AI path draws on them — an AI Studio API key
does not.

```bash
gcloud auth application-default login
```

```bash
export GOOGLE_CLOUD_PROJECT=<your-project-id>
```

Credentials stay in `~/.config/gcloud`. **Never put a key in this repo — it is public.**

## Running

The agentic loop, the model driving through tools until it finishes or the budget
runs out:

```bash
uv run halo run --task pairwise_distances --agent agentic
```

The single-shot loop, one proposal per step:

```bash
uv run halo run --task naive_attention --agent single-shot --steps 3
```

No LLM, no network, no cost — a scripted agent with a known-good rewrite:

```bash
uv run halo run --task naive_attention --agent fast
```

See exactly what the model would be sent, without sending it:

```bash
uv run halo run --task naive_attention --dry-run
```

Other agents: `unstable` (a faster but numerically broken rewrite, to exercise the
correctness gate) and `echo` (proposes no change — the control condition that measures
the harness noise floor). `uv run halo tasks` lists the benchmarks. `--model` picks the
Gemini model (`gemini-2.5-flash-lite` by default for development; use
`gemini-2.5-flash` for reported sweeps); `--evaluations` and `--patience` size an
agentic run's budget.

Every run writes `runs/<timestamp>-<task>/` containing `config.json`, `env.json`,
`result.json` (best attempt, lineage, stop reason, cost), and per attempt the exact
`candidate.py` measured, `measurement.json`, `decision.json`, and the raw `jaxpr.txt`,
`stablehlo.txt` and `hlo.txt`. A single-shot run also keeps each prompt and raw
response; an agentic run keeps the whole tool conversation in `transcript.json`. The
report's tables come from these files.

## How it works

```
single-shot agent  ──propose──>  ┐
                                 ├──>  session  (ledger, budget, the only path to accept)
agentic agent  ──tool calls──>   ┘        │
                                          └──>  harness  (subprocess: compile, verify,
                                                          benchmark, extract every IR level)
```

Neither agent benchmarks anything itself. The session owns every side effect and
every decision, which makes each step loggable and replayable, and makes a scripted
agent a drop-in substitute for a model in tests. Each candidate is judged against the
*current best*, not the seed — comparing to the seed would accept a later candidate
slower than an earlier one and the loop would walk backwards reporting progress.

### The agentic loop

The model gets four tools and a budget:

| tool | what the session does |
|---|---|
| `evaluate(source, hypothesis, parent)` | compile, check against the oracle, time against the current best in interleaved rounds, rule on it. The only way to be accepted. |
| `check(source)` | correctness only, no timing. Cheap; never counts toward patience. |
| `inspect(attempt, what)` | the raw jaxpr / StableHLO / optimized HLO / buffer assignment of any earlier attempt, or its source or diff. Context is *pulled* when the model wants it. |
| `finish(summary)` | stop. The result is the best attempt the harness verified, whatever the summary says. |

The budget belongs to the harness: a maximum number of evaluations, tool calls and
dollars, and a patience of consecutive rejections. The model is told what remains
after every call but never asked whether it agrees. `parent` makes the ledger a tree:
building on a rejected attempt is a branch, not a lost result, and every candidate is
still judged against the verified best.

One tool call is executed per turn. `gemini-2.5-flash-lite` will otherwise emit six
evaluations and a `finish` in a single response — a whole plan guessed without seeing
a single result. The first call runs; the rest are recorded and the model is told.

Live, on Apple M5: `pairwise_distances` reached 4.62x (the measured ceiling is 4.58x)
in two model calls for $0.0007; on the `softmax` control the agent spent one
evaluation, read the rejection, and finished — $0.0006.

### Why the harness is a subprocess

JAX caches compilations process-wide and holds device state, and candidate code is
written by a language model — it can hang, exhaust memory, or abort the interpreter.
Isolation means a bad proposal costs one measurement instead of the whole run.

Note the honest limitation: this is isolation, **not a sandbox**. The wall-clock
timeout is enforced; a memory cap is not, because `RLIMIT_AS` interacts badly with
XLA's virtual-address reservations on macOS. Do not point HALO at untrusted code.

### What the agent is shown

Not the raw IR — the optimized HLO alone is ~700 lines for a toy attention kernel.
The agent gets each level of the lowering pipeline as an op histogram, so it can see
what XLA fixed on its own and what survived to the end:

| level | source | what it reveals |
|---|---|---|
| jaxpr | `jax.make_jaxpr` | the program as written; an unrolled Python loop shows up here as repeated primitives |
| StableHLO | `lower().as_text()` | what JAX handed the compiler |
| optimized HLO | `compile().as_text()` | what XLA built: fusions, surviving ops, flops, arithmetic intensity |
| buffer report | `--xla_dump_to` | which tensors were actually materialised, **by shape** |

The buffer report is the one that names names. `memory_analysis()` reports the seed
attention's scratch as a single number; the report says
`f32[4,256,256]x17` — seventeen live attention-score matrices, which is the
per-head loop stated in one line. The accepted rewrite leaves only
`f32[4,8,256,64]x4`: the three inputs and the output.

Nested jaxprs are walked, so a `scan` body is counted rather than skipped. That
matters because the streaming-softmax reformulation is a `scan`, and counting only
the top level would report such a candidate as a two-operation program.

### Why the timings are trustworthy

Three things, all easy to get wrong:

1. **`block_until_ready()` on every timed call.** JAX dispatches asynchronously.
   Without this you time the dispatch, not the computation.
2. **Interleaved A/B.** Baseline and candidate alternate within each round and the
   order flips between rounds, so thermal drift and frequency scaling hit both equally
   rather than favouring whichever ran first.
3. **A paired bootstrap CI on the ratio.** A candidate is accepted only when the 95%
   lower bound clears 1.05x. Measured on this machine, proposing *no change at all*
   yields 1.010x with a CI of [0.997, 1.025] — that band is the noise floor the
   threshold has to clear.

Compile time is measured as `lower()` + `compile()`, before the correctness pass, so
warming jit's cache cannot make a candidate's compile time read as zero.

Timings are only comparable within one device. Every measurement carries a device
fingerprint and the controller **refuses** to compare across them.

### Why the agent cannot cheat

The oracle is a NumPy **float64** implementation, deliberately independent of JAX, so
it also catches errors in the compiler rather than only in the rewrite.

- The agent may only write `candidate.py`. `task.py` is hash-checked every run.
- The graded inputs are withheld: `correctness_cases` is stripped from the source the
  agent is shown, and the seed that generates them never appears in a prompt.
- Adversarial cases are included by construction. For attention, a large common
  component in `q` and `k` pushes pre-softmax scores to ~200 — far past float32's `exp`
  overflow at 88 — so dropping the max-subtraction produces `inf` and is rejected,
  while the scores stay in a narrow band so a *correct* float32 implementation still
  matches the oracle. (Simply scaling the inputs up instead makes attention nearly
  one-hot, and then no float32 implementation can match float64 — a test that fails
  correct code.)
- Shape, dtype and finiteness are checked, not just values.
- Correctness is evaluated before performance, and a correctness failure is final.

Relative error is reported against the array's own scale rather than per element;
dividing by near-zero outputs reports a "178% error" on a result correct to 6e-05.

## Adding a benchmark

Create `src/halo/tasks/<name>/` with three files.

`task.py` is immutable and defines `make_inputs(rng)`, `reference(*inputs)` (NumPy
float64), `correctness_cases(rng)` returning `Case` objects, and a `DESCRIPTION` shown
to the agent. Keep the docstring neutral — this file goes into the prompt, so any hint
at the intended optimization invalidates the experiment.

`candidate.py` defines `candidate(*inputs)` and is the only file the agent rewrites.

`ceiling.py` is the best implementation you know of, and is never shown to the agent.
It is what makes the task's headroom a measured number rather than an assumption —
including for controls, where it proves there is nothing to find. For a control, copy
the seed. Then add the task to `CLASSIFICATION` and `MEASURED_HEADROOM` in
`tasks/ceilings.py`.

A `Case` may override `atol`/`rtol` for inputs that are ill-conditioned in float32, and
a task may set module-level `ATOL`/`RTOL` to raise its floor for every case. Reductions
need this: an output element that cancels toward zero still carries the accumulated
error of the whole reduction, so a 512-term float32 dot product cannot meet a tolerance
that an elementwise operation meets easily. Five tasks here set `ATOL` for that reason,
each stating the measured error that justifies the value. Always say why — a silently
loosened tolerance is how a broken candidate gets accepted.

`uv run pytest` asserts that both the seed and the ceiling pass their own oracle, so a
tolerance too tight to admit a correct rewrite fails loudly rather than showing up as a
task with no headroom.

## Hardware

Develop on CPU — JAX's CPU backend is real XLA, so the HLO, the fusion decisions and
the compiler feedback are all genuine; only the absolute numbers are laptop-grade.

For GPU numbers: **JAX has no native Windows GPU support.** NVIDIA laptops need
WSL2 + Ubuntu + CUDA, then `uv sync --extra cuda` (add the extra when that machine is
set up) or `pip install "jax[cuda12]"`. Never compare a GPU timing against a CPU
baseline; the fingerprint guard will refuse it, and it is meaningless anyway.

## Tests

```bash
uv run pytest
```

Everything except the `slow` marker runs without JAX or a subprocess. Nothing in the
suite makes a network call or costs money.

```bash
uv run pytest -m "not slow"
```

## Not built yet

Block- and model-level benchmarks (embedding, cross-entropy, GQA, scan hoisting, a
small transformer forward pass), per-op profiling, GPU measurements and Pallas
kernels. See the roadmap in the project plan.
