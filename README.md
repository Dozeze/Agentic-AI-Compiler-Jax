# HALO — HLO-Aware Lowering Optimizer

An agentic optimization loop around JAX/XLA. Given a JAX function, HALO measures it,
shows a language model the code plus the compiler's own feedback, takes the model's
rewrite, measures it again, and accepts it only if it is both **correct** and
**measurably faster**.

KTH DD2430 Group 19 · Project 1 (Ericsson, *Agentic AI Compiler*).

Current scope is the MVP: **one** improvement step. The loop is written as a loop, so
going autonomous means raising the step count and adding a stopping rule, not
restructuring anything.

## Status

Verified end to end on Apple M5 (CPU backend), `gemini-2.5-flash-lite` via Vertex AI:

| task | outcome | speedup (95% CI) | cost |
|---|---|---|---|
| `naive_attention` | accepted | **1.726x** [1.691, 1.761] | $0.000508 |
| `softmax` | rejected (performance) | 1.018x [0.996, 1.042] | $0.000276 |

The softmax result is the expected one and is not a failure. XLA already fuses the
standard formulation; the model correctly identified the kernel as memory-bound and
proposed no change, and the harness declined to call a 1.8% difference a speedup.

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

```bash
uv run halo run --task naive_attention --agent vertex --steps 1
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
the harness noise floor). `uv run halo tasks` lists the benchmarks.

Every run writes `runs/<timestamp>-<task>/` containing `config.json`, `env.json`, and
per attempt the exact `candidate.py` measured, `measurement.json`, `decision.json`,
the optimized `hlo.txt`, and the prompt and raw response. The report's tables come
from these files.

## How it works

```
controller ──> agent      (pure function: Context -> Proposal)
     │
     └───────> harness    (subprocess: compile, verify, benchmark, extract HLO)
```

The agent never benchmarks anything itself. The controller owns every side effect,
which makes each step loggable and replayable, and makes a scripted agent a drop-in
substitute for a model in tests.

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

Create `src/halo/tasks/<name>/` with two files.

`task.py` is immutable and defines `make_inputs(rng)`, `reference(*inputs)` (NumPy
float64), `correctness_cases(rng)` returning `Case` objects, and a `DESCRIPTION` shown
to the agent. Keep the docstring neutral — this file goes into the prompt, so any hint
at the intended optimization invalidates the experiment.

`candidate.py` defines `candidate(*inputs)` and is the only file the agent rewrites.

A `Case` may override `atol`/`rtol` for inputs that are genuinely ill-conditioned in
float32. Always say why in `note`: a silently loosened tolerance is how a broken
candidate gets accepted.

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

Multi-step loops, parallel candidate exploration, tool-calling agents, Pallas/Triton
kernels, multi-provider comparison, RAG over JAX docs. The interfaces leave room for
each.
