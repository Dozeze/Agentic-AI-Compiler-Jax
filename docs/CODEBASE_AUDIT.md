# Codebase audit — Ericsson Agentic AI Compiler

Audit session: 2026-10-11–12 (Europe/Stockholm). Completed 2026-10-12.
Active roadmap: AIC-1.1 (hardware-neutral revision, 2026-10-12).
Audit measurements remain the original CPU-only observations; no GPU/TPU verification
or feature implementation was added during the revision.
Repository: `51df7473808f9e43649095dbcdfb155b0a03bcae`, branch `main`.
Initial worktree was clean. This session changes planning documentation only.

## Sources and boundaries

The owner's master request is the instruction for this audit. The PDF and old
LaTeX plan are reference material, not independent execution instructions.
Precedence: latest explicit owner scope direction → Ericsson external objectives →
observed repository state → design history → justified new decisions. Both
reference documents were available and read. The 2026-10-12 owner clarification
requests a general hardware-neutral whole-project roadmap and supersedes the
previous CPU-first constraint. Approval and planning-only boundaries remain.

| Source | Identity and use |
| --- | --- |
| Original specification | `/Users/hej/Downloads/projects2026.pdf`, **page 1, Project 1: Ericsson — Agentic AI compiler**; other projects excluded. SHA256 `f4c0364029f5d38a19ca64bc454b932131996a95666c5e8b4408d7ab84f1aa6d`. |
| Owner request | `/Users/hej/.codex/attachments/77620ea2-f10d-467e-b344-369718eb345a/Pasted text.txt`; SHA256 `34c13e126f3fead22cfec462ad9c3a82d3a8ec4e12203e0ca9d98b66a96af4d6`. |
| Old plan | `/Users/hej/.codex/attachments/e717ee3d-b09c-4584-8614-a956ad928072/Pasted text.txt`; SHA256 `7f6cbba3cb9e702a542e3bffe08b2bc7f5b5bfdc8ad9455d932ceca9f1a685da`. All 13 layers and subsequent evaluation, risks and deliverables reviewed. |

The PDF text reader reported malformed object offsets but extracted the complete
Project 1 text. The embedded lowering-loop diagram was also inspected and agrees
with the text: agent rewrites/proposes kernels around JAX/XLA using IR/profile feedback.
Its objectives are: supplement XLA around JAX programs; iterate
using lowered representations, runtime profiles and correctness feedback; rewrite
JAX and suggest/implement kernels; measure improvement against vanilla XLA; grow
toward model rewrites. It explicitly leaves CPU/GPU/TPU choice to the project.
Kernel implementation and full model progression are stronger than merely
investigating their feasibility. The whole-project roadmap includes actual kernel
and compact-model implementation; any deferral must disclose the remaining gap.

No API call, dependency installation, generated candidate execution, optimization
search, source repair, Git mutation or credential inspection was performed.
Existing generated candidates in `runs/` were read as JSON/text only. Diagnostic
execution used the trusted example source; mocked loop evaluation never executed
candidate text. Temporary artifacts stayed under `/private/tmp` and were removed
by temporary-directory contexts, apart from a disposable Matplotlib cache and
an extracted PDF diagram used for visual inspection.

## Existing architecture

There is a flat Python script project, not the old plan's `src/` package tree.

| Path / symbols | Actual responsibility |
| --- | --- |
| `agent.py:25` `_load_source_workload` | Reads a trusted Python file and executes it to find a callable; uses `make_test_args`, `TEST_ARGS`, or guessed one/two-argument arrays; wraps the callable with `jax.jit`. |
| `agent.py:59` `build_controller` | Builds the reference and a raw-source prompt; passes an injected Anthropic client to `Controller`. Defaults to 32-element addition. |
| `agent.py:102` `main` | CLI, `.env.local` loading, device printing, optional GPU requirement, API-key check, live client, loop and optional plot. |
| `controller.py:21` `Controller` | Constructor immediately evaluates baseline; keeps best source/runtime; tool dispatch, conversational loop, decisions, JSON persistence and plotting are combined. |
| `benchmark.py:35` `evaluate` | Python syntax compile, unrestricted in-process `exec`, candidate JIT, one reference comparison, timing. |
| `benchmark.py:8` `timeit_version_1` | Per-batch warmup, synchronized single-array results, repeated batch means and their standard deviation. |
| `workloads/example.py:7` | Fixed `float32` inputs and addition; infrastructure example, not a research suite. |
| `algorithm.py:31` | Unconnected experimental gradient script with an invalid call; not a supported workload. |
| `test.py`, `test1.py`–`test3.py` | Environment dump and interactive provider/tool examples, not assertions or automated tests. |
| `testrun.py`, `runningscripts.py`, `utils.py` | Duplicate timer, stale imports, unrelated utility functions respectively. |
| `pyproject.toml`, `.python-version`, `uv.lock` | uv project, Python 3.12 preference, Python >=3.12 declaration, JAX 0.4.38, locked CPU dependencies; CUDA 12 extra conditional on Linux. |
| `README.md`, `how_to_run.md` | Current CPU/GPU CLI instructions and known limitations. |
| `runs/example/` | Ignored local four-entry history and per-iteration JSON. No provenance manifest. |
| `tests/` | Only bytecode cache; no tracked or readable test source. Cache is not test coverage. |

Data flow today: trusted source → reference JIT → constructor baseline evaluation
→ Claude tools (`get_current_code`, `get_history`, `benchmark_code`) → in-process
evaluation → strictly lower mean accepted → JSON → next tool result. Thus an
iterative architecture and feedback channel already exist. They need hardening
and compiler/runtime context, not wholesale replacement.

No applicable `AGENTS.md` was found in the repository or inspected parent
directories; `.agents/` and `.codex/` are absent here. Future session procedure is
defined in [CODEX_WORKFLOW.md](CODEX_WORKFLOW.md). No instruction file was overwritten.

## Capability assessment

`VERIFIED_WORKING` means demonstrated within the stated scope below, not that the
entire future requirement is satisfied. `IMPLEMENTED_UNVERIFIED` means code exists
without execution evidence. The other labels describe gaps or observed defects.

| Capability | Status | Evidence / practical limit |
| --- | --- | --- |
| JAX/XLA CPU integration | VERIFIED_WORKING | D02; original example compiles, executes and matches addition on `TFRT_CPU_0`. |
| Trusted source loading | VERIFIED_WORKING | D02, `agent.py:25`; explicit example works. Arbitrary workload signatures, static arguments and import failures are not comprehensively covered. |
| Workload management | PARTIALLY_IMPLEMENTED | One fixed case; no seed/shape registry, edge cases, hidden split or contract. Guessed inputs can misrepresent a function. |
| Baseline timing | VERIFIED_WORKING | D02; baseline goes through JIT and synchronized timer. This establishes execution only, not statistically reliable performance. |
| General correctness | PARTIALLY_IMPLEMENTED | D02 rejects value/shape/dtype mismatches, but single array and single input only. A tuple reference fails synchronization. No structured per-case diagnostics or workload tolerance policy. |
| Benchmark methodology | PARTIALLY_IMPLEMENTED | `benchmark.py:8`; warmup excludes first compilation and blocks every timed result. Raw samples, medians, compile durations, environment, pairing and drift control absent. |
| Compiler feedback | MISSING | No repo extraction/summary path. D03 proves suitable APIs are available in the installed environment. |
| Runtime profiling | MISSING | Repo measures wall time only. D04 proves a trace can be collected externally with observed CPU execution events. No parser/context integration exists. |
| Live provider connection | IMPLEMENTED_UNVERIFIED | `controller.py:277`, `agent.py:127`; Anthropic installed. No live call made; credentials/model availability not validated. |
| Structured proposals | PARTIALLY_IMPLEMENTED | Tool schema defines a `code` string; dispatch checks that string only. No hypothesis, local full-schema validation, size limits or constraints enforcement. |
| Candidate source generation | IMPLEMENTED_UNVERIFIED | Provider tool path and old stored candidates exist. No audit evidence that a live LLM generated/discovered an improvement. |
| Candidate execution | PARTIALLY_IMPLEMENTED | Trusted original text works. Generated execution shares controller privileges and credentials; no limits or isolation. Unsafe for autonomous generated code. |
| Iterative controller and feedback | VERIFIED_WORKING | D05 injected fake client yields two evaluations, three history entries and second-request feedback. This verifies plumbing only. |
| Runtime budgets | BROKEN | D06: zero requested evaluations still produces one model request and one candidate evaluation. Negative values likewise lack input validation by inspection. |
| Performance decision | BROKEN | D05 accepts a synthetic 1e-8-second improvement amid 0.2-second reported deviation. `controller.py:86` compares means without uncertainty. D07 shows real archived consequence. |
| JSON history / console report | VERIFIED_WORKING | D05 confirms exact JSON roundtrip, baseline plus two iteration files and reporting. Atomic individual replacement exists at `controller.py:193`. |
| Reproducibility / recovery | PARTIALLY_IMPLEMENTED | Source, diff and status retained; no seeds, raw timing, versions, model messages, usage, run ID, source hash, recovery or best-source artifact. Default `runs/latest` can overwrite results and leave stale iteration files. |
| Plot | IMPLEMENTED_UNVERIFIED | `controller.py:327`; failure rows contain `None`, so must be tested before claiming robust reporting. Importing controller unnecessarily imports Matplotlib. |
| Automated tests | MISSING | D01: zero tests; pytest unavailable. Interactive scripts deliberately not collected. |
| Custom kernels | MISSING | No FFI/native build or kernel agent action. Installed FFI API present, native integration untested (D03). |
| GPU configuration | IMPLEMENTED_UNVERIFIED | Optional Linux CUDA dependency exists; no GPU compile/execute/profile or isolation diagnostics attempted. Packaging does not prove support. |
| TPU integration and accelerator kernels | MISSING | No TPU-specific setup or implemented GPU/TPU kernel path found; no target has been selected for the revised roadmap. |

## Diagnostics and evidence register

Executed on the current host, with no package installation. CPU commands used
`JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1`; controller imports also used
`MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/agentic-audit-mpl`. Benchmark numbers
below are audit observations on tiny inputs and must not appear as speedup claims.

### D00 — environment and Git

Commands: `git status --short --branch`, `git rev-parse HEAD`, `git ls-files`,
`rg --files`, `.venv/bin/python` with `platform`, `importlib.metadata` and
`jax.devices()`. Results:

- Python 3.12.13; macOS 12.7.6; x86_64; four reported logical CPUs.
- JAX/jaxlib 0.4.38; NumPy 2.5.2; Anthropic 1.3.0; claude-agent-sdk 0.2.151;
  python-dotenv 1.2.3. JAX backend CPU, device `TFRT_CPU_0`, x64 disabled.
- pytest, TensorBoard and TensorBoard profile plugin not installed.
- `.venv` works; clean installation from `uv.lock` was **not tested**.
- `sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu` was denied by the
  execution sandbox. CPU model and RAM remain unverified; no invented hardware.
- Docker CLI exists; `docker info --format '{{.ServerVersion}}'` reported socket
  permission denial. Wrapper reported exit 0 despite error output; daemon access
  and containment are **unverified**, not proven unavailable on the user's host.
- `.env` and `.env.save` are tracked; `.env.local` is ignored. Contents were not
  read. This is a credential-hygiene concern, not evidence of leaked credentials.

### D01 — automated test availability

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v`
reported **Ran 0 tests / NO TESTS RAN**, exit 5 in this environment.
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q` failed at startup,
exit 1: **No module named pytest**. There were no executed test assertions and
there is no basis to report a passing suite or coverage percentage.

Root test discovery was deliberately not attempted: `test.py:5` prints the whole
environment; `test1.py:7`, `test2.py:29`, `test3.py:89` start interactive paid-API
loops. Those scripts were inspected as text only.

### D02 — bounded trusted workload and evaluator

Core diagnostic, executed through `.venv/bin/python -`:

```python
from agent import _load_source_workload
import benchmark
source, reference, args = _load_source_workload('workloads/example.py', 'bad_func')
output = reference(*args).block_until_ready()
result = benchmark.evaluate(source, reference, args,
                            num_runs=10, num_tests=3, function_name='bad_func')
```

Output shape `(32,)`, dtype `float32`, exact match with input addition. Evaluator
returned syntax/correctness true; mean `1.337776678459098e-05` seconds, batch-mean
std `4.504888098232872e-07` seconds. No raw timings were returned.

With the **same trusted candidate source**, reference callbacks computing
`x-y`, `(x+y)[:-1]`, and `(x+y).astype(int32)` each returned correctness false
and no timing. A tuple reference `(x+y, x-y)` returned the caught error
`'tuple' object has no attribute 'block_until_ready'`. This tests rejection
branches without running archived/model-generated candidates; tuple-to-tuple
equivalence is also unsupported by the same synchronization implementation.

### D03 — pinned compiler APIs and FFI availability

Using the D02 reference: `lowered = reference.lower(*args)`;
`compiled = lowered.compile()`. Observed:

| API | Actual result |
| --- | --- |
| `str(jax.make_jaxpr(reference)(*args))` | 177-character Jaxpr (contains outer JIT wrapper). |
| `str(lowered.compiler_ir(dialect='stablehlo'))` | 301-character StableHLO. |
| `lowered.compiler_ir(dialect='hlo').as_hlo_text()` | 227-character pre-optimization HLO. |
| `compiled.as_text()` | 579-character scheduled compiled HLO; distinguish from lowered HLO. |
| `compiled.cost_analysis()` | List containing a dict, including FLOPs 32 and bytes accessed 384. Static estimates. |
| `compiled.memory_analysis()` | `CompiledMemoryStats`: argument 256 B, output 128 B, temporary 0 B, generated code 0 B; not a measured peak-RSS report. |
| `jax.extend.ffi` | `ffi_call`, `register_ffi_target`, `include_dir` callable; no C++ build or FFI execution attempted. |

The installed `jax/_src/pallas/pallas_call.py:1482` explicitly rejects CPU
lowering without interpret mode. Pallas interpretation cannot establish native
CPU kernel optimization. Current documentation may expose APIs not in 0.4.38;
the tested pinned interfaces take precedence.

### D04 — CPU runtime trace feasibility

Compiled/warmed `workloads/example.py` before collecting five calls:

```python
with jax.profiler.trace(directory, create_perfetto_link=False,
                       create_perfetto_trace=True):
    for _ in range(5):
        with jax.profiler.TraceAnnotation('audit_cpu_call'):
            reference(*args).block_until_ready()
```

Produced XPlane protobuf, gzipped trace JSON and `perfetto_trace.json.gz`.
The Perfetto JSON contained 57 events, including five annotations, ten
`TfrtCpuExecutable::Execute`, five `ExecuteHelper`, five `add.3`, and five
`ThunkExecutor::Execute (wait for completion)` events. The library logged that
it could not import `tensorflow.python.profiler.trace`, yet emitted usable data.
These counts prove trace collection, **not** reliable source-op attribution,
non-overlapping hotspot percentages, or a bottleneck diagnosis. No viewer,
TensorFlow installation or trace upload was needed. Profiling was separate from
the unprofiled D02 timing.

### D05 / D06 — deterministic controller probes

`unittest.mock.patch('benchmark.evaluate', side_effect=...)` supplied evaluation
records. A fake `messages.create` returned the known example source in a
`benchmark_code` block, and deep-copied request messages. No model or generated
code executed. With `max_iterations=2`, supplied means were `1.0`, `1.0-1e-8`,
`2.0`, all std `0.2`. Observations:

- Two requests, three history entries, accepted flags `[true, true, false]`.
- Request two included the previous evaluation as `tool_result`.
- Four files: `history.json`, `iteration_0000.json` through `0002.json`;
  deserialized history equaled returned history.
- With `max_iterations=0`, still one request and one evaluation. The loop's
  post-evaluation budget check and `max(1, iterations*3)` cause this defect.

No mock timings are performance measurements or LLM research evidence.

### D07 — pre-existing artifacts

Read `runs/example/history.json` without executing its code. Four entries:
baseline `12.1892802417 µs`, accepted candidate `12.1874009104 µs`, then two
rejections. The accepted textual rewrite changes `x + y` to `jnp.add(x, y)`.
Its measured difference is about **0.001879 µs (0.0154%)**, while the reported
std values are approximately **0.475 and 0.480 µs**. The decision provides no
credible speedup evidence. No raw timings, model provenance, environment or
independent confirmation are stored. Later comments claim potential contiguous
memory/fusion benefits; those are unverified candidate claims, not audit facts.

### D08 — CLI and static defects

`... .venv/bin/python agent.py --help` completed and lists `--iterations`,
`--model`, `--source`, `--function`, `--output-dir`, `--plot`, `--require-gpu`.
The documented live CLI was not run because it requests paid model calls and
executes generated code without isolation.

- `algorithm.py:32`: `func(x,c)` is called with four arguments, then its result
  would be passed to `jax.grad` with invalid `argnums=4`. Broken by inspection;
  not executed, and not part of the functioning path.
- `runningscripts.py:6`: imports absent `bad_script` and `Functester`. Broken by
  repository inspection; not executed.
- `controller.py:38`: baseline failure does not abort `agent_loop`; provider
  exceptions have no durable run-level failure handling.
- `benchmark.py:96`: timing exceptions lie outside the earlier exception handler.
- `controller.py:227` `benchmark_code` evaluates without recording, unlike the
  tool dispatcher. Preserve one authoritative evaluation/recording route later.
- `agent.py:14` refers to absent `controller.md`; stale comments contradict the
  implemented loop. `--function` without `--source` can disagree with hard-coded
  `bad_func`. These are targeted repairs, not reasons for a package rewrite.

## Critical review of the previous plan under the revised scope

The old LaTeX plan and the superseded unapproved CPU-1.0 draft are distinct.
The following decisions are the active hardware-neutral interpretation, not a
claim of new implementation. Historical task aliases map CPU-NNN to AIC-NNN;
no completed task was renumbered.

| Old proposal / decision | Disposition | Reason / revised location |
| --- | --- | --- |
| JAX/XLA, iterative feedback, correctness and vanilla baseline | Retain | Direct Ericsson objectives; existing loop is a useful starting point. |
| Open hardware selection | Retain and make executable | AIC-031 compares actual CPU/GPU/TPU access, versions, profiling, isolation and kernel feasibility; choose one primary target. |
| CPU-only completion scope from first draft | Superseded by owner | General whole-project roadmap; CPU observations remain audit facts, not target selection. |
| Thirteen layers / fixed sequence | Consolidate | Seven main milestones and one optional portability milestone; reuse implemented loop and history rather than rebuilding. |
| Large src/ tree and four config files | Simplify | Existing flat modules plus cohesive contracts/runner/feedback/storage, one resolved run config and explicit target config. |
| Environment and testing | Retain / repair | Lock exists but clean install and meaningful test suite do not; preserve CPU smoke without equating it to accelerator verification. |
| Workload progression | Retain, bound by resources | Addition → softmax → normalization/dense → attention → block → compact complete model. No production-size model required initially. |
| Numerical correctness | Retain / strengthen | Multi-case PyTrees, nonfinite policy, frozen workload tolerances and independent reference outputs. |
| Two-percent acceptance threshold | Replace | Audit shows tiny noise differences accepted; use target-specific A/A calibration and independent confirmation. |
| CPU/GPU timing details | Generalize explicitly | Synchronized same-device end-to-end JAX measurement; CPU threads and accelerator allocation/driver/memory as selected-target controls. |
| Compiler IR extraction | Retain / probe per target | CPU APIs verified; changed software/accelerator behavior must be checked, optional metrics not assumed. |
| Normalized profiling | Retain with coverage | Distinguish actual CPU/device events, host-only traces and unavailable observations; never substitute estimates for profiles. |
| Provider abstraction | Simplify | Reuse injected Anthropic client through one validated proposal boundary and deterministic fake. No multi-provider framework prerequisite. |
| Hill-climbing controller | Retain / repair | Correctness-gated decisions, budgets before actions, baseline abort and durable outcome before next context. |
| SQLite/history layer | Simplify / merge early | Atomic JSON and unique run records suffice; history informs the MVP instead of becoming a later disconnected feature. |
| Subprocess as sandbox alternative | Strengthen | Lifecycle management alone is insufficient; prove the actual selected host/device containment and cleanup boundary. |
| Custom kernels | Restore to main-project target | Choose one actual native/device path, implement manual integration, then live agent kernel edits before final evaluation. Investigation-only needs scoped exception. |
| Complex workload/full model | Include bounded progression | Attention and block followed by a compact full forward model. Explicit limits distinguish it from production LLM coverage. |
| Research conditions C0–C5 | Bound and prioritize | Source/feedback/compiler comparisons, matched kernel condition and full-model coverage; optional extra profile ablation only with real data/budget. |
| Report, demo and final DoD | Retain / evidence-gate | Functional, correctness, research, performance and original scope are assessed independently. |
| Team allocation and calendar | Keep unresolved | No invented capacity; relative effort and approved gates. |
| Multi-hardware capability | Defer additional target implementation | Neutral goals do not mean simultaneous CPU/GPU/TPU support. AIC-033 qualifies a second target only after delivery. |

## Principal debt, blockers and conclusion

The largest gaps are safe selected-target evaluation, broad correctness, defensible measurements,
compiler/profile context and durable experimental provenance. Existing work
already proves the basic mechanism of an iterative loop. The plan should extend
it, retaining source loading, provider injection, history semantics and atomic
JSON writes where appropriate.

Immediate blockers are absent automated tests, unverified isolation access and
unknown final measurement host. Model/spend decisions block future live runs,
not planning. No evidence establishes a reproducible improvement over XLA.
Kernel generation and larger model rewrites are absent. The revised main-project target includes kernel and compact whole-model capability.
A reduced implementation may still provide functional/research value, but any
missing kernel/model/profile scope and unachieved speedup must remain explicit.
Hardware selection is open; only CPU measurements have current execution evidence.

The actionable design and backlog are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
and [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md). No implementation task is
marked DONE from these narrow diagnostics.
