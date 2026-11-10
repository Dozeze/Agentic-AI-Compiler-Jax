# AI_CONTEXT — shared context for anyone (human or AI) working on this repo

> **Team rules (meeting, Nov 2026):**
> 1. If you use an AI tool on this project, it must read this file first, and **every change you push must update
>    this file in the same commit** — at least §5 Status, §6 Decisions and §9 Change log. If your change makes
>    anything here wrong, fix it.
> 2. **All documentation and code comments are in English.**
>
> Keep this file short: it is the current state, not a history. Detailed evidence goes in
> `docs/IMPLEMENTATION_PROGRESS.md`; the full plan is `docs/IMPLEMENTATION_PLAN.md`.

Last updated: 2026-11-10 by Sergio (Sergio Medina Hernandez)

---

## 1. What this project is
KTH DD2430, Group 19, Project 1 (Ericsson): **Agentic AI Compiler**. An LLM agent iteratively rewrites pure JAX
functions; a trusted harness compiles them with XLA, checks correctness against a frozen reference, measures
runtime, and keeps a candidate only if it is correct **and** measurably faster.

Supervisors: Alexander Kravberg (Ericsson), Danica Kragic (KTH).
Team: Jonathan Larsson (`Dozeze`, repo owner), Jonathan Tadesse (`NeguseNegest`),
Sergio Medina Hernandez (`SergioMedinaH`), Weizhuo "Harry" Sun.

## 2. Which document wins
1. Latest explicit team decision (recorded in §6).
2. `docs/IMPLEMENTATION_PLAN.md` — roadmap **AIC-1.1** (2026-10-12): 8 milestones, 33 tasks, approval gates.
   *(The plan exists as a PDF; `docs/IMPLEMENTATION_PLAN.md`, the 33-task backlog and the other process documents do not exist yet — §6 P1.)*
3. The 2026-09-04 LaTeX project plan is **design history only** (AIC-1.1 says so). Its dates and task list are
   superseded; the course deliverables it mentions (report, weekly diaries, presentation) still apply.

## 3. Repository map
| Branch | Role |
|---|---|
| `main` | Base for all AIC-1.1 work. Flat Python modules, Anthropic client, JAX 0.4.38. |
| `harry/batch-01` | Harry's batch 01: AIC-001, 031, 002, 003, 004. Harry's plan is local only (`docs/plans/` is git-ignored). |
| `sergio/batch-02` | Sergio's batch 02: AIC-005, 006, 007, 008, 009 (from `harry/batch-01`). |
| `claude-experiment` | Sergio's separate HALO prototype (Gemini/Vertex, JAX 0.11, 18 tasks, results). Shares no code with `main`; AIC-1.1 does not use it. Reference only. |
| `new_ver` | Stale ancestor of `main`. Ignore. |

Key files on `main`: `agent.py` (CLI entry), `controller.py` (loop, history, tools), `benchmark.py` (correctness +
timing), `workloads/example.py`. Root `test*.py`, `testrun.py`, `runningscripts.py`, `utils.py`, `algorithm.py` are
scratch scripts — **never import them from tests**.

## 4. Environment and how to run
- Python 3.12, `uv`, **JAX/jaxlib pinned to 0.4.38**. Always `uv run --locked ...`. Do not upgrade JAX without a team decision.
- **Development:** Harry's Mac (CPU only) — write code, run CPU tests.
- **Target hardware:** NVIDIA GPU via CUDA on Harry's **Windows** NV host, through **WSL2 Ubuntu** (JAX has no native
  Windows GPU support; WSL2 is "experimental" in the JAX docs). All official measurements run there, in one fixed
  environment. Setup: `docs/WSL2_CUDA_SETUP.md` (written in AIC-031).
- CPU tests: `uv sync --locked && uv run --locked pytest -q` (no API key, no network).
- GPU (inside WSL2): `uv sync --locked --extra cuda12 && uv run --locked --extra cuda12 pytest -q --require-gpu`
  (`--require-gpu` turns the `needs_gpu` skip into a setup error, non-zero exit, when CUDA is broken).
- Isolation: start Docker, then `uv run --locked python scripts/build_worker_image.py` (add `--variant cuda12` on the NV host).
- Run (needs a key in `.env.local`): `uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1` (`--config run.json` for a RunConfig v1 file; `--runner subprocess` without Docker: baseline only)
- Freeze δ: `uv run --locked python scripts/calibrate_aa.py --workload softmax --backend cpu --runner docker --label <population>`, then `--calibration <file>`.
- NV host, everything GPU in one go: `scripts/phase_b.sh`.

## 5. Status (update every push)
| Milestone | Tasks | Owner | Status |
|---|---|---|---|
| M1 Foundations & target | AIC-001, 031, 002, 003 | Harry | batch 01 AWAITING_ACCEPTANCE, pushed to `origin/harry/batch-01` (D-07; report: `docs/IMPLEMENTATION_PROGRESS.md`): AIC-001 done locally (offline pytest suite, fake Anthropic client, CI); AIC-031 Mac part done (`environment.py`, `scripts/probe_target.py`, `evidence/targets/mac-cpu.json`, `docs/TARGET_DECISION.md` **DRAFT**) — NV-host probe (phase B) pending; AIC-002 done locally (`contracts.py` RunConfig v1, `--backend`/`--config`, `run_config.json` + `manifest.json`); AIC-003 done locally (`workloads/registry.py` WorkloadSpec v1, seeded `softmax`, float64 oracle, legacy `example_add`); dual-backend rules D-06 + guard tests (`tests/test_backend_portability.py`) |
| M2 Reliable, isolated evaluation | AIC-004 … 009 | AIC-004: Harry; AIC-005 … 009: Sergio | batch 02 AWAITING_ACCEPTANCE on `sergio/batch-02` (report: `docs/IMPLEMENTATION_PROGRESS.md`): AIC-004 done (Harry); AIC-005 timing protocol + A/A calibration, AIC-006 decision rule, AIC-007 contracts + source policy, AIC-008 worker/runner, AIC-009 Docker containment + controller integration done on the Mac (CPU). Pending on the NV host: GPU containment and the official δ (`scripts/phase_b.sh`) |
| M3 – M8 | AIC-010 … 033 | AIC-010: Sergio; rest TBD | AIC-010 in progress on `sergio/batch-03` (Sergio); the rest not started |

Course deadlines (fixed by KTH, not by AIC-1.1): report draft **2026-11-20 17:00**; presentation + final report
**2026-12-15 … 21**.

## 6. Decisions
**Decided**
- D-01 Work in batches of five tasks; batch 01 = AIC-001, 031, 002, 003, 004 (Harry).
- D-02 Primary target: CUDA on an NVIDIA GPU (Harry's NV host). Mac is for development only.
- D-03 AI use is allowed with this file kept up to date.
- D-04 English only for documentation and code comments.
- D-05 The NV host runs Windows with an NVIDIA RTX 40-series GPU (Ada, newer than Volta, so CUPTI profiling under WSL2 is supported); the GPU path is WSL2 Ubuntu + Windows NVIDIA driver (no Linux driver inside WSL). Exact model and driver version are recorded by the AIC-031 probe.
- D-06 All code and tests run on both CPU and CUDA, so every member can test locally (Mac, CI: CPU). CPU runs
  prove correctness only; CUDA correctness and **all** timing evidence come from the NV host. Rules: §7.
- D-07 Hand-off: the author verifies locally (full suite + phase A on the Mac), then pushes their own branch;
  the rest of the team runs it from there (CI, and phase B on the NV host). Nobody pushes to `main` directly.

**Pending** (default in brackets — used until the team decides)
- P1 `docs/IMPLEMENTATION_PLAN.md`, the 33-task backlog, `CODEBASE_AUDIT.md`, `CODEX_WORKFLOW.md` do not exist yet; the team creates and maintains them from now on. Batch 01 creates `docs/IMPLEMENTATION_PROGRESS.md` (its first entry is the batch-01 report).
- P2 Who is the "owner" who approves milestones? [whole team]
- P3 Keep WSL2, or dual-boot Ubuntu on the NV host? [WSL2; revisit only if a hard requirement fails]
- P4 `--iterations` default 10 → 3 per AIC-1.1 §7? [3; implemented in AIC-002 as `Budgets.max_candidates = 3` — set it back to 10 if the team disagrees]
- P5 Softmax large-offset edge case: 200 or 1e4? Must be fixed **before any search**. [200: still catches a non-stabilised softmax; offsets ≥ 1000 reject the valid `exp(x - logsumexp(x))` rewrite at rtol 1e-5; implemented as `workloads/registry.OFFSET = 200.0`, re-confirmed on the Mac by `test_logsumexp_rewrite_admitted` / `test_edge_cases_have_teeth`]
- P6 New flat module `environment.py` (backend selection + env capture)? [yes; added by AIC-031 — merge into `contracts.py` if the team objects, interface unchanged]
- P7 What must be done by the 2026-11-20 report draft? [unset]
- P8 LLM provider and money cap (no live calls without a cap, AIC-1.1 §7). [Anthropic; cap unset → live disabled]
- P9 What to do with `claude-experiment` results (Sergio). [reference only]
- P10 Move root scratch scripts into `scratch/`? [leave them; AIC-001 isolates them with `testpaths = ["tests"]`]
- P11 Record the hostname in run manifests? [yes (`environment.platform.node`); hash it if anyone objects]
- P12 Isolation technology for model-written code (AIC-009)? [Docker: same path on the Mac, in CI and on the NV host
  (`--gpus all` under WSL2); `runner.DockerRunner`, image from `uv.lock`]
- P13 What if isolation is unavailable? [block: exit 2 `ISOLATION_UNAVAILABLE` before any LLM call; `--runner subprocess`
  runs the baseline and refuses model candidates (the run ends as INTERRUPTED)]
- P14 When is δ calibrated? [lazily, before the first candidate measurement, unless a frozen `--calibration` file is given;
  official files come from `scripts/calibrate_aa.py` on the NV host]
- P15 New flat modules `source_policy.py` and `array_io.py`? [yes; each is tested on its own]

## 7. Rules for code changes
- One branch per person/batch; never push straight to `main`. One task per commit/PR; message starts with the task ID (`AIC-001: ...`).
- Every PR: tests pass offline (`uv run --locked pytest -q`), this file updated, entry in `docs/IMPLEMENTATION_PROGRESS.md`.
- English only for docs, comments, docstrings, test names and commit messages.
- **The repo is public:** never commit keys, `.env*` (except `.env.example`), `runs/`, large traces.
- Never dump `os.environ`; record only allowlisted variables.
- Use AI incrementally: small, explainable steps the committer can present (1.5 h final presentation).
- Correctness is a hard gate before timing; never loosen a tolerance after seeing a failing candidate.

**Dual-backend code (D-06).** `tests/test_backend_portability.py` has best-effort AST guards for rules 1 and 5,
tests the `--require-gpu` wiring of rule 2 and demonstrates rule 3; rules 2 (subprocess) and 4 are review-only.
1. **Never choose a backend in harness code** (root modules, `workloads/`, `scripts/`). It comes from
   `RunConfig.backend` through `environment.select_backend()`; code that only forwards that configured name (to a runner,
   the evaluator or a worker request) passes it as `selected_backend=`, never `backend=`. Outside `environment.py`: no `jax.devices("gpu")`,
   no `backend=` or `device=` arguments, no `jax.default_device`, no `jax.config.update("jax_platforms", ...)`, no
   `JAX_PLATFORMS`. To ask "is this a GPU?", use `environment.has_gpu(devices)`. Test setup (`tests/conftest.py`,
   `tests/backends.py`) and CI may set `JAX_PLATFORMS`.
2. **Tests run on CPU by default.** A test that only makes sense on a GPU is marked `@pytest.mark.needs_gpu`
   and runs JAX in a subprocess with `tests/backends.gpu_env()`. It is skipped without CUDA; on the NV host run
   `pytest --require-gpu` so a broken CUDA setup errors instead of skipping.
3. **Pallas: `interpret=jax.default_backend() == "cpu"`.** Hard-coded `False` does not compile on CPU; hard-coded
   `True` would time the interpreter on the GPU. Interpret-mode results are correctness evidence only.
   Exception: `scripts/probe_target.py` hard-codes both values on purpose, to probe each mode.
4. **One tolerance per workload, fixed before any search** (in `WorkloadSpec`, covered by `policy_hash`). If a
   backend needs a different one — GPU float32 matmuls may default to TF32 (to confirm in phase B) — declare it in
   the spec up front. Current workloads have no matmul and use the same tolerance on both backends.
5. **No speed claims in ordinary tests.** Checking that a timing exists (`> 0`, finite) is fine; comparing runtimes
   or speedups belongs in `slow` or `needs_gpu` tests. Mac numbers are never compared with NV-host numbers.

## 8. Verified pitfalls (keep current)
JAX 0.4.38:
- Selecting an unavailable backend: `cuda` raises a bare `AssertionError`; `gpu` blames ROCm. Select the backend
  **before any JAX computation** and **verify actual devices** — selecting after JAX initialised is **silently
  ignored** (stays on CPU).
- `compiled.cost_analysis()` returns a **list** (take `[0]`).
- CPU profiler traces contain executable-level events only (`TfrtCpuExecutable::Execute`), not per-op device time.
  The probe classifies this as `CPU_EXECUTION_EVENTS` (granularity `executable`); confirmed on the Mac (arm64, 208 events).
  Two harmless `Can't import tensorflow.python.profiler.trace` log lines are expected.
- Use `environment.select_backend(name)` with `cpu|cuda|rocm|tpu` (never `gpu`); it raises `BackendUnavailable`
  and verifies the actual device platform. Tests that select a backend must run in a subprocess.
- On macOS `os.sched_getaffinity` does not exist: recorded under `unavailable`, never guessed.

Run configuration (AIC-002):
- `agent.main()` order matters: resolve config → select backend → `--require-gpu` check → key check → write
  `run_config.json` + `manifest.json` → build controller (execs the workload) → loop. Building the controller first
  would initialise JAX and make the backend selection silently ineffective.
- `bool` is a subclass of `int`: `contracts._coerce` rejects it (and floats) for int fields.
- Budgets are recorded, not enforced (AIC-013/015); only `max_candidates` (= `--iterations`) changes behaviour.
  `max_candidates = 0` makes no provider request.
- `CLAUDE_MODEL` sits between `--model` and the config file in precedence.

Workloads (AIC-003):
- Inputs come only from `workloads/registry.py` (NumPy `default_rng(seed)`, float32); agent-visible reference
  sources (`workloads/softmax.py`) contain no inputs or seeds. Edge cases use the fixed seed 12345.
- The float64 oracle (`workloads/oracles.py`) validates the reference only; candidates are compared with the
  jit-compiled reference (AIC-004). JAX x64 is off, so the oracle runs in NumPy.
- The registry is wired into the run path: `--workload softmax|example_add` (or `workload_id` in a config file) selects a registered workload; `--source/--function` overrides it with a non-scientific `registry.legacy_source_spec`. An unknown workload id is a config error.

Correctness (AIC-004):
- `check_correctness` compares candidates with the **jit-compiled frozen reference**, not the oracle. A failing or
  non-finite reference raises `ReferenceInvalid` (harness bug, never a candidate failure).
- Categories: `COMPILE_FAILED`, `EXECUTION_FAILED`, `STRUCTURE_MISMATCH` (tree, shape, dtype — bf16 for f32 fails),
  `NONFINITE`, `NUMERIC_MISMATCH`. A compile failure is not retried for the same shape/dtype; other shapes are.
- `build_cases(..., include_heldout_shapes=True)` is a smoke test only, never AIC-018 held-out evidence.
- Relative error uses denominator `max(|e|, max(1e-3·max|e|, tiny))`; it is reported only, not used for the verdict.
- `policy_hash` covers tolerance, non-finite policy, case names, workload id and reference sha256.
- Pallas on CPU: interpret mode only (`ValueError` otherwise) — not native-kernel evidence.
- Legacy `benchmark.evaluate` (in-process, accepts noise: same code 4.80 / 4.00 / 4.03 µs) is kept for its tests only; the controller no longer uses it. Since AIC-006/009 the controller evaluates through `Evaluator` + a runner and accepts only via `judge()` (CI lower bound > 1 + δ, confirmed in a fresh session).
- Since AIC-004, `benchmark.evaluate` rejects non-finite output even when the reference is non-finite too
  (`error` = "reference produced non-finite values; …"), and supports PyTree outputs; timing now uses
  `jax.block_until_ready(out)` so tuple/dict outputs can be timed. The five legacy result keys keep their meaning;
  a size-bounded `correctness` key is added (it reaches the LLM via `get_history`; never inputs or seeds).
- `exp(x - logsumexp(x))` is a valid softmax rewrite but fails rtol 1e-5 when logits are offset by ≥ 1000.
- `controller.agent_loop` passes the **same** `messages` list object on every API call (mutated in place);
  `tests/fakes.py` stores a snapshot per call.

Tests (AIC-001):
- Only `tests/` is collected (`testpaths`). Importing a root `test*.py` creates an Anthropic client and blocks on `input()`.
- `agent.main()` loads `.env.local` from the cwd: tests use the `no_key` fixture (key removed, cwd = `tmp_path`).
- pytest exits with code 5 when it collects no tests; the anyio plugin is disabled (`-p no:anyio`) to avoid a pytest 9.2 warning.

Dual backend (D-06):
- `conftest.py` forces `JAX_PLATFORMS=cpu` (assignment, not `setdefault`), so the pytest process runs on CPU even
  on the NV host and even if the shell exports `JAX_PLATFORMS=cuda`. GPU checks run in a child process without it.
- `needs_gpu` skipping is done by `conftest.pytest_runtest_setup` (one CUDA probe subprocess per session, only when
  a `needs_gpu` test runs); a marker alone skips nothing. The skip is reported at `conftest.py`, not at the test.
- `Device.platform` for CUDA may be `gpu` or `cuda` (phase B confirms which); never compare it with a literal.

WSL2 + CUDA (NVIDIA CUDA-on-WSL guide; to be confirmed on the host):
- Install only the Windows NVIDIA driver; **never** a Linux driver inside WSL; avoid the `cuda`/`cuda-drivers` apt packages.
- `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi` and often not on PATH; many NVML queries are unsupported; GPU clocks cannot be locked.
- GPU profiler events need CUPTI, which under WSL2 requires a Volta-or-newer GPU.
- Keep the repo inside the WSL filesystem (`~/...`), not under `/mnt/c/...` (slow cross-filesystem I/O).
- Mac CPU and WSL2 GPU results are different measurement populations — never compare them.

Evaluation pipeline (AIC-005 … 009):
- `controller.Evaluator`: frozen reference outputs (trusted) → candidate outputs in a worker → host comparison
  (`benchmark.report_from_outputs`) → lazy A/A calibration (or a frozen `--calibration` file, checked) → `judge()`.
  The controller never executes candidate code; `Controller.__init__` has no side effects, `start()` runs the baseline.
- Runners: `--runner docker` (default; the only path for model-written code) or `--runner subprocess` (trusted code
  only: model candidates end the run as INTERRUPTED). Docker unavailable → exit 2 `ISOLATION_UNAVAILABLE` before any
  LLM call. Build the image with `uv run --locked python scripts/build_worker_image.py`; its tag is a content hash.
- On a busy Mac, tiny workloads (x + y, µs per call) usually fail the 10 % / 5 % instability limits: the run then
  stops with "no usable A/A calibration" instead of guessing a threshold. That is the protocol working.
- A measurement population is device + JAX version + runner/image: a change mid-run stops the run (INTERRUPTED).

## 9. Change log (newest first, one line per push)
- 2026-11-10 Sergio — P8: `gemini.py` (Gemini on Vertex AI behind the Anthropic Messages surface: request/response translation, function responses named by call id, thought signatures kept, 429/5xx retried); `google-genai` dependency; controller loop runs unchanged on it.
- 2026-11-10 Sergio — P8: `providers.py` + `--provider anthropic|anthropic-vertex|gemini-vertex`, `model.project/region`; anthropic (default) unchanged; anthropic-vertex = `anthropic.AnthropicVertex` billed to the GCP project via ADC; per-provider default model; `.env.example` GCP variables.
- 2026-11-10 Sergio — AIC-003: `agent.py --workload softmax|example_add` (→ `RunConfig.workload_id`; `--source` overrides); unknown workload id is a config error.
- 2026-11-10 Sergio — AIC-002: manifest `scientific` only with explicit backend + scientific workload + isolated runner (`non_scientific_reasons` lists what is missing); `run_config.json` carries `correctness_policy_hash`; manifest records workload hashes and held-out commitment.
- 2026-11-10 Sergio — AIC-018: softmax held-out split v1 frozen (commitment b2871080… in `workloads/registry.py`; salt kept outside the repo until the final results).
- 2026-11-10 Sergio — AIC-018: `heldout.py` + `scripts/heldout.py` (commit-reveal: secret salt in git-ignored `heldout/`, sha256 committed; ≥ 20 seeds disjoint from search/edge seeds × 2 unseen shapes; host-side verdict; burn ledger `evidence/heldout/ledger.jsonl`); `WorkloadSpec.heldout_commitment/heldout_version`.
- 2026-11-10 Sergio — AIC-010: worker outputs run returns compiler feedback for the measurement case (raw levels in `ir/`); recorded for baseline and candidates; shown to the agent only under C3/C4/K; `feedback.py` added to the worker image.
- 2026-11-09 Sergio — AIC-010: `feedback.py` (jaxpr / StableHLO / HLO / compiled op counts and groups, versioned parser, provenance; cost and memory estimates in separate namespaces; null + reason when unavailable; capped raw artifacts).
- 2026-11-09 Sergio — batch 02 report in `docs/IMPLEMENTATION_PROGRESS.md`; P12–P15 defaults; `scripts/phase_b.sh` (all NV-host steps in one command); WSL2 doc §5 Docker + NVIDIA Container Toolkit; GPU containment test (`needs_gpu`).
- 2026-11-09 Sergio — AIC-005: `scripts/calibrate_aa.py` freezes an A/A calibration per workload and population (exit 1 on a noisy host); first Mac Docker trial: softmax δ = 0.076 (not committed: official δ comes from the NV host).
- 2026-11-09 Sergio — AIC-009: controller rewritten around `Evaluator` (reference → isolated candidate outputs → host comparison → A/A calibration → `judge()`); `agent.py --runner docker|subprocess`, `--calibration`; ISOLATION_UNAVAILABLE exits before any LLM call; Attempt v1 records with raw blocks in history; configured backend forwarded as `selected_backend=` (rule 1).
- 2026-11-09 Sergio — AIC-009: CI job `containment` (builds the CPU worker image, runs `-m needs_docker --require-docker`); the `test` job excludes needs_docker.
- 2026-11-09 Sergio — AIC-009: `runner.DockerRunner` (no network, read-only root, no capabilities, no-new-privileges, nobody, CPU/memory/PID limits, tmpfs scratch, only the work dir mounted, OOM → RESOURCE_LIMIT, verified removal); `needs_docker` marker + `--require-docker`; `tests/test_containment.py` hostile probes pass on Docker Desktop.
- 2026-11-09 Sergio — AIC-009: `docker/worker.Dockerfile` (locked deps via uv, harness modules only, user nobody; `EXTRA=cuda12` GPU variant) and `scripts/build_worker_image.py` (minimal build context; tag = content hash from `runner.image_reference`).
- 2026-11-09 Sergio — AIC-008: `benchmark.compare_flat` / `report_from_outputs` (host-side verdict on worker outputs; the candidate never sees reference outputs or tolerances); the 18-fixture matrix reproduced through the worker case by case.
- 2026-11-09 Sergio — AIC-008: `runner.py` (`execute`: own session, deadline, process-group cleanup, bounded log tails via temp files; result validation; env allowlist without credentials or JAX_PLATFORMS) and `TrustedSubprocessRunner` (trusted code only; model code → ISOLATION_UNAVAILABLE).
- 2026-11-09 Sergio — AIC-008: `worker.py` (worker_request/v1 → worker_result/v1; backend via `select_backend` first; modes `outputs` and `measure`; raw outputs and raw blocks only, never a verdict).
- 2026-11-09 Sergio — AIC-008: `array_io.py` (npz with allow_pickle=False + JSON index of tree structure, leaf paths, dtypes, shapes; rejects undeclared arrays, changed dtypes, object arrays, symlinks, paths outside the work dir, oversize files).
- 2026-11-09 Sergio — AIC-007: `benchmark_code` takes a Proposal v1; `Controller.submit_proposal` records PROPOSAL_INVALID / POLICY_REJECTED without running the code; `tests/fakes.proposal` helper.
- 2026-11-09 Sergio — AIC-007: `source_policy.py` (AST allowlist: jax/jnp/lax/nn/scipy/math/functools imports, defs, assignments; rejects dunders, dynamic code, file/process/callback/device/config access, module mutation); all 18 fixtures pass it.
- 2026-11-09 Sergio — AIC-007: `contracts.Proposal` v1 + `parse_proposal` (strict, size-limited, no paths or evaluation parameters), `Evaluation` v1, `Attempt` v1 (host-built, terminal states only); `ContractError` base class.
- 2026-11-09 Sergio — AIC-006: terminal states in `contracts.py`; `controller.decide` (stable + CI lower bound > 1 + δ), `controller.judge` (one retry, fresh confirmation), `controller.verify_winner` (≥ 3 sessions vs the original baseline on every shape).
- 2026-11-09 Sergio — AIC-005: `benchmark.compile_timed` (ahead-of-time lower + compile, timed separately, `Lowered` kept for AIC-010).
- 2026-11-09 Sergio — AIC-005: `benchmark.aa_threshold` (δ = max(p95 of A/A median differences, timer floor); unstable A/A fails) and `benchmark.Calibration` (aa_calibration/v1 record, `check()` refuses another workload/shape/protocol/population).
- 2026-11-09 Sergio — AIC-005: `benchmark.stability` (IQR/median > 10 %, baseline drift > 5 %) and `benchmark.analyze_session` (host-side analysis of raw blocks; refuses a different protocol hash).
- 2026-11-09 Sergio — AIC-005: `benchmark.summarize` (median, mean, std, quartiles, min/max, IQR/median) and `benchmark.paired_speedup` (geometric-mean ratio, fixed-seed bootstrap over paired log ratios by block).
- 2026-11-09 Sergio — AIC-005: `benchmark.measure_paired` (balanced seeded AB/BA blocks, frozen repeat count, raw `Block` records), `PairedSession` JSON round-trip.
- 2026-11-09 Sergio — AIC-005: `benchmark.warm_up`, `benchmark.calibrate_repeats` (repeat count from the baseline only, 10–50 ms blocks, capped); fake-clock tests in `tests/test_measurement.py`.
- 2026-11-09 Sergio — AIC-005: `contracts.MeasurementProtocol` (paired_blocks/v1: warmup ≥ 5, 10–50 ms blocks, ≥ 20 paired blocks, seeded order and bootstrap, ≥ 5 A/A sessions, 1 retry, ≥ 3 final sessions), `validate_protocol`, `protocol_hash`; `RunConfig.measurement`.
- 2026-11-09 Harry — pushed `harry/batch-01` (batch 01 + D-06) for the team to run (D-07); first CI run passed (179 passed, 1 `needs_gpu` skipped); CI result recorded in `docs/IMPLEMENTATION_PROGRESS.md`; phase B still pending.
- 2026-11-06 Harry — D-06 dual-backend rules (§7, `CLAUDE.md`); `needs_gpu` auto-skip and `pytest --require-gpu` in `tests/conftest.py`; forced CPU pin; `environment.has_gpu` (used by `agent.py --require-gpu`); guard tests `tests/test_backend_portability.py`, revised after review.
- 2026-11-06 Harry — batch 01 report in `docs/IMPLEMENTATION_PROGRESS.md` (status AWAITING_ACCEPTANCE); phase A passed on the Mac; phase B (NV host) pending.
- 2026-11-06 Harry — AIC-004: `benchmark.py` PyTree comparator, `build_cases`, `check_correctness`, result types and policy hash; legacy `evaluate()` uses it (inf vs inf now rejected; PyTree outputs supported); softmax fixture matrix in `tests/fixtures/`.
- 2026-11-06 Harry — AIC-003: `workloads/registry.py` (WorkloadSpec v1, `softmax` + legacy `example_add`), `workloads/softmax.py` reference, `workloads/oracles.py` float64 oracle; softmax `OFFSET = 200` (P5 default).
- 2026-11-06 Harry — AIC-002: `contracts.py` (RunConfig v1, strict parsing, CLI > file > defaults, sha256 hash); `agent.py --backend/--config`, writes `run_config.json` + `manifest.json` before any LLM call; `--iterations` default 10 → 3.
- 2026-11-06 Harry — AIC-031 (Mac part): `environment.py` backend selection + environment capture; `scripts/probe_target.py` (`target_probe/v1`); `evidence/targets/mac-cpu.json`; `docs/WSL2_CUDA_SETUP.md`; `docs/TARGET_DECISION.md` (DRAFT).
- 2026-11-06 Harry — AIC-001: pytest dev dependency, offline test suite (`tests/`), fake Anthropic client, GitHub Actions CI; `.env`/`.env.save` untracked, `.env.example` added; `.DS_Store` git-ignored.
- 2026-11-06 Harry — created this file; process documents recorded as not yet existing; English-only rule; branch `harry/batch-01`; NV host is Windows (WSL2 path); personal plans in `docs/plans/` are git-ignored.
