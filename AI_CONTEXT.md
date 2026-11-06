# AI_CONTEXT — shared context for anyone (human or AI) working on this repo

> **Team rules (meeting, Nov 2026):**
> 1. If you use an AI tool on this project, it must read this file first, and **every change you push must update
>    this file in the same commit** — at least §5 Status, §6 Decisions and §9 Change log. If your change makes
>    anything here wrong, fix it.
> 2. **All documentation and code comments are in English.**
>
> Keep this file short: it is the current state, not a history. Detailed evidence goes in
> `docs/IMPLEMENTATION_PROGRESS.md`; the full plan is `docs/IMPLEMENTATION_PLAN.md`.

Last updated: 2026-11-06 by Harry (Weizhuo Sun)

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
- GPU (inside WSL2): `uv sync --locked --extra cuda12 && uv run --locked --extra cuda12 pytest -q`.
- Run (needs a key in `.env.local`): `uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1` (`--config run.json` for a RunConfig v1 file)

## 5. Status (update every push)
| Milestone | Tasks | Owner | Status |
|---|---|---|---|
| M1 Foundations & target | AIC-001, 031, 002, 003 | Harry | in progress on `harry/batch-01`: AIC-001 done locally (offline pytest suite, fake Anthropic client, CI); AIC-031 Mac part done (`environment.py`, `scripts/probe_target.py`, `evidence/targets/mac-cpu.json`, `docs/TARGET_DECISION.md` **DRAFT**) — NV-host probe (phase B) pending; AIC-002 done locally (`contracts.py` RunConfig v1, `--backend`/`--config`, `run_config.json` + `manifest.json`); AIC-003 done locally (`workloads/registry.py` WorkloadSpec v1, seeded `softmax`, float64 oracle, legacy `example_add`) |
| M2 Reliable, isolated evaluation | AIC-004 … 009 | AIC-004: Harry; rest TBD | not started |
| M3 – M8 | AIC-010 … 033 | TBD | not started |

Course deadlines (fixed by KTH, not by AIC-1.1): report draft **2026-11-20 17:00**; presentation + final report
**2026-12-15 … 21**.

## 6. Decisions
**Decided**
- D-01 Work in batches of five tasks; batch 01 = AIC-001, 031, 002, 003, 004 (Harry).
- D-02 Primary target: CUDA on an NVIDIA GPU (Harry's NV host). Mac is for development only.
- D-03 AI use is allowed with this file kept up to date.
- D-04 English only for documentation and code comments.
- D-05 The NV host runs Windows with an NVIDIA RTX 40-series GPU (Ada, newer than Volta, so CUPTI profiling under WSL2 is supported); the GPU path is WSL2 Ubuntu + Windows NVIDIA driver (no Linux driver inside WSL). Exact model and driver version are recorded by the AIC-031 probe.

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

## 7. Rules for code changes
- One branch per person/batch; never push straight to `main`. One task per commit/PR; message starts with the task ID (`AIC-001: ...`).
- Every PR: tests pass offline (`uv run --locked pytest -q`), this file updated, entry in `docs/IMPLEMENTATION_PROGRESS.md`.
- English only for docs, comments, docstrings, test names and commit messages.
- **The repo is public:** never commit keys, `.env*` (except `.env.example`), `runs/`, large traces.
- Never dump `os.environ`; record only allowlisted variables.
- Use AI incrementally: small, explainable steps the committer can present (1.5 h final presentation).
- Correctness is a hard gate before timing; never loosen a tolerance after seeing a failing candidate.

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
- The registry is not wired into `agent.py` yet (M3); `--source/--function` remains the run path.
- Pallas on CPU: interpret mode only (`ValueError` otherwise) — not native-kernel evidence.
- Current `benchmark.evaluate` accepts a candidate that is faster only by noise (same code: 4.80 / 4.00 / 4.03 µs) — fixed by AIC-005/006, not before.
- Current `benchmark.evaluate` treats reference `inf` == candidate `inf` as correct — fixed by AIC-004.
- `exp(x - logsumexp(x))` is a valid softmax rewrite but fails rtol 1e-5 when logits are offset by ≥ 1000.
- `controller.agent_loop` passes the **same** `messages` list object on every API call (mutated in place);
  `tests/fakes.py` stores a snapshot per call.

Tests (AIC-001):
- Only `tests/` is collected (`testpaths`). Importing a root `test*.py` creates an Anthropic client and blocks on `input()`.
- `agent.main()` loads `.env.local` from the cwd: tests use the `no_key` fixture (key removed, cwd = `tmp_path`).
- pytest exits with code 5 when it collects no tests; the anyio plugin is disabled (`-p no:anyio`) to avoid a pytest 9.2 warning.

WSL2 + CUDA (NVIDIA CUDA-on-WSL guide; to be confirmed on the host):
- Install only the Windows NVIDIA driver; **never** a Linux driver inside WSL; avoid the `cuda`/`cuda-drivers` apt packages.
- `nvidia-smi` is at `/usr/lib/wsl/lib/nvidia-smi` and often not on PATH; many NVML queries are unsupported; GPU clocks cannot be locked.
- GPU profiler events need CUPTI, which under WSL2 requires a Volta-or-newer GPU.
- Keep the repo inside the WSL filesystem (`~/...`), not under `/mnt/c/...` (slow cross-filesystem I/O).
- Mac CPU and WSL2 GPU results are different measurement populations — never compare them.

## 9. Change log (newest first, one line per push)
- 2026-11-06 Harry — AIC-003: `workloads/registry.py` (WorkloadSpec v1, `softmax` + legacy `example_add`), `workloads/softmax.py` reference, `workloads/oracles.py` float64 oracle; softmax `OFFSET = 200` (P5 default).
- 2026-11-06 Harry — AIC-002: `contracts.py` (RunConfig v1, strict parsing, CLI > file > defaults, sha256 hash); `agent.py --backend/--config`, writes `run_config.json` + `manifest.json` before any LLM call; `--iterations` default 10 → 3.
- 2026-11-06 Harry — AIC-031 (Mac part): `environment.py` backend selection + environment capture; `scripts/probe_target.py` (`target_probe/v1`); `evidence/targets/mac-cpu.json`; `docs/WSL2_CUDA_SETUP.md`; `docs/TARGET_DECISION.md` (DRAFT).
- 2026-11-06 Harry — AIC-001: pytest dev dependency, offline test suite (`tests/`), fake Anthropic client, GitHub Actions CI; `.env`/`.env.save` untracked, `.env.example` added; `.DS_Store` git-ignored.
- 2026-11-06 Harry — created this file; process documents recorded as not yet existing; English-only rule; branch `harry/batch-01`; NV host is Windows (WSL2 path); personal plans in `docs/plans/` are git-ignored.
