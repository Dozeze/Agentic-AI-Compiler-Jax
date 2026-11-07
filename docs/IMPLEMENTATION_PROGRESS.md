# Implementation progress

Evidence log: what each task did, the commands run, their actual output, and deviations.
The current-state snapshot is `AI_CONTEXT.md`.

## Batch 01 (AIC-001, 031, 002, 003, 004) · 2026-11-06 · owner: Harry · status: AWAITING_ACCEPTANCE

Branch `harry/batch-01` (from `main` at `51df747`), local commits only, not pushed. Implemented in one wave on
Harry's Mac (macOS 15 / Darwin 24.6.0, arm64, Python 3.12.16, jax/jaxlib 0.4.38, numpy 2.5.2, uv 0.13.7).

### Summary
- An offline pytest suite (136 tests, no key, no network) with a fake Anthropic client and Linux CPU CI.
- `environment.py` + `scripts/probe_target.py`: explicit backend selection that never falls back silently, and
  `target_probe/v1` evidence; Mac CPU evidence recorded; the CUDA/WSL2 target decision is a DRAFT awaiting phase B.
- `contracts.py` RunConfig v1; `agent.py --backend/--config` writes `run_config.json` + `manifest.json` before any LLM call.
- `workloads/registry.py` WorkloadSpec v1 with a seeded softmax workload, float64 oracle and legacy `example_add`.
- `benchmark.py` PyTree correctness gate with separate failure categories; an 18-fixture correctness matrix.

### Per task

#### AIC-001 — Offline reproducible test foundation
- Commit: `289e613`
- Changed files: `.env` and `.env.save` (untracked, local files kept), `.env.example`, `.github/workflows/ci.yml`,
  `.gitignore`, `AI_CONTEXT.md`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, `tests/conftest.py`, `tests/fakes.py`,
  `tests/test_foundation.py`
- Gate: `uv run --locked pytest -q` → `13 passed in 13.23s`; same with `HTTPS_PROXY=http://127.0.0.1:9` and
  `uv run --offline` → `13 passed in 6.01s`; `env -u ANTHROPIC_API_KEY uv run --locked python agent.py --help` → exit 0;
  `git ls-files .env .env.save` → no output.
- Deviations from the batch plan:
  - `.DS_Store` added to `.gitignore` (untracked Finder files would otherwise keep the tree dirty for phase A).
  - `MPLBACKEND=Agg` set in CI pre-emptively (plan key check #8 suggested it only if CI complains).
  - `FakeAnthropic.__init__` accepts and ignores keyword arguments, so the AIC-002 driver can subclass it.
  - `uv` was not installed on the Mac; installed with Homebrew (`uv 0.13.7`) with Harry's approval.
- Known limitations: CI has not run yet (nothing is pushed); `astral-sh/setup-uv@v6` and `actions/checkout@v4`
  were confirmed to exist with `git ls-remote --tags`.

#### AIC-031 — Target selection (Mac part)
- Commit: `15b47de`
- Changed files: `environment.py`, `scripts/probe_target.py`, `evidence/targets/mac-cpu.json`,
  `docs/WSL2_CUDA_SETUP.md`, `docs/TARGET_DECISION.md`, `tests/test_environment.py`, `tests/test_probe_target.py`,
  `AI_CONTEXT.md`
- Gate: `uv run --locked pytest -q tests/test_environment.py tests/test_probe_target.py` → 12 passed, 1 skipped
  (`test_cuda_on_host`: "no CUDA here: AssertionError"); full suite → `25 passed, 1 skipped in 18.70s`;
  `scripts/probe_target.py --backend cpu --label mac-cpu --out evidence/targets` → `cpu: AVAILABLE`.
- JAX behaviour re-confirmed on the Mac: `cuda` → bare `AssertionError`; `rocm` → `RuntimeError: Unable to initialize
  backend 'rocm'`; `tpu` → `Failed to open libtpu.so`; `jax.devices()` first, then selecting `cuda` → silently stays
  on CPU; `cost_analysis()` → `list`; `os.sched_getaffinity` absent on macOS.
- Mac CPU probe: IR APIs all ok (jaxpr 17 lines, StableHLO 16, HLO 33, compiled 65, flops 524032);
  profiler `CPU_EXECUTION_EVENTS` (granularity executable, 208 complete events, no device lanes); Pallas native →
  `ValueError: Only interpret mode is supported on CPU backend.`, interpret mode correct; `jax.extend.ffi` available
  with headers; Apple clang 17; no Docker daemon, no cgroup v2, no user namespaces; 16 GiB RAM; dispatch probe
  median ≈ 3.7 µs (informative only).
- Deviations from the batch plan:
  - `collect_environment` also records `repo.on_windows_mount` (plan key check #11) and lists a failed
    `git rev-parse` under `unavailable`; the probe's `isolation` section records `cwd_on_windows_mount`.
  - Added `test_classify_trace_rules`, a unit test of the profiler classification with synthetic device-lane events.
  - The probe skips `nvidia_container` unless `--try-docker-gpu` is given, and records why.
- Known limitations: `evidence/targets/nvhost-wsl2-cuda.json` does not exist yet (phase B). The device-lane marker
  strings of profiler rule 2 and `device.platform` under CUDA are unverified. `mac-cpu.json` contains the Mac's
  hostname (P11 default) and was captured with a dirty tree at commit `289e613` (the probe ran before its own commit).

#### AIC-002 — RunConfig v1 and environment manifest
- Commit: `9e3fa1f`
- Changed files: `contracts.py`, `agent.py`, `README.md`, `how_to_run.md`, `tests/test_contracts.py`,
  `tests/test_agent_cli.py`, `AI_CONTEXT.md`
- Gate: `uv run --locked pytest -q tests/test_contracts.py tests/test_agent_cli.py` → 33 passed; full suite →
  `58 passed, 1 skipped in 44.97s`. A fake-client run directory was checked by hand: `run_config.json` readable;
  `manifest.json` has versions, devices, lock and pyproject hashes, git commit and dirty flag, and only the
  allowlisted variables.
- Deviations from the batch plan:
  - `--function`, `--output-dir`, `--model` argparse defaults are now `None` (as for `--iterations`), so a config
    file can set them; the effective defaults are unchanged (`bad_func`, `runs/latest`, the Haiku model id).
  - `CLAUDE_MODEL` is placed between `--model` and the config file in precedence.
  - `--require-gpu` with backend `tpu` is also an argument error (not only `cpu`).
  - `max_candidates == 0` skips the agent loop, so no provider request is made (previously `--iterations 0` made one).
  - `atomic_write_json` lives in `agent.py`; `contracts.from_dict` added for strict round-trips.
  - Extra validation: `benchmark.num_runs/num_tests` and `model.max_tokens` must be positive.
  - Extra tests: config file used, bad config exits 2 without traceback, zero candidates makes no provider request,
    float fields accept ints, invalid split/context/seeds/schema rejected, invalid JSON is a `ConfigError`.
- Known limitations: budgets other than `max_candidates` are recorded only (AIC-013/015). The manifest marks a run
  `scientific` purely on an explicit backend, even for the non-scientific `example_add` workload and the in-process
  runner (`runner: in-process/v0` is recorded). `correctness_policy_hash` stays null until the registry is wired in (M3).

#### AIC-003 — Seeded softmax workload and registry
- Commit: `1041a95`
- Changed files: `workloads/registry.py`, `workloads/softmax.py`, `workloads/oracles.py`, `tests/test_workloads.py`,
  `AI_CONTEXT.md`
- Gate: `uv run --locked pytest -q tests/test_workloads.py` → `38 passed in 3.52s`; full suite →
  `96 passed, 1 skipped in 46.02s`. `workloads/example.py` unchanged.
- Input regression hash (seed 0, shape (32,128), computed on the first run, not invented):
  `afb9825a6df7150ed3c35dbaed712813eced3e3fd34175de6ef3db4921f6127a`.
- Deviations from the batch plan:
  - The `example_add` float64 oracle is `oracles.add_f64` (instead of a bare `np.add`), so it upcasts like the
    softmax oracle.
  - Extra test `test_reference_source_has_no_inputs`.
  - `workloads/` stays a namespace package (no `__init__.py`), matching `main`.
- Known limitations: tolerances are AIC-1.1 starting values (provisional) and must be frozen before any search.

#### AIC-004 — Correctness checking
- Commit: `119f084`
- Changed files: `benchmark.py`, `tests/fixtures/__init__.py`, `tests/fixtures/softmax_candidates.py`,
  `tests/test_correctness.py`, `AI_CONTEXT.md`
- Gate: `uv run --locked pytest -q tests/test_correctness.py` → `40 passed in 4.07s`; full suite →
  `136 passed, 1 skipped in 46.34s` (all AIC-001 characterization tests still pass).
- Correctness matrix (26 cases: 10 development, 6 edge, 10 held-out-shape smoke; 2.9 s for all 18 fixtures on CPU):

  | fixture | passed | category | failed cases |
  |---|---|---|---|
  | REFERENCE, JAX_NN, RECIPROCAL_MUL, LOGSUMEXP, NUMERIC_TINY_OK | yes | — | 0/26 |
  | WRONG_TREE, EXTRA_LEAF, WRONG_SHAPE, WRONG_DTYPE_BF16, WRONG_DTYPE_INT | no | STRUCTURE_MISMATCH | 26/26 |
  | WRONG_AXIS, NUMERIC_OFFSET | no | NUMERIC_MISMATCH | 26/26 |
  | NAIVE_UNSTABLE | no | NONFINITE | 3/26 (edge cases only) |
  | INJECT_NAN | no | NONFINITE | 26/26 |
  | OVERFIT_SHAPE | no | NUMERIC_MISMATCH | 15/26 |
  | OVERFIT_VALUE | no | NUMERIC_MISMATCH | 20/26 (seed 0 and the zeros/equal-logits edges pass) |
  | COMPILE_ERROR, TRACE_ERROR | no | COMPILE_FAILED | 26/26 |

  Largest report: 14 095 bytes (limit 16 KiB). `EXECUTION_FAILED` is tested by substituting `benchmark._compile`.
- Intentional behaviour changes of `evaluate()` (the five legacy keys keep their meaning):
  1. reference `inf` and candidate `inf` → now `output_correct=False`, `error` "reference produced non-finite values;
     workload contract must declare masks" (was `True`);
  2. PyTree outputs are supported (a tuple output used to fail with an `AttributeError` inside the try);
  3. new key `correctness` (size-bounded report, `None` when the comparison is not reached).
- Deviations from the batch plan:
  - `timeit_version_1` now synchronises with `jax.block_until_ready(out)` instead of `out.block_until_ready()`
    (identical for arrays; needed so PyTree outputs can be timed). The timing protocol is otherwise unchanged.
  - `compare_outputs` returns a `CaseResult` (with an empty case name) and itself raises `ReferenceInvalid` on a
    non-finite reference, as a second guard; all leaves are compared, the case category is the first failing leaf's.
  - Held-out-shape smoke cases are named `heldout-smoke(...)/seedN` to make their status visible in reports.
  - Extra tests: `test_build_cases_counts`, `test_matrix_details`, `test_evaluate_supports_pytree_outputs`,
    `test_reference_nonfinite_raises_in_compare`, `test_report_contains_no_inputs`.
- Known limitations: candidates are still `exec`'d in-process (AIC-007…009). `evaluate()` still uses a single input
  and the noisy acceptance rule (AIC-005/006).

### Defaults taken for [TEAM DECISION] items
| item | default used | where in code |
|---|---|---|
| P4 `--iterations` default | 3 (was 10); README and how_to_run note the change | `contracts.Budgets.max_candidates`, `agent.DEFAULT_ITERATIONS` |
| P5 softmax large offset | `OFFSET = 200.0` | `workloads/registry.py` (`OFFSET`, `SOFTMAX_EDGE_CASES`); `tests/test_workloads.py::test_logsumexp_rewrite_admitted` |
| P6 new `environment.py` module | yes (merge into `contracts.py` if the team objects; interface unchanged) | `environment.py` |
| P10 root scratch scripts | left untouched; isolated by `testpaths = ["tests"]`. Proposal: move `test.py`, `test1-3.py`, `testrun.py`, `runningscripts.py`, `utils.py` into `scratch/` (`test.py` prints every environment variable) | `pyproject.toml` `[tool.pytest.ini_options]` |
| P11 hostname in manifests | recorded as-is | `environment.collect_environment` → `platform.node`; also in `evidence/targets/mac-cpu.json` |

### M1 exit evidence (AIC-1.1 §10)
- AIC-001/031/002/003 done on the Mac: **yes** — commits `289e613`, `15b47de`, `9e3fa1f`, `1041a95`; full suite
  `136 passed, 1 skipped` offline; `evidence/targets/mac-cpu.json` recorded.
- Target availability and profiler/kernel/isolation feasibility: **pending phase B** on the NV host
  (`docs/WSL2_CUDA_SETUP.md` §4). `docs/TARGET_DECISION.md` is `DRAFT`.

### Phase A acceptance output (Mac)
`/tmp/probe` was replaced by a session scratch directory for the smoke probe; everything else as in §6.
```
$ git status --porcelain
(exit 0)
$ uv sync --locked
Resolved 78 packages in 5ms
Checked 61 packages in 7ms
$ env -u ANTHROPIC_API_KEY uv run --locked pytest -q
.........s.......................................................        [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_environment.py:87: no CUDA here: AssertionError
136 passed, 1 skipped in 44.98s
$ HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 env -u ANTHROPIC_API_KEY uv run --locked --offline pytest -q
136 passed, 1 skipped in 44.77s
$ env -u ANTHROPIC_API_KEY uv run --locked python agent.py --help
(exit 0)
$ uv run --locked python scripts/probe_target.py --backend cpu --label smoke --out <scratch>/probe-smoke
cpu: AVAILABLE -> <scratch>/probe-smoke/smoke.json
$ git ls-files .env .env.save docs/plans
(no output = ok)
$ git log --oneline main..harry/batch-01
119f084 AIC-004: PyTree correctness checking and fixture matrix
1041a95 AIC-003: seeded softmax workload and registry
9e3fa1f AIC-002: RunConfig v1 and environment manifest
15b47de AIC-031: backend selection, target probe and Mac CPU evidence
289e613 AIC-001: offline pytest foundation and fake provider
$ for c in $(git rev-list main..harry/batch-01); do ... grep -qx AI_CONTEXT.md || echo "missing context update: $c"; done
(no output = ok)
```
Per test file: foundation 13 passed (5.2 s); environment 8 passed, 1 skipped (8.3 s); probe_target 4 passed (4.0 s);
contracts 24 passed (0.04 s); agent_cli 9 passed (23.6 s); workloads 38 passed (2.8 s); correctness 40 passed (3.9 s).
Total under 2 minutes on CPU; no test is marked `slow`.

### Still to do
- §6 phase B on the NV host inside WSL2: `uv sync --locked --extra cuda12`, full suite with `--require-gpu`,
  `probe_target.py --backend cuda --label nvhost-wsl2-cuda`; then fill `docs/TARGET_DECISION.md`, confirm the
  CUDA `device.platform` value and profiler device-lane names (update `environment._ACCEPTED_PLATFORMS`,
  `probe_target._DEVICE_LANE_MARKERS` and `AI_CONTEXT.md` §8 if they differ), and add the results here.
- Team sign-off of `docs/TARGET_DECISION.md`; team confirmation of the defaults above (P4, P5, P6, P10, P11).
- First CI run: triggered by the push of `harry/batch-01` on 2026-11-07 (D-07); result to be added here.

## Post-batch 01: dual-backend rules (D-06) · 2026-11-06 · owner: Harry · status: AWAITING_REVIEW

Team decision: all code and tests must run on both CPU and CUDA so every member can test locally. Rules are in
`AI_CONTEXT.md` §7 (summary in `CLAUDE.md`). Changes:
- `tests/conftest.py`: `needs_gpu` tests are now really skipped without CUDA (before, the marker alone skipped
  nothing and the one GPU test skipped itself); new `pytest --require-gpu` makes them error instead, for the NV
  host. `JAX_PLATFORMS=cpu` is now forced, not `setdefault`.
- `tests/backends.py`: `gpu_env()` and a cached `cuda_status()` probe (one child process per session; a crash
  reports the last stderr line).
- `environment.has_gpu(devices)`: accepts every GPU platform name (`gpu`, `cuda`, `rocm`); `agent.py --require-gpu`
  and `tests/test_foundation.py` use it instead of `platform == "gpu"`.
- `tests/test_backend_portability.py`: best-effort AST guards for rules 1 and 5, each first run on bad and good
  snippets; `--require-gpu` error path; a Pallas softmax candidate with `interpret=jax.default_backend() == "cpu"`
  passes the correctness gate on CPU, while `interpret=False` gives `COMPILE_FAILED`
  ("Only interpret mode is supported on CPU backend").

Review (separate session, report `D-06-dual-backend-review.md`, not committed): no blockers; all findings were
reproduced and fixed:
- S1: forced CPU pin. Before the fix, `JAX_PLATFORMS=cuda pytest -k pallas` failed 2 tests; now 8 pass.
- S2, S10: wording: which rules are guarded, rule 1 scope, `--require-gpu` errors rather than fails.
- S3: the rule-5 guard checks each pair of a chained comparison; `> 0` is allowed only on a single value.
- S4: the rule-1 guard matches the last name segment and `backend=` / `device=` on any call.
- S5: the rule-5 guard adds `*_s` / `*_time` names, `== pytest.approx`, class and module `pytestmark` markers;
  ignores `len(...)` and `timeout` names.
- S6: crash reason keeps stderr.
- S7: one subprocess run instead of two.
- S8: case-count asserts.
- S9: `has_gpu`.
Known limits: asserts in helper functions and `if ...: raise AssertionError` are not inspected; rule 4 has no
machine check; the TF32 default on GPU is to be confirmed in phase B.

Mac result after the fixes: `uv run --locked pytest -q -rs` → 179 passed, 1 skipped
(`needs_gpu: CUDA unavailable (AssertionError: <no message>)`), 50 s;
`pytest --require-gpu tests/test_environment.py::test_cuda_on_host` → 1 error (exit 1), as intended.
