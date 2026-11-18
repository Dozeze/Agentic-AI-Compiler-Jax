# Implementation progress

Evidence log: what each task did, the commands run, their actual output, and deviations.
The current-state snapshot is `AI_CONTEXT.md`.

## AIC-009 fix and first phase-B run · 2026-11-18

- First phase-B run on the NV host (output pasted by the team; the log and `evidence/targets/nvhost-wsl2-cuda.json` are
  not in the repository yet). A first attempt failed in step 2: XLA called an older system `ptxas` (CUDA 12.0,
  PTX ISA 8.0) for PTX 8.3, in the first test that really compiles on the GPU
  (`test_cuda_profile_requires_real_device_execution_events`); the other GPU tests only select the device. After
  the host environment was fixed, a second run passed steps 1–5: GPU suite, target probe, worker images, CPU and
  GPU containment.
- Step 6 (A/A calibration, softmax (128, 1024), CUDA container) did not calibrate: IQR/median per session
  0.290/0.330, 0.152/0.144, 0.107/0.117, 0.145/0.190 (drift 0.116), 0.196/0.124 against limits 0.10 and 0.05. No
  official δ; the decision on the protocol is pending (`AI_CONTEXT.md` P26).
- The script then crashed while deleting its temporary directory: the container user (65534) had written
  `profile/perfetto_trace.json.gz` in a 0755 directory it created, which the host user cannot empty. Fix in
  `worker.share_with_owner`: when the worker does not own the work directory, it clears its umask, so everything it
  creates stays removable by the owner (the work directory itself was already 0777). Tests: a unit test of the
  ownership rule in `tests/test_worker.py`, and `test_runtime_profile_inside_the_container` now deletes the work
  directory after the container run (meaningful on Linux, where CI runs it; Docker Desktop on macOS hides the
  problem). `worker.py` is part of the image hash, so the worker images must be rebuilt.

## AIC-015–018 — GitHub handoff · 2026-11-17

The user requested pushing the accumulated implementation to the current branch, `sergio/batch-03`.
Publish the integrated, already verified code and documentation in one batch commit above `46f5089`;
shared-file changes remain in the exact tested snapshot. `AI_CONTEXT.md` records this handoff exception
to the usual task-per-commit convention. The final shared offline suite remains **781 passed / 2
host-CUDA skips / 20 target-sandbox deselected**; no implementation changed after that verification.
Local `runs/` evidence and credentials are excluded. This publication does not satisfy the outstanding
CUDA, target/live or milestone acceptance gates and authorizes no paid calls or provisioning.

## AIC-018 — Frozen final evaluation, release and replay · 2026-11-17 · status: BLOCKED for target acceptance

- Authorization: after AIC-017 and discussion of the outstanding target integration, the user explicitly
  requested the next task. Complete independent final-evaluation software/offline checks. Prior M3/M4,
  hardware and live spending gates remain open. Existing AIC-015–017 changes remain uncommitted above
  `46f5089` on `sergio/batch-03`; no commits/pushes, paid calls, provisioning or official split consumption.
  Starting shared suite: 749 passed / 2 host-CUDA skips / 17 target-sandbox deselected.
- Added `SplitCommitment` and a deterministic digest binding reference, input-recipe source, workload
  version, split seeds/shapes, development population, dtype/static semantics and numeric policy.
  CLI and direct-controller manifests save only the opaque commitment before search dispatch. The
  positive proposal-context allowlist still excludes final seeds/shapes/arrays/results. P16 public seed
  declarations remain unchanged; this is evaluation separation, not repository-secrecy protection.
- New `agent.py --finalize-run SEARCH_RUN`: requires a completed run's verified terminal best and its
  pre-search commitment. Missing old commitments, reference/input/split/lock drift or changed runner/
  image/device/JAX population refuse reuse. Inherit frozen config/protocol/budgets, never construct a
  provider or allow search overrides. Baseline fallback may run as trusted reference; model-written
  winners still require isolation. Python tests explicitly inject reviewed fixtures where needed.
- Reserve the final split once before final input publication/worker launch. Success, failed
  correctness, interruptions, correctness-only evaluation and partial publication all seal that version
  against further search/re-evaluation. Changing only the version cannot relabel exposed data: fresh
  seeds and shapes are required. Malformed/partial prior state fails closed. Legacy failure burns stay
  honored. Automatic reconciliation/resume remains AIC-019; tests use private state, not the official
  `evidence/heldout/publications/` registry.
- `finalization.py` uses actual evaluator workers for all forty reserved correctness cases per family.
  Keep every shape/case and block all timing on any correctness failure. Otherwise, default final mode
  records baseline-only A/A calibration and at least three fresh paired baseline/winner sessions on
  each unseen shape (timing uses its first reserved seed). Keep raw blocks, host uncertainty/stability,
  source identities and decisions; raw invalid blocks cannot grant a claim. Budget/time/noise failures
  remain explicit. A baseline fallback is valid but never promoted as a measured improvement.
- Final evidence is a new store: source snapshots, frozen parent selection/result, split release,
  per-shape results, raw workers/calibration, `final_report.json` and commit-last `release.json`. The
  original search best/source/scope are untouched. `storage.verified_final_result` and
  `scripts/heldout.py replay --run FINAL_RUN` verify hashes, complete coverage, native session links,
  recalculated analysis/decisions and derived flags without importing saved candidate code. This
  checks the recorded chain, not malicious replacement of every file. The old arbitrary-candidate
  correctness helper remains explicitly a legacy diagnostic, not canonical final research evidence.
- `--final-correctness-only` supports explicitly scoped correctness checks and discloses missing
  performance verification. It still consumes the split; documented rehearsal uses offline tests with
  private state. New contracts/commands/limitations are in `docs/FINAL_EVALUATION.md` and README.

### Verification and retained evidence

- First focused run: 18 passed / 10 failed. Native execution/results were correct; fixed the final
  writer returning an artifact reference instead of its report and an exception assertion in a test.
  Affected regression then passed 234 checks with two serialization-comparison failures: normalize the
  returned report to its persisted JSON form (calibration shape tuples become lists).
- Corrected finalization suite: **30 passed / 1 target deselected in 80.55 s**. Two later state/replay
  checks and three actual-target cases are included in final shared verification. New tests cover
  pre-dispatch commitment, all three native families, development-valid overfit, tampering of source/
  arrays/report/selection, no source imports in replay, sealed-state/fresh-version rules, inherited
  zero-compile caps, failed publication and provider-free CLI. Synthetic timing fixtures exercise
  stable/noisy/calibration-failed/incomplete-block behavior and remain non-scientific.
- Shared full offline command:
  `uv run --locked --offline pytest tests/ -m 'not target and not sandbox and not live' -q`
  → **781 passed, 2 host-CUDA skips, 20 target-sandbox deselected, 2 unchanged google-genai warnings
  in 387.09 s**. Full output: `runs/aic-018-native-validation-20261117/offline-test-output.txt`.
  Existing environment-only OpenSSL/cache workaround used; `uv.lock` and dependency pins unchanged.
- Native final-correctness evidence (private evaluation state, synthetic parent selection metadata,
  actual worker outputs, no synthetic timing), retained with hash review at
  `runs/aic-018-native-validation-20261117/validation.json`:
  softmax **20261117T005604.693407Z-cb46d3d7c1a4**, layer norm
  **20261117T005626.240974Z-1ed08367fa4a**, dense exact GELU
  **20261117T005638.261049Z-322ada418792**. Each passes **40 cases / 2 shapes / 20 seeds**;
  correctness-only release, no performance claim. Parent exports/source hashes remain unchanged.
- Reviewed shape-overfit fixture passes all development cases, then fails **40/40 final cases**, twenty
  on each unseen shape. Final ID **20261117T011416.622456Z-4a076732eb1f** under `overfit-final/`;
  status **CORRECTNESS_FAILED**, timing blocked for both shapes. Its parent selection metadata is
  deliberately synthetic and is not a live optimizer or speedup example.
- Additional actual baseline-only CLI + native final-timing rehearsal, private state:
  `runs/aic-018-native-measurement-20261117/validation.json`, final ID
  **20261117T111525.425711Z-c2c238e21e83**. Correctness passes; default A/A collects **ten actual raw
  sessions**, five per shape. Both shapes fail the unchanged IQR/drift checks, so status is
  **TIMING_INCOMPLETE**, both timings explicitly UNAVAILABLE and zero final comparison sessions.
  Retain the first noise refusal; no rerun, relaxed threshold, performance claim or provider request.
- Mandatory new target command:
  `AIC_TARGET_CONFIG=configs/target.json uv run --locked --offline pytest tests/test_holdout.py -m target -q`
  → **exit 1, 3 CUDA setup errors, 32 deselected, no skips, 3.24 s**. All three actual CUDA/container
  final-session tests remain unexecuted. No skipped mandatory check is counted as acceptance.

### Remaining acceptance and next work

Finalization/replay software and native CPU correctness are verified. **AIC-018 is not DONE for target
acceptance**, and earlier AIC-016/017/M3/M4 gates remain open. On the selected host, complete the live
MVP and reference/resource/calibration prerequisites, freeze selection, then run final correctness and
fresh per-shape comparisons without model access. Obtain the target/M4 review. Commit the new workload
declarations before research search; do not backfill missing commitments in old runs. Execution recovery
and general artifact-only reporting remain AIC-019 and AIC-020. No official evaluation version was used
by these tests/rehearsals, and no paid request was made.

## AIC-017 — Representative workloads and calibration handoff · 2026-11-16 · status: BLOCKED for target acceptance

- Authorization: user requested the next task after the AIC-016 offline handoff and explicit target/live
  block. Implement independent AIC-017 software and credential-free checks. This does not accept the
  AIC-016 dependency/M3 gate, complete M4 or authorize live calls/provisioning. Starting shared offline
  suite: 643 passed / 2 host-CUDA skips / 15 target-sandbox deselected. Prior AIC-015/016 edits preserved
  uncommitted above pushed `46f5089` on `sergio/batch-03`.
- Added `layer_norm(x, scale, bias)`: affine last-axis population normalization, two-pass variance,
  epsilon **1e-5** inside the square root. Added `matmul_gelu(x, weight, bias)`: explicit
  **HIGHEST** dot precision and float32 accumulation, column bias and exact erfc GELU. Neither
  reference adds intentional slowdown or approximate/reduced precision. Both reject non-float32
  arguments and invalid input layouts. Pure agent-visible source contains no input arrays or seeds.
- Registered plan development/unseen shapes, deterministic PCG64 factories, explicit parameters,
  nonfinite rejection and 8 MiB combined input caps checked before oversized allocation. New factories
  reject invalid dimensions/dtypes. Development cases: **17 layer norm / 16 dense**, including nearly
  constant/offset/cancellation cases and a GELU activation sweep. Disjoint final split slots preserve
  the existing split contract; commit these new declarations before search. This work does not perform
  held-out candidate selection or generalization evaluation.
- Declared LN rtol/atol **1e-4/1e-4** for epsilon-conditioned mean roundoff and dense **1e-4/1e-5** for
  float32 dot/GELU roundoff. Policies were set before reference checks and not changed afterward.
  Added independent NumPy float64 oracles (Python math.erfc for GELU), eager/JIT/oracle checks on all
  development/edge cases and reference shape smoke on the two unseen declarations. Negative fixtures
  reject altered epsilon, omitted affine parameters, unstable variance, approximate GELU, omitted bias
  and reduced output dtype. JAXPR inspection verifies both HIGHEST operands/float32 accumulation.
- The shared evaluator/CLI/context already supports explicit multi-input workloads; integration checks
  cover them without special controller branches. `runner.IMAGE_FILES` includes both source modules,
  so worker image content hashes change. No dependency/JAX/lock upgrade. `docs/WORKLOAD_SUITE.md`
  records numerical policies, shapes, input byte calculations, limitations and actual-target commands.
- Calibration CLI adds optional **--work-dir <new-path>**: retains actual worker inputs/source/results,
  raw A/A blocks, logs and profile traces, including noisy failures. `calibration_context.json` records
  config, budgets, protocol, reference/policy hashes, primary shape and runner identity. Retained work
  directories and existing aggregate calibrations are refused. The CLI now passes configured budgets
  to the evaluator; zero-compile-cap tests prove that no worker launches. Without --work-dir the
  existing temporary-work behavior remains available. The native real A/A mechanics regression now
  checks that raw sessions survive success or a noise refusal.

### Actual verification and artifacts

- Initial new workload file: **100 passed, 2 target tests deselected in 19.25 s**. After adding CLI
  cap/retention checks, affected regression across workload suite, runner, original workloads,
  portability, split and context tests: **232 passed, 2 target tests deselected in 64.48 s**.
  Command: `uv run --locked --offline pytest tests/test_workload_suite.py tests/test_runner.py
  tests/test_workloads.py tests/test_backend_portability.py tests/test_heldout.py tests/test_context.py
  -q -m 'not target and not sandbox'`.
- Preserved native baseline-only CLI evidence:
  `runs/aic-017-native-validation-20261116/validation.json`, with `layer_norm/` and `matmul_gelu/` run
  stores and CLI logs. The report verifies each actual worker output against the independent oracle,
  baseline/best/original byte equality, source/policy identities and implementation hashes. Both use
  C4 and report **CPU_EXECUTION_EVENTS**, executable granularity only. No candidate search, provider
  requests, paid calls, timing/gain claim or selected-target acceptance.
- Layer norm run **20261116T190927.700924Z-5312c932f8f6**: 17 development/edge reference cases pass;
  maximum absolute oracle error **4.2117111e-5**, maximum tolerance fraction **0.411804**.
  Reference SHA-256 `9a69bc9b3f555cfe9b49e1b7b57aac3cd20d938b3c985cceada1f238af15d74e`.
- Dense run **20261116T190931.562444Z-fbf1e18b54f3**: 16 development/edge reference cases pass;
  maximum absolute oracle error **2.7784521e-6**, maximum tolerance fraction **0.042146**.
  Reference SHA-256 `f6d542ec63dcdf2fe10fbfd335c95073dfa8194385f86c62af9893d95c4bb385`.
  These numbers characterize CPU reference correctness, not device timings or CUDA error bounds.
- Mandatory new target checks:
  `AIC_TARGET_CONFIG=configs/target.json uv run --locked --offline pytest tests/test_workload_suite.py
  -m target -q` → **exit 1, 2 CUDA capability setup errors, 102 deselected, no skips, 2.14 s**.
  No target test is counted as passed.
- Actual calibration preflights for **softmax, layer_norm and matmul_gelu**, each with `--backend cuda
  --runner docker --config configs/target.json --label aic-017-nvhost-20261116 --work-dir
  runs/aic-017-target-calibration-20261116/<workload>` → **exit 2**: CUDA worker image
  `aic-worker:cuda12-7678d1eebe492949` is not available. Preflight refuses before raw work creation,
  calibration or any provider call. No target A/A evidence or resource-fit claim was manufactured.
- Shared full offline regression:
  `uv run --locked --offline pytest tests/ -m 'not target and not sandbox and not live' -q`
  → **749 passed, 2 host-CUDA skips, 17 target-sandbox deselected, 2 unchanged google-genai warnings
  in 263.48 s**. Full output retained as `runs/aic-017-native-validation-20261116/offline-test-output.txt`.
  Mandatory target checks above failed separately; exclusions do not establish target acceptance.
  `git diff --check` is clean. Commands use the existing environment-only OpenSSL fallback, temporary
  UV/Matplotlib caches and PYTHONDONTWRITEBYTECODE. No lock/package changes.

### Remaining acceptance

The three-family software/reference/edge-policy items are implemented and CPU-verified. **AIC-017 is
not DONE.** AIC-016/M3 acceptance remains open. On the selected host, rebuild current worker images,
run actual target/reference/containment checks and retain A/A sessions per primary workload shape.
Confirm CUDA tolerances and complete compilation/device/profile memory fit inside the frozen resource
caps before search. Keep the plan shapes unless pre-search measurements justify a documented resize;
do not choose shapes or loosen tolerance based on candidate performance. M4 is not accepted.

## AIC-016 — Selected-target offline/live MVP verification · 2026-11-16 · status: BLOCKED

- Authorization: user requested the next task after AIC-015. Implement verification and run reviewed,
  credential-free fixtures. The NV-host access/execution handoff and explicit provider/model, money
  amount/currency and dated price ceilings were requested; no answer or spending inputs were supplied.
  No paid request, provisioning, remote access attempt or later task was performed.
- Starting state: AIC-015 changes remain uncommitted on `sergio/batch-03`, above pushed `46f5089`.
  Preserve them. Previous full suite: 635 passed / 16 Docker-CUDA skips. This task does not accept
  pending M1/M2/AIC-011 target checks or mark the target decision DECIDED.

### Implemented checks and handoff

- `configs/target.json` is the nonsecret D-02 execution contract: CUDA, Docker, registered softmax, C4,
  three candidate slots and three provider requests. Normal frozen protocol/budget defaults apply.
  It contains no guessed GPU identity, price or money cap. It cannot dispatch a live request unchanged.
- `tests/test_end_to_end.py` sends reviewed compile-error, nonfinite and forbidden-import sources
  through the actual CLI/provider boundary and native CPU workers, explicitly enabling the test-only
  reviewed-fixture runner. Real compilation/correctness reports reach later bounded contexts. The run
  returns the verified frozen reference and never executes generated live code in a trusted process.
- The separate `target` test runs compile-error / `jax.nn.softmax` / logsumexp fixtures through actual
  selected CUDA container workers. It requires all development cases and two measured rewrites under
  real A/A calibration, raw paired timings and normal host decision/confirmation. No outcome or speedup
  is scripted. Calibration noise blocks the check rather than changing the protocol after results.
- Existing `needs_docker` probes are also marked `sandbox` during collection, making the plan's
  `-m 'target or sandbox'` command select the containment tests. With `AIC_TARGET_CONFIG`, unavailable
  CUDA/Docker produces failure instead of a skip. Ordinary CPU-only tests remain available locally.
- `scripts/verify_mvp.py` reads stored JSON/bytes only. It checks committed hashes and the best export,
  resolved CLI config/manifest identity, exact dispatched contexts and previous-result provenance,
  provider mode, native worker source/device observations, request IDs/usage and profile coverage.
  `--target-mode live` requires the recorded live mode, matching prices/cap, CUDA/container observations
  and two to three evaluated proposals. The verifier cannot authenticate arbitrarily fabricated files
  or replace actual containment probes, profile review or M3 approval. Final/held-out checks remain
  separate. README documents the new offline/target/live commands and unique evidence directories.

### Actual results

- Focused end-to-end and portability checks:
  `uv run --locked --offline pytest tests/test_end_to_end.py tests/test_backend_portability.py -q -m 'not target'`
  → **45 passed, 1 deselected in 15.59 s**. The eight new offline cases cover real failure/context/export
  paths, tampered best/context/worker rejection, the target contract and refusing declared capabilities
  without native source/device/funding evidence. AIC-015's 66 checks supply explicit synthetic acceptance
  and confirmation branches; they do not substitute for a selected-target win.
- Native run ID **20261116T182356.869516Z-49c6ec758c8a**, retained at
  `runs/aic-016-native-validation-20261116/native/run`; immutable review/provenance snapshot at
  `native/validation.json`, CLI logs alongside the run. States:
  **ACCEPTED (reference), COMPILE_FAILED, CORRECTNESS_FAILED, POLICY_REJECTED**. Three offline replay
  requests, two actual candidate correctness evaluations, zero paid calls, no candidate timing/gain claim.
  Context 2 contains the compile failure and context 3 the actual NONFINITE result across 16 cases.
  Baseline/best/original source SHA-256:
  `74fe461f76bb239bad8f2f5d638ff61dd31b8020bb5232991ee85390776a0c6f`.
  Profile: **CPU_EXECUTION_EVENTS**, executable events only, no per-HLO-op attribution. Manifest explicitly
  records a nonisolated CPU mechanics run. Both selected-target modes refuse it.
- Mandatory actual-target/sandbox command:
  `AIC_TARGET_CONFIG=configs/target.json uv run --locked --offline pytest tests/ -m 'target or sandbox' -q`
  → **exit 1, 15 setup errors, 645 deselected, no successful skips, 12.73 s**. Docker worker image cannot
  be built/accessed and CUDA selection raises BackendUnavailable/AssertionError. A separate authorized
  read-only `docker info` check outside the filesystem sandbox also reports that the daemon is not
  reachable. Docker Desktop was not started and no unrelated services were changed.
- The first focused attempt exposed a test-driver mistake (`--context-condition` is not a CLI option);
  changed the driver to supply C4 through its JSON config, then verified it. An initial regression command
  also referenced nonexistent `tests/test_config.py`; corrected to the actual affected files. Neither
  failed invocation changed production behavior or weakened a correctness/timing policy.
- Shared full offline suite:
  `uv run --locked --offline pytest tests/ -m 'not target and not sandbox and not live' -q`
  → **643 passed, 2 host-CUDA skips, 15 target/sandbox deselected, 2 unchanged google-genai warnings
  in 280.98 s**. The mandatory target/sandbox command above failed separately, so these exclusions do
  not establish target acceptance. Full offline output retained beside the native evidence as
  `offline-test-output.txt`. `git diff --check` is clean. The verifier CLI also returns exit 2 when this
  CPU replay is offered as the live target MVP.
  Local commands use the existing environment-only OpenSSL fallback, temporary UV/Matplotlib cache
  paths and `PYTHONDONTWRITEBYTECODE=1`; dependency versions and `uv.lock` are unchanged.

### Remaining acceptance gates

1. Access to Harry's NVIDIA WSL2 host; actual target probe/model/driver and containment evidence from
   phase B. Run the new selected-target replay on that population and retain its real measurements.
2. Explicit live provider/model, dated price ceilings/provenance, approved money cap/currency and secure
   credentials. On that same host, evaluate **at least two and at most three genuine live proposals**;
   external failures/budget exhaustion do not turn fewer proposals into a passed gate.
3. Replay live artifacts without source imports, review next-request feedback, real profile coverage
   and limitations and obtain the M3 review. No speedup is required; a verified baseline is valid.

Offline software work is implemented. **AIC-016 and M3 remain BLOCKED**, not DONE. No GPU, container,
live-optimization or profiling-benefit claim follows from native CPU or AIC-015 synthetic evidence.

## AIC-015 — Bounded current-best controller · 2026-11-16 · status: DONE

- Authorization: user requested the next task after AIC-014 was committed/pushed as `46f5089` on
  `sergio/batch-03`. Implement controller integration and offline verification; no paid calls or target
  provisioning. Starting suite: 568 passed / 16 Docker-CUDA skips; clean working tree.
- Scope: explicit baseline initialization, consistent terminal attempts, current-best confirmation,
  pre-launch budgets, shared run/worker/provider deadlines and a verified terminal best artifact.
  Live/sandbox selected-target proof remains AIC-016. All five task acceptance criteria evidenced below.

### Implemented behavior and scope

- `Controller.start()` starts a shared monotonic episode clock and verifies the immutable reference once.
  Native compile failures, missing cases/outputs or nonfinite reference data abort before a provider call.
  `baseline_failure.json` retains diagnostic/raw evidence; `run_result.json` explicitly has no best artifact.
  Zero candidates verifies the baseline and makes no model call. Negative/nonfinite/wrong-type budgets
  and candidate limits fail before artifacts or trusted source initialization.
- `optimize()` returns `run_result/v1`, including a verified best reference, output directory, counts,
  budget snapshot and stop status/reason. `agent_loop()` preserves the history-list API and sets the
  same `controller.result`. Terminal calls are idempotent and verify best again; direct new submissions
  are refused after termination. The CLI prints attempt counts, stop status and best path, including zero.
- Each unique submission consumes one candidate slot, including schema/policy/parent rejection and
  duplicates. A malformed/empty/refused provider response without a locally validated submission is a
  request outcome, not an invented candidate. Every candidate record is committed before another request.
  Exact hashes plus existing syntax-tree equivalence prevent duplicate evaluation. Optional parent IDs
  must match the current incumbent, whose source is used for measurements.
- Promotions require every declared development correctness case under the frozen policy, matching
  candidate/incumbent source identities, protocol and calibration, complete positive raw measurement
  blocks, matching host analysis and fresh confirmation under AIC-006. The controller reruns the host
  decision from evidence before accepting an evaluator's promotion claim. `Evaluation.decision` records
  threshold/calibration/protocol/incumbent identity; bounded context includes the safe threshold summary.
  Promotion happens only after the attempt commit succeeds. Reporting-only baseline-comparison failures
  now produce a durable candidate outcome instead of escaping unrecorded.
- `Evaluator._run` checks wall/stage/source/artifact limits before launch and rechecks after preparing
  inputs. Stage allowances travel in the host-built request. Parent runners monitor atomic phase markers
  at 20 ms cadence: compilation/startup/input preparation/IR is cumulative per request, correctness is
  cumulative per declared case, and measurement covers warmup/session work. Markers never replenish a
  budget. Absolute host deadline is rechecked after runner verification and clips the launch timeout;
  late required results are discarded. A profile that exhausts remaining time does not invalidate a win
  already confirmed within budget. No generated source executes in the controller.
- Model evaluation requires isolated-runner preflight before requesting a proposal and before execution.
  Trusted subprocess mode verifies only the baseline; positive-candidate CLI runs stop before dispatch.
  Explicit reviewed fixtures remain available for tests. Runner failures, interruption, timeout and
  resource limits stop with the last verified source. Process-group cleanup runs on exceptions/interrupts;
  Docker removal/absence checks run even when launch or inspection raises. Source/log/artifact ceilings
  apply before side effects; zero log bytes really captures zero. Publication/terminal storage failures
  are sticky and cannot resume. Active-worker filesystem quotas and execution recovery are not added.
- Provider waiting is independently capped by request timeout and remaining episode time, in addition
  to SDK I/O timeouts. An outstanding SDK call retains its full uncertain reservation, is never retried
  and cannot use a late response to launch evaluation or another request. Its daemon only finishes the
  SDK operation. A reservation expiring before dispatch records `NOT_DISPATCHED`, known zero usage and
  released token/money debit. Every request still consumes a request slot. No remote-cancellation or final
  billing guarantee is claimed. Cleanup can extend elapsed time past the work deadline; it must complete
  before further work. Real cap/prices remain unset and no paid calls are enabled.

### Validation and retained evidence

Commands used the same environment-only Homebrew OpenSSL fallback and temporary UV/Matplotlib caches
as AIC-014, with `PYTHONDONTWRITEBYTECODE=1`; no dependency or JAX/jaxlib/lockfile changes.

- Required `uv run --locked --offline pytest tests/test_controller.py -q`: **66 passed in 28.77 s**.
  Deterministic state matrix covers zero/invalid limits, failed baseline, invalid schema/policy/parent,
  exact/semantic duplicates, compile/correctness failure, slower/no-change/unstable/confirmed/unconfirmed,
  timeout/resource/interruption, request/token/money/wall caps, late required versus optional results,
  corrupted promotion evidence, storage failure and pre-dispatch/isolation/cleanup checks.
- Two tests actually run invalid reference code in trusted CPU workers (nonfinite output and invalid
  reshape), retain failure artifacts and assert zero model calls/no best/source untouched. Parent deadline
  tests actually kill reviewed sleeping processes; deterministic phase-clock tests verify cumulative limits.
  No synthetic timing is substituted for selected-target performance evidence.
- Affected regression run: **314 passed / 3 target skips**, 169.32 s. An earlier full run had 634 passed /
  16 target skips and one D-06 timing-marker guard failure: new functional timeout assertions needed
  `slow` markers. Added the markers; the unchanged guard then passed (1 passed, 0.88 s).
- Final `uv run --locked --offline pytest -q`: **635 passed / 16 skipped**, 274.28 s; two unchanged
  google-genai SyntaxWarnings. Skips are 12 Docker image/daemon checks and 4 unavailable-CUDA checks;
  these do not waive selected-target/sandbox acceptance. No required controller test was skipped.
- Retained synthetic state episode: `runs/aic-015-synthetic-validation-20261116/validation.json`.
  States: baseline `ACCEPTED` -> `PROPOSAL_INVALID` -> confirmed `ACCEPTED` -> `CORRECTNESS_FAILED`;
  best attempt 2 is preserved and the third prompt receives that incumbent. Timing and correctness
  observations are explicitly scripted; host decision, confirmation recheck and durable storage are real.
  This is not a performance or native-execution claim.
- Retained native CPU episode: `runs/aic-015-native-validation-20261116/validation.json`.
  States: baseline `ACCEPTED` -> `COMPILE_FAILED` -> `CORRECTNESS_FAILED` -> `DUPLICATE`; real reviewed
  fixtures run in CPU workers and prior failures enter subsequent C4 contexts. Final best is the verified
  original baseline. Both episodes stop at three candidate attempts/three offline requests and preserve
  original sources; exact request/context/evaluation/commit/export references were independently checked.
- Retained validation snapshots precede the timing-marker-only test edit; production source hashes still
  match. Each episode's `reviewed_checks.json` links the original report hash, verifies production hashes
  and the best export again, and records the final test-fixture hash. The marker changes do not alter the
  scripted evaluator or candidate fixture behavior.
- Offline transition matrix (`tests/test_controller.py`; no real speedup claim):

| Trigger/observation | Candidate terminal result | Export/stop behavior |
| --- | --- | --- |
| Zero candidates | No candidate | Verified baseline, zero requests |
| Invalid reference/nonfinite/compile failure | No candidate | BASELINE_FAILED, no best, zero requests |
| Invalid schema/policy/current parent | PROPOSAL_INVALID / POLICY_REJECTED | Counts one slot; prior result enters next context |
| Exact/semantic source repeat | DUPLICATE | No evaluation; incumbent retained |
| Compile or numeric failure | COMPILE_FAILED / CORRECTNESS_FAILED | Incumbent retained; diagnostics supplied next |
| Stable slower/noise-level session | VALID_SLOWER / VALID_NO_CHANGE | Incumbent retained |
| Unstable sessions through fixed retry | MEASUREMENT_UNSTABLE | No promotion |
| Correct win plus fresh confirmation | ACCEPTED | Recheck raw proof, commit, then promote |
| Unconfirmed or corrupted win | VALID_NO_CHANGE / INTERRUPTED | No promotion |
| Timeout/resource/runner/user interruption | TIMEOUT / RESOURCE_LIMIT / INTERRUPTED | Durable stop, verified incumbent returned |
| Protocol/provider/budget failure | Request event, no invented candidate | Durable reason and verified incumbent |
| Publication failure | No uncommitted promotion | Sticky storage failure; no new dispatch/resume |

- Limits: no live provider, Docker/CUDA acceptance, actual speedup or final/held-out result is claimed.
  Those gates remain separate (AIC-016 and later evaluation tasks). No automatic resume, artifact-only
  reporting feature or source overwrite was introduced. Next immediate task after this scope: AIC-016.
- Acceptance: zero-model baseline mode, one terminal result per candidate, prior outcome in next request,
  confirmed-correct promotion and verified fallback/terminal export all verified. No paid calls made.

## AIC-014 — Bounded proposal context · 2026-11-16 · status: DONE

- Authorization: user requested "continue with the next task" after AIC-013. Implement bounded context
  selection and offline evidence; no paid calls, provisioning or pending target acceptance.
- Starting point: AIC-013 full suite 537 passed / 16 Docker-CUDA skips. Preserve previous working-tree
  changes and planning documents. JAX/jaxlib and dependencies remain unchanged.
- Policy: 24 KiB serialized metadata plus current source under its separate source cap; the full provider
  request must also fit remaining conservative token/money reservations. Keep constraints and latest
  outcome, remove historical snippets first, then optional older outcomes. Save exact prompt bytes,
  included fields and omitted IDs before dispatch. All four acceptance criteria evidenced below.

### Implemented behavior and scope

- `context_builder.py`: deterministic `proposal_context/v1` and external `context_selection/v1`. Positive
  allowlists expose the development workload contract, selected host environment/runner and worker population,
  frozen correctness rules, current source, baseline/incumbent hashes, remaining budgets and measured feedback.
  Five recent nonpromoted outcomes include slower/no-change rejections; at most three accepted rewrites are
  selected. Baseline, incumbent and latest result are mandatory. Full history remains in durable artifacts.
- Paired medians/standard deviations and speedup confidence intervals retain protocol/source identities
  and hashed evaluation references. Evaluator session records now identify both measured sources explicitly.
  Baseline/incumbent timing comes only from identified sessions in mandatory evidence, with an explicit
  unavailable reason otherwise. These are search measurements, not final verification or a gain claim.
- Compiler summaries preserve parser/backend/signature provenance and label estimates separately from
  measurements. C3/C4/K receive them; only C4 receives runtime coverage/reasons and per-lane/name union
  durations, counts and artifact hashes. CPU/device/host-only/unavailable labels are preserved. Raw traces
  and IR remain in evidence files. No model-generated performance facts or retrieval service is introduced.
- Serialized metadata is capped at 24 KiB, current source separately at `Budgets.max_source_bytes` (64 KiB
  default). Historical source snippets are at most 2 KiB; diagnostics are bounded to 2 KiB, proposal text and
  hotspot labels to 512 bytes. Truncations are recorded. Remove optional snippets before older optional
  outcomes; preserve mandatory summaries/constraints or stop with `CONTEXT_BLOCKED` before a call.
  `ProposalProvider.context_fits` checks the exact encoded whole body against payload, remaining token and
  Decimal money ceilings without consuming a request. The existing reservation check still runs per retry.
- Known environment/SDK credentials are redacted before accounting; current source containing a credential
  is refused rather than rewritten. Raw logs, raw IR, raw timing blocks, input arrays and split seeds are
  excluded. Development-case allowlists and diagnostic split guards refuse other evaluation populations.
  Source comments, proposal text and diagnostics are untrusted data beneath the provider system policy.
- `RunStore.reserve_request` publishes exact context bytes and selection manifest under `contexts/NNNN.*`
  before the reservation and SDK call. The manifest records included field paths, omitted IDs/reasons,
  truncations, source/metadata/payload bytes and SHA256. Reservations reference both files; committed
  attempts also hash-link them, so best-artifact verification detects context tampering. Publication or
  snapshot/hash mismatch stops dispatch. Existing raw evidence is preserved. Proposal-only CLI uses the
  same bounded context policy without importing/evaluating source.

### Validation and retained evidence

- Focused context suite: `uv run --locked --offline pytest tests/test_context.py -q`:
  **31 passed in 17.00 s**. Covers actual native compile/correctness failures in subsequent prompts,
  deterministic window/truncation, a 601-record history, UTF-8/encoded payload/token/money caps, provenance,
  baseline/incumbent distinction, all context conditions/coverage labels, secret/split exclusion, source
  injection as data, pre-dispatch publication, publication failure and context tampering.
- Affected controller/CLI/Gemini/runtime regression run: **109 passed / 3 target skips**, 134.22 s.
- Final full suite: `uv run --locked --offline pytest -q`: **568 passed / 16 skipped**, 257.57 s.
  Skips are 12 Docker checks (worker image build unavailable) and 4 CUDA checks (backend unavailable).
  Two existing google-genai SyntaxWarnings remain. These skips do not waive pending target acceptance.
- Retained actual C4 CPU run: `runs/aic-014-context-validation-20261116/validation.json` and its three
  context/selection/reservation files. Offline model fixtures propose reviewed invalid-reshape and wrong
  subtraction code; real CPU workers produce `COMPILE_FAILED` then `CORRECTNESS_FAILED`. Request 2 sees
  the compile failure and request 3 sees the correctness failure. The final empty provider response is
  durably recorded; best remains the verified original baseline. No timing or improvement claim is made.
- Exact metadata byte counts: **6335 / 7934 / 10817**, all <= 24576. Whole request byte counts:
  **10293 / 12048 / 15222**; reservations (input bytes + 4096 framing, plus 2048 output) each fit the
  configured 30000-token ceiling. All context/selection hashes, evaluation provenance and baseline export
  independently verified. Original source before/after SHA256 is
  `bd07354a813c3153d69f975a9f445512c4dd3c7b55476c326793932c5e096df3`.
- The initial evidence driver completed its run/assertions but used the wrong best-index key while writing
  the final summary; fixed the verification helper and independently read the existing artifacts to publish
  `validation.json`. No worker/provider execution resumed and no committed evidence was overwritten.
- Environment-only test prefix is unchanged from AIC-013: Homebrew OpenSSL fallback, temporary UV and
  Matplotlib caches and `PYTHONDONTWRITEBYTECODE=1`. JAX/jaxlib remain 0.4.38; `uv.lock` is unchanged.
  CPU reviewed-fixture evidence is not Docker/CUDA acceptance, live provider proof, billing certification
  or profiling-benefit evidence. No paid calls were made. Full controller budgets remain AIC-015;
  live proof AIC-016, execution recovery AIC-019, reporting AIC-020.
- Acceptance: prior outcomes in subsequent requests, size caps, metric provenance and separate original/
  current-best identities verified. Next immediate task: **AIC-015**, bounded controller integration.

## AIC-013 — Structured proposals and provider budgets · 2026-11-16 · status: DONE (offline software scope)

- Authorization: user requested "proceed with the next task" after AIC-012. Scope is the structured
  provider boundary, request budgets/accounting, offline replay and proposal-only mode. No paid provider
  call, provider migration, later task or pending hardware acceptance is authorized by this instruction.
- Starting baseline: AIC-012 full suite 470 passed / 16 Docker-CUDA skips; preserve all previous changes
  and pre-existing untracked planning documents. JAX/jaxlib and the lockfile stay unchanged.
- Prices are explicit configuration with dated provenance; no real rate or money cap is invented or
  enabled by default. HTTP mock accounting fixtures are explicitly synthetic, with zero remote calls.

### Implemented behavior and scope

- `providers.ProposalProvider.propose(context) -> Proposal` is the sole controller SDK dispatch boundary.
  Exactly one `submit_proposal` tool value is accepted, with local Proposal v1 validation independent of
  the tool schema. Unknown/multiple calls, malformed/refused/truncated/oversize responses and unexpected
  response models stop dispatch. Invalid proposal fields create `PROPOSAL_INVALID` attempts without
  evaluating source. The host alone applies source policy and invokes the evaluator.
- `controller.py` removes provider-driven inspection/evaluation tools. Each call gets a fresh context
  containing current source, compact measured history, constraints and budget summary. Deterministic context
  selection/trimming remains AIC-014; this task applies hard request byte limits and conservative token checks.
  CLI and direct-loop zero candidates make no request. CLI zero candidates construct no provider client.
- `contracts.py`: strict optional `ModelPrices` (`model_prices/v1`), provider/model/currency identity, positive
  input/output price ceilings, ISO checked date and HTTPS provenance URL; transport/retry/payload settings;
  `proposal_only` and offline response path. Live mode requires explicit cap, currency and dated matching
  prices, before client construction in the CLI and before dispatch for injected SDK callers. Default
  provider/model IDs are unchanged. Real prices/cap remain unset, so default live calls are disabled.
- Every attempt reserves serialized request bytes + 4096 framing tokens for input, configured output ceiling
  and Decimal cost at the declared price ceilings. `provider_reservation/v1` is atomically saved under
  `request_reservations/NNNN.json` before dispatch, with full bounded/redacted structured parameters, SHA256,
  local ID, retry number, timeout and price provenance. This conservative input policy is not a vendor-certified
  tokenizer bound or billing guarantee. Overrun is recorded/charged and stops the run, not silently clipped.
- Known usage reconciles reservations; missing components retain their full debit. Failed/unknown calls
  retain their debit, including 429/timeouts. Requests, cumulative tokens, money and remaining time are
  checked before each retry. SDK retries are disabled; `model.max_retries` defaults to 0. Default I/O timeout
  is 30 s, shortened to remaining time; late responses are rejected. Full optimizer/worker deadlines remain
  AIC-015, and a client timeout cannot guarantee remote cancellation or final billing.
- Cached, priority or server-tool billing is not enabled/supported; unexpected usage stops the loop instead
  of assuming free tokens/features. Bounded outcome records retain IDs or unavailability reasons, normalized
  redacted payloads, usage, accounting, refusals/errors and timing. Literal SDK keys, environment credentials,
  sensitive fields and credential-bearing dictionary keys are redacted. Source containing a known credential
  is refused; other proposal metadata is redacted without modifying executable source bytes.
- `storage.py`: immutable reservation/outcome references, hashed attempt links and provider stop artifacts.
  Failed publication prevents another dispatch. Reservations without outcomes remain uncertain, with no
  automatic replay; recovery is AIC-019. `save_proposal` publishes unverified proposal-only artifacts, with
  no attempt/best export or implied correctness/performance result.
- `agent.py`: `--offline-responses FILE` (deterministic local Messages-shaped JSON array) and `--proposal-only`
  (reads reference text, skips workload import/runner/evaluation, writes `proposed.py`, `proposal.json` and
  `proposal_only_result.json` with `verified: false`). The live-money gate also applies to live proposal-only.
  Checked-in `tests/fixtures/provider_responses.json` provides an explicit offline/synthetic protocol example.
- `gemini.py` keeps the existing provider behind the same boundary, disables hidden SDK/application retries
  in its transport, applies per-request HTTP options and preserves missing usage rather than inventing zeros.
  Existing adapter/controller fixtures pass. No provider migration or dependency/lockfile changes.
- Installed Anthropic SDK 1.3.0 has no `temperature` argument. The real SDK/HTTP-mock test caught this, and
  Anthropic requests now omit it; an explicit temperature is supported only for Gemini. Thinking is disabled
  for Anthropic requests and first-party requests explicitly select standard service tier.

### Verification and retained evidence

All commands used the existing host SSL workaround and temporary caches, without changing project dependencies:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/usr/local/opt/openssl@3/lib \
UV_CACHE_DIR=/private/tmp/aic-011-uv-cache MPLCONFIGDIR=/private/tmp/aic-011-mpl \
PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline pytest -q tests/test_provider.py
```

- Required standalone provider command: **67 passed in 20.15 s**; no secrets, network or skipped cases.
  Cases cover installed SDK response objects and actual SDK calls through local HTTP mocks, malformed tools,
  refusals, truncation, timeouts, rate limits/auth failures, partial/unknown/invalid usage, cap exhaustion,
  missing cap/prices, literal-key redaction, pre-dispatch/outcome storage failures and direct-loop zero budgets.
- Focused provider/foundation command: **90 passed in 72.97 s**. Earlier affected regression commands passed
  **200 tests** (CLI/foundation/storage/contracts/provider/Gemini) and **125 tests** (provider/contracts/Gemini).
- An initial full run had **534 passed, 1 failed, 16 skipped**: CPU contention made an A/A session's
  IQR/median 1.026 exceed the test protocol's 1.0 limit. The harness correctly stopped. The provider/history
  integration test now scripts timing while retaining real correctness workers; production calibration,
  tolerances and decision rules are unchanged, with their independent tests retained. This removes an
  accidental host-speed dependency rather than weakening production acceptance.
- Final full-suite command, same prefix: `uv run --locked --offline pytest -q` — **537 passed, 16 skipped,
  2 warnings in 261.23 s**. Twelve skips require Docker/a buildable worker image; four require CUDA. Two
  unchanged google-genai SyntaxWarnings. No AIC-013 mandatory test was skipped; earlier target gates remain open.
- Actual CLI smoke, same prefix: `uv run --locked --offline python agent.py --backend cpu --proposal-only
  --offline-responses tests/fixtures/provider_responses.json --output-dir runs/aic-013-proposal-only-validation-20261116`.
  Exit 0; retained unverified source/result, no `work/` or `best.py`, no original import/execution or remote call.
- Retained SDK/accounting evidence: `runs/aic-013-sdk-validation-20261116-reviewed/validation.json`, generated
  by `/private/tmp/aic-013-sdk-evidence.py` through the installed Anthropic SDK and `httpx2.MockTransport`.
  Run ID `20261116T131701.080274Z-141c7d9455fa`. Two local HTTP attempts: 429 then valid proposal; both SDK
  retry headers `0`, both read timeouts 30 s; server fixture IDs captured. Synthetic prices/provenance are
  explicitly marked; event `mode: live` exercises the authorization branch, while manifest/validation identify
  the transport as an offline HTTP mock. This is not a live provider/availability or optimization claim.
- Accounting fixture: 8961 tokens charged conservatively (8944 reserved for unknown first-call usage + 17
  reported on success), synthetic debit `0.017181 USD`, **zero actual spend/remote calls**. Redacted
  request/reservation files are hash-linked and independently rechecked. Original source before/after SHA256:
  `bd07354a813c3153d69f975a9f445512c4dd3c7b55476c326793932c5e096df3` (343 bytes).
  Validation fingerprints match all current core files and the unchanged lockfile; original source is preserved.
- Primary protocol checks: installed SDK `Messages.create`/`with_options` signatures, Usage schema, and
  google-genai `HttpOptions`/`HttpRetryOptions`; Anthropic's [token-counting docs](https://platform.claude.com/docs/en/build-with-claude/token-counting)
  describe counting as an estimate, and [prompt-caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
  distinguish cache input categories. No live token-counting endpoint or price lookup is part of dispatch.
- `git diff --check` passes. Prior changes and pre-existing untracked planning files remain preserved; no
  commit, push, dependency upgrade, paid provider call, cloud provisioning or milestone acceptance.
- Acceptance: all five AIC-013 criteria verified offline. G-07 remains partial for AIC-014 context selection
  and AIC-016 live proof. Stage-wide budgets/deadlines remain AIC-015, recovery AIC-019. Next task: AIC-014.

## AIC-012 — Durable run artifacts · 2026-11-16 · status: DONE (offline storage scope)

- Authorization: the user requested "proceed with the next task" after the AIC-011 implementation report;
  the next task is AIC-012. This authorizes storage implementation and offline tests, not acceptance of
  pending Docker/CUDA checks, paid calls, recovery implementation (AIC-019) or later tasks.
- Starting HEAD `12c805e`, branch `sergio/batch-03`, with all previous AIC-011 source/documentation changes
  and the six original untracked planning files preserved. The AIC-011 final suite was 421 passed / 16 skipped.
- Scope: unique non-overwriting runs, shared strict atomic writes, immutable source/request/attempt evidence,
  commit-record-last persistence, derived history and validated best-source export.

### Implemented behavior

- `storage.py`: default timestamp/UUID run IDs, exclusive run reservation, shared strict JSON/byte writers
  with file fsync and atomic publication. Immutable artifacts use exclusive publication; collisions never
  replace existing runs or evidence. Parent-controlled paths reject traversal, symlinks and special files.
- Manifest/config/environment/baseline have exact SHA256 identities. `manifest/v1` records the run ID,
  creation time, config hash and references, alongside the existing environment/device/lock provenance.
  CLI config stays `run_config/v1`; direct Python API snapshots use `direct_run_config/v1` and an explicit
  non-scientific reason instead of claiming a resolved CLI config.
- Immutable `provider_request_event/v1` metadata lives in `requests/NNNN.json`; `requests.json` is derived.
  `attempts/NNNN/` retains `source.py`, proposal/evaluation, raw measurements, feedback and request references
  as available. Invalid submissions have a bounded `proposal_submission/v1` artifact with an omission reason.
  `attempt_commit/v1` in `record.json` is published last, with identities/hashes for source, metadata and raw
  worker artifacts. Sources are never imported during index generation or reporting.
- `history.json`, compatibility `iteration_NNNN.json`, `best.py` and `best/v1` are derived only after a commit.
  Source/evaluation/proposal hashes and host acceptance/correctness evidence are checked. Best falls back to
  the verified baseline. Its scope names the development cases/policy and explicitly leaves final and
  held-out verification unperformed. `storage.verified_best(path)` rejects corrupt evidence or stale exports.
- `agent.py` / `controller.py`: lazy construction, reservation before baseline/LLM requests, configured byte
  budgets forwarded, persistence before in-memory promotion and request append. Publication failure prevents
  further execution on that controller; no automatic retry/replay. Host-side runner outcome and bounded,
  redacted stdout/stderr artifacts are retained even for worker failure.
- Baseline source is frozen in `Evaluator`; `controller.py`, `workloads/registry.py` and `worker.py` preserve
  UTF-8 bytes including CRLF for consistent source/profile hashes. The user's source is never written.
- Metadata filters credential fields and known credential values; executable source containing a known
  credential is refused rather than rewritten. Host source/artifact limits fail closed without deleting
  correctness/timing evidence. An active worker has no filesystem quota: its files can exceed the host cap
  before the post-worker publication check. This limitation is recorded, not represented as hard disk isolation.
- Each derived file is atomic, not a transaction across all indexes. Failure after a commit can leave stale
  history/best; verification rejects a stale best pointer. Valid commits can regenerate indexes through
  `RunStore.derive_indexes()`; execution resume/request recovery remains AIC-019, reporting remains AIC-020.
- README/how_to_run, `AI_CONTEXT.md` sections 5/6/9, task status and the G-09/O-05 entries are updated.
  Prior AIC-011 changes and the pre-existing untracked planning files remain preserved. No dependency,
  lockfile, commit, push or live provider changes.

### Verification

Tests use this environment prefix (the same host SSL workaround as AIC-011; no project/dependency changes):

```bash
DYLD_FALLBACK_LIBRARY_PATH=/usr/local/opt/openssl@3/lib \
UV_CACHE_DIR=/private/tmp/aic-011-uv-cache MPLCONFIGDIR=/private/tmp/aic-011-mpl \
PYTHONDONTWRITEBYTECODE=1 uv run --locked --offline pytest -q tests/test_storage.py
```

- Final storage command: **47 passed in 12.44 s**. Includes real CPU baseline/profile workers, CRLF source
  hashes, in-run worker ownership and original-byte preservation; no statistical speed assertion.
- Earlier affected regressions: **99 passed in 81.72 s** (storage, CLI, backend portability and held-out
  separation). A full run before the final CRLF/direct-API coverage additions was **468 passed / 16 skipped**
  in 228.17 s.
- Final full-suite command, with the same prefix: `uv run --locked --offline pytest -q` — **470 passed,
  16 skipped, 2 warnings in 209.67 s**. Twelve skips require Docker/a built worker image (image build is
  unavailable here); four require CUDA. Two unchanged google-genai SyntaxWarnings. No storage check is skipped.
- Retained native CPU evidence: `runs/aic-012-cpu-validation-20261116-final/validation.json`. Generated using
  `/private/tmp/aic-012-cpu-evidence.py` with the same environment prefix and `uv run --locked --offline python`.
  One offline `FakeAnthropic` request; real trusted workers; states `ACCEPTED` (baseline),
  `CORRECTNESS_FAILED` (32/32 numerical mismatches). `best.py` is the baseline, with its verification scope.
  All commit references and best hashes re-read and verified. Raw artifact inventories: 23 for baseline,
  37 cumulative after the rejected candidate. The validation file fingerprints the actual core sources and lockfile.
- Run ID `20261116T122558.389126Z-c6d32432806e`; baseline/best/original-before/original-after SHA256 all equal
  `bd07354a813c3153d69f975a9f445512c4dd3c7b55476c326793932c5e096df3` (343 bytes).
  No original source, lockfile or dependency versions changed. Sample artifacts remain ignored under `runs/`.
- `git diff --check` passed. Docker/CUDA acceptance for earlier tasks remains separate; this task's storage
  invariants are verified offline on CPU. No live provider call or accelerator performance claim.
- Acceptance: independent paths/collision refusal, complete committed identity/evidence, validated best
  source/baseline fallback, and exclusion of partial attempts all verified. AIC-012 is done; G-09 remains
  partial until AIC-019 recovery and AIC-020 reporting. The next immediate task is AIC-013; no later task
  or pending target milestone has been marked accepted.

## AIC-011 — Runtime profile feedback · 2026-11-16 · status: BLOCKED for selected-target acceptance

- Authorization: the user requested implementation of the next immediate task after the assessment identifying
  G-06 / AIC-011. This authorizes this task's implementation and offline verification; it does not accept M1/M2,
  authorize paid provider calls, or establish CUDA verification.
- Starting source: `12c805e`, branch `sergio/batch-03`. Six untracked planning documents already existed; they
  are preserved. AIC-010 compiler feedback is present; selected-target acceptance remains pending on the NV host.
- Scope: bounded runtime traces in separate worker requests, conservative coverage and overlap accounting,
  baseline/promoted-incumbent profiles, and profile context under C4. Preserve unprofiled timing and C2/C3.
- `uv sync --locked` installed the missing locked test dependencies without changing
  the lockfile. Initial test collection exposed a pre-existing Homebrew Python SSL library path mismatch
  (`openssl@3/3.6.1` absent, installed `3.6.4`); verification uses
  `DYLD_FALLBACK_LIBRARY_PATH=/usr/local/opt/openssl@3/lib`, with no system or repository workaround.

### Implemented behavior and files

- `contracts.py`: `runtime_trace/v1`, explicit coverage/reason, source/device provenance, artifact identity and
  bounded summaries; `Evaluation.runtime_trace` separates observed events from compiler estimates/timing.
- `feedback.py`: five annotated synchronized calls after five warmups; raw Perfetto gzip capped at 8 MiB,
  decoded JSON at 32 MiB and 100,000 events. Temporary duplicate exports are discarded. Complete-event
  timestamps, annotation scope and explicit device metadata are validated. Foreign device lanes are
  unattributed, host dispatch cannot establish device execution, and CPU observations remain executable-level.
- Interval unions remove overlap within each lane/name. Inclusive durations across names/lanes may overlap;
  no percentages, exclusive-duration claims or HLO-op attribution are produced. HOST_ONLY/UNAVAILABLE have no
  hotspots; corrupt/unsupported/oversize/permission-denied profiles have explicit reasons.
- `worker.py`: separate `profile` mode; lowering/compilation/warmup precede capture, never benchmark windows.
- `controller.py`: profile baseline and promoted incumbents in fresh workers, after performance confirmation
  for a candidate. Verify source hash, raw artifact path and SHA256 on the host. Optional profile failure
  preserves the correctness/performance decision. Persist run-relative artifacts and expose observations only
  under C4; C2/C3/K remain profile-blind. Held-out-only evaluation does not launch a development profile.
- `tests/test_runtime_profile.py`: parser/corruption/bounds/foreign-device/overlap fixtures, capture ordering,
  real CPU worker and baseline persistence, artifact forgery rejection, promotion/fallback and C4 visibility.
  Real native CUDA and CPU/CUDA Docker checks are included for the target environment.
- Updated `AI_CONTEXT.md` sections 5/6/9, README, task status, G-06 and R-06 evidence. The pre-existing untracked
  planning files remain untracked; no commits, pushes, provider requests or dependency-version changes.

### CPU evidence and verification

Test environment prefix used for the commands below:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/opt/openssl@3/lib
export UV_CACHE_DIR=/private/tmp/aic-011-uv-cache
export MPLCONFIGDIR=/private/tmp/aic-011-mpl
export PYTHONDONTWRITEBYTECODE=1
```

- `uv run --locked --offline pytest -q tests/test_runtime_profile.py -m "not needs_docker"`:
  **42 passed, 1 skipped (CUDA unavailable), 2 deselected (container checks)**, 4.61 s.
- `uv run --locked --offline pytest -q`: **421 passed, 16 skipped**, 165.19 s, exit 0 on the final source.
  Skips: 12 Docker checks (daemon unavailable) and 4 CUDA-marked checks (CUDA unavailable, including the GPU
  container cases). Two existing google-genai invalid-escape SyntaxWarnings remain. These skips are unexecuted
  acceptance checks, not evidence of containment/device support. `git diff --check` also passes.
- Trusted CPU reference validation: `uv run --locked --offline python /private/tmp/aic-011-cpu-evidence.py
  /Users/hej/Agentic-AI-Compiler-Jax runs/aic-011-cpu-validation-20261116` (temporary driver, no provider).
  Replayable evidence is in `runs/aic-011-cpu-validation-20261116/validation.json` and `history.json`, including
  environment/JAX/lock identities and SHA256 of all four changed harness modules. Raw trace retained at
  `work/0002-baseline-profile/profile/perfetto_trace.json.gz` relative to that directory.
  Coverage **CPU_EXECUTION_EVENTS**, 5 annotated calls, 228 events;
  raw SHA256 `d49e9e46f07ef54a0ab0bfcbda4d3934bc65c996d7f418073899e83c5b24ba73`;
  reference SHA256 `74fe461f76bb239bad8f2f5d638ff61dd31b8020bb5232991ee85390776a0c6f`.
  These are CPU observation/implementation checks, not speedup or official CUDA evidence. The ignored run
  artifacts are retained locally; tests reproduce them in temporary directories.

### Remaining acceptance checks (not substituted by CPU or mock results)

Docker CLI is installed, but the daemon is not running (also checked outside the filesystem sandbox).
CUDA is unavailable here. Task implementation is CPU-verified; task acceptance and R-06 remain partial,
and M1/M2/M3 gates are not passed by this change. A funded C4 live-context demonstration/study belongs to
AIC-016/022–024; none was run or claimed here. No task beyond AIC-011 was started.

On the NV host, with Docker and NVIDIA Container Toolkit running:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 pytest -q tests/test_runtime_profile.py -m "not needs_docker" --require-gpu
uv run --locked python scripts/build_worker_image.py --variant cpu
uv run --locked python scripts/build_worker_image.py --variant cuda12
uv run --locked --extra cuda12 pytest -q tests/test_runtime_profile.py -m needs_docker --require-docker --require-gpu
```

The CUDA checks require actual DEVICE_EXECUTION_EVENTS; missing CUPTI/device coverage fails instead of
passing a host-only trace. Build hashes automatically include the changed worker, contracts and feedback.
If the verified target cannot expose execution events, preserve explicit fallback coverage and request
acceptance of that R-06 scope limitation at the M3 review; do not mark the original target check complete.


## Batch 03 (AIC-010, AIC-018, G-01 gaps, P8) · 2026-11-10 · owner: Sergio · status: AWAITING_ACCEPTANCE

Branch `sergio/batch-03` (from `sergio/batch-02` at `7c6d745`). Same Mac and tools as batch 02; `google-genai`
3.0.0 added. Every commit passed the offline suite (`-m "not needs_docker"`); the Docker tests were run again at
the end of the batch because Docker Desktop on this Mac restarts on its own every few minutes (it also runs two
unrelated Supabase stacks), which stalled whole-suite runs.

### Summary
- AIC-010 (G-05): `feedback.py` summarises jaxpr, StableHLO, pre-optimisation HLO and compiled HLO (op totals,
  dots/reductions/transposes/broadcasts/custom calls/fusions, top ops, parser version, provenance), keeps XLA's
  cost and memory estimates in their own namespaces (null + reason when unavailable), writes capped raw text.
  The worker produces it for the measurement case in the same run as correctness; it is recorded for the
  baseline and every candidate and shown to the agent only under C3/C4/K.
- AIC-018 (G-02): `heldout.py` + `scripts/heldout.py`: fixed `heldout_seeds` (≥ 20, disjoint from the search and
  edge seeds) × ≥ 2 unseen shapes in the registry, a final-evaluation entry point the search cannot reach, host-side
  verdict, burn ledger. Softmax split v1: seeds 1000–1019 × (64, 257), (256, 2048), frozen before any search.
  (First implemented as a salted commit-reveal scheme, then simplified in `4143f54`: the model never reads the
  repository, so public seeds are not given to it, and the commit already proves the freeze.)
- G-01 gaps: honest `scientific` flag (explicit backend + scientific workload + isolated runner, reasons listed),
  `correctness_policy_hash` recorded (AIC-002); `--workload` (AIC-003).
- P8: `--provider anthropic|anthropic-vertex|gemini-vertex`. The Anthropic API stays the default and unchanged;
  Claude on Vertex through `anthropic.AnthropicVertex`; Gemini on Vertex through `gemini.py`, an adapter with the
  Anthropic Messages surface, so the loop and the fakes are provider-neutral; usage per request in `requests.json`.

### Commits
| task | commit | element |
|---|---|---|
| AIC-010 | `5da1206` | compiler feedback at every lowering level |
| AIC-010 | `0a5d63c` | feedback for baseline and candidates, C3-only visibility |
| AIC-018 | `dfe0d35` | held-out split, commit-reveal, burn ledger |
| AIC-018 | `7ff40ab` | softmax split v1 frozen (salted) |
| AIC-018 | `4143f54` | split simplified to fixed seeds in the registry |
| AIC-002 | `6bfafab` | honest scientific flag, policy hash in the record |
| AIC-003 | `ffc893f` | `--workload` |
| P8 | `28c7ce7` | provider setting; Claude on Vertex |
| P8 | `14a2723` | Gemini on Vertex adapter |
| P8 | `6a2dadd` | per-request usage |

### Gates (Mac)
- `env -u ANTHROPIC_API_KEY uv run --locked pytest -q -m "not needs_docker"` → `374 passed, 1 skipped`.
- With Docker Desktop: `uv run --locked python scripts/build_worker_image.py` (new tag after `uv.lock` and
  `feedback.py`), then `pytest -q -m needs_docker` → `11 passed, 1 skipped` (the skip: `needs_gpu`). The
  end-to-end test now also produces the compiler feedback inside the container.
- `scripts/heldout.py check --workload softmax` → 40 cases (20 seeds × 2 unseen shapes), not burned.
- New test files: feedback 6, heldout 6, providers 13, gemini 5 (stand-in client, no network).

### Deviations and decisions (defaults recorded in `AI_CONTEXT.md` §6)
- P8 default: provider `anthropic` unchanged; two GCP providers added (P8, P16). Claude on Vertex is a partner
  model: whether the education credits cover it is not verified; `gemini-vertex` is billed as a Google model.
- AIC-018 lands before M4 because G-02 lists it; held-out evaluation is a separate entry point that the search
  loop and its tools cannot reach (asserted by a test).
- Calibration dtype and the IR are taken on the primary development shape only (`development_shapes[-1]`).

### Live provider runs · 2026-11-10 (GCP project `dd2430-508012`, Docker runner, Mac CPU)
- `gemini-vertex` (`gemini-2.5-flash`), softmax, `--iterations 2`: 2 requests (the first called `get_current_code`
  and `get_history`, the second `benchmark_code`), 847 input / 498 output tokens. Proposal: replace the
  implementation with `jax.nn.softmax`. It passed the source policy and all 16 correctness cases (edge cases
  included) inside the container, and its compiler feedback was recorded (compiled HLO: 39 ops, 4 reductions,
  2 fusions). The run then stopped as designed: the A/A calibration on the Mac was unstable (one session at
  IQR/median 0.112 > 0.10), so nothing was measured or accepted. All Gemini runs that day used about
  2.3k input / 2.7k output tokens, under one US cent at list price.
- With `--iterations 1` the inherited turn limit (3 × iterations) was spent on reading tools before any proposal;
  `requests.json` now lists the tools called per request so this is visible (the loop itself is AIC-015).
- Re-calibrating on the Mac with 40 ms blocks, and with a declared development limit of IQR/median 0.15, still
  failed (IQR 0.107, then baseline drift 0.126): the interactive Mac (WindowServer, desktop apps, the Docker VM) is
  not a measurement host. Not loosened further: timing evidence comes from the NV host (D-06).
- `anthropic-vertex` (`claude-haiku-4-5@20251001`): authentication and routing work, but Vertex returns 429
  "Quota exceeded … base model: anthropic-claude-haiku-4-5" on `global` (quota 0) and 404 on `us-east5` and
  `europe-west1` (model not available to the project there). This needs a quota increase request in the GCP
  console (P17). The 429 used to crash agent.py; since `e90fb64` it ends the run as a recorded request event.
- Docker Desktop on macOS refused a bind mount of a directory that had been deleted and recreated under the same
  name ("bind source path does not exist"); a new run directory name avoids it.

### Known limitations
- The runtime trace (AIC-011) is reported as `UNAVAILABLE`; budgets beyond candidates, deadlines and sizes, and
  the money cap, are still not enforced (AIC-013).
- GPU evidence (phase B, GPU containment, official δ, IR on CUDA) is still `scripts/phase_b.sh` on the NV host.

## Batch 02 (AIC-005, 006, 007, 008, 009) · 2026-11-09 · owner: Sergio · status: AWAITING_ACCEPTANCE

Branch `sergio/batch-02` (from `harry/batch-01` at `bff744d`). Implemented on Sergio's Mac (macOS, arm64, Python
3.12, jax/jaxlib 0.4.38, uv 0.11.26, Docker Desktop 29.6.1 linux/arm64). One small commit per element; every
commit passed the full offline suite before it was made.

### Summary
- AIC-005: paired-block timing protocol (`contracts.MeasurementProtocol`, `benchmark.measure_paired`), raw blocks,
  summary statistics, paired log-ratio bootstrap, stability diagnostics, A/A calibration (`benchmark.aa_threshold`,
  `benchmark.Calibration`, `scripts/calibrate_aa.py`), AOT compile timing.
- AIC-006: `controller.decide` / `judge` / `verify_winner`: stable session, CI lower bound > 1 + δ, one retry,
  confirmation in a fresh session, final check against the original baseline on every shape.
- AIC-007: Proposal v1 / Evaluation v1 / Attempt v1 (`contracts.py`), `source_policy.py`, `benchmark_code` takes a
  validated proposal.
- AIC-008: `array_io.py` (pickle-free exchange), `worker.py`, `runner.py` (`TrustedSubprocessRunner`), host-side
  correctness from worker outputs.
- AIC-009: worker image (`docker/worker.Dockerfile`, `scripts/build_worker_image.py`), `runner.DockerRunner`,
  hostile containment probes, CI `containment` job, and the controller rewritten around `Evaluator`: it never
  runs candidate code itself.

### Commits
| task | commit | element |
|---|---|---|
| AIC-005 | `8160e08` | measurement protocol v1 in the run config |
| AIC-005 | `bd7d2e5` | repeat count calibrated from the baseline only |
| AIC-005 | `ee9abc6` | paired AB/BA blocks with raw records |
| AIC-005 | `e6e6bb3` | summary statistics, paired bootstrap speedup |
| AIC-005 | `4f08e8d` | stability diagnostics, session analysis |
| AIC-005 | `10dd15f` | A/A calibration of delta, `Calibration` record |
| AIC-005 | `139f32a` | lowering and compilation timed separately |
| AIC-006 | `d87ebf0` | acceptance rule, one retry, fresh confirmation, final verification |
| AIC-007 | `d40c8a4` | Proposal / Evaluation / Attempt contracts |
| AIC-007 | `2241fb2` | source policy |
| AIC-007 | `ba0aeb6` | `benchmark_code` takes a validated Proposal v1 |
| AIC-008 | `445e0e3` | pickle-free, bounded array exchange |
| AIC-008 | `c8ff063` | worker (`outputs`, `measure`) |
| AIC-008 | `b89ad5c` | runner boundary, trusted subprocess runner |
| AIC-008 | `a0e9c7c` | host-side correctness from worker outputs |
| AIC-009 | `faaad18` | worker image from the lock file |
| AIC-009 | `c1cca65` | `DockerRunner`, real containment tests |
| AIC-009 | `3544bb8` | CI containment job |
| AIC-009 | `fed6a03` | controller evaluates only through runners and `judge()` |
| AIC-005 | `10f9e5d` | `scripts/calibrate_aa.py` |

### Gates (Mac, Docker Desktop running)
- Full suite: `env -u ANTHROPIC_API_KEY uv run --locked pytest -q` → `349 passed, 1 skipped` (the skip: `needs_gpu`).
- Per new test file: measurement 27, decision 10, source_policy 49, array_io 8, worker 7, runner 12,
  containment 10 (+1 `needs_gpu` skipped), isolated_run 1 — all passed.
- Containment probes on Docker Desktop (linux/arm64), all held with the source policy bypassed: no network or DNS
  (`OSError` / `gaierror`), uid 65534, `CapEff` 0, `NoNewPrivs` 1, no credentials in the environment (only the
  base image's public `GPG_KEY`), no host paths, read-only root (`/app`, `/etc` not writable), `/tmp` scratch
  writable, fewer than 128 processes, memory exhaustion contained, a hung container killed at the deadline, no
  container left behind.
- The 18-fixture correctness matrix gives the same verdicts, case by case, through the worker + host comparison
  as in-process.
- End to end in Docker (`tests/test_isolated_run.py`): softmax, `jax.nn.softmax` rewrite measured with raw blocks;
  the naive softmax rejected as `CORRECTNESS_FAILED` (NONFINITE).
- First calibration trial, Docker Desktop on the Mac, softmax (128, 1024): 111 calls per block, δ = 0.076 from
  5 A/A sessions, 11 s. Not committed: official calibrations come from the NV host (D-06).

### Deviations and decisions (defaults recorded as P12–P15 in `AI_CONTEXT.md` §6)
- Isolation is Docker (P12); without it, model-written code is refused and the run stops before any LLM call
  (P13). `--runner subprocess` keeps baseline and proposal-only work available.
- The A/A calibration is taken lazily, before the first candidate measurement, unless a frozen `--calibration`
  file is given; a noisy host stops the run instead of loosening the threshold (P14).
- `UNCALIBRATED` is not a candidate state: a missing or unusable calibration ends the run as `INTERRUPTED`.
- New flat modules beyond the plan's list: `source_policy.py` and `array_io.py` (each testable on its own; P15).
- The configured backend is forwarded as `selected_backend=` so the rule-1 AST guard keeps rejecting `backend=`.
- Tests that need short sessions use loosened instability limits; the defaults (10 %, 5 %) are unchanged and the
  guard against speed claims in ordinary tests still holds (synthetic numbers use `math.isclose`).

### Known limitations
- All timing here is CPU, in a Docker VM on macOS: correctness and mechanics only, never a speed claim (D-06).
- GPU containment, the GPU A/A calibration and phase B of AIC-031 need the NV host: `scripts/phase_b.sh`.
- The bind-mounted work directory is writable and not size-limited by Docker; result files are size-checked
  before they are read.
- Budgets other than candidates, deadlines and sizes (requests, tokens, money) are still recorded, not enforced
  (AIC-013).

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
- Known limitations: CI had not run when this was written (first run: see "First CI run" below); `astral-sh/setup-uv@v6` and `actions/checkout@v4`
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

### First CI run · 2026-11-09
- Run 42599232878 (`ci` workflow, push of `harry/batch-01` at c50cb1e, ubuntu-latest, CPython 3.12.3): success.
- `uv sync --locked` then `uv run --locked pytest -q`: 179 passed, 1 skipped in 32 s. The skip is
  `test_cuda_on_host` (`needs_gpu`: CUDA unavailable), the same result as on the Mac.

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
