# Codex implementation backlog — whole project

Version **AIC-1.1**, 2026-10-12. **PLAN APPROVED** by the owner on 2026-10-12;
M1 entry authorization remains pending. See PLAN-APPROVAL in
[IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md).

33 tasks: **32 main-project tasks in M1–M7**, one optional second-target task in
M8. No implementation is authorized or DONE. All test commands below are future
verification instructions; audit results are in [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md).
The [plan](IMPLEMENTATION_PLAN.md) defines common numerical, measurement, safety,
profile, budget and approval contracts.

## Identity and scope revision

The owner requested a general, hardware-neutral whole-project roadmap. Each
unstarted draft task `CPU-NNN` (001–030) now has canonical ID `AIC-NNN` with the
same numeric identity; this one-to-one mapping preserves traceability. No task
was completed, removed or merged. Old CPU-prefixed IDs are historical aliases,
not active CPU-only scope. New AIC-031 selects the target, AIC-032 establishes a
compact complete model, and AIC-033 qualifies optional second-target use. Kernel
and model tasks AIC-027–030 are now main-project REQUIRED tasks before evaluation,
not automatic deferrals. The listing is in execution order, not numeric order.

All tasks start NOT_STARTED except AIC-033 DEFERRED. READY requires both task
prerequisites and authorization of its milestone. The progress tracker is the
execution record; update these Status fields together. A K2/K3 kernel deferral
requires an explicit amended task/milestone/dependency plan, never silently
marking unmet prerequisites DONE.

## Test command convention

Offline suite after AIC-001:
`uv run --locked pytest tests/ -m 'not target and not sandbox and not live' -q`.
The existing CPU smoke may run here as a development convenience; it does not
verify another hardware target.

Actual selected-target/sandbox suite after AIC-002/009:
`AIC_TARGET_CONFIG=configs/target.json uv run --locked pytest tests/ -m 'target or sandbox' -q`.
AIC-001 fixtures/markers and AIC-002 config must implement this contract, setting
backend selection before importing JAX and checking actual device. Missing target
or sandbox capability is a failed/blocked acceptance gate, not a successful skip.
Live tests are explicit, separately funded runs; never automatic in either suite.

Single-file commands in tasks exercise focused tests; where criteria require real
target evidence, also run the shared selected-target command or the documented
actual-target diagnostic. No ordinary unit test may require a wall-clock speedup.
New filenames below are proposed paths, not claimed existing modules.

## Task ID: AIC-001

**Title:** Establish a safe, offline test entry point.

**Milestone:** M1.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Make automated verification runnable without importing interactive scripts or reading credentials.

**Reason:** Audit D01 found no test source and no pytest; root test.py dumps the environment.

**Current codebase state:** pyproject.toml has no test dependency/config; tests/ contains only cache; test1.py–test3.py run provider loops.

**Dependencies:** None (plan approved; M1 authorization still required).

**Implementation instructions:** Add a pinned pytest development dependency through uv without upgrading unrelated packages. Restrict collection to tests/. Define offline, target, sandbox and live markers with live execution opt-in and explicit cap checks. Add safe import/CLI tests and the already verified CPU addition smoke as a portable development check. Target acceptance uses AIC-031/002 configuration separately. Add secret filename ignores; review tracked .env filenames without exposing contents, preserve local files and never rewrite Git history.

**Expected files:** pyproject.toml, uv.lock, .gitignore, README.md; new tests/test_smoke.py and tests/conftest.py (safe offline test harness).

**Implementation constraints:** No live API calls or generated code. CPU is a convenient existing smoke-test environment, not the selected project target. No broad dependency upgrade.

**Testing instructions:** uv sync --locked; uv run --locked pytest --collect-only -q; uv run --locked pytest tests/test_smoke.py -q. Verify collection includes only intended tests and all credentials can be absent.

**Acceptance criteria:**

- [ ] More than zero offline assertions execute.
- [ ] Interactive/environment-dump scripts excluded.
- [ ] Offline tests need no credentials.
- [ ] Target/sandbox/live markers cannot imply verification from skips.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** Network/setup unavailable: record startup blocker; do not report a passing suite from zero tests.

**Out of scope:** Provider experiments, module reorganization, repairs to unused algorithm.py.

**Completion evidence:** Test/dependency diff, collected test count, actual commands/exit codes, marker policy and credential-hygiene decision without secret values.

## Task ID: AIC-031

**Title:** Choose and prove the primary hardware target.

**Milestone:** M1.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Select one practical CPU, GPU or TPU target from actual available resources.

**Reason:** Hardware-neutral architecture needs an explicit implementation target before meaningful timing, isolation and kernel work.

**Current codebase state:** Only CPU JAX 0.4.38 execution/IR/trace is verified; optional CUDA dependency is unverified GPU packaging. Hardware availability beyond the audited machine is unknown.

**Dependencies:** AIC-001.

**Implementation instructions:** Inventory owner-provided hardware/access without provisioning or spending implicitly. Compare availability, supported JAX/runtime, isolation feasibility, profiler events, native/kernel route, memory and cost. Run bounded trusted JAX compile/execute/IR/trace probes on accessible candidates; record unsupported/untested options. Recommend one primary target and minimal pinned environment changes, resource limits and candidate execution boundary. Obtain owner selection; preserve CPU audit results as dated evidence even if selecting an accelerator. No need to implement multiple backends.

**Expected files:** New docs/TARGET_DECISION.md and tests/test_target_preflight.py; README.md and progress/plan decision record. Config implementation follows AIC-002.

**Implementation constraints:** Actual target access and allocation authorization required. No presumed hardware, cloud purchase, generated code, arbitrary remote execution or silent CPU fallback. Investigate isolation/kernel feasibility here; implement and verify them in their later tasks.

**Testing instructions:** Record actual device identity, pinned Python/JAX/jaxlib/plugin/driver versions, a trusted addition/softmax compile/execute, IR and trace coverage probe, relevant resource limits and access restrictions. Test preflight rejects an unavailable requested backend.

**Acceptance criteria:**

- [ ] Options compared using actual access/evidence.
- [ ] Selected target runs a trusted JAX program.
- [ ] Profiler/isolation/kernel feasibility and unresolved risks explicit.
- [ ] Owner selection and supported environment recorded.

**Definition of Done:** Every acceptance item is evidenced; implementation and documentation agree; listed tests and affected regressions pass; actual results/limitations are recorded in the tracker. No skipped mandatory selected-target check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** No accessible suitable target: finish inventory/offline tests, mark BLOCKED and request access or revised scope; do not fabricate accelerator evidence.

**Out of scope:** Full sandbox/kernel implementation or selecting a second target.

**Completion evidence:** Target decision, probe commands/results, actual device/software identity, owner choice and minimal environment delta.

## Task ID: AIC-002

**Title:** Resolve target configuration and environment provenance.

**Milestone:** M1.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Create a validated run configuration for the explicitly selected hardware and capture reproducibility metadata.

**Reason:** Existing artifacts cannot identify hardware, flags, versions or model settings.

**Current codebase state:** agent.py:102 parses scattered CLI options; agent.py:119 only prints devices; pyproject.toml/uv.lock already pin JAX.

**Dependencies:** AIC-031.

**Implementation instructions:** Implement RunConfig v1 with CLI > JSON > defaults, strict budgets and selected backend/device. Use AIC-031 decision to pin any approved version/runtime adjustment; retain old audit provenance. Apply backend selection before JAX initialization, verify actual device identity and fail if unavailable. Add credential-free baseline mode. Record source/lock/runner hashes, software, CPU host metadata and accelerator model/driver/runtime/topology where relevant, with explicit unavailable reasons. Save configs/target.json as the nonsecret selected-target contract.

**Expected files:** agent.py, pyproject.toml/uv.lock only for target-decision-approved changes; new contracts.py, configs/target.json, tests/test_config.py, tests/test_environment.py; README.md.

**Implementation constraints:** Implement one selected target configuration; no silent CPU fallback, environment dump or generic backend class hierarchy. Unknown hardware fields remain explicit.

**Testing instructions:** uv run --locked pytest tests/test_config.py tests/test_environment.py -q; selected-target preflight using configs/target.json. Simulate missing requested device and wrong runtime; assert failure before any LLM request.

**Acceptance criteria:**

- [ ] Resolved config roundtrips.
- [ ] Actual requested device proven before evaluation.
- [ ] Zero budgets make no provider calls.
- [ ] Versions/device/driver/runner identity recorded.
- [ ] Missing target cannot silently run on CPU.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Target metadata or runtime unavailable: record blocker and retain offline checks; any primary-target switch needs owner decision.

**Out of scope:** A second backend, full containment implementation or unapproved dependency changes.

**Completion evidence:** Redacted example config/manifest, preflight outputs, tests and documented missing host fields.

## Task ID: AIC-003

**Title:** Define seeded workloads and a stable softmax reference.

**Milestone:** M1.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Replace guessed scientific inputs with an explicit workload contract.

**Reason:** The fixed 32-element addition cannot test correctness breadth or meaningful reductions.

**Current codebase state:** workloads/example.py has TEST_ARGS only; agent.py:25 supports trusted source loading and guesses fallback inputs.

**Dependencies:** AIC-002.

**Implementation instructions:** Add immutable WorkloadSpec records and simple registry with input(seed, shape, dtype), fixed reference source/hash, static parameters, declared splits and edge cases. Adapt the existing example without rewriting it. Add last-axis numerically stable float32 softmax using max subtraction. Define development shapes/seeds and output invariants. Preserve trusted-file loading but require explicit cases for research runs; no inferred argument arrays there.

**Expected files:** agent.py, contracts.py, workloads/example.py if adapter metadata needed; new workloads/registry.py, workloads/softmax.py, tests/test_workloads.py.

**Implementation constraints:** Pure forward functions and explicit inputs/static parameters; no dynamic shape discovery, gradients or arbitrary plugin imports. Validate the registered workloads on the selected target using the target test command.

**Testing instructions:** uv run --locked pytest tests/test_workloads.py -q. Check seed reproducibility/distinction, eager/JIT equivalence, sums near one, finite extreme-offset cases and a small NumPy reference.

**Acceptance criteria:**

- [ ] Addition and softmax registered.
- [ ] Seeds/shapes/dtypes explicit.
- [ ] Reference source hash stable.
- [ ] Edge cases finite and mathematical checks pass.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Starting shapes unsuitable for selected device: calibrate within resource budget before search and record adjusted shapes.

**Out of scope:** Layer norm, attention, Transformer and native kernels until their scheduled tasks.

**Completion evidence:** Workload contract, source hashes, shape/seed list, reference tests and declared unsupported semantics.

## Task ID: AIC-004

**Title:** Separate compilation and multi-case numerical validation.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Reject incorrect implementations with structured evidence over the supported workload domain.

**Reason:** Single-array single-input allclose is insufficient and tuple outputs currently fail.

**Current codebase state:** benchmark.py:35 combines loading, JIT, comparison and timing; D02 verifies narrow rejections and tuple failure.

**Dependencies:** AIC-003.

**Implementation instructions:** Refactor trusted callable evaluation into explicit trace/lower/compile/execute/compare stages. Compare PyTree structure, leaf shape/dtype, integer equality and workload-specific floating tolerances. Synchronize all leaves. Record errors/nonfinite masks per case and catch stage failures. Keep reference outputs independent of candidate globals. Cover five development seeds at two shapes plus edge cases; timing cannot start after any failure.

**Expected files:** benchmark.py, contracts.py; new tests/test_correctness.py.

**Implementation constraints:** Trusted reviewed fixtures only until AIC-009 passes. Do not execute model output or relax tolerances to obtain speedups.

**Testing instructions:** uv run --locked pytest tests/test_correctness.py -q. Fixtures: valid tuple/dict, wrong tree, shape, dtype, values, unexpected NaN/Inf, import/trace failure; timer spy asserts zero calls after rejection.

**Acceptance criteria:**

- [ ] All rejection categories distinct.
- [ ] Valid PyTrees pass.
- [ ] Multiple cases and workload tolerances enforced.
- [ ] Failed correctness never reaches timing.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Numerical extremes expose baseline flaws: fix/reference-review the workload before optimizing; policy changes documented.

**Out of scope:** Security sandbox and performance selection.

**Completion evidence:** Fixture/result matrix, tolerance rationale, commands/results, failed-case artifact example.

## Task ID: AIC-005

**Title:** Implement paired measurements and A/A calibration on the selected target.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Produce raw synchronized baseline/candidate measurements separated from compilation and host/device transfer.

**Reason:** The current mean/std timer loses samples and cannot distinguish drift from improvement.

**Current codebase state:** benchmark.py:8 warms and blocks single-array calls but returns only batch mean/std; audit D07 exposes noise.

**Dependencies:** AIC-004.

**Implementation instructions:** Implement plan section 5: target-verified device placement; input transfer/synchronization outside timing; explicit lowering/compile durations; all-leaf warmup; common batch count; balanced AB/BA blocks; raw durations/order and summary. Record CPU thread/allocation or accelerator device/memory/driver controls selected in AIC-031. Run at least five independent A/A sessions for example and softmax primary shape. Freeze empirical delta and instability thresholds before search. Use synchronized end-to-end JAX latency for every target; optional device events are additional metrics only.

**Expected files:** benchmark.py, contracts.py; new tests/test_benchmark.py; README.md measurement instructions.

**Implementation constraints:** No live LLM or concurrent compilation/benchmarks; no profiles during timing. Never compare baseline on a different device/runner or treat asynchronous enqueue time as runtime.

**Testing instructions:** uv run --locked pytest tests/test_benchmark.py -q for fake-clock/statistical fixtures; selected-target tests and baseline/calibration CLI using configs/target.json in idle sessions. Verify device completion and transfer/compile/warmup exclusion explicitly.

**Acceptance criteria:**

- [ ] >=20 paired blocks/session retained.
- [ ] Compilation excluded and recorded.
- [ ] A/A noise floor/thresholds derived and saved.
- [ ] Unstable measurement classified.
- [ ] Protocol identical for both sides.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Dispatch overhead, shared-device contention or drift: bounded longer blocks/larger predeclared shapes or dedicated allocation; no performance claim if instability persists.

**Out of scope:** Claimed speedup, native-kernel microbenchmark replacement of end-to-end latency or multi-device comparison.

**Completion evidence:** Raw calibration sessions, config/environment hashes, median/IQR/drift results and selected delta justification.

## Task ID: AIC-006

**Title:** Make acceptance uncertainty-aware and baseline-preserving.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Replace any-lower-mean decisions with correctness-gated, calibrated comparisons.

**Reason:** controller.py currently accepts timing differences much smaller than measured noise.

**Current codebase state:** controller.py:86 compares runtime_mean < best_runtime; original baseline is not a separate comparison record.

**Dependencies:** AIC-005.

**Implementation instructions:** Add a pure decision function using paired-block ratio interval, calibrated delta and stability flags. Require second fresh-session confirmation before promotion; otherwise no-change/unstable. Preserve immutable baseline and incumbent identities. Distinguish slower, inconclusive, invalid and accepted. Bound retries to one and never update best on missing/nonfinite data.

**Expected files:** controller.py, benchmark.py, contracts.py; new tests/test_decisions.py.

**Implementation constraints:** No old 2% constant and no automatic performance assertions from wall-clock unit tests. Correctness cannot be outweighed.

**Testing instructions:** uv run --locked pytest tests/test_decisions.py -q. Synthetic datasets include clear improvement, exact tie, slower, drift, high variance, missing data, failed correctness and contradictory confirmation.

**Acceptance criteria:**

- [ ] Audit D05 noise-only case rejected.
- [ ] Incorrect or unstable candidate cannot win.
- [ ] Clear synthetic gain requires confirmation.
- [ ] Baseline immutable.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** Interval estimates fragile on low sample count: classify inconclusive and collect only bounded protocol-approved confirmation.

**Out of scope:** LLM/controller integration and multiple-hypothesis research claims.

**Completion evidence:** Decision fixture outcomes, statistical parameters, M2 review with calibration evidence.

## Task ID: AIC-007

**Title:** Validate proposals and restrict candidate source policy.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Create a narrow, explicit boundary before any candidate execution.

**Reason:** Tool schemas and a code-string check do not enforce local semantics or safe actions.

**Current codebase state:** controller.py:147 declares one string; _dispatch_tool validates only that string; agent.py prompt constraints are unenforced.

**Dependencies:** AIC-006.

**Implementation instructions:** Define Proposal v1 and local validator for all plan fields, enums, lengths, unknown keys, parent identity and target. Parse source AST without execution. Document/implement an allowlist for pure JAX imports, definitions, numeric literals and approved operations sufficient for current workloads; reject top-level actions, arbitrary imports, dunder access, eval/exec and global/module mutation. Validate signature and distinguish schema from policy failure. Hash candidates for duplicate detection.

**Expected files:** contracts.py, agent.py, controller.py; new tests/test_proposals.py.

**Implementation constraints:** AST validation is not a security sandbox. No execution of rejected source. Native/FFI targets remain disabled.

**Testing instructions:** uv run --locked pytest tests/test_proposals.py -q. Accept reviewed softmax rewrites; reject malformed JSON, extra keys, fences, oversize source, missing function, import os, top-level calls and forbidden attributes.

**Acceptance criteria:**

- [ ] Full schema enforced locally.
- [ ] Valid restricted sources pass.
- [ ] Invalid/policy failures have no side effects.
- [ ] Duplicate identity deterministic.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Allowlist too narrow: smallest tested pure construct extension; never silently allow arbitrary Python.

**Out of scope:** Multi-file changes, agent tools that write files, native kernels.

**Completion evidence:** Schema examples, policy document and accept/reject fixture matrix.

## Task ID: AIC-008

**Title:** Implement bounded worker lifecycle and artifact protocol.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Move evaluation out of the controller with explicit inputs, timeouts and outputs.

**Reason:** In-process exec shares controller credentials/state and cannot recover from hangs.

**Current codebase state:** benchmark.py:51 executes candidate; controller.py:168 directly calls it; JSON records already provide an adaptation point.

**Dependencies:** AIC-007.

**Implementation instructions:** Add runner.py lifecycle and worker.py selected-target evaluation with bounded JSON/array protocol. Parent controls source/config/inputs, independent reference comparisons and acceptance. Capture/truncate logs; classify stage/timeout/resource/protocol errors; terminate and reclaim the entire job and target allocation. Validate non-pickle array sizes/paths/hashes and ignore worker-supplied correctness flags. Local subprocess mode supports reviewed trusted fixtures only; generated execution requires AIC-009. Account for driver calls that may outlive a process and test cleanup on the chosen runtime.

**Expected files:** controller.py, benchmark.py, contracts.py; new runner.py, worker.py, tests/test_runner.py.

**Implementation constraints:** Subprocess is lifecycle isolation only. No untrusted autonomous execution in this task; reviewed fixtures only. Reference outputs not provided to candidate.

**Testing instructions:** uv run --locked pytest tests/test_runner.py -q. Reviewed fixtures exercise crash, timeout, excessive logs, malformed/oversized output and cleanup, with tight resource-safe bounds.

**Acceptance criteria:**

- [ ] Controller never imports reviewed candidate into its own namespace.
- [ ] Stage deadlines and output caps enforced.
- [ ] Invalid worker output cannot update best.
- [ ] Child cleanup verified.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Platform/device cleanup differs: implement only the selected job boundary; inability to reclaim it blocks autonomous execution, not permission to use a weaker path.

**Out of scope:** Live provider integration and sandbox security approval.

**Completion evidence:** Protocol samples, lifecycle failure results, process cleanup evidence and trusted-only mode guard.

## Task ID: AIC-009

**Title:** Verify the selected-target execution boundary.

**Milestone:** M2.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Permit generated execution only after actual host/device containment and cleanup are demonstrated.

**Reason:** CPU Docker access was unverified in the audit; accelerator access additionally needs target-specific containment evidence.

**Current codebase state:** No container configuration or isolation preflight exists; AIC-008 supplies lifecycle only.

**Dependencies:** AIC-008.

**Implementation instructions:** Implement the one execution boundary selected in AIC-031. For container deployment use non-root, read-only root, dropped capabilities, no-new-privileges, tested seccomp, bounded memory/CPU/PIDs/scratch, narrow mounts and only required device exposure. For a dedicated VM/managed worker prove equivalent controls. Deny candidate egress and metadata access; constrain any required runtime traffic to verified endpoints outside candidate authority. Exclude credentials/project/home/control sockets. Preflight returns ISOLATION_UNAVAILABLE if a required control fails. Use the same allocation and timing boundary for baseline/candidate; cleanup includes accelerator jobs/contexts.

**Expected files:** runner.py, worker.py, README.md; new tests/test_sandbox.py; Containerfile.worker and .dockerignore if container chosen, otherwise a nonsecret configs/worker.json describing the chosen worker policy. These paths are proposed, not existing infrastructure.

**Implementation constraints:** No broad privileged device container or unrestricted subprocess fallback. Provisioning/spend needs actual authorization; installing a runtime is not implied by writing the plan.

**Testing instructions:** AIC_TARGET_CONFIG=configs/target.json uv run --locked pytest tests/test_sandbox.py -m sandbox -q on the real selected worker. Reviewed fixtures exercise network/metadata/host-file denial, missing credentials, read-only writes, bounded resource exhaustion and descendant/device cleanup.

**Acceptance criteria:**

- [ ] Real target containment tests pass without skips.
- [ ] Selected JAX workload runs under declared limits.
- [ ] Failed control blocks generated code.
- [ ] Device jobs/resources reclaimed after timeout.
- [ ] Reference/candidate allocation and settings match.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** No safe target access: complete independent proposal/trusted-harness work, mark BLOCKED and request revised target/runtime decision; M2 exit cannot pass.

**Out of scope:** Multiple execution backends, remote service platform or second device support.

**Completion evidence:** Actual target/worker/image/runtime identity, launch policy, denial/cleanup results and accepted trust boundary.

## Task ID: AIC-010

**Title:** Extract pinned compiler artifacts and compact summaries.

**Milestone:** M3.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Give the agent verifiable selected-target JAX/XLA information.

**Reason:** IR APIs work locally but repository never collects them.

**Current codebase state:** Audit D03 verified Jaxpr, StableHLO, HLO, cost list and memory stats under 0.4.38; no feedback.py exists.

**Dependencies:** AIC-009.

**Implementation instructions:** Add feedback.py using D03 APIs as a CPU-verified starting point, then verify on selected target/pinned software. Retain Jaxpr/StableHLO/lowered-HLO/compiled-HLO artifacts with IDs and byte caps. Normalize optional cost/memory fields and availability status. Summarize operation categories conservatively with parser version. Return small summaries plus artifact references to the controller.

**Expected files:** worker.py, contracts.py; new feedback.py, tests/test_compiler_feedback.py.

**Implementation constraints:** No HLO rewriting, API upgrade or treating static metrics as runtime observations.

**Testing instructions:** uv run --locked pytest tests/test_compiler_feedback.py -q for parser/availability fixtures; selected-target addition/softmax extraction with configs/target.json. Verify actual artifacts and separate lowered/compiled representations.

**Acceptance criteria:**

- [ ] Nonempty selected-target IR and summary.
- [ ] Raw hashes identify actual candidate/environment.
- [ ] Unsupported estimates are null with reasons.
- [ ] Context size limits respected.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Selected version/backend differs from audit: test actual APIs; omit unsupported optional estimates rather than copying latest examples blindly.

**Out of scope:** Optimization of IR text or compiler-pass tuning.

**Completion evidence:** Example artifact inventory, API versions, extraction tests and cost-versus-runtime labeling.

## Task ID: AIC-011

**Title:** Collect runtime traces and report actual device coverage.

**Milestone:** M3.

**Priority:** REQUIRED.

**Status:** BLOCKED for selected-target acceptance (2026-11-16). Implementation and CPU verification complete;
CUDA/device-event and container verification require the NV host and a running Docker daemon.

**Goal:** Add observed runtime/profile feedback on the selected target, with an explicit limitation path.

**Reason:** Runtime timing and compiler estimates cannot substitute for actual profile observations.

**Current codebase state (2026-11-16):** bounded Perfetto capture/parser and RuntimeTrace v1 implemented;
separate profile workers collect baseline/promoted-incumbent evidence; C4 exposes observed coverage. Real CPU
worker capture, overlap/corruption/identity fixtures and timing separation are verified. Original D04 was only
a standalone probe. Selected-target/container acceptance remains pending; see IMPLEMENTATION_PROGRESS.md.

**Dependencies:** AIC-010.

**Implementation instructions:** Probe jax.profiler.trace on the selected runtime after compile/warmup; retain raw data and parse bounded event summaries. Distinguish CPU_EXECUTION_EVENTS, DEVICE_EXECUTION_EVENTS, HOST_ONLY and UNAVAILABLE. Validate timestamps, nesting and overlap; separate host dispatch from device execution and avoid summed overlapping percentages. Install an additional profiler component only if target evidence justifies it and scope permits. Benchmark without trace instrumentation; reprofile promoted best.

**Expected files:** feedback.py, worker.py, contracts.py; new tests/test_runtime_profile.py.

**Implementation constraints:** No synthesized hotspots, no attribution beyond event evidence, no TensorFlow/advanced profiler install just to silence optional warnings.

**Testing instructions:** uv run --locked pytest tests/test_runtime_profile.py -q for parser fixtures, plus a real selected-target trace test. Check event type/device identity, nested intervals, corruption, unsupported fields and missing profiler permission.

**Acceptance criteria:**

- [x] Actual CPU trace summary or explicit unavailable reason produced.
- [x] Timing/estimates/profile separated; measurement mode cannot start a profiler.
- [x] No overlapping-event percentage inflation; per-lane/name inclusive unions verified.
- [x] Agent-consumable coverage field, exposed only under C4.

These checks establish CPU behavior. Mandatory selected-target verification and result acceptance remain open;
this task is not DONE and does not pass the M3 gate.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Host-only/unavailable profiling: record wall time plus IR, R-06 partial and requested scope acceptance; it does not establish device hotspots or profiling-benefit claims.

**Out of scope:** Multiple vendor profiler implementations or claiming causality from event names alone.

**Completion evidence:** CPU commands/results, hashed raw artifact and source identity, coverage/overlap/fallback
fixtures, exact Docker/CUDA follow-up commands in the 2026-11-16 AIC-011 entry of IMPLEMENTATION_PROGRESS.md.

## Task ID: AIC-012

**Title:** Persist unique runs and immutable attempt artifacts.

**Milestone:** M3.

**Priority:** REQUIRED.

**Status:** DONE (2026-11-16; offline storage scope verified, 47 storage checks and full suite 470 passed / 16 existing Docker-CUDA skips).

**Goal:** Make every baseline, request and attempt traceable and export the best source.

**Reason:** runs/latest can overwrite history and lacks provenance or standalone candidates.

**Current codebase state:** storage.py owns unique run reservation, immutable hashed evidence, commit-last
attempts and derived/validated best export. agent.py/controller.py integrate publication before promotion;
worker.py and workloads/registry.py preserve exact source bytes. Offline storage and full-suite verification
pass. Evidence: IMPLEMENTATION_PROGRESS.md, AIC-012.

**Dependencies:** AIC-009.

**Implementation instructions:** Extract reusable atomic writing to storage.py. Create non-overwriting run IDs, schema-versioned manifest, config/env/source/lock hashes, baseline and candidate .py, per-attempt result, proposal/request refs, raw measurement and feedback artifacts. Parent controls filenames. Commit attempt record last, then derive history/best indexes; interrupted partial output cannot become best. Export best.py with reference scope, including baseline when nothing wins. Never overwrite user's original.

**Expected files:** controller.py, contracts.py; new storage.py, tests/test_storage.py.

**Implementation constraints:** Use JSON/files, no SQLite prerequisite. Secret filtering and artifact caps; never load candidate source while reporting.

**Testing instructions:** uv run --locked pytest tests/test_storage.py -q. Verify collision refusal, atomic failure boundaries, exact source hashes, best=baseline fallback and raw-artifact references.

**Acceptance criteria:**

- [x] Independent runs cannot overwrite each other.
- [x] Every committed attempt has identity/evidence.
- [x] Best export resolves to verified source.
- [x] Partial attempts not accepted.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Large artifacts: bounded retention with explicit omissions; never discard acceptance raw timing/correctness evidence.

**Out of scope:** Resume implementation, dashboards or relational database.

**Completion evidence:** 47 offline storage checks pass, including collisions, atomic-failure boundaries,
tamper rejection, baseline fallback, request/raw references, caps/credentials, CRLF source identity and no
execution while reporting. Native CPU evidence: runs/aic-012-cpu-validation-20261116-final/validation.json
(fake provider only, incorrect candidate rejected, baseline export, original hash unchanged). Full suite:
470 passed / 16 existing Docker-CUDA skips; see IMPLEMENTATION_PROGRESS.md.

## Task ID: AIC-013

**Title:** Add a structured Anthropic proposal boundary and budgets.

**Milestone:** M3.

**Priority:** CRITICAL.

**Status:** DONE (2026-11-16; offline scope verified: 67 provider checks, full suite 537 passed / 16 existing Docker-CUDA skips; live proof remains AIC-016).

**Goal:** Use one testable provider call with explicit limits and offline fakes.

**Reason:** Existing injection is reusable, but raw tool calls have no cost accounting or full proposal validation.

**Current codebase state:** providers.ProposalProvider owns the sole structured SDK dispatch boundary;
controller.py validates/persists proposals before host-controlled evaluation. Live cap/price gating,
durable per-attempt reservations, conservative reconciliation, bounded retries/transport and offline replay
are implemented. CLI proposal-only mode skips source import/evaluation. Offline verification passes;
live service proof remains AIC-016.

**Dependencies:** AIC-012, AIC-007.

**Implementation instructions:** Implement propose(context) using current Anthropic SDK and Proposal v1. Record model parameters, request/response IDs, bounded redacted payloads, usage and refusals. Reserve worst-case tokens/cost before calls; explicit dated prices/money cap required for live mode. Bound transport timeouts/retries and charge every retry. Provide deterministic fake client and proposal-only mode. Remove unnecessary tool-driven arbitrary evaluation authority.

**Expected files:** agent.py, controller.py, contracts.py, storage.py; new tests/test_provider.py.

**Implementation constraints:** No extra provider framework or default paid call. Do not change model provider without reason/approval.

**Testing instructions:** uv run --locked pytest tests/test_provider.py -q. Fake SDK cases include malformed tool response, refusal, timeout, rate limit, partial usage, cap exhaustion and no configured money cap.

**Acceptance criteria:**

- [x] Locally validated proposals only.
- [x] Missing cap blocks live calls.
- [x] Retry usage bounded/accounted.
- [x] Tests require no secrets/network.
- [x] Provider failure persisted.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** SDK/model compatibility: test fixtures against installed response schema, proposal-only error; live proof deferred to AIC-016.

**Out of scope:** Provider shopping, fine-tuning and live optimization experiments.

**Completion evidence:** 67 provider checks (including installed SDK with local HTTP mocks), full suite
537 passed / 16 existing Docker-CUDA skips; checked-in offline fixture and real proposal-only CLI smoke.
Retained redacted request/reservation and synthetic price-provenance evidence:
runs/aic-013-sdk-validation-20261116-reviewed/validation.json. Both retry attempts accounted, unknown
usage reserved, zero remote calls, original source unchanged. Details: IMPLEMENTATION_PROGRESS.md AIC-013.

## Task ID: AIC-014

**Title:** Construct bounded compiler/runtime/history context.

**Milestone:** M3.

**Priority:** REQUIRED.

**Status:** DONE (2026-11-16; bounded context and offline/CPU evidence).

**Goal:** Supply relevant measured feedback for the next proposal without unbounded history.

**Reason:** provide_hist returns the entire history; raw tool prompts omit compiler/profile context.

**Current codebase state:** context_builder.py selects bounded, provenance-tagged development context;
controller/CLI use it before requests. storage.py persists exact contexts and selection manifests before dispatch.

**Dependencies:** AIC-010, AIC-011, AIC-012, AIC-013.

**Implementation instructions:** Build deterministic context from workload, baseline, incumbent, selected-target environment, correctness rules, compiler/profile summaries, last five failures, up to three accepted transformations and remaining budgets. Store exact included fields and omitted IDs. Enforce byte/token caps, never dropping constraints or latest result before old snippets. Keep source comments/logs as data. Exclude held-out data and credentials.

**Expected files:** controller.py, contracts.py, storage.py; new tests/test_context.py.

**Implementation constraints:** No vector database, separate memory service or model-generated performance facts.

**Testing instructions:** uv run --locked pytest tests/test_context.py -q. Verify compilation/correctness failures appear in next context, deterministic truncation, coverage labels, baseline/current distinction and secret/holdout exclusion.

**Acceptance criteria:**

- [x] Later context includes prior outcome: native compile/correctness failures appear in requests 2/3.
- [x] Size caps honored: 24 KiB metadata, separate source, encoded request/token/money checks and trimming tests.
- [x] Metrics have provenance: source/protocol/device/parser identities and hashed evaluation/trace references.
- [x] Current-best and original baseline remain distinct: separate IDs/hashes/timing, promoted-fixture tests.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** IR too large: retain raw artifact outside prompt and bounded summary; record omissions.

**Out of scope:** Sophisticated retrieval and experiment ablation execution.

**Completion evidence:** tests/test_context.py: 31 passed; full suite 568 passed / 16 Docker-CUDA skips.
Two successive failure-informed C4 contexts and accounting retained under
runs/aic-014-context-validation-20261116/validation.json; exact hashes and original source verified.
No paid calls or target/live acceptance claimed. Details: IMPLEMENTATION_PROGRESS.md AIC-014.

## Task ID: AIC-015

**Title:** Integrate the bounded current-best optimization controller.

**Milestone:** M3.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Connect proposal, sandbox evaluation, decision, storage and feedback into one loop.

**Reason:** Existing loop proves plumbing but budgets, baseline failure and error recovery are weak.

**Current codebase state:** Controller.agent_loop, updateHist, _dispatch_tool and initHist are existing integration points; D06 zero-budget defect.

**Dependencies:** AIC-006, AIC-009, AIC-012, AIC-013, AIC-014.

**Implementation instructions:** Make initialization explicit and abort on invalid baseline. Check every budget before request/worker launch, dispatch one validated proposal at a time, count rejected/malformed attempts consistently, deduplicate hashes, evaluate only in verified runner, confirm promotion using AIC-006 and commit outcome before next proposal. Persist terminal failures/stop reasons; return best artifact. Remove direct generated eval bypasses, including benchmark_code inconsistency. Preserve testable client/evaluator injection.

**Expected files:** controller.py, agent.py, storage.py; new tests/test_controller.py.

**Implementation constraints:** No generated execution in controller, no automatic source overwrite and no unbounded tool conversation.

**Testing instructions:** uv run --locked pytest tests/test_controller.py -q. Deterministic cases cover baseline failure, zero/negative budgets, duplicate, invalid proposal, compile/correctness failure, slower/no-change/accepted, timeout and exhausted request/cost/time caps.

**Acceptance criteria:**

- [ ] Zero candidates invokes no model.
- [ ] Every attempt records one terminal result.
- [ ] Second iteration consumes prior result.
- [ ] Only confirmed correct candidate replaces best.
- [ ] Stop returns verified baseline/incumbent.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Legacy tools conflict with new flow: one internal evaluate-and-record route; compatibility wrapper only if safe.

**Out of scope:** Advanced search and multi-agent optimization.

**Completion evidence:** State transition matrix, regression tests, bounded fake-run artifacts and source untouched proof.

## Task ID: AIC-016

**Title:** Verify the offline loop and bounded live MVP on the selected target.

**Milestone:** M3.

**Priority:** CRITICAL.

**Status:** NOT_STARTED.

**Goal:** Demonstrate genuine multi-iteration optimization on the explicitly chosen hardware.

**Reason:** Fake-client evidence proves plumbing, not LLM-based optimization.

**Current codebase state:** Audit D05 is mocked only; integrated system after AIC-015 must demonstrate real evaluation.

**Dependencies:** AIC-015.

**Implementation instructions:** Add fake-provider end-to-end tests exercising actual selected-target sandbox compilation/correctness/timing using reviewed fixtures. With explicit model/spend cap and M3 authorization, evaluate at least two and at most three live softmax proposals. Store next-request feedback, actual device/containment evidence, artifacts and best export. Distinguish fake decision fixtures from live results; a speedup is not required.

**Expected files:** tests/test_end_to_end.py (new), README.md, implementation tracker; runtime artifacts under runs/<id>/.

**Implementation constraints:** No automatic paid run in tests. Sandbox suite cannot be skipped to pass Level A. Live model availability and cap are hard entry conditions.

**Testing instructions:** Run the shared offline test command and real target/sandbox command defined above. Separately execute the bounded optimize CLI with configs/target.json and approved model/run config; validate reports without importing stored candidate text.

**Acceptance criteria:**

- [ ] Offline compile/failure/reject/accept paths pass.
- [ ] >=2 genuine proposals evaluated on selected device.
- [ ] Later context includes actual earlier result.
- [ ] Best source and manifest replay.
- [ ] Profile coverage/limitations reviewed.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** No model/access/funding: offline portion completes but task/M3 remain BLOCKED; no mocked discovery claim.

**Out of scope:** Research-scale search, another hardware target or speedup claims from MVP alone.

**Completion evidence:** Full test output, redacted live request IDs/usage, run ID, manifest, source hashes and M3 review.

## Task ID: AIC-017

**Title:** Expand representative workloads and calibrate target-specific shapes.

**Milestone:** M4.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Cover reduction, normalization and dense compound computations under one evaluator.

**Reason:** A single simple example cannot establish project utility or workload progression.

**Current codebase state:** Only example exists today; AIC-003 establishes softmax/registry.

**Dependencies:** AIC-016.

**Implementation instructions:** Add pure reference/input/edge-case definitions for layer norm and matmul+bias+GELU. Fix epsilon, axis, GELU variant and matmul precision. Register plan shapes, verify reference with independent checks, calibrate workload-specific tolerance/noise policy before search. Preserve baseline semantics and reject unsupported dtypes.

**Expected files:** workloads/registry.py; new workloads/layer_norm.py, workloads/matmul_gelu.py, tests/test_workload_suite.py.

**Implementation constraints:** No deliberate baseline slowdown, approximate precision changes or full models.

**Testing instructions:** uv run --locked pytest tests/test_workload_suite.py -q, then selected-target baseline/calibration for all three families using configs/target.json and declared resource caps.

**Acceptance criteria:**

- [ ] Three workload families supported.
- [ ] Eager/JIT and reference checks pass.
- [ ] Edge cases/tolerances recorded.
- [ ] All fit isolated resource limits or preregistered smaller shapes.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Latency/device-memory/compile cost exceeds budget: resize before protocol freeze, retain all three computation families and disclose shape changes.

**Out of scope:** Attention, Transformer and kernel rewrites.

**Completion evidence:** Reference correctness, configs/shape policy, raw A/A evidence per primary shape.

## Task ID: AIC-018

**Title:** Separate held-out correctness and shape generalization.

**Milestone:** M4.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Evaluate selected implementations on inputs not used to guide search.

**Reason:** Existing fixed inputs and future search feedback otherwise encourage overfitting.

**Current codebase state:** No split infrastructure exists in current code; WorkloadSpec defines slots after AIC-003.

**Dependencies:** AIC-017.

**Implementation instructions:** Add trusted final-evaluation mode with at least twenty disjoint seeds and two unseen shapes/workload, frozen split digest before search. Keep arrays/seeds out of prompts and candidate context; release only after selection and evaluation. Evaluate winner versus immutable baseline and record per-shape correctness/speedups. Refuse search reuse of final feedback under same evaluation version.

**Expected files:** workloads/registry.py, agent.py, contracts.py, storage.py; new tests/test_holdout.py.

**Implementation constraints:** Held-out failure invalidates affected claims; never tune tolerances or candidates on heldout and reuse the same label.

**Testing instructions:** uv run --locked pytest tests/test_holdout.py -q. Use reviewed seed/shape-overfit fixture, split overlap rejection, context leak checks and replay after final split publication.

**Acceptance criteria:**

- [ ] Splits disjoint/frozen.
- [ ] Deliberately overfit fixture fails.
- [ ] No heldout fields in proposal context.
- [ ] All per-shape results retained.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Seed secrecy alone insufficient: candidate sees actual arrays; pure policy plus broader hidden cases and transparent finite-domain claims.

**Out of scope:** Universal equivalence, gradients or inferring second-target correctness from the primary target.

**Completion evidence:** Pre-search split hash, release manifest, negative fixture results and generalization report schema.

## Task ID: AIC-019

**Title:** Recover safely from interruption and repository drift.

**Milestone:** M4.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Resume compatible runs without duplicating spend or promoting incomplete output.

**Reason:** Atomic individual files alone do not form a recoverable run transaction.

**Current codebase state:** controller.py rewrites history with no resume; AIC-012 adds committed attempt records.

**Dependencies:** AIC-016, AIC-012.

**Implementation instructions:** Implement resume based on committed records, source/config/lock/environment compatibility and recorded usage. Mark unfinished worker attempt INTERRUPTED; terminate leftovers through runner identity. Reconcile missing derived indexes from committed data. Persist request-start before paid dispatch; uncertain provider completion consumes reserved budget and requires explicit recovery policy, not automatic blind retry. Refuse incompatible environments; start a new linked run.

**Expected files:** storage.py, controller.py, runner.py, agent.py; new tests/test_resume.py.

**Implementation constraints:** Do not deserialize executable objects or replay paid requests automatically. No stale result promotion after source changes.

**Testing instructions:** uv run --locked pytest tests/test_resume.py -q. Inject interrupts before/after request, artifact write, commit and index update; test changed config/hash and existing directory collisions.

**Acceptance criteria:**

- [ ] Committed history preserved.
- [ ] Interrupted attempts never become best.
- [ ] Budgets monotonic.
- [ ] Incompatible resume rejected.
- [ ] Next request uses recovered history.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Provider usage uncertain: conservative reserved debit and owner reconciliation; prefer new run over guessed recovery.

**Out of scope:** Distributed transactions or cross-host transparent migration.

**Completion evidence:** Fault-injection table, resumed artifact diff, stop reasons and budget reconciliation examples.

## Task ID: AIC-020

**Title:** Generate artifact-only reports and validated best exports.

**Milestone:** M4.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Explain each optimization run without executing its stored code.

**Reason:** Console and optional plots do not provide reproducible analysis or provenance.

**Current codebase state:** controller.py graph_it/print report exist; source currently stored inside JSON only.

**Dependencies:** AIC-018, AIC-019.

**Implementation instructions:** Add report mode reading validated manifest/committed records. Emit Markdown/JSON summary of baseline/current-best, measured ratios/uncertainty, correctness scope, failure categories, usage, stop reason, profile coverage and source diff. Handle failures/null timings. Verify best source hash and output a replay command/config. Preserve all workloads/runs for later aggregate report; no dashboard.

**Expected files:** storage.py, agent.py, README.md; new tests/test_reporting.py.

**Implementation constraints:** Reports never import candidate modules; untrusted strings escaped/bounded and no hidden failure exclusion.

**Testing instructions:** uv run --locked pytest tests/test_reporting.py -q; report a successful, no-improvement and interrupted fixture run; tampered source/evidence rejected.

**Acceptance criteria:**

- [ ] Report reconstructs run from disk.
- [ ] All attempts included.
- [ ] Missing evidence disclosed.
- [ ] Best source/hash valid.
- [ ] No-improvement returns baseline.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** Legacy artifacts lack fields: explicit legacy/unverified display, no fabricated provenance.

**Out of scope:** Interactive site, publication plots before final data.

**Completion evidence:** Three report examples, artifact consistency tests, selected-target labels and M4 review.

## Task ID: AIC-021

**Title:** Select the concrete custom-kernel technology and scope.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Make an evidence-based kernel implementation decision for the selected primary target.

**Reason:** The complete Ericsson project requires agent kernel capability, not just a feasibility memo.

**Current codebase state:** No native/device kernel implementation exists. D03 found CPU FFI API availability and CPU Pallas interpretation limits; GPU/TPU paths remain unverified. AIC-031 makes preliminary feasibility part of target selection.

**Dependencies:** AIC-020.

**Implementation instructions:** Within two focused investigation sessions, use selected-target evidence and representative workload measurements to choose one operation, technology, layout/shape/precision contract, build/launch policy and integration route. Default operation is established layer norm; CPU may use native FFI, GPU/TPU a verified Pallas or justified native path. Compare engineering cost, profiling opportunity and end-to-end overhead. Prepare concrete AIC-027/028 implementation scope and request owner K1 approval. K2 investigation-only or K3 later hardware are explicit scope amendments leaving R-11 incomplete and altering downstream gates; neither silently passes M5.

**Expected files:** docs/IMPLEMENTATION_PLAN.md, docs/REQUIREMENTS_MATRIX.md, docs/IMPLEMENTATION_PROGRESS.md; new docs/KERNEL_DECISION.md with chosen API/toolchain/operation/build policy and probes.

**Implementation constraints:** No full kernel implementation in this investigation task; no unapproved install/spend. Do not confuse CPU interpretation, library wrapping or a manual kernel with live agent kernel capability.

**Testing instructions:** Record exact selected-target API/toolchain/device probes and compare against pinned official docs. Review K1/K2/K3, containment requirements and precise operation semantics; owner decision is separate evidence.

**Acceptance criteria:**

- [ ] Concrete chosen operation/API/toolchain and resource bounds recorded.
- [ ] Three outcomes and requirement impact compared.
- [ ] Actual owner decision recorded.
- [ ] Main-project default is implementation; deferral has explicit amended gates.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** No viable path within bound: owner may select K2/K3 and amend scope; deferred kernel tasks stay incomplete and full original-objective satisfaction cannot be claimed.

**Out of scope:** Unapproved native kernel coding or switching project hardware automatically.

**Completion evidence:** Kernel decision memo, feasibility probes, accepted build/run contract and explicit owner outcome.

## Task ID: AIC-027

**Title:** Integrate one real custom kernel on the selected target.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Prove a lower-level operation executes natively on the selected hardware within the common evaluator.

**Reason:** K1 requires native kernel implementation beyond feasibility/interpretation.

**Current codebase state:** No kernel source exists. AIC-021 will select a precise operation/API/build contract; audit CPU FFI API availability alone does not prove execution.

**Dependencies:** AIC-021.

**Implementation instructions:** Implement exactly the K1-approved operation and shape/precision/layout contract, preferably layer norm matching AIC-017. Use the selected native FFI or Pallas/device path; keep existing JAX reference unchanged. Add isolated reproducible compilation/registration or lowering, fixed build arguments, bounded memory/launch checks and target/library/compiler hashes. Validate through existing correctness, measured end-to-end JAX timing and containment. Forbid accidental interpretation/CPU fallback and explicitly document unsupported transforms/dtypes.

**Expected files:** runner.py, worker.py, contracts.py; new kernels/normalization.cc for chosen native path OR kernels/normalization.py for chosen Pallas path, kernels/build.py only if needed, tests/test_kernel_integration.py; docs/KERNEL_DECISION.md. Create only files needed by selected technology.

**Implementation constraints:** Requires K1 approval and M5 authorization. One technology, one operation; no host loading of untrusted libraries and no agent-generation/speedup claim from manual integration.

**Testing instructions:** Build/lower in verified target worker; run tests/test_kernel_integration.py with target/sandbox configuration. Prove actual native/device execution, reject layout/shape errors, verify edge/nonfinite cases and resource/cleanup behavior.

**Acceptance criteria:**

- [ ] Real custom operation executes on selected target in JAX workload.
- [ ] Shared correctness suite passes.
- [ ] Raw end-to-end timings retained.
- [ ] Target/source/build identity recorded.
- [ ] Isolated compile/run and timeout cleanup pass.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** API/ABI/device mismatch or excessive effort: stop for K2/K3 scope amendment; never substitute interpreter execution as kernel-performance completion.

**Out of scope:** Live agent kernel edits, multiple kernel technologies or second-target port.

**Completion evidence:** Actual lowering/registration/device proof, source/build hashes, correctness/timing artifacts and strengthened containment tests.

## Task ID: AIC-028

**Title:** Enable bounded agent proposals for the chosen kernel target.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Demonstrate iterative LLM proposals and evaluation for the actual selected-target custom operation.

**Reason:** A manual native/device kernel alone does not satisfy Ericsson's agent kernel capability.

**Current codebase state:** No kernel-agent capability today; AIC-027 will supply one verified manual integration and fixed build/run boundary.

**Dependencies:** AIC-027.

**Implementation instructions:** Version the proposal schema for one explicit custom-kernel target with constrained source/launch parameters. Build commands and dependencies remain harness-owned; compile/lower and run only inside verified limits. Feed build/compile/correctness/timing feedback across >=2 real LLM proposals within approved M5 cap. Apply same reference and confirmation protocol as JAX candidates. Record source/binary or lowered-artifact/toolchain/device identities, usage and failures.

**Expected files:** contracts.py, agent.py, controller.py, runner.py, selected kernels/ files; new tests/test_kernel_agent.py; kernel/evaluation docs.

**Implementation constraints:** No arbitrary scripts/dependency downloads, direct host native execution or promised speedup. Target kernel policy cannot expand arbitrary Python authority.

**Testing instructions:** Offline schema/build-failure/wrong-output/timeout fixtures; real selected-target kernel suite; >=2 separately funded live kernel proposals with next-request feedback. Fresh confirmation required for speedup claims.

**Acceptance criteria:**

- [ ] >=2 live kernel proposals evaluated with feedback.
- [ ] Incorrect/crashing/over-budget kernels rejected.
- [ ] Native/device artifacts reproducible.
- [ ] R-11 status reflects demonstrated capability, not speedup expectation.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Unstable toolchain/provider: preserve failed attempts, keep capability incomplete and use explicit scope amendment if necessary.

**Out of scope:** Multiple languages, device selection by the agent or additional hardware.

**Completion evidence:** Live proposal→build/lower→evaluation→feedback chain, target/budget identities, tests and R-11 evidence.

## Task ID: AIC-029

**Title:** Evaluate scaled dot-product attention rewrites.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Extend the same optimizer to a meaningful algorithmic workload on the selected target.

**Reason:** Ericsson seeks progression toward larger rewrites; attention adds meaningful compound behavior.

**Current codebase state:** No attention workload exists; M4 will supply reliable primitives, evaluator and registry.

**Dependencies:** AIC-021.

**Implementation instructions:** Define one forward float32 attention variant with fixed mask/scaling semantics, bounded shapes, stable immutable reference, disjoint splits and numeric policy. Add to registry, verify reference and run the same selected-target optimizer under M5 cap. Permit JAX-level tiling/streaming only within semantics. Record actual memory/runtime/compile tradeoffs; no assumption that a published hardware-specific gain transfers to this device.

**Expected files:** workloads/registry.py; new workloads/attention.py, tests/test_attention.py; evaluation/progress docs.

**Implementation constraints:** Single selected target and bounded memory/time. No cache/training/feature expansion. Source-only attention rewriting remains independently useful alongside kernel work.

**Testing instructions:** uv run --locked pytest tests/test_attention.py -q for semantic fixtures; selected-target correctness/heldout and bounded live optimization with configs/target.json. Confirm any speedup independently.

**Acceptance criteria:**

- [ ] Attention semantics fixed/tested.
- [ ] Full loop and heldout evaluation execute.
- [ ] Memory/resource limits obeyed.
- [ ] Results/negative findings documented.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Compile or device-memory growth: reduce declared dimensions before freeze; if loop cannot run, task remains incomplete rather than substituting another workload.

**Out of scope:** Transformer composition, new kernels or another backend.

**Completion evidence:** Contract/reference checks, run artifacts and comparison to immutable attention baseline.

## Task ID: AIC-030

**Title:** Evaluate a small Transformer block and composed semantics.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Verify optimization of a complete model component before the full compact model.

**Reason:** A component demo advances model progression without pretending to optimize an entire model.

**Current codebase state:** No Transformer block exists; AIC-029 adds attention and AIC-017 supplies normalization/dense primitives.

**Dependencies:** AIC-029.

**Implementation instructions:** Compose one deterministic forward block with fixed normalization/residual/activation semantics and explicit weights. Define independent reference checks and target resource bounds. Run bounded optimization and heldout evaluation through the common loop; include compilation costs. Keep this task limited to one block; AIC-032 handles a full model forward pass.

**Expected files:** workloads/registry.py; new workloads/transformer_block.py, tests/test_transformer_block.py; evaluation/requirements/progress docs.

**Implementation constraints:** One fixed component, selected target and approved M5 budget. No model download/training/framework.

**Testing instructions:** uv run --locked pytest tests/test_transformer_block.py -q for reference fixtures; selected-target block correctness, bounded loop, heldout shapes and performance confirmation if a gain is claimed.

**Acceptance criteria:**

- [ ] Fixed component passes correctness.
- [ ] Full-loop artifacts produced within budget.
- [ ] Compile/runtime costs distinct.
- [ ] Remaining full-model scope explicit.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Resource bound exceeded: reduce dimensions before freeze; omitting the block requires explicit scope amendment and leaves model progression incomplete.

**Out of scope:** Complete language model, distributed inference, training and GPU backend.

**Completion evidence:** Block contract, reference/heldout results, full-loop artifacts and readiness for AIC-032.

## Task ID: AIC-032

**Title:** Demonstrate whole-model optimization on a compact complete model.

**Milestone:** M5.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Exercise the optimizer on a complete declared model forward pass, beyond a single block.

**Reason:** Ericsson asks for progression toward full model rewrites; component-only results leave that objective partial.

**Current codebase state:** No complete model exists. AIC-030 will provide one tested Transformer block, not a full model.

**Dependencies:** AIC-030.

**Implementation instructions:** Build a compact two-block Transformer classifier from the verified block, with embedded-vector input, mean pooling and fixed linear classification head producing logits. Use explicit immutable weights/parameters supplied by trusted harness. Start B=2,S=32,D=64,heads=4,classes=8, calibrate resource bounds then freeze development/heldout shapes and tolerance. Verify complete reference forward and selected intermediate invariants. Permit whole-forward JAX proposals through the same source policy, isolated evaluator and history loop; do not enable gradients/downloads. Run >=2 bounded live proposals, preserve no-gain outcomes and prepare model records for final protocol.

**Expected files:** workloads/registry.py; new workloads/compact_model.py, tests/test_compact_model.py; model contract in documentation and experiment artifacts.

**Implementation constraints:** Small complete forward model, not production LLM support. All declared layers/head must execute; candidate cannot replace weights or cache known outputs. Selected-target resource/spend limits apply.

**Testing instructions:** uv run --locked pytest tests/test_compact_model.py -q for reference/structure fixtures; real selected-target cases and hidden weights/inputs, bounded multi-iteration live run, independent confirmation for any claimed gain. Check incomplete-forward or constant-output fixtures are rejected.

**Acceptance criteria:**

- [ ] Full two-block-plus-head reference validated.
- [ ] Complete-model candidate cannot skip semantics/replace parameters.
- [ ] >=2 real proposals evaluated with feedback within bounds.
- [ ] Heldout cases and artifact provenance retained.
- [ ] Compact-model versus production-model limits explicit.

**Definition of Done:** Every acceptance item is evidenced; implementation and documentation agree; listed tests and affected regressions pass; actual results/limitations are recorded in the tracker. No skipped mandatory selected-target check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Memory/context/compile cost excessive: reduce model dimensions before freeze; omitting complete-model evidence needs approved scope amendment and partial R-12.

**Out of scope:** Training, production-size LLMs, distributed execution or model downloads.

**Completion evidence:** Complete model contract/source hash, actual target/run IDs, reference/heldout tests, request/usage and performance/negative findings.

## Task ID: AIC-022

**Title:** Freeze a feasible, funded experimental protocol.

**Milestone:** M6.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Define comparable trials before collecting research data.

**Reason:** Uncontrolled search and selective reporting cannot establish effectiveness.

**Current codebase state:** No protocol or research dataset; M4 provides suite/report infrastructure.

**Dependencies:** AIC-028, AIC-032, AIC-020.

**Implementation instructions:** Freeze C0/C1/C2/C3 for four workloads (three representative operations plus compact complete model), three search runs and five-proposal ceilings; add three five-proposal kernel runs on the chosen operation matched to C3 JAX-only runs. Proposed bound: 195 proposal slots in 39 searches, plus baselines and confirmation. Estimate device/runtime/model cost and time; choose smaller justified protocol before collection if needed. Freeze source/lock/target/split hashes, order and A/A acceptance policy. Optional profile C4 requires additional funded scope and actual coverage. Obtain owner approval for exact protocol/cap; an approved K2/K3 amendment must explicitly revise kernel conditions/dependencies.

**Expected files:** contracts.py, agent.py; new docs/EXPERIMENT_PROTOCOL.md, configs/evaluation.json, tests/test_experiment_config.py.

**Implementation constraints:** 195 slots is a proposed upper bound, not authorization. No pooling timing across devices or selecting only favorable shapes/workloads after observing outcomes. Exact budget/model/target required.

**Testing instructions:** uv run --locked pytest tests/test_experiment_config.py -q; dry-run matrix enumeration validates counts, limits, hashes and split disjointness without model calls.

**Acceptance criteria:**

- [ ] Concrete protocol and total spend approved.
- [ ] Conditions and target controls frozen.
- [ ] Every planned run/seed/order enumerated.
- [ ] Kernel/model coverage or approved scope gaps explicit.
- [ ] Negative-result/stop/exclusion policy defined.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Budget/time insufficient: preregister smaller pilot and narrower conclusions; preserve all workload families/confirmation.

**Out of scope:** Final trial execution or unapproved extra conditions/targets.

**Completion evidence:** Frozen config/target hashes, exact matrix count, cost projection and owner protocol authorization.

## Task ID: AIC-023

**Title:** Execute the bounded evaluation and independent confirmation.

**Milestone:** M6.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Collect the approved whole-project optimization dataset on the selected target.

**Reason:** Software tests do not answer whether an LLM improves XLA programs.

**Current codebase state:** No validated empirical speedup exists; archived runs lack provenance.

**Dependencies:** AIC-022.

**Implementation instructions:** Execute only approved matrix using verified runner; serialize measurements, randomize/block condition order as frozen and enforce global budget. Freeze each search winner before heldout evaluation. Confirm against original baseline in at least three fresh sessions and all evaluation shapes. Retain raw blocks, candidates, failures, usage and unexpected stops. Mark unavailable trials explicitly; critical harness bugs invalidate affected runs and require a new protocol version.

**Expected files:** Runtime outputs runs/<evaluation-id>/; implementation progress; fixes only if separately reviewed within scope.

**Implementation constraints:** No selective reruns, post-result tuning or cross-device speedup claims. Full-model and kernel records need the same provenance/correctness standard as ordinary JAX trials.

**Testing instructions:** Run documented protocol commands and artifact validator/report for every run; reconcile planned versus completed/failed/excluded counts and budget totals.

**Acceptance criteria:**

- [ ] Every trial accounted for.
- [ ] Heldout/correctness evidence retained.
- [ ] Fresh confirmations completed or claims withheld.
- [ ] No budget overrun.
- [ ] Baseline and candidate environment hashes compatible.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Host noise/service failure: use predefined retry/abort rules; partial dataset must be labeled and research gate reviewed.

**Out of scope:** Expanding search budget or kernel/Transformer work.

**Completion evidence:** Evaluation manifest, run inventory, raw session artifacts, request/usage totals and deviations.

## Task ID: AIC-024

**Title:** Analyze source, kernel and model results and remaining gaps.

**Milestone:** M6.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Produce a defensible project evaluation across the declared optimization levels.

**Reason:** Best-case timing alone hides invalid candidates, overfitting and negative results.

**Current codebase state:** Current artifacts show no credible performance gain; AIC-023 will supply real evidence.

**Dependencies:** AIC-023, AIC-021.

**Implementation instructions:** Compute metrics from all retained records with explicit denominators and uncertainty. Compare C2/C3 compiler effects, qualify C1/C2 feedback/search effects, and compare kernel-enabled versus JAX-only runs under declared budgets. Report primitive/compound/full-model results, heldout failures, build/compile/search costs and every no-gain trial. Explain selected IR/profile observations without inventing causal bottlenecks. Answer RQs or mark unsupported; track R-11 kernel, R-12 model and R-14 speedup evidence independently.

**Expected files:** storage.py reporting helpers if needed; new docs/EVALUATION_REPORT.md and tests/test_analysis.py; requirements/progress docs.

**Implementation constraints:** No fabricated tables, p-hacking or claims that negative results meet the speedup objective. Optional C4 absent means RQ2 unresolved.

**Testing instructions:** uv run --locked pytest tests/test_analysis.py -q; regenerate report from evaluation manifest, cross-check totals and hand-check one run/shape per workload.

**Acceptance criteria:**

- [ ] Metrics reproducible from artifacts.
- [ ] Failures/negative results included.
- [ ] RQs bounded by actual evidence.
- [ ] Functional/research/performance/custom-kernel completion distinguished.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Small sample count: descriptive pilot conclusions and explicit confounds, not strong causal generalization.

**Out of scope:** Inventing additional experiments after favorable/unfavorable outcomes without new approval.

**Completion evidence:** Reproducible report, complete trial totals, kernel/model source examples, requirement evidence and M6 review.

## Task ID: AIC-025

**Title:** Validate supported setup, operation and replay.

**Milestone:** M7.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Make the selected-target project reproducible by another user/session.

**Reason:** Working only in the audit host's existing venv is insufficient delivery.

**Current codebase state:** README/how_to_run describe unsafe prototype path; new implementation requires updated workflow.

**Dependencies:** AIC-024.

**Implementation instructions:** Document selected hardware/runtime, pinned dependencies, isolation, offline/target/live test commands, workload/proposal contracts, model cap and baseline/optimize/report/resume workflow. Validate in a clean supported environment without model credentials; replay saved source/kernel/model results without paid requests and verify hashes. Run complete offline and real target/sandbox checks, including kernel integration. Remove unused SDK/viewer dependencies only with usage evidence and no unrelated upgrade.

**Expected files:** README.md, how_to_run.md, pyproject.toml/uv.lock only if justified; docs/CODEX_WORKFLOW.md and progress; tests as needed.

**Implementation constraints:** No dependency refresh for its own sake. Replay is not a fresh live experiment. Release tagging/publishing requires explicit owner instruction.

**Testing instructions:** uv sync --locked in a fresh supported environment with selected-target extras from its decision record; run shared offline and target/sandbox commands. Exercise documented replay/report for JAX, kernel and model artifacts.

**Acceptance criteria:**

- [ ] Fresh selected-target setup succeeds.
- [ ] Offline tests need no keys.
- [ ] Real target/sandbox/kernel tests execute.
- [ ] Source, kernel and model replay verified.
- [ ] Unsupported platforms labeled honestly.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Clean environment inaccessible: record blocked check and resolve before task DONE; existing-venv pass is not equivalent.

**Out of scope:** Publishing/tagging/pushing, second-target support or broad packaging redesign.

**Completion evidence:** Fresh-environment identity, setup/test commands, replay hashes and remaining documented limitations.

## Task ID: AIC-026

**Title:** Deliver the full project demonstration and completion review.

**Milestone:** M7.

**Priority:** REQUIRED.

**Status:** NOT_STARTED.

**Goal:** Present the optimizer, kernel/model progression and final evidence for owner acceptance.

**Reason:** The owner needs reviewable evidence to accept the delivered project.

**Current codebase state:** No final system/demo; AIC-016 supplies earlier live MVP evidence and AIC-024 research report.

**Dependencies:** AIC-025.

**Implementation instructions:** Prepare a reproducible demonstration of baseline, compiler/profile feedback, >=2 live-origin iterations, correctness/decisions/history and best source; include selected-target kernel capability and whole-model evaluation. Labeled replay may use prior real artifacts; a new live demo requires funded scope. Reconcile final requirements and any approved kernel/model/profile gaps; do not claim performance success without repeated evidence. Produce 13-item M7 review, mark AWAITING_APPROVAL and stop.

**Expected files:** README.md, docs/EVALUATION_REPORT.md, docs/REQUIREMENTS_MATRIX.md, docs/IMPLEMENTATION_PROGRESS.md; new docs/DEMONSTRATION.md.

**Implementation constraints:** No invented discoveries/gains or automatic M8 work. Plan approval and accepted results do not authorize additional hardware/spend.

**Testing instructions:** Follow demo instructions from clean supported environment; verify every displayed number/source against artifact hashes; run final required regression checks once after any changes.

**Acceptance criteria:**

- [ ] End-to-end demo reproducible.
- [ ] Kernel/model scope evidenced or approved gaps explicit.
- [ ] Completion dimensions assessed independently.
- [ ] Deliverables linked and owner review pending.

**Definition of Done:** Every acceptance item is evidenced; the implementation and relevant documentation agree; the listed tests and affected regression checks pass; actual results and limitations are recorded in the progress tracker. No skipped mandatory check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** SMALL.

**Risks and fallback:** API unavailable: labeled replay with previous live evidence; cannot replace missing original live evidence with mocks.

**Out of scope:** Optional portability or additional features after release.

**Completion evidence:** Demo source/run IDs, final required tests, target support statement and M7 result-approval request.

## Task ID: AIC-033

**Title:** Qualify a second target for the existing JAX optimization loop.

**Milestone:** M8.

**Priority:** OPTIONAL.

**Status:** DEFERRED.

**Goal:** Establish bounded portability on another actually available hardware target.

**Reason:** Hardware-neutral design is extensible, but claims of multi-target support require independent evidence.

**Current codebase state:** No second-target verification today; main project will support its selected primary target only.

**Dependencies:** AIC-026.

**Implementation instructions:** After M7 and separate access/budget authorization, select one second target and scope this task to the existing softmax JAX-rewrite loop. Verify environment/isolation, adapt only device-specific configuration/extraction if needed, calibrate measurement separately and run >=2 live proposals. Keep primary-target regressions passing and record per-target artifacts. If portability requires a substantial redesign or native kernel port, stop with a concrete new backlog proposal before that work.

**Expected files:** contracts.py, runner.py, worker.py, feedback.py only as justified; new configs/second_target.json, tests/test_portability.py and target-decision update.

**Implementation constraints:** No universal three-backend framework, native-kernel port or cross-device speedup ratio. This task demonstrates a bounded source-rewrite capability only on its verified second target.

**Testing instructions:** Run original offline/primary-target checks and separate real second-target containment, IR/profile coverage, A/A calibration and softmax loop. No primary-device result may substitute for the second-device gate.

**Acceptance criteria:**

- [ ] Actual second device and containment verified.
- [ ] Separate raw baseline/calibration retained.
- [ ] >=2 live feedback-informed source proposals run.
- [ ] Primary target unchanged by regressions.
- [ ] Support limitations explicit.

**Definition of Done:** Every acceptance item is evidenced; implementation and documentation agree; listed tests and affected regressions pass; actual results/limitations are recorded in the tracker. No skipped mandatory selected-target check, implicit approval or mock substituted for required real evidence.

**Estimated complexity:** MEDIUM.

**Risks and fallback:** Unavailable runtime or broad redesign required: remain DEFERRED/BLOCKED and propose focused follow-up tasks, without affecting main-project completion.

**Out of scope:** Second-target kernel capability, larger model port or additional providers/search algorithms.

**Completion evidence:** Second-target selection/access approval, actual environment/run artifacts, primary regression checks and M8 review.
