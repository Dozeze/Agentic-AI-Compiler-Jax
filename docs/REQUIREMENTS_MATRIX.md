# Whole-project requirements traceability

Version **AIC-1.1**, 2026-10-12. **PLAN APPROVED** by the owner on 2026-10-12;
no implementation milestone authorized. See PLAN-APPROVAL in
[IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md).
The latest owner instruction requests a general, hardware-neutral roadmap for the
entire project. It supersedes the CPU-first restriction of the previous request
and unapproved CPU-1.0 draft. Planning-only work, correctness and explicit human
approvals remain required. No primary target has been selected.

External source: **projects2026.pdf, page 1, Project 1: Ericsson — Agentic AI
compiler**; fingerprint and repository evidence D00–D08 in
[CODEBASE_AUDIT.md](CODEBASE_AUDIT.md). Only CPU behavior has execution evidence
from that audit. GPU/TPU support, live discovery, kernel capability and credible
speedup are unverified or absent. The one-page source sets objectives; evaluation,
security and provenance controls below operationalize those objectives and the
owner's additional requirements.

Statuses: SATISFIED means evidenced within the stated scope; PARTIALLY_SATISFIED
means a material gap remains; NOT_SATISFIED means required capability/evidence
is absent or inadequate; UNVERIFIED means an implementation/claim awaits checks;
DEFERRED means an explicitly scoped postponement, never implementation completion.
No target milestone is evidence of current satisfaction.

Scope A: functional optimizer through M3. Scope B: main project through M7,
including selected-target kernels and compact complete-model progression.
Scope C: optional second-target qualification in M8 and separately scoped later
extensions. B-research denotes the speedup objective, which is assessed rather
than guaranteed. Deferring a B capability requires an explicit plan amendment.

| ID | Original objective / operational requirement | Current status / evidence | Planned tasks | Verification procedure | Required deliverable | Final target / scope | Limitation or decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-01 | Accept models or algorithms written in JAX | SATISFIED for trusted addition only; D02, agent.py source loader | AIC-003,017,029,030,032 | Seeded reference tests and actual selected-target execution for every advertised workload | JAX suite from simple operations through compact model | M5 / A basic, B progression | Arbitrary Python/gradients are outside initial contract; current evidence is CPU only. |
| R-02 | Supplement XLA while retaining its compilation pipeline | SATISFIED for audited CPU example, D02/D03 | AIC-031,002,004,010 | Verify selected device and pinned environment; lower/compile/execute actual JAX workloads and retain IR | Verified target configuration, compiled reference/candidate artifacts | M3 / A | No second-target inference from CPU probes; no replacement compiler or mandatory HLO mutation. |
| R-03 | Iterative observation, rewrite, evaluation and feedback | PARTIALLY_SATISFIED (2026-11-16): AIC-013 structured requests and AIC-014 bounded feedback verified offline with real CPU compile/correctness failures in later requests; controller integration/live proof pending | AIC-013,014,015,016 | >=2 genuine live proposals evaluated on selected target; later request includes earlier result; offline transition tests | Complete bounded optimization episode | M3 / A | Sandbox, model and spending gates required. Mock demonstration does not prove live optimization. |
| R-04 | Propose high-level JAX rewrites | PARTIALLY_SATISFIED; code-string tool and archived candidates, live generation unverified | AIC-007,013,015,016,029,030,032 | Validate actual proposals, signature/source policy and semantics; evaluate whole-forward model changes at later scope | Proposal/source/diff/hypothesis/outcome records | M5 / A function, B algorithm/model | Pure forward subset initially; no arbitrary filesystem or precision changes. |
| R-05 | Inspect lowered program and compiler IR such as HLO/StableHLO | PARTIALLY_SATISFIED (2026-11-16): AIC-010 capture and AIC-014 bounded C3/C4/K summaries verified with actual CPU compiler provenance; selected-target validation and benefit assessment pending | AIC-010,014,022,023,024 | Actual selected-target IR and compact summaries saved and included in context; C2/C3 comparison | Raw IR, bounded summaries and compiler-feedback analysis | M3 implementation, M6 assessment / A-B | Changed target/software must be probed. Static estimates are not measured runtime. |
| R-06 | Inspect runtime profile and use profiling feedback | PARTIALLY_SATISFIED (2026-11-16): AIC-011 capture/parser and AIC-014 bounded C4 coverage/hotspots verified with real CPU traces and retained request snapshots; CUDA/container verification pending | AIC-031,011,014,016,022,024 | Real selected-target trace; CPU/device versus host-only coverage checked; overlap-aware summary in context; optional C4 for benefit claim | Trace, coverage/provenance summary and measured feedback | M3 implementation, M6 assessment / A-B | CPU evidence does not establish CUDA support or profiling benefit. Host-only/unavailable fallback has no hotspots; selected-target and live-context validation remain open. |
| R-07 | Preserve correctness under transformation | PARTIALLY_SATISFIED; D02 narrow checks, one input, PyTree failure | AIC-003,004,007,018,027,029,030,032 | Reject wrong structure/dtype/shape/numerics/nonfinite/overfit fixtures; validate JAX, kernels and full model on development and heldout inputs | Frozen contracts, tolerance policy and case-level evidence | M5 / A core, B generalization | Testing is finite-domain evidence, not universal proof. Precision and target settings must match baseline. |
| R-08 | Recompile and execute modified implementations | PARTIALLY_SATISFIED; trusted source works, generated execution unsafe | AIC-007,008,009,015,016,027,028 | Actual selected-target isolation, compilation/execution, resource failure and cleanup tests; native/device path separately verified | Runner/worker, build/lowering integration and per-stage artifacts | M5 / A JAX, B kernels | Device allocation or subprocess alone is not containment. Missing controls block autonomy. |
| R-09 | Analyze performance to decide subsequent optimization | PARTIALLY_SATISFIED (2026-11-16): AIC-005/006 calibrated host decisions and AIC-014 provenance-tagged paired uncertainty in next context implemented; complete controller/final verification remains AIC-015 | AIC-005,006,014,015 | A/A calibration; noisy fixtures rejected; correct stable improvement independently confirmed before promotion; next context records decision | Calibrated decision records and uncertainty | M3 / A | No universal percentage threshold or stale fastest-sample comparison; final selection requires new measurements. |
| R-10 | Compare against vanilla XLA | PARTIALLY_SATISFIED; original JIT baseline exists but protocol/provenance weak | AIC-002,005,006,012,018,022,023 | Same target, allocation, inputs, precision and measurement scope; immutable original reference; raw paired data and independent sessions | Per-target reproducible baseline and candidate ratios | M6 / A measurement, B research | A CPU-versus-GPU ratio is hardware comparison, not an optimizer speedup; no deliberately weakened baseline. |
| R-11 | Agent suggests and implements custom kernels | NOT_SATISFIED; no native/device kernel or action | AIC-031 feasibility; AIC-021,027,028 implementation; AIC-022–024 research | K1 concrete scope; actual custom operation runs natively on chosen target; >=2 live kernel proposals evaluated with feedback and shared correctness/timing | Implemented selected-target kernel action, build/source evidence and results | M5 capability, M6 assessment / B | Investigation, Pallas CPU interpretation or manual wrapping alone insufficient. K2/K3 require explicit scope amendment, leave R-11 incomplete and do not pass original B scope. |
| R-12 | Progress from simple optimizations toward full model rewrites | PARTIALLY_SATISFIED at simple addition starting point | AIC-003,017,029,030,032 | Same optimizer handles primitives, attention, block and full compact model; complete-forward semantics, heldout cases and live proposal records | Increasing-complexity suite and whole-model demonstration | M5 / B | Compact two-block classifier is a complete bounded model, not production LLM/training/distributed support. Omitting it leaves model progression partial. |
| R-13 | Demonstrate effectiveness on a set of models/algorithms with defensible experiments | NOT_SATISFIED; D07 lacks raw data/provenance | AIC-012,017–020,022–026,032 | Frozen funded matrix, target/model/source/split hashes, all trials accounted for, artifact-only report regeneration and replay | Dataset, protocol, quantitative report and demonstration | M7 / B | Reproducibility/control operationalize the demonstration objective. No promise that a live provider produces identical new responses. |
| R-14 | Produce measurably faster correct programs than vanilla XLA | NOT_SATISFIED by evidence; D07 accepted ~0.0154% amid larger variation | AIC-005,006,018,022,023,024,026 | Correct selected candidate beats original same-target baseline beyond calibrated noise in independent final sessions; heldout results and regressions reported | Confirmed speedup evidence or explicit unmet-objective/negative-result report | M6 evidence, M7 final assessment / B-research | Cannot guarantee success. Functional/research completion without a credible gain leaves R-14 NOT_SATISFIED. |

Task ranges include every integer ID in the range; detailed dependencies and
execution order are in [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md).
AIC-031 precedes AIC-002. Kernel/model tasks precede final protocol AIC-022.
Historical CPU-NNN aliases retain numeric identity; they imply no active CPU-only
scope. Optional AIC-033 is not a dependency of the main project.

## Owner-required controls

| ID | Requirement / origin | Current state | Tasks / evidence | Verification / deliverable |
| --- | --- | --- | --- | --- |
| O-01 | General hardware-neutral whole-project roadmap, latest owner clarification | Planning revised; no target selected | AIC-031,002 and all main-project tasks | Choose one actually available target; common goals/controller, target-specific verified settings; no mandatory simultaneous CPU/GPU/TPU support. Earlier CPU-first mandate is superseded. |
| O-02 | Execution isolation, resource limits and credential protection | NOT_SATISFIED | AIC-007–009,027,028; D00 runtime access unverified | Real selected-target conformance including device cleanup; fail closed, no unsafe fallback. |
| O-03 | Tests without paid API calls | PARTIALLY_SATISFIED (2026-11-16): replay/installed SDK mocks and 31 bounded-context checks verified; full suite 568 passed / 16 target skips; later optimizer/recovery tests remain | AIC-001,013–016,025; IMPLEMENTATION_PROGRESS.md AIC-013/014 | Positive offline test count, distinct real target/sandbox/live evidence. CPU smoke cannot stand in for accelerator checks. |
| O-04 | Explicit request/candidate/time/token/money budgets | PARTIALLY_SATISFIED (2026-11-16): AIC-013 reservations/transport limits and AIC-014 pre-dispatch context byte/token/money fit active; zero-budget defect fixed; stage-wide deadlines/recovery pending | AIC-002,013,014,015,019,022 | Checks before side effects; retries charged; uncertain request recovery conservative; no implicit hardware/cloud spend. |
| O-05 | Preserve history/source and return best verified implementation | PARTIALLY_SATISFIED (2026-11-16): AIC-012 immutable evidence, exact source/hash and baseline fallback verified; recovery/reporting pending | AIC-012,019,020; IMPLEMENTATION_PROGRESS.md AIC-012 | Original source unchanged; immutable artifacts, best source/hash and baseline fallback. |
| O-06 | Initial plan and separate milestone approvals | Overall AIC-1.1 plan approved; no milestone authorized | PLAN-APPROVAL in IMPLEMENTATION_PROGRESS.md; all milestone gates | Owner's explicit “yes” approves the annotated overall-plan question. Record separate milestone entry/result decisions; stop at AWAITING_APPROVAL. |
| O-07 | Persistent execution state and recovery | Documentation state initialized; runtime recovery absent | AIC-019,025, CODEX_WORKFLOW.md | Startup reads plan/task/matrix/tracker, checks actual code and valid authorization; drift revalidation and safe interrupted-job recovery. |
| O-08 | Six internally consistent planning documents | Revised documentation delivered; overall AIC-1.1 plan approved | Audit, plan, backlog, matrix, progress, workflow | Validate 33 IDs/templates, DAG, milestones, scopes/statuses and links; documentation checks are not implementation tests. |

## Scope exceptions and support claims

- Planned main scope includes actual kernel implementation and a compact complete
  model. K1 is the planned route, not a recorded owner outcome. K2/K3 require an
  approved amendment to tasks, dependencies, milestone gates and final claims.
- Primary target is open. CPU/GPU/TPU are selection options, not promised
  simultaneous support. AIC-033 remains DEFERRED until separate authorization.
- Profiling fallback retains measured wall time and compiler analysis, while
  R-06 stays partial if real execution/profile feedback is unavailable.
- Larger production models, gradients, distributed training and advanced search
  remain later scope; none is implied by a compact forward-model demonstration.
- No gain means R-14 remains unsatisfied. Never substitute successful software
  tests, one lucky sample, mock timings or hardware changes for performance proof.

Update a status only with actual command/result, source/target/config identity,
run artifacts and relevant review evidence. A proposed milestone or completed
planning document is never proof of implemented capability.
