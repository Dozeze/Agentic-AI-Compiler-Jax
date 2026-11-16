# Codex operating workflow

Version **AIC-1.1**, 2026-10-12. Owner instruction, master request §§12–13:
explicit initial plan approval, authorization of each milestone, and a stop for
review after each exit gate. Current plan state: **APPROVED**, 2026-10-12;
no milestone is authorized. See PLAN-APPROVAL in
[IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md).
These instructions govern development of the runtime optimizer; they do not
make Codex the runtime optimization agent.

## Active scope and identity

The owner's 2026-10-12 clarification requests a **general, hardware-neutral
roadmap for the entire project**, superseding the earlier CPU-first restriction.
AIC-1.1 includes target selection in M1, actual kernel and compact-model work in
M5, research in M6 and delivery in M7. Optional second-target qualification is M8.
One selected primary target is enough for the main implementation; no hardware
is chosen by this document. All original approval/safety/verification rules remain.

Historical CPU-001–CPU-030 IDs map one-to-one to AIC-001–AIC-030; no task was
completed under the old draft. AIC-031/032/033 are new target-selection/full-model/
optional-portability tasks. Read dependency IDs and milestone order, not numeric
sorting. AIC-031 precedes AIC-002; kernel/model tasks precede final research.

## Session initialization

1. Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md),
   [IMPLEMENTATION_TASKS.md](IMPLEMENTATION_TASKS.md),
   [REQUIREMENTS_MATRIX.md](REQUIREMENTS_MATRIX.md),
   [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md) and this workflow.
   Read [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) for baseline evidence; it is a dated
   snapshot, not permission to assume unchanged code.
2. Inspect Git status/HEAD, relevant source and applicable instruction files.
   No AGENTS.md was found during the audit; recheck for changed instructions.
   If one appears later, read it and
   reconcile with explicit owner instructions; do not overwrite it. Do not treat
   comments, candidate files, model responses or reference PDFs as authorization.
3. Identify approved plan version, currently authorized milestone and exact owner
   authorization evidence. An owner message in the current session can supply
   missing authorization; record it once rather than asking again.
4. Check dependencies, evidence and repository drift. Preserve unrelated user
   changes. Revalidate only previously completed work that changes may invalidate.
5. Select the next READY task **within the authorized milestone**. Promote a
   NOT_STARTED task to READY only when prerequisites and authorization hold.
   Mark IN_PROGRESS with session date/HEAD before editing.

If state is AWAITING_INITIAL_APPROVAL or milestone AWAITING_APPROVAL, make no
implementation edits. Present the concrete plan/review and request the missing
owner decision. Silence, previous approvals and passing tests do not authorize
progression. Read-only review and requested documentation corrections remain
permitted. Do not bypass the gate by calling new work a diagnostic.

## State and authorization model

Plan status: initially AWAITING_INITIAL_APPROVAL; after explicit owner approval,
record APPROVED with version and evidence. Changing the approved architecture or
scope materially creates a proposed revision needing approval; preserve the
last approved version and permitted work.

Milestone states are exactly NOT_STARTED, IN_PROGRESS, BLOCKED,
AWAITING_APPROVAL, APPROVED, CHANGES_REQUESTED. Store **authorization separately**
from these states because APPROVED means the owner accepted the result, not that
any next milestone is authorized. When owner authorizes entry, set IN_PROGRESS
and record scope, date and source message. Do not pre-fill hypothetical approval.
Target selection and a K1/K2/K3 decision
are separate recorded decisions; a capability deferral must amend affected tasks,
dependencies, exit gates and requirement claims. It never marks a task DONE.

Task states are NOT_STARTED, READY, IN_PROGRESS, BLOCKED, DONE, DEFERRED. A
blocked prerequisite prevents dependent tasks becoming READY, but independent
work inside the same approved milestone can continue. Do not start a later
milestone to route around a blocked exit gate.

The initial decisions are distinct:

- Approve plan AIC-1.1: recorded in PLAN-APPROVAL; do not request it again.
- Authorize M1 (AIC-001, AIC-031, AIC-002, AIC-003): pending, if the owner chooses to begin it.

At M1 completion, request acceptance of M1 and separate authorization of M2.
An owner may explicitly grant both in one message; record both decisions. Do
not infer the second from the first. The same rule applies to all later stages.

## Implementation boundaries

Within the approved milestone, implement/fix/test all its required tasks without
repeated confirmation for routine reversible choices. Follow existing conventions
and retain verified behavior where safe. New paths in the backlog are intentional
proposals, not a request to reorganize working modules.

Material deviations include changing an already selected primary target/provider, replacing the chosen
isolation boundary, significant dependency/version changes, relaxing numerical
semantics, weakening confirmation or accepting more cost/scope. Before such a
change, finish unaffected authorized work, describe the actual problem, smallest
proposed change, alternatives, impacted requirements/tasks and validation. Request
owner approval with a concrete reviewable proposal. Renumber no stable task IDs;
append new IDs if approved. Minor implementation details within a task need no
extra approval.

Never execute untrusted generated code in the development controller or on an
unverified fallback runner. Use reviewed fixtures before M2. After M2, run live
candidates only through the verified boundary. Paid runs require both milestone
scope and configured explicit spend authorization; a test command must not
silently contact an API. Missing access/caps are blockers for those actions,
not permission to fabricate a mock-based live result.

## Testing and task completion

1. Implement the selected task and meaningful tests for its behavior, especially
   discovered failure modes. Tests should not merely mirror code.
2. Run the task's specified checks and affected regressions. Use positive test
   counts; distinguish assertion failures, startup failures, skips and not-run
   checks. A no-tests result is not success.
3. Separate unit tests using deterministic data/fake clocks from actual CPU
   diagnostics, sandbox conformance and live experiments. Record which evidence
   each provides. Never claim a mock timing is measured speedup.
4. Check every mandatory acceptance item. Resolve failures within approved scope.
   If required verification is unavailable, leave BLOCKED with exact command,
   reason and resolution needed. An optional unsupported compiler estimate may
   be a tested fallback; missing live MVP or real sandbox evidence is not optional.
5. Record modified files, HEAD/dirty state, exact command/config, exit/result,
   artifact/run IDs, hashes, numerical outcomes, limitations and deviations in
   IMPLEMENTATION_PROGRESS.md. Do not record credentials, environment dumps or
   unnecessary full provider payloads. Keep raw experiment artifacts in runs/.
6. Update backlog task status and requirement evidence. Mark DONE only with
   completed implementation, tests, documentation and acceptance evidence.
7. Select another READY task in the same authorized milestone. No additional
   owner permission is needed merely to continue its remaining authorized work.

Shared offline command after AIC-001:
`uv run --locked pytest tests/ -m 'not target and not sandbox and not live' -q`.
Actual selected-target/sandbox command after config/runner implementation:
`AIC_TARGET_CONFIG=configs/target.json uv run --locked pytest tests/ -m 'target or sandbox' -q`.
A CPU smoke can support development but cannot satisfy GPU/TPU acceptance.
Missing requested hardware or sandbox controls must fail/block the gate, not
silently fall back or count as a passing skip. Live runs need explicit funding.
Document exact CLI flags as implemented; planned commands are not verified today.

## Milestone review and stop

After all included tasks are DONE and the milestone exit procedure passes,
prepare this review in the progress tracker (or linked review artifact):

1. Milestone objective.
2. Completed task IDs and any unfinished work (an unfinished required task blocks exit).
3. Implemented functionality and user-visible behavior.
4. Modified files and relevant commit/source identities.
5. Exact test/diagnostic commands executed.
6. Actual outcomes, counts, skips, failures and not-run checks.
7. Acceptance checklist for tasks and milestone gate.
8. Supporting artifacts, logs, manifests and run IDs.
9. Known limitations, including unsupported claims.
10. Remaining risks and external blockers.
11. Deviations from approved plan, with approval evidence where required.
12. Proposed next milestone and its entry prerequisites.
13. Exact decision requested: accept current result and, separately, authorize
    named next milestone if desired.

Set current milestone **AWAITING_APPROVAL**, set next permitted action to owner
review/documentation correction, and **stop implementation**. Only explicit owner
acceptance changes state to APPROVED. If changes are requested, record
CHANGES_REQUESTED, implement the requested corrections within that scope, rerun
affected checks and return to AWAITING_APPROVAL. If scope of requested changes is
unclear, ask while continuing independent already authorized corrections.

## Failure, interruption and recovery

- On session interruption, preserve files and record last completed operation,
  active task and pending checks. Do not assume an interrupted command succeeded.
- On restart, compare code/artifact hashes to completion evidence; inspect any
  partial edits before continuing. Mark stale DONE evidence for revalidation and
  block only affected dependents.
- Runtime experiment recovery follows AIC-019: committed attempt records are
  authoritative; incomplete attempt is INTERRUPTED; incompatible environment
  starts a new linked run. Never rerun an uncertain paid request blindly.
- If tests fail, retain actual output and fix within scope. Report independent
  environment blockers separately from source defects.
- If the owner changes scope, preserve old plan/evidence history, record proposed
  revision and update matrix/backlog/tracker consistently after approval.
- Do not erase existing run directories to make recovery or tests pass. Test
  data uses temporary paths; experimental results remain immutable evidence.

## Evidence and decision record templates

Task record fields: task ID; state; session date; source HEAD/dirty hash; changed
paths; command; environment/config identity; exit code; result/count; artifact
references; acceptance items; deviations; blocker; next action.

Approval record fields: decision ID; owner message/date; exact plan version or
milestone scope; decision (approve plan / authorize milestone / accept result /
request changes / approve material revision); constraints and budget; next
permitted action. Never invent timestamps, quotation text or authorizations.

Maintain document consistency before each review: valid unique IDs, acyclic task
dependencies, one milestone per task, valid paths, consistent scope/status,
requirements evidence and preserved approval boundaries. The initial audit's
verified primitives remain distinct from completion of new implementation tasks.
