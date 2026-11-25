# Simplification proposal (draft for team review, 2026-11-25)

Status: PROPOSAL. Nothing below is applied yet. Code written by another team member is only changed
after that person agrees. Measurement, correctness tolerances and the decision rule are out of scope.

## Why
About three weeks remain before the final presentation (2026-12-15 … 21). The code base is about
7,300 lines of Python, 6,000 lines of tests and 5,100 lines of documentation. Yet no live proposal
has been evaluated on the GPU and no speedup over XLA has been measured. Every mechanism that guards
against a risk this project does not have costs review time, presentation time and test time. Each
of us must be able to explain the code in the 1.5 h presentation (AI_CONTEXT.md §7).

Rule used to pick candidates: keep a guard when it protects correctness, timing validity, the
held-out split or the public repository. Remove it when it only re-checks our own trusted code,
or defends against an unlikely failure at a high complexity cost.

## A. Dead code (safe; no behavior change)
| Item | Evidence |
|---|---|
| `algorithm.py`, `utils.py` | imported nowhere; `algorithm.py` does not even run (`jax.grad` applied to an array) |
| `test.py`, `test1.py`, `test2.py`, `test3.py`, `testrun.py`, `runningscripts.py` | root scratch scripts from `main`; not collected by pytest (`testpaths = ["tests"]`), not imported (P10) |
| `Controller.provideTools`, `provide_prompt`, `import_info` | no caller anywhere |
| `benchmark.timeit_version_1` and the legacy `benchmark.evaluate` | used only by `runningscripts.py` and legacy tests; the real path is `measure_paired` |

## B. Duplicate mechanisms (choose one)
1. **Held-out reuse is blocked twice.** `heldout.py` keeps a burn ledger (`evidence/heldout/ledger.jsonl`,
   `check_not_burned`, `evaluate`, used by `scripts/heldout.py`). `finalization.py` adds a second,
   directory-based one-use reservation (`reserve`, `check_search_allowed`, used by `agent.py --finalize-run`).
   There are also two entry points for the same final evaluation.
   Proposal: keep `agent.py --finalize-run` (it inherits the run's population and protocol) and
   **one** guard: the ledger, appended before the final worker starts. Remove `scripts/heldout.py` and
   the reservation directories.
2. **`Controller._confirmed` (49 lines) re-derives a promotion that `Evaluator.evaluate` → `judge()`
   has just made**, from the same process's own data: same hashes, same raw blocks, re-run
   bootstrap. It defends against "an injected evaluator bypassing judge()", but there is no
   untrusted evaluator; the evaluator is our code. Model-written code never runs in the host process.
   Proposal: keep the cheap source/incumbent hash check, drop the re-analysis.

## C. Over-engineered for this threat model (largest gains; needs Jonathan's agreement)
1. **`ProposalProvider` (323 lines).** Each request is dispatched on a daemon thread with a host-side
   queue timeout, on top of the SDK timeout. A durable "reservation" ledger is written before each
   dispatch, with worst-case input tokens = UTF-8 bytes + 4096, and money is tracked as `Decimal`.
   Our runs are 3–10 requests that cost cents.
   Proposal: SDK `timeout` + `max_retries=0`, count requests, record the usage reported by the API,
   and stop after the first response that exceeds the cap. Keep the money cap itself (P8), with a
   simple estimate. Expected size: about 100 lines.
2. **`RunStore` (399 lines) + `verify_mvp.py` (151).** Atomic writes, fsyncs of directories, symlink
   refusal, hash-verified indexes rebuilt from commits, and a separate verifier that re-reads all of it.
   Proposal: keep atomic JSON writes and SHA-256 of sources and evidence (cheap, useful in the report).
   Drop the directory fsyncs, symlink checks, derived-index rebuild and `verify_mvp.py`; the run's
   `record.json` files are the evidence.
3. **Context selection (AIC-014, `context_builder.py` + P21).** The 24 KiB metadata cap, the
   remaining-reservation fit check and the snippet-eviction order exist partly to serve the
   reservation ledger that C.1 removes. Proposal: keep the allowlist (no split inputs, no credentials, no raw logs),
   the last N attempts and a single byte cap.

## D. Documentation
1. `docs/IMPLEMENTATION_TASKS.md` still says AIC-001 … 010 `NOT_STARTED`. The gate table in
   `docs/IMPLEMENTATION_PLAN.md` still marks G-04 and G-05 `MISSING`. Both were done in batches 02/03.
   Fix the statuses.
2. `AI_CONTEXT.md` §6 P18–P25 are paragraph-long policies (up to 12 lines each). This file is meant
   to be read first by every person and every AI. Proposal: one or two lines per decision, with a link
   to the task section that holds the detail.
3. `docs/CODEBASE_AUDIT.md` and `docs/CODEX_WORKFLOW.md` describe the state before AIC-011.
   Proposal: move both to `docs/archive/`.

## E. What stays as it is
The measurement protocol, A/A calibration, decision rule, tolerances, source policy, Docker isolation,
correctness gate, fixed held-out seeds and offline provider replay. These are what make a speed
claim credible.

## Estimated effect
A + B + C remove roughly 1,000–1,500 lines of code (rough estimate) and the tests that only cover the removed
mechanisms. Order: A and D first (no behavior change, one commit each), then B, then C after review.
