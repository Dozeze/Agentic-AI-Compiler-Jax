"""Optimization controller: the current-best loop, the evaluation pipeline and the acceptance decision.

The controller never runs candidate code itself (AIC-1.1 §3). Every compile, run and timing happens in a
worker process through a runner (runner.py): the frozen reference is trusted and may use a local process;
model-written code runs only in an isolated container. The host compares outputs, analyses the raw timing
blocks and decides. A worker only reports.

Search keeps two sources apart: the immutable original baseline (the frozen reference) and the current
incumbent, the best verified candidate so far. Each candidate is measured against the incumbent.
"""
import ast
import dataclasses
import difflib
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import array_io
import benchmark
import feedback
import storage
import runner as runners
import worker
from contracts import (ACCEPTED, COMPILE_FAILED, CORRECTNESS_FAILED, DUPLICATE, EXECUTION_FAILED, INTERRUPTED,
                       MEASUREMENT_UNSTABLE, POLICY_REJECTED, PROPOSAL_INVALID, RESOURCE_LIMIT, TIMEOUT,
                       VALID_NO_CHANGE, VALID_SLOWER, Attempt, Budgets, ContractError, Evaluation,
                       parse_proposal)
from contracts import ModelSettings
from providers import ProposalFailure, ProposalProvider, proposal_tool
from context_builder import (build_context, workload_contract, check_development_cases,
                             compiler_view, COMPILER_CONTEXT, ContextError)
from source_policy import check_source


# ---------------------------------------------------------------------------
# Acceptance decision (AIC-006, AIC-1.1 §5.7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    state: str        # a terminal state from contracts
    reason: str


def decide(analysis, delta) -> Decision:
    """Rule on ONE stable-or-not measurement session of a correct candidate against the incumbent.

    Correctness is decided before this is ever called. A session must be stable; then the candidate is
    provisionally better only if the lower bound of its speedup exceeds 1 + delta, where delta is the
    frozen A/A calibration. It is slower only if the upper bound is below 1 / (1 + delta); anything in
    between is indistinguishable from noise.
    """
    if not analysis.stability.stable:
        return Decision(MEASUREMENT_UNSTABLE, "; ".join(analysis.stability.reasons))
    s = analysis.speedup
    interval = f"speedup {s.speedup:.3f}x, {s.confidence:.0%} CI [{s.ci_low:.3f}, {s.ci_high:.3f}]"
    if s.ci_low > 1 + delta:
        return Decision(ACCEPTED, f"{interval}; lower bound clears 1 + delta = {1 + delta:.3f}")
    if s.ci_high < 1 / (1 + delta):
        return Decision(VALID_SLOWER, f"{interval}; upper bound below 1 / (1 + delta)")
    return Decision(VALID_NO_CHANGE, f"{interval}; not distinguishable from noise (delta = {delta:.3f})")


def judge(measure, delta, protocol):
    """(Decision, analyses) for a correct candidate, from fresh sessions produced by `measure()`.

    An unstable session gets `protocol.unstable_retries` fixed retries, never unbounded reruns looking for
    a win. A provisional win must be repeated by `protocol.confirmation_sessions` further fresh sessions
    before promotion; all of them must meet the same rule.
    """
    analyses = []
    for _ in range(1 + protocol.unstable_retries):
        analyses.append(measure())
        decision = decide(analyses[-1], delta)
        if decision.state != MEASUREMENT_UNSTABLE:
            break
    if decision.state != ACCEPTED:
        return decision, analyses
    for i in range(protocol.confirmation_sessions):
        analyses.append(measure())
        confirmation = decide(analyses[-1], delta)
        if confirmation.state != ACCEPTED:
            state = MEASUREMENT_UNSTABLE if confirmation.state == MEASUREMENT_UNSTABLE else VALID_NO_CHANGE
            return Decision(state, f"provisional win not confirmed in fresh session {i + 1}: "
                                   f"{confirmation.reason}"), analyses
    return Decision(ACCEPTED, f"{decision.reason}; confirmed in {protocol.confirmation_sessions} fresh "
                              f"session(s)"), analyses


def verify_winner(measure_against_baseline, shapes, delta_for, protocol):
    """Final check of a frozen winner (AIC-1.1 §5.8): `protocol.final_sessions` fresh sessions against the
    ORIGINAL baseline on every declared shape. Returns (reproduced, {shape: [Decision, ...]}); a single
    session that does not clear 1 + delta on any shape means the effect did not reproduce.
    """
    per_shape = {}
    for shape in shapes:
        per_shape[shape] = [decide(measure_against_baseline(shape), delta_for(shape))
                            for _ in range(protocol.final_sessions)]
    reproduced = all(d.state == ACCEPTED for decisions in per_shape.values() for d in decisions)
    return reproduced, per_shape


def read_python_source(source_path):
    """Read a Python workload file and return its complete source as text."""
    path = Path(source_path)
    if path.suffix != ".py":
        raise ValueError(f"Expected a .py source file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Python source file does not exist: {path}")
    return path.read_bytes().decode("utf-8")


def same_program(a, b):
    """Are two sources the same program, ignoring formatting and comments? (compared as syntax trees)"""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return a.strip() == b.strip()


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Evaluation pipeline (AIC-006, AIC-008, AIC-009)
# ---------------------------------------------------------------------------

class EvaluationError(RuntimeError):
    """A worker request did not produce a usable result; `state` is the candidate's terminal state."""

    def __init__(self, state, detail):
        super().__init__(f"{state}: {detail}")
        self.state, self.detail = state, detail


class BaselineFailed(RuntimeError):
    """The frozen reference itself failed: the run stops before any proposal is requested."""


_OUTCOME_STATE = {runners.TIMEOUT: TIMEOUT, runners.RESOURCE_LIMIT: RESOURCE_LIMIT,
                  runners.CRASHED: EXECUTION_FAILED, runners.ISOLATION_UNAVAILABLE: INTERRUPTED}
_CATEGORY_STATE = {"COMPILE_FAILED": COMPILE_FAILED, "EXECUTION_FAILED": EXECUTION_FAILED}


@dataclass(frozen=True)
class Verdict:
    """What the pipeline concluded about one candidate."""
    state: str
    reason: str
    evaluation: Evaluation
    output_correct: bool = False
    latency_s: float | None = None          # candidate median latency in its first session
    latency_std_s: float | None = None


class Evaluator:
    """One workload, one runner, one measurement population.

    Correctness cases come from the workload spec; the measurement case is the primary (largest)
    development shape with the first search seed. The A/A calibration is either a frozen record (checked
    against this population) or taken lazily, before the first candidate measurement, from
    `protocol.aa_sessions` sessions of the baseline against itself.
    """

    def __init__(self, spec, runner, protocol, workdir, *, selected_backend, budgets=None, calibration=None,
                 cases=None):
        """`cases` overrides the search cases (the held-out evaluation of AIC-018 passes its own)."""
        self.spec, self.runner, self.protocol, self.backend = spec, runner, protocol, selected_backend
        self.budgets = budgets or Budgets()
        self._baseline_source = spec.reference_source
        self.baseline_sha256 = storage.sha256(self._baseline_source.encode("utf-8"))
        self.workdir = Path(workdir)
        self.cases = list(cases) if cases is not None else benchmark.build_cases(spec)
        self.case_names = [name for name, _ in self.cases]
        self.policy_hash = benchmark.policy_hash(spec.tolerance, self.case_names, spec.workload_id,
                                                 self.baseline_sha256)
        shape = spec.development_shapes[-1]
        self.measure_case = f"dev{benchmark._shape_name(shape)}/seed{spec.search_seeds[0]}"
        self.measure_shape = tuple(shape)
        identity = runner.identity()
        self.runner_label = identity.get("image") or identity["runner"]
        self.calibration = calibration
        self._calibration_checked = calibration is None
        self.fingerprint = None
        self.timer_resolution_s = None
        self.reference = None
        self.baseline_ir = None
        self.baseline_trace = None
        self._requests = 0

    @property
    def baseline_source(self):
        return self._baseline_source

    # -- worker requests --------------------------------------------------------------------------

    def _run(self, label, mode, sources, *, trusted, cases, timeout_s, **extra):
        for source in sources.values():
            storage.check_source_credentials(source)
        self._requests += 1
        work = self.workdir / f"{self._requests:04d}-{label}"
        work.mkdir(parents=True)
        files = {}
        for role, source in sources.items():
            (work / f"{role}.py").write_text(source, encoding="utf-8")
            files[role] = f"{role}.py"
        array_io.save(work / "inputs.npz", {name: array_io.flatten(args) for name, args in cases})
        request = {"schema_version": worker.REQUEST_SCHEMA, "mode": mode, "backend": self.backend,
                   "function_name": self.spec.function_name, "sources": files, "inputs": "inputs.npz", **extra}
        if mode == "outputs":
            request["analyze_case"] = self.measure_case
        outcome = self.runner.run(work, request, trusted=trusted, timeout_s=timeout_s)
        try:
            for name in ("stdout", "stderr"):
                data = storage.redact(getattr(outcome, name)).encode("utf-8")[-self.budgets.max_log_bytes:]
                storage.atomic_write_bytes(work / f"{name}.log", data, exclusive=True)
            storage.atomic_write_json(work / "runner_outcome.json", storage.redact({
                "schema_version": "runner_outcome/v1", "status": outcome.status, "detail": outcome.detail,
                "wall_s": outcome.wall_s, "stdout": "stdout.log", "stderr": "stderr.log"}), exclusive=True)
        except (OSError, ValueError) as error:
            raise storage.StorageError("cannot retain worker outcome") from error
        if outcome.status != runners.OK:
            raise EvaluationError(_OUTCOME_STATE[outcome.status], outcome.detail or outcome.status)
        if outcome.result.get("status") != "ok":
            raise EvaluationError(EXECUTION_FAILED, f"worker error: {outcome.result.get('error')}")
        self._same_population(outcome.result["device"])
        return outcome.result, work

    def _same_population(self, device):
        """Timings are only comparable within one device and runner; anything else stops the run."""
        fingerprint = f"{device['platform']}|{device['device_kind']}|jax {device['jax']}|{self.runner_label}"
        if self.fingerprint is None:
            self.fingerprint, self.timer_resolution_s = fingerprint, device["timer_resolution_s"]
        elif fingerprint != self.fingerprint:
            raise EvaluationError(INTERRUPTED, f"measurement population changed from {self.fingerprint} "
                                               f"to {fingerprint}; timings are not comparable")

    def _outputs(self, source, *, trusted, label):
        timeout = self.budgets.compile_timeout_s + len(self.cases) * self.budgets.correctness_case_timeout_s
        result, work = self._run(label, "outputs", {"candidate": source}, trusted=trusted, cases=self.cases,
                                 timeout_s=timeout)
        outputs = array_io.load(work / "outputs.npz", base=work, max_bytes=self.budgets.max_artifact_bytes)
        return result, outputs

    def reference_outputs(self):
        """Run the frozen reference once, trusted. Any failure is a harness bug: BaselineFailed."""
        try:
            result, outputs = self._outputs(self.baseline_source, trusted=True, label="reference")
        except EvaluationError as error:
            raise BaselineFailed(f"reference run failed: {error}") from error
        failed = [c for c in result["cases"] if c["status"] != "ok"]
        if failed:
            raise BaselineFailed(f"reference failed on {failed[0]['name']}: {failed[0].get('detail')}")
        for name, tree in outputs.items():
            for path, leaf in tree.leaves:
                if leaf.dtype.kind == "f" and not np.all(np.isfinite(leaf)):
                    raise BaselineFailed(f"reference produced non-finite values in {name}{path}")
        self.reference = outputs
        self.baseline_ir = result.get("ir")
        self.baseline_trace = self.profile(self.baseline_source, trusted=True, label="baseline-profile")
        return result

    def correctness(self, source, *, trusted=False):
        result, outputs = self._outputs(source, trusted=trusted, label="candidate")
        report = benchmark.report_from_outputs(self.case_names, self.reference, result["cases"], outputs,
                                               self.spec.tolerance, self.policy_hash)
        return report, result["compile"], result.get("ir")

    def session(self, a_source, b_source, *, trusted, repeats=None):
        case = [(name, args) for name, args in self.cases if name == self.measure_case]
        if repeats is None and self.calibration is not None:
            repeats = self.calibration.repeats
        result, _ = self._run("session", "measure", {"a": a_source, "b": b_source}, trusted=trusted, cases=case,
                              timeout_s=self.budgets.measurement_session_timeout_s, case=self.measure_case,
                              protocol=dataclasses.asdict(self.protocol), repeats=repeats)
        session = benchmark.PairedSession.from_json(result["session"])
        return session, benchmark.analyze_session(session, self.protocol)

    def profile(self, source, *, trusted, label="incumbent-profile"):
        """Collect optional observations in a fresh worker, outside all benchmark sessions."""
        import hashlib
        from contracts import RuntimeTrace

        provenance = {"candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "backend_requested": self.backend, "fingerprint": self.fingerprint}
        cases = [(name, args) for name, args in self.cases if name == self.measure_case]
        if not cases:
            return feedback.unavailable_trace("no development measurement case in this evaluation split", provenance)
        try:
            result, work = self._run(label, "profile", {"candidate": source}, trusted=trusted, cases=cases,
                                     timeout_s=self.budgets.compile_timeout_s + self.budgets.correctness_case_timeout_s,
                                     case=self.measure_case)
            trace = RuntimeTrace(**result["runtime_trace"]).to_json()
            if trace["provenance"].get("candidate_sha256") != provenance["candidate_sha256"]:
                raise ValueError("profile source identity differs from the requested source")
            trace["provenance"].update(provenance)
            if trace.get("artifact"):
                path = array_io.checked_path(work / trace["artifact"], work, feedback.MAX_TRACE_BYTES)
                if hashlib.sha256(path.read_bytes()).hexdigest() != trace["artifact_sha256"]:
                    raise ValueError("profile artifact hash mismatch")
                trace["artifact"] = str(Path(self.workdir.name) / path.relative_to(self.workdir.resolve()))
            return trace
        except (EvaluationError, ValueError, TypeError, KeyError, OSError) as error:
            return feedback.unavailable_trace(f"profile unavailable: {type(error).__name__}: {error}", provenance)

    def calibrate(self):
        """The frozen minimum effect delta for this population (AIC-005)."""
        from contracts import protocol_hash
        if self.calibration is not None:
            if not self._calibration_checked:
                self.calibration.check(workload_id=self.spec.workload_id, shape=self.measure_shape,
                                       dtype="float32", protocol_hash=protocol_hash(self.protocol),
                                       fingerprint=self.fingerprint)
                self._calibration_checked = True
            return self.calibration
        sessions, repeats = [], None
        for _ in range(self.protocol.aa_sessions):
            session, _ = self.session(self.baseline_source, self.baseline_source, trusted=True, repeats=repeats)
            repeats = session.repeats
            sessions.append(session)
        delta, diffs = benchmark.aa_threshold(sessions, self.protocol, self.timer_resolution_s)
        self.calibration = benchmark.Calibration(
            schema_version=benchmark.CALIBRATION_SCHEMA, label="run-start", workload_id=self.spec.workload_id,
            shape=self.measure_shape, dtype="float32", protocol_hash=protocol_hash(self.protocol),
            fingerprint=self.fingerprint, repeats=repeats, delta=delta, aa_relative_differences=diffs,
            timer_resolution_s=self.timer_resolution_s, created_utc=_utc())
        self.calibration.save(self.workdir / "calibration.json")
        return self.calibration

    # -- one candidate ------------------------------------------------------------------------------

    def evaluate(self, candidate_source, incumbent_source):
        """correctness -> (calibration) -> fresh sessions against the incumbent -> judge()."""
        import hashlib
        stages = {"policy": "ok", "correctness": "skipped", "measurement": "skipped"}
        evaluation = dict(candidate_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(), stages=stages)
        try:
            report, compile_times, ir = self.correctness(candidate_source)
        except EvaluationError as error:
            stages["correctness"] = "failed"
            return Verdict(error.state, error.detail, Evaluation(**evaluation, failure_category=error.state))
        evaluation.update(correctness=report.to_json(), compile=compile_times, ir=ir,
                          environment={"fingerprint": self.fingerprint, "runner": self.runner_label})
        if not report.passed:
            stages["correctness"] = "failed"
            state = _CATEGORY_STATE.get(report.category, CORRECTNESS_FAILED)
            first = next(c for c in report.cases if not c.passed)
            return Verdict(state, f"{report.category} on {first.case}: {first.detail}",
                           Evaluation(**evaluation, failure_category=report.category))
        stages["correctness"] = "ok"

        try:
            calibration = self.calibrate()
        except (benchmark.CalibrationError, EvaluationError) as error:
            # No trustworthy delta (noisy host, mismatched frozen record): stop, never guess a threshold.
            return Verdict(INTERRUPTED, f"no usable A/A calibration: {error}",
                           Evaluation(**evaluation, failure_category="CALIBRATION"), output_correct=True)
        sessions = []

        def measure():
            session, analysis = self.session(incumbent_source, candidate_source, trusted=False)
            sessions.append({"against": "incumbent", "a_sha256": storage.sha256(incumbent_source.encode()),
                             "b_sha256": storage.sha256(candidate_source.encode()),
                             "raw": session.to_json(), "analysis": analysis.to_json()})
            return analysis

        try:
            decision, analyses = judge(measure, calibration.delta, self.protocol)
        except EvaluationError as error:
            stages["measurement"] = "failed"
            return Verdict(error.state, error.detail,
                           Evaluation(**evaluation, sessions=tuple(sessions), failure_category=error.state),
                           output_correct=True)
        stages["measurement"] = "ok"
        if decision.state == ACCEPTED and not same_program(incumbent_source, self.baseline_source):
            # Recorded for the report, never used to rank: the decision is against the incumbent.
            session, analysis = self.session(self.baseline_source, candidate_source, trusted=False)
            sessions.append({"against": "baseline", "a_sha256": self.baseline_sha256,
                             "b_sha256": storage.sha256(candidate_source.encode()),
                             "raw": session.to_json(), "analysis": analysis.to_json()})
        first = analyses[0].b
        trace = (self.profile(candidate_source, trusted=False) if decision.state == ACCEPTED
                 else feedback.unavailable_trace("profiles retained for baseline and promoted incumbents"))
        stages["profile"] = "skipped" if trace["coverage"] == "UNAVAILABLE" else "ok"
        unavailable = {"runtime_trace": trace["reason"]} if trace["coverage"] == "UNAVAILABLE" else {}
        return Verdict(decision.state, decision.reason,
                       Evaluation(**evaluation, sessions=tuple(sessions), runtime_trace=trace, unavailable=unavailable),
                       output_correct=True, latency_s=first.median_s, latency_std_s=first.std_s)


def _compact(entry, context_condition="C2"):
    """The history entry as the model sees it: no raw blocks, no per-case lists, just the evidence.
    The compiler summary is included only under the context conditions that provide it."""
    view = {k: v for k, v in entry.items() if k != "attempt"}
    attempt = entry.get("attempt") or {}
    evaluation = attempt.get("evaluation") or {}
    if context_condition in COMPILER_CONTEXT:
        view["compiler"] = compiler_view(evaluation.get("ir"))
    if context_condition == "C4":
        trace = evaluation.get("runtime_trace") or feedback.unavailable_trace("this attempt has no profile evidence")
        view["runtime_trace"] = {key: trace.get(key) for key in
                                 ("coverage", "reason", "provenance", "artifact", "hotspots", "parser_version",
                                  "invalid_events", "ignored_events", "omitted_lanes")}
    correctness = evaluation.get("correctness")
    if correctness:
        failing = [c for c in correctness["cases"] if not c["passed"]][:1]
        view["correctness"] = {"passed": correctness["passed"], "category": correctness["category"],
                               "n_failed": correctness["n_failed"], "n_cases": correctness["n_cases"],
                               "first_failure": failing[0] if failing else None}
    view["sessions"] = [{"against": s["against"], "speedup": s["analysis"]["speedup"],
                         "stable": s["analysis"]["stability"]["stable"]}
                        for s in evaluation.get("sessions", ())]
    return view


class Controller:
    def __init__(self, agent: str, prompt, evaluator, client=None, function_name="kernel", max_iterations=3,
                 output_dir=None, source_path=None, context_condition="C2", store=None, model_settings=None):
        """Store the agent, prompt and evaluator. No side effects: start() evaluates the baseline."""
        self.agent = agent
        self.init_prompt = prompt
        self.client = client
        self.evaluator = evaluator
        self.function_name = function_name
        self.max_iterations = max_iterations
        self.history = []
        self.iteration = 0
        self.best_code = None
        self.best_runtime = None
        self.best_iteration = None
        self.stop_reason = None
        self._storage_failed = False
        self.model_settings = model_settings or ModelSettings()
        self._provider = None
        self._started = time.monotonic()
        self.requests = []          # one record per provider request: model, stop reason, tokens, latency
        self.context_condition = context_condition
        self.store = store or storage.RunStore(output_dir, budgets=evaluator.budgets)
        self.output_dir = self.store.path
        if evaluator._requests:
            raise storage.StorageError("a new run requires an unused evaluator")
        self.evaluator.workdir = self.output_dir / "work"
        self.source_path = str(source_path) if source_path is not None else None

    @property
    def initial_code(self):
        return self.evaluator.baseline_source

    def start(self):
        """Evaluate the immutable baseline once. Raises BaselineFailed before any proposal is requested."""
        if self.history:
            return
        if not self.store.initialized:
            from environment import collect_environment
            self.store.initialize(self.initial_code, config={
                "schema_version": "direct_run_config/v1", "output_dir": str(self.output_dir),
                "function_name": self.function_name, "context_condition": self.context_condition,
                "max_iterations": self.max_iterations, "backend": self.evaluator.backend,
                "model_settings_at_construction": dataclasses.asdict(self.model_settings),
                "measurement": dataclasses.asdict(self.evaluator.protocol),
                "budgets": dataclasses.asdict(self.evaluator.budgets)}, manifest={
                "scientific": False, "non_scientific_reasons": ["direct API; no resolved CLI RunConfig"],
                "workload": {"id": self.evaluator.spec.workload_id,
                             "reference_sha256": self.evaluator.baseline_sha256,
                             "correctness_policy_hash": self.evaluator.policy_hash,
                             "split": "development", "cases": self.evaluator.case_names},
                "runner": self.evaluator.runner.identity(),
                "environment": collect_environment(Path(__file__).resolve().parent)})
        self.evaluator.reference_outputs()
        import hashlib
        trace = self.evaluator.baseline_trace
        profile_available = trace["coverage"] != "UNAVAILABLE"
        evaluation = Evaluation(candidate_sha256=hashlib.sha256(self.initial_code.encode()).hexdigest(),
                                stages={"reference": "ok", "profile": "ok" if profile_available else "skipped"},
                                ir=self.evaluator.baseline_ir, runtime_trace=trace,
                                unavailable={} if profile_available else {"runtime_trace": trace["reason"]})
        attempt = Attempt(attempt_id=0, state=ACCEPTED, reason="immutable baseline: the frozen reference",
                          evaluation=evaluation, started_utc=_utc(), finished_utc=_utc())
        self._append({"iteration": 0, "state": ACCEPTED, "reason": attempt.reason, "code": self.initial_code,
                      "runtime_mean": None, "runtime_std": None, "syntax_correct": True, "output_correct": True,
                      "accepted": True, "changes": "No changes", "error": None, "attempt": attempt.to_json()})
        self.best_code, self.best_iteration = self.initial_code, 0

    def provideTools(self):
        return [proposal_tool(self.evaluator.budgets.max_source_bytes)]


    def changesPython(self, old_code, new_code):
        """Return a readable diff between the current best code and new code."""
        if old_code is None:
            return "No previous valid code"

        changes = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), fromfile="best.py", tofile="candidate.py", lineterm="")
        result = "\n".join(changes)
        return result if result else "No changes"

    def _report_runtime(self, result):
        """Print one explicit runtime/decision line for every evaluated iteration."""
        runtime = result["runtime_mean"]
        if runtime is None:
            timing = "runtime unavailable"
        else:
            timing = f"median {runtime * 1e6:.2f} µs ± {(result['runtime_std'] or 0) * 1e6:.2f} µs"
        print(f"Iteration {result['iteration']}: {timing}; {result['state']}: {result['reason']}")

    def _append(self, entry, *, submitted_proposal=None):
        entry.update(source_path=self.source_path, function_name=self.function_name)
        if entry.get("code") is not None:
            storage.check_source_credentials(entry["code"])
        entry = storage.redact(entry)
        try:
            self.store.commit_attempt(entry, submitted_proposal=submitted_proposal)
        except storage.StorageError:
            self._storage_failed = True
            raise
        self.history.append(entry)
        self.iteration = entry["iteration"]
        self._report_runtime(entry)
        return entry

    def provide_hist(self):
        """The bounded history supplied to a proposal, with measured summaries and explicit omissions."""
        return self.proposal_context()["history"] if self.history else []

    def provide_prompt(self):
        """ Provides all prompts to agent """
        return self.init_prompt

    def import_info(self, source_path="algorithm.py"):
        """ Imports the running function """

        return read_python_source(source_path)

    def get_current_code(self):
        return self.best_code

    def submit_proposal(self, tool_input):
        """Validate a proposal and its source before anything runs; then evaluate it in a worker.

        An invalid proposal, a policy violation or a duplicate is a terminal candidate state recorded in
        the history, and the candidate's code is never executed.
        """
        started = _utc()
        if self._storage_failed:
            raise storage.StorageError("run stopped after a storage failure; execution cannot resume")
        max_source_bytes = self.evaluator.budgets.max_source_bytes
        try:
            proposal = parse_proposal(tool_input, max_source_bytes=max_source_bytes)
        except ContractError as error:
            return self._record(PROPOSAL_INVALID, str(error), None, None, None, started, submitted_proposal=tool_input)
        if proposal.parent_id is not None and proposal.parent_id not in {e["iteration"] for e in self.history}:
            return self._record(PROPOSAL_INVALID, "proposal.parent_id must reference an earlier attempt",
                                None, None, None, started, submitted_proposal=tool_input)
        source = proposal.candidate_source
        storage.check_source_credentials(source)
        policy = check_source(source, self.function_name, max_source_bytes=max_source_bytes)
        if not policy.ok:
            return self._record(POLICY_REJECTED, policy.summary(), source, proposal, None, started)
        duplicate = next((e["iteration"] for e in self.history if e["code"] and same_program(e["code"], source)), None)
        if duplicate is not None:
            return self._record(DUPLICATE, f"same program as iteration {duplicate}; not measured again",
                                source, proposal, None, started)
        verdict = self.evaluator.evaluate(source, self.best_code)
        entry = self._record(verdict.state, verdict.reason, source, proposal, verdict, started)
        if verdict.state == ACCEPTED:
            self.best_code, self.best_iteration, self.best_runtime = source, entry["iteration"], verdict.latency_s
        if verdict.state == INTERRUPTED:
            self.stop_reason = verdict.reason
        return entry

    def _record(self, state, reason, code, proposal, verdict, started, *, submitted_proposal=None):
        number = self.iteration + 1
        parent = proposal.parent_id if proposal and proposal.parent_id is not None else self.best_iteration
        attempt = Attempt(attempt_id=number, state=state, reason=reason, proposal=proposal, parent_id=parent,
                          evaluation=verdict.evaluation if verdict else None,
                          provider=self.requests[-1] if self.requests else None, started_utc=started,
                          finished_utc=_utc())
        failed = state not in (ACCEPTED, VALID_NO_CHANGE, VALID_SLOWER)
        return self._append({
            "iteration": number, "state": state, "reason": reason, "code": code,
            "runtime_mean": verdict.latency_s if verdict else None,
            "runtime_std": verdict.latency_std_s if verdict else None,
            "syntax_correct": state != PROPOSAL_INVALID and "syntax error" not in reason,
            "output_correct": bool(verdict and verdict.output_correct), "accepted": state == ACCEPTED,
            "changes": self.changesPython(self.best_code, code) if code else None,
            "error": reason if failed else None, "attempt": attempt.to_json()}, submitted_proposal=submitted_proposal)

    def _persist_request(self, record):
        """Only the provider boundary dispatches calls; persist its reconciled outcome."""
        try:
            record = self.store.record_request(record)
        except storage.StorageError:
            self._storage_failed = True
            raise
        self.requests.append(record)
        return record

    def usage_totals(self):
        def total(key):
            return sum(r[key] or 0 for r in self.requests)
        return {"requests": len(self.requests), "input_tokens": total("input_tokens"),
                "output_tokens": total("output_tokens"),
                "accounted_tokens": sum((r.get("input_tokens_accounted") or 0) +
                                         (r.get("output_tokens_accounted") or 0) for r in self.requests),
                "unknown_usage_requests": sum(bool(r.get("usage_uncertain")) for r in self.requests)}

    def proposal_context(self, *, with_selection=False):
        """Select bounded measured evidence; never discard rules or the latest outcome to fit."""
        snapshot = self.store.context_inputs()
        remaining = self._provider.remaining_budget() if self._provider else None
        if remaining is not None:
            remaining["candidates"] = min(remaining["candidates"], max(0, self.max_iterations - self.iteration))
        try:
            check_development_cases(self.evaluator.spec, self.evaluator.case_names)
            snapshot["measurement_population"] = self.evaluator.fingerprint
            context = build_context(function_name=self.function_name, workload=workload_contract(self.evaluator.spec),
                baseline_sha256=self.evaluator.baseline_sha256, current_source=self.best_code,
                current_parent_id=self.best_iteration, history=self.history,
                environment=snapshot.pop("environment"), selected_backend=self.evaluator.backend, evidence=snapshot,
                context_condition=self.context_condition, max_source_bytes=self.evaluator.budgets.max_source_bytes,
                remaining_budgets=remaining,
                fits=self._provider.context_fits if self._provider else None,
                redact=self._provider._redact if self._provider else storage.redact)
        except ContextError as error:
            raise ProposalFailure("CONTEXT_BLOCKED", str(error)) from error
        return context if with_selection else context.payload

    def agent_loop(self, model=None, max_tokens=None):
        """Request one locally validated proposal at a time; the host alone invokes evaluation."""
        if self._storage_failed:
            raise storage.StorageError("run stopped after a storage failure; execution cannot resume")
        if min(self.max_iterations, self.evaluator.budgets.max_candidates) == 0:
            self.start()
            return deepcopy(self.history)
        if self.client is None:
            raise RuntimeError("A provider client is required for agent_loop")
        settings = dataclasses.replace(self.model_settings,
                                       model_id=model if model is not None else self.model_settings.model_id,
                                       max_tokens=max_tokens if max_tokens is not None else self.model_settings.max_tokens)
        if self._provider is None:
            self._provider = ProposalProvider(self.client, settings, self.evaluator.budgets, system=self.init_prompt,
                                               reserve=self.store.reserve_request, record=self._persist_request,
                                               started=self._started)
        elif self._provider.settings != settings:
            raise RuntimeError("provider parameters cannot change during a run")
        self.start()
        while self.iteration < min(self.max_iterations, self.evaluator.budgets.max_candidates) and not self.stop_reason:
            try:
                proposal = self._provider.propose(self.proposal_context(with_selection=True))
            except ProposalFailure as error:
                if error.status == PROPOSAL_INVALID:
                    self._record(PROPOSAL_INVALID, str(error), None, None, None, _utc(),
                                 submitted_proposal=error.submitted)
                    continue
                self.stop_reason = storage.redact(str(error))
                try:
                    self.store.record_provider_stop(error.status, self.stop_reason, self._provider.summary())
                except storage.StorageError:
                    self._storage_failed = True
                    raise
                break
            except storage.StorageError:
                self._storage_failed = True
                raise
            self.submit_proposal(dataclasses.asdict(proposal))
        return deepcopy(self.history)

    def graph_it(self):
        """ Graphs the measured candidate latencies vs iterations (unmeasured iterations are skipped) """
        points = [(e["iteration"], e["runtime_mean"], e["runtime_std"] or 0.0)
                  for e in self.history if e["runtime_mean"] is not None]
        if not points:
            print("No measured iterations to plot.")
            return
        iterations, means, stds = (list(column) for column in zip(*points))
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.plot(iterations, means, marker="o", label="median latency per iteration")
        plt.fill_between(iterations, lower, upper, alpha=0.5, label=r"$\pm 1$ standard deviation")
        plt.grid()
        plt.legend()
        plt.show()
