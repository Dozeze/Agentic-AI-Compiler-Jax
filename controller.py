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
import math
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
from contracts import ModelSettings, ConfigError, validate_budgets, protocol_hash
from providers import ProposalFailure, ProposalProvider, proposal_tool
from context_builder import (build_context, workload_contract, check_development_cases,
                             compiler_view, COMPILER_CONTEXT, ContextError, evaluation_commitment,
                             check_search_access)
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


class RunStopped(RuntimeError):
    """The episode is terminal; a new controller/run is required for further proposals."""


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
        validate_budgets(self.budgets)
        if hasattr(runner, "max_log_bytes"):
            runner.max_log_bytes = min(runner.max_log_bytes, self.budgets.max_log_bytes)
        self.clock, self.deadline, self.store = time.monotonic, None, None
        self.completed_required_at = None
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
        if identity.get("cpu_control"):     # P29: a CPU quota, fixed cores or neither are different populations
            self.runner_label += f"|cpu {identity['cpu_control']}"
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

    def remaining_s(self):
        if self.deadline is None:
            self.deadline = self.clock() + self.budgets.run_wall_s
        return max(0., self.deadline - self.clock())

    def candidate_preflight(self):
        if not (self.runner.isolated or getattr(self.runner, "reviewed_fixtures", False)):
            return "model-written code requires verified isolation; subprocess mode is baseline-only"
        return self.runner.preflight()

    def _run(self, label, mode, sources, *, trusted, cases, timeout_s, **extra):
        validate_budgets(self.budgets)
        remaining = self.remaining_s()
        if remaining <= 0 or timeout_s <= 0 or self.budgets.compile_timeout_s <= 0:
            raise EvaluationError(TIMEOUT, "run or worker budget exhausted before launch")
        if not trusted and (reason := self.candidate_preflight()):
            raise EvaluationError(INTERRUPTED, reason)
        stage_limits = {"compile": self.budgets.compile_timeout_s,
                        "correctness": self.budgets.correctness_case_timeout_s,
                        "measure": self.budgets.measurement_session_timeout_s,
                        "profile": self.budgets.correctness_case_timeout_s}
        stage = {"outputs": "correctness", "measure": "measure", "profile": "profile"}[mode]
        if stage_limits[stage] <= 0:
            raise EvaluationError(TIMEOUT, f"{stage} budget exhausted before launch")
        for source in sources.values():
            storage.check_source_credentials(source)
            if len(source.encode()) > self.budgets.max_source_bytes:
                raise EvaluationError(RESOURCE_LIMIT, "source byte budget exceeded before launch")
        if self.store is not None:
            self.store.check_capacity()
        self._requests += 1
        work = self.workdir / f"{self._requests:04d}-{label}"
        work.mkdir(parents=True)
        files = {}
        for role, source in sources.items():
            (work / f"{role}.py").write_text(source, encoding="utf-8")
            files[role] = f"{role}.py"
        array_io.save(work / "inputs.npz", {name: array_io.flatten(args) for name, args in cases})
        request = {"schema_version": worker.REQUEST_SCHEMA, "mode": mode, "backend": self.backend,
                   "function_name": self.spec.function_name, "sources": files, "inputs": "inputs.npz",
                   "stage_limits": stage_limits, "stage_cases": [name for name, _ in cases],
                   "host_deadline_monotonic": self.deadline, **extra}
        if mode == "outputs":
            request["analyze_case"] = self.measure_case
        if self.store is not None:
            self.store.check_capacity()
        timeout_s = min(timeout_s, self.remaining_s())
        if timeout_s <= 0:
            raise EvaluationError(TIMEOUT, "run wall budget exhausted while preparing worker inputs")
        try:
            outcome = self.runner.run(work, request, trusted=trusted, timeout_s=timeout_s)
        except Exception as error:
            if isinstance(error, storage.StorageError):
                raise
            outcome = runners.RunOutcome(runners.CRASHED, None, "", "", 0.,
                                         f"runner failed ({type(error).__name__}): {error}")
        try:
            for name in ("stdout", "stderr"):
                data = storage.redact(getattr(outcome, name)).encode("utf-8")
                data = data[-self.budgets.max_log_bytes:] if self.budgets.max_log_bytes else b""
                storage.atomic_write_bytes(work / f"{name}.log", data, exclusive=True)
            storage.atomic_write_json(work / "runner_outcome.json", storage.redact({
                "schema_version": "runner_outcome/v1", "status": outcome.status, "detail": outcome.detail,
                "wall_s": outcome.wall_s, "stdout": "stdout.log", "stderr": "stderr.log"}), exclusive=True)
        except (OSError, ValueError) as error:
            raise storage.StorageError("cannot retain worker outcome") from error
        if outcome.status != runners.OK:
            state = INTERRUPTED if outcome.detail and outcome.detail.startswith("runner failed") else _OUTCOME_STATE[outcome.status]
            raise EvaluationError(state, outcome.detail or outcome.status)
        if self.remaining_s() <= 0:
            raise EvaluationError(TIMEOUT, "worker completed after the run deadline; result discarded")
        if not isinstance(outcome.result, dict):
            raise EvaluationError(EXECUTION_FAILED, "worker returned no result object")
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
        except (EvaluationError, array_io.ArrayIOError, ValueError, KeyError, TypeError, OSError) as error:
            if isinstance(error, storage.StorageError):
                raise
            raise BaselineFailed(f"reference run failed: {error}") from error
        if (set(outputs) != set(self.case_names) or
                sorted(c.get("name", "") for c in result["cases"]) != sorted(self.case_names)):
            raise BaselineFailed("reference did not return exactly the declared cases")
        failed = [c for c in result["cases"] if c["status"] != "ok"]
        if failed:
            raise BaselineFailed(f"reference failed on {failed[0]['name']}: {failed[0].get('detail')}")
        for name, tree in outputs.items():
            for path, leaf in tree.leaves:
                if benchmark._is_float(leaf.dtype) and not np.all(np.isfinite(leaf)):
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
                              timeout_s=self.budgets.compile_timeout_s + self.budgets.measurement_session_timeout_s,
                              case=self.measure_case,
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
        evaluation["decision"] = {"delta": calibration.delta, "protocol_hash": protocol_hash(self.protocol),
                                  "incumbent_sha256": storage.sha256(incumbent_source.encode()),
                                  "calibration": dataclasses.asdict(calibration)}

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
            try:
                session, analysis = self.session(self.baseline_source, candidate_source, trusted=False)
            except EvaluationError as error:
                stages["measurement"] = "failed"
                return Verdict(error.state, error.detail, Evaluation(**evaluation, sessions=tuple(sessions)), True)
            sessions.append({"against": "baseline", "a_sha256": self.baseline_sha256,
                             "b_sha256": storage.sha256(candidate_source.encode()),
                             "raw": session.to_json(), "analysis": analysis.to_json()})
        first = analyses[0].b
        self.completed_required_at = self.clock()
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
                 output_dir=None, source_path=None, context_condition="C2", store=None, model_settings=None,
                 clock=time.monotonic):
        """Store the agent, prompt and evaluator. No side effects: start() evaluates the baseline."""
        self.agent = agent
        self.init_prompt = prompt
        self.client = client
        self.evaluator = evaluator
        validate_budgets(evaluator.budgets)
        if type(max_iterations) is not int or max_iterations < 0:
            raise ConfigError("max_iterations must be a nonnegative integer")
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
        self.clock, self._started = clock, None
        self.result = None
        self._seen_sources = {}
        self.requests = []          # one record per provider request: model, stop reason, tokens, latency
        self.context_condition = context_condition
        self.store = store or storage.RunStore(output_dir, budgets=evaluator.budgets)
        self.evaluator.clock, self.evaluator.store = self.clock, self.store
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
        if self.candidate_limit:
            check_search_access(self.evaluator.spec)
        if self.history:
            return
        if self._storage_failed:
            raise storage.StorageError("run stopped after a storage failure; execution cannot resume")
        if self.result is not None:
            raise BaselineFailed(self.result["reason"])
        if self._started is None:
            self._started = self.clock()
            self.evaluator.deadline = self._started + self.evaluator.budgets.run_wall_s
        if not self.store.initialized:
            from environment import collect_environment
            self._storage_failed = True
            self.store.initialize(self.initial_code, config={
                "schema_version": "direct_run_config/v1", "output_dir": str(self.output_dir),
                "function_name": self.function_name, "context_condition": self.context_condition,
                "max_iterations": self.max_iterations, "backend": self.evaluator.backend,
                "model_settings_at_construction": dataclasses.asdict(self.model_settings),
                "measurement": dataclasses.asdict(self.evaluator.protocol),
                "budgets": dataclasses.asdict(self.evaluator.budgets)}, manifest={
                "scientific": False, "non_scientific_reasons": ["direct API; no resolved CLI RunConfig"],
                "evaluation_split": evaluation_commitment(self.evaluator.spec),
                "workload": {"id": self.evaluator.spec.workload_id,
                             "reference_sha256": self.evaluator.baseline_sha256,
                             "correctness_policy_hash": self.evaluator.policy_hash,
                             "split": "development", "cases": self.evaluator.case_names},
                "runner": self.evaluator.runner.identity(),
                "environment": collect_environment(Path(__file__).resolve().parent)})
            self._storage_failed = False
        try:
            self.evaluator.reference_outputs()
            if self.evaluator.reference is None:
                raise BaselineFailed("reference produced no validated outputs")
        except storage.StorageError:
            self._storage_failed = True
            raise
        except Exception as error:
            reason = storage.redact(f"baseline failed ({type(error).__name__}): {error}")[:2000]
            self._finish("BASELINE_FAILED", reason, baseline_failure=True)
            raise BaselineFailed(reason) from error
        import hashlib
        trace = self.evaluator.baseline_trace or feedback.unavailable_trace("baseline has no profile evidence")
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

    @property
    def candidate_limit(self):
        return min(self.max_iterations, self.evaluator.budgets.max_candidates)

    def _budget_snapshot(self):
        return {"candidates": self.iteration, "candidate_limit": self.candidate_limit,
                "elapsed_s": max(0., self.clock() - self._started) if self._started is not None else 0.,
                "provider": self._provider.summary() if self._provider else None}

    def _finish(self, status, reason, *, baseline_failure=False):
        if self.result is None:
            try:
                self.result = self.store.finish_run(status, storage.redact(reason), self._budget_snapshot(),
                                                     baseline_failure=baseline_failure)
            except storage.StorageError:
                self._storage_failed = True
                raise
            self.stop_reason = self.result["reason"]
        return deepcopy(self.result)

    def optimize(self, model=None, max_tokens=None):
        """Run the episode and return run_result/v1 with its verified baseline/incumbent artifact."""
        self.agent_loop(model=model, max_tokens=max_tokens)
        return deepcopy(self.result)

    def _confirmed(self, verdict, source):
        """Recheck a claimed promotion from raw host evidence; injected evaluators cannot bypass judge()."""
        evidence = verdict.evaluation
        proof = evidence.decision or {}
        if (not verdict.output_correct or evidence.candidate_sha256 != storage.sha256(source.encode())
                or not evidence.correctness or evidence.correctness.get("passed") is not True
                or evidence.correctness.get("policy_hash") != self.evaluator.policy_hash
                or proof.get("incumbent_sha256") != storage.sha256(self.best_code.encode())
                or proof.get("protocol_hash") != protocol_hash(self.evaluator.protocol)):
            raise ContractError("claimed promotion lacks matching correctness/source/protocol evidence")
        delta = proof.get("delta")
        if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
            raise ContractError("claimed promotion lacks a finite frozen threshold")
        calibration = self.evaluator.calibration
        if calibration is None or delta != calibration.delta or proof.get("calibration") != dataclasses.asdict(calibration):
            raise ContractError("claimed promotion threshold differs from the frozen calibration")
        cases = evidence.correctness.get("cases", ())
        if (evidence.correctness.get("n_cases") != len(self.evaluator.case_names)
                or evidence.correctness.get("n_failed") != 0
                or sorted(case.get("case", "") for case in cases) != sorted(self.evaluator.case_names)
                or any(case.get("passed") is not True for case in cases)):
            raise ContractError("claimed promotion lacks correctness on every declared development case")
        analyses = []
        for session in evidence.sessions:
            if session["against"] != "incumbent":
                continue
            if (session.get("a_sha256") != proof["incumbent_sha256"]
                    or session.get("b_sha256") != evidence.candidate_sha256):
                raise ContractError("claimed promotion measured different sources")
            raw = session["raw"]
            blocks = raw["blocks"]
            if (type(raw.get("repeats")) is not int or raw["repeats"] != calibration.repeats
                    or len(blocks) != self.evaluator.protocol.n_blocks
                    or [block.get("index") for block in blocks] != list(range(len(blocks)))
                    or any(block.get("order") not in ("AB", "BA") or block.get("repeats") != raw["repeats"]
                           or any(type(block.get(key)) is not int or block[key] <= 0 for key in ("a_ns", "b_ns"))
                           for block in blocks)):
                raise ContractError("claimed promotion has incomplete or invalid measurement blocks")
            analysis = benchmark.analyze_session(benchmark.PairedSession.from_json(raw), self.evaluator.protocol)
            if analysis.to_json() != session["analysis"]:
                raise ContractError("claimed promotion analysis differs from its raw measurements")
            analyses.append(analysis)
        iterator = iter(analyses)
        try:
            decision, used = judge(lambda: next(iterator), delta, self.evaluator.protocol)
        except StopIteration as error:
            raise ContractError("claimed promotion lacks fresh confirmation sessions") from error
        if decision.state != ACCEPTED or len(used) != len(analyses):
            raise ContractError("claimed promotion is not confirmed by the host decision rule")

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
        if entry.get("code") is not None:
            self._seen_sources[storage.sha256(entry["code"].encode())] = entry["iteration"]
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
        if self.result is not None:
            raise RunStopped(self.result["reason"])
        self.start()
        if self.iteration >= self.candidate_limit:
            self._finish("CANDIDATE_LIMIT", "candidate budget exhausted")
            raise RunStopped("candidate budget exhausted")
        max_source_bytes = self.evaluator.budgets.max_source_bytes
        try:
            proposal = parse_proposal(tool_input, max_source_bytes=max_source_bytes)
        except ContractError as error:
            return self._record(PROPOSAL_INVALID, str(error), None, None, None, started, submitted_proposal=tool_input)
        if proposal.parent_id is not None and proposal.parent_id != self.best_iteration:
            return self._record(PROPOSAL_INVALID, "proposal.parent_id must reference the current incumbent",
                                None, None, None, started, submitted_proposal=tool_input)
        source = proposal.candidate_source
        storage.check_source_credentials(source)
        if self.evaluator.remaining_s() <= 0:
            return self._record(TIMEOUT, "run wall budget exhausted before evaluation", source, proposal, None, started)
        policy = check_source(source, self.function_name, max_source_bytes=max_source_bytes)
        if not policy.ok:
            return self._record(POLICY_REJECTED, policy.summary(), source, proposal, None, started)
        duplicate = self._seen_sources.get(storage.sha256(source.encode()))
        if duplicate is None:
            duplicate = next((e["iteration"] for e in self.history if e["code"] and same_program(e["code"], source)), None)
        if duplicate is not None:
            return self._record(DUPLICATE, f"same program as iteration {duplicate}; not measured again",
                                source, proposal, None, started)
        try:
            self.evaluator.completed_required_at = None
            verdict = self.evaluator.evaluate(source, self.best_code)
            if self.evaluator.remaining_s() <= 0 and (self.evaluator.completed_required_at is None
                    or self.evaluator.completed_required_at > self.evaluator.deadline):
                verdict = dataclasses.replace(verdict, state=TIMEOUT, reason="required evaluation exceeded the run deadline")
            if verdict.state == ACCEPTED:
                self._confirmed(verdict, source)
        except KeyboardInterrupt:
            verdict = Verdict(INTERRUPTED, "evaluation interrupted by user", Evaluation(storage.sha256(source.encode()),
                              {"evaluation": "interrupted"}, failure_category=INTERRUPTED))
        except storage.StorageError:
            self._storage_failed = True
            raise
        except Exception as error:
            reason = f"evaluation interrupted ({type(error).__name__}): {error}"
            verdict = Verdict(INTERRUPTED, reason, Evaluation(storage.sha256(source.encode()),
                              {"evaluation": "failed"}, failure_category="EVALUATOR_INVALID"))
        entry = self._record(verdict.state, verdict.reason, source, proposal, verdict, started)
        return entry

    def _record(self, state, reason, code, proposal, verdict, started, *, submitted_proposal=None):
        number = self.iteration + 1
        parent = proposal.parent_id if proposal and proposal.parent_id is not None else self.best_iteration
        attempt = Attempt(attempt_id=number, state=state, reason=reason, proposal=proposal, parent_id=parent,
                          evaluation=verdict.evaluation if verdict else None,
                          provider=self.requests[-1] if self.requests else None, started_utc=started,
                          finished_utc=_utc())
        failed = state not in (ACCEPTED, VALID_NO_CHANGE, VALID_SLOWER)
        entry = self._append({
            "iteration": number, "state": state, "reason": reason, "code": code,
            "runtime_mean": verdict.latency_s if verdict else None,
            "runtime_std": verdict.latency_std_s if verdict else None,
            "syntax_correct": state != PROPOSAL_INVALID and "syntax error" not in reason,
            "output_correct": bool(verdict and verdict.output_correct), "accepted": state == ACCEPTED,
            "changes": self.changesPython(self.best_code, code) if code else None,
            "error": reason if failed else None, "attempt": attempt.to_json()}, submitted_proposal=submitted_proposal)
        if state == ACCEPTED:
            self.best_code, self.best_iteration, self.best_runtime = code, number, verdict.latency_s
        if state in (INTERRUPTED, TIMEOUT, RESOURCE_LIMIT):
            self._finish(state, reason)
        elif self.evaluator.remaining_s() <= 0:
            self._finish("TIME_LIMIT", "run wall budget exhausted after committed outcome")
        elif self.iteration >= self.candidate_limit:
            self._finish("CANDIDATE_LIMIT", "candidate budget exhausted")
        return entry

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
        """Compatibility history return; optimize() returns the durable verified terminal result."""
        if self._storage_failed:
            raise storage.StorageError("run stopped after a storage failure; execution cannot resume")
        validate_budgets(self.evaluator.budgets)
        if self.result is not None:
            if self.result["best"] is not None:
                self.store.verified_best()
            return deepcopy(self.history)
        self.start()
        if self.candidate_limit == 0:
            self._finish("CANDIDATE_LIMIT", "zero candidate budget; verified baseline returned")
            return deepcopy(self.history)
        if self.evaluator.remaining_s() <= 0:
            self._finish("TIME_LIMIT", "run wall budget exhausted")
            return deepcopy(self.history)
        try:
            reason = self.evaluator.candidate_preflight()
        except Exception as error:
            reason = f"runner preflight failed ({type(error).__name__}): {error}"
        if reason:
            self._finish(INTERRUPTED, reason)
            return deepcopy(self.history)
        if self.client is None:
            self._finish("PROVIDER_UNAVAILABLE", "a provider client is required for optimization")
            return deepcopy(self.history)
        settings = dataclasses.replace(self.model_settings,
                                       model_id=model if model is not None else self.model_settings.model_id,
                                       max_tokens=max_tokens if max_tokens is not None else self.model_settings.max_tokens)
        if self._provider is None:
            from providers import ProviderError
            try:
                self._provider = ProposalProvider(self.client, settings, self.evaluator.budgets, system=self.init_prompt,
                    reserve=self.store.reserve_request, record=self._persist_request,
                    started=self._started, clock=self.clock)
            except ProviderError as error:
                self._finish("PROVIDER_UNAVAILABLE", str(error))
                return deepcopy(self.history)
        elif self._provider.settings != settings:
            raise RuntimeError("provider parameters cannot change during a run")
        while self.iteration < self.candidate_limit and self.result is None:
            try:
                if self.evaluator.remaining_s() <= 0:
                    self._finish("TIME_LIMIT", "run wall budget exhausted")
                    break
                if reason := self.evaluator.candidate_preflight():
                    self._finish(INTERRUPTED, reason)
                    break
                self.store.check_capacity()
                proposal = self._provider.propose(self.proposal_context(with_selection=True))
            except ProposalFailure as error:
                if error.status == PROPOSAL_INVALID:
                    self._record(PROPOSAL_INVALID, str(error), None, None, None, _utc(),
                                 submitted_proposal=error.submitted)
                    continue
                self.stop_reason = storage.redact(str(error))
                try:
                    self.store.record_provider_stop(error.status, self.stop_reason, self._provider.summary())
                    self._finish(error.status, self.stop_reason)
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
