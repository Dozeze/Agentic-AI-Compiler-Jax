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
import runner as runners
import worker
from contracts import (ACCEPTED, COMPILE_FAILED, CORRECTNESS_FAILED, DUPLICATE, EXECUTION_FAILED, INTERRUPTED,
                       MEASUREMENT_UNSTABLE, POLICY_REJECTED, PROPOSAL_INVALID, RESOURCE_LIMIT, TIMEOUT,
                       VALID_NO_CHANGE, VALID_SLOWER, Attempt, Budgets, ContractError, Evaluation,
                       parse_proposal)
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
    return path.read_text(encoding="utf-8")


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

    def __init__(self, spec, runner, protocol, workdir, *, selected_backend, budgets=None, calibration=None):
        self.spec, self.runner, self.protocol, self.backend = spec, runner, protocol, selected_backend
        self.budgets = budgets or Budgets()
        self.workdir = Path(workdir)
        self.cases = benchmark.build_cases(spec)
        self.case_names = [name for name, _ in self.cases]
        self.policy_hash = benchmark.policy_hash(spec.tolerance, self.case_names, spec.workload_id,
                                                 spec.reference_sha256)
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
        self._requests = 0

    @property
    def baseline_source(self):
        return self.spec.reference_source

    # -- worker requests --------------------------------------------------------------------------

    def _run(self, label, mode, sources, *, trusted, cases, timeout_s, **extra):
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
        outcome = self.runner.run(work, request, trusted=trusted, timeout_s=timeout_s)
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
        return result

    def correctness(self, source, *, trusted=False):
        result, outputs = self._outputs(source, trusted=trusted, label="candidate")
        report = benchmark.report_from_outputs(self.case_names, self.reference, result["cases"], outputs,
                                               self.spec.tolerance, self.policy_hash)
        return report, result["compile"]

    def session(self, a_source, b_source, *, trusted, repeats=None):
        case = [(name, args) for name, args in self.cases if name == self.measure_case]
        if repeats is None and self.calibration is not None:
            repeats = self.calibration.repeats
        result, _ = self._run("session", "measure", {"a": a_source, "b": b_source}, trusted=trusted, cases=case,
                              timeout_s=self.budgets.measurement_session_timeout_s, case=self.measure_case,
                              protocol=dataclasses.asdict(self.protocol), repeats=repeats)
        session = benchmark.PairedSession.from_json(result["session"])
        return session, benchmark.analyze_session(session, self.protocol)

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
            report, compile_times = self.correctness(candidate_source)
        except EvaluationError as error:
            stages["correctness"] = "failed"
            return Verdict(error.state, error.detail, Evaluation(**evaluation, failure_category=error.state))
        evaluation.update(correctness=report.to_json(), compile=compile_times,
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
            sessions.append({"against": "incumbent", "raw": session.to_json(), "analysis": analysis.to_json()})
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
            sessions.append({"against": "baseline", "raw": session.to_json(), "analysis": analysis.to_json()})
        first = analyses[0].b
        return Verdict(decision.state, decision.reason, Evaluation(**evaluation, sessions=tuple(sessions)),
                       output_correct=True, latency_s=first.median_s, latency_std_s=first.std_s)


def _compact(entry):
    """The history entry as the model sees it: no raw blocks, no per-case lists, just the evidence."""
    view = {k: v for k, v in entry.items() if k != "attempt"}
    attempt = entry.get("attempt") or {}
    evaluation = attempt.get("evaluation") or {}
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
                 output_dir=None, source_path=None):
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
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.source_path = str(source_path) if source_path is not None else None

    @property
    def initial_code(self):
        return self.evaluator.baseline_source

    def start(self):
        """Evaluate the immutable baseline once. Raises BaselineFailed before any proposal is requested."""
        if self.history:
            return
        self.evaluator.reference_outputs()
        self.best_code, self.best_iteration = self.initial_code, 0
        attempt = Attempt(attempt_id=0, state=ACCEPTED, reason="immutable baseline: the frozen reference",
                          started_utc=_utc(), finished_utc=_utc())
        self._append({"iteration": 0, "state": ACCEPTED, "reason": attempt.reason, "code": self.initial_code,
                      "runtime_mean": None, "runtime_std": None, "syntax_correct": True, "output_correct": True,
                      "accepted": True, "changes": "No changes", "error": None, "attempt": attempt.to_json()})

    def provideTools(self):
        return [
            {
                "name": "get_history",
                "description": (
                    "Get the complete optimization history. "
                    "Use this to see previous candidate implementations, "
                    "their correctness, runtime, and whether they were accepted."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_current_code",
                "description": (
                    "Get the current best JAX implementation being optimized."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "benchmark_code",
                "description": (
                    "Submit one proposal: a hypothesis and a complete candidate source. The harness "
                    "checks the source against its policy (pure JAX only), then for correctness, and "
                    "only then measures it. A correct candidate becomes the current best only when it "
                    "is faster than the incumbent."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string",
                                       "description": "Why this change should be faster."},
                        "transformation": {"type": "string",
                                           "description": "The single change made, e.g. 'vectorise the loop'."},
                        "rationale": {"type": "string",
                                      "description": "The evidence (code, history) behind the hypothesis."},
                        "expected_effect": {"type": "string",
                                            "description": "Predicted effect; a hypothesis, not a measurement."},
                        "candidate_source": {
                            "type": "string",
                            "description": (
                                "Complete Python source of the candidate module: imports from jax, "
                                "jax.numpy, jax.lax, jax.nn, jax.scipy, math or functools only, and the "
                                "function itself. Raw source without Markdown fences, escaped quotes or "
                                "@jax.jit decorators; the evaluator applies JIT."
                            ),
                        },
                        "parent_id": {"type": "integer",
                                      "description": "Iteration this builds on (default: current best)."},
                        "constraints": {"type": "array", "items": {"type": "string"},
                                        "description": "Semantics the rewrite preserves."},
                        "notes": {"type": "string"},
                    },
                    "required": ["hypothesis", "transformation", "rationale", "expected_effect",
                                 "candidate_source"],
                    "additionalProperties": False,
                },
            },
        ]


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

    def _write_iteration_json(self, result):
        """Persist each attempt and the complete history as JSON artifacts."""
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        iteration_path = self.output_dir / f"iteration_{result['iteration']:04d}.json"
        history_path = self.output_dir / "history.json"
        for path, value in ((iteration_path, result), (history_path, self.history)):
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
            temporary_path.replace(path)

    def _append(self, entry):
        entry.update(source_path=self.source_path, function_name=self.function_name)
        self.history.append(entry)
        self._report_runtime(entry)
        self._write_iteration_json(entry)
        return entry

    def provide_hist(self):
        """The history as the agent sees it: compact, without raw timing blocks or per-case lists."""
        return [_compact(entry) for entry in deepcopy(self.history)]

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
        max_source_bytes = self.evaluator.budgets.max_source_bytes
        try:
            proposal = parse_proposal(tool_input, max_source_bytes=max_source_bytes)
        except ContractError as error:
            return self._record(PROPOSAL_INVALID, str(error), None, None, None, started)
        source = proposal.candidate_source
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

    def _record(self, state, reason, code, proposal, verdict, started):
        self.iteration += 1
        parent = proposal.parent_id if proposal and proposal.parent_id is not None else self.best_iteration
        attempt = Attempt(attempt_id=self.iteration, state=state, reason=reason, proposal=proposal, parent_id=parent,
                          evaluation=verdict.evaluation if verdict else None, started_utc=started,
                          finished_utc=_utc())
        failed = state not in (ACCEPTED, VALID_NO_CHANGE, VALID_SLOWER)
        return self._append({
            "iteration": self.iteration, "state": state, "reason": reason, "code": code,
            "runtime_mean": verdict.latency_s if verdict else None,
            "runtime_std": verdict.latency_std_s if verdict else None,
            "syntax_correct": state != PROPOSAL_INVALID and "syntax error" not in reason,
            "output_correct": bool(verdict and verdict.output_correct), "accepted": state == ACCEPTED,
            "changes": self.changesPython(self.best_code, code) if code else None,
            "error": reason if failed else None, "attempt": attempt.to_json()})

    def _dispatch_tool(self, name, tool_input):
        """Run one model-requested tool and return JSON-serializable data."""
        if name == "get_history":
            return {"history": self.provide_hist()}

        if name == "get_current_code":
            return {"code": self.get_current_code()}

        if name == "benchmark_code":
            return _compact(self.submit_proposal(tool_input))

        return {"error": f"Unknown tool: {name}"}

    def agent_loop(self, model="claude-haiku-4-5-20251001", max_tokens=2048):
        """Ask the model for bounded candidates and keep only measured, confirmed improvements."""
        if self.client is None:
            raise RuntimeError("An Anthropic client is required for agent_loop")
        self.start()

        messages = [{
            "role": "user",
            "content": (
                "Start the optimization run. Inspect the current implementation and "
                "history with the tools. Propose a complete replacement source file "
                f"for `{self.function_name}` and call benchmark_code to evaluate it. "
                "Only measured, correct candidates can become the best implementation. "
                f"You may make at most {self.max_iterations} candidate evaluations."
            ),
        }]
        max_turns = max(1, self.max_iterations * 3)
        candidate_count = 0

        for _ in range(max_turns):
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self.init_prompt,
                tools=self.provideTools(),
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            evaluated_this_turn = False
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                name = block.name
                if name == "benchmark_code" and (evaluated_this_turn or self.stop_reason):
                    result = {"error": "Only one benchmark_code call is allowed per turn"
                              if evaluated_this_turn else f"run stopped: {self.stop_reason}"}
                else:
                    result = self._dispatch_tool(name, block.input)
                    if name == "benchmark_code":       # every submission spends a candidate slot
                        evaluated_this_turn = True
                        candidate_count += 1

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            if not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})
            if candidate_count >= self.max_iterations or self.stop_reason:
                break

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
