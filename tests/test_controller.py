"""AIC-015 state transitions use synthetic reports/timings and real host decisions/storage.

No candidate executes in the scripted evaluator. Native worker correctness is separately covered in
test_context/test_foundation; selected-target sandbox/live acceptance remains AIC-016.
"""
import dataclasses
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark
import feedback
import runner
import storage
from contracts import (ACCEPTED, COMPILE_FAILED, CORRECTNESS_FAILED, DUPLICATE, INTERRUPTED,
    MEASUREMENT_UNSTABLE, POLICY_REJECTED, PROPOSAL_INVALID, RESOURCE_LIMIT, TIMEOUT,
    VALID_NO_CHANGE, VALID_SLOWER, Budgets, ConfigError, Evaluation, MeasurementProtocol, ModelPrices,
    ModelSettings, protocol_hash)
from controller import BaselineFailed, Controller, EvaluationError, Evaluator, RunStopped, Verdict
from fakes import FakeAnthropic, proposal, tool_use
from fixtures.softmax_candidates import JAX_NN, LOGSUMEXP, COMPILE_ERROR
from providers import OfflineClient, ProposalProvider, ProposalFailure
from workloads.registry import SOFTMAX

P = MeasurementProtocol(warmup_calls=2, n_blocks=6, bootstrap_samples=500)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FixtureRunner:
    isolated = True

    def identity(self):
        return {"runner": "synthetic-fixture/v1", "isolated": True}

    def preflight(self):
        return None

    def run(self, *args, **kwargs):
        raise AssertionError("scripted evaluator must not execute source")


class ScriptedEvaluator(Evaluator):
    """Scripts only observations; production Evaluator.evaluate and judge decide the outcome."""
    def __init__(self, work, *, budgets=None, reports=(), ratios=(), baseline_error=None):
        super().__init__(SOFTMAX, FixtureRunner(), P, work, selected_backend="cpu", budgets=budgets)
        self.reports, self.ratios = list(reports), list(ratios)
        self.baseline_error = baseline_error
        self.baseline_calls, self.evaluated, self.measured = 0, [], []
        self.fingerprint = "synthetic observation fixture"
        self.calibration = benchmark.Calibration(benchmark.CALIBRATION_SCHEMA, "synthetic fixture", SOFTMAX.workload_id,
            self.measure_shape, "float32", protocol_hash(P), self.fingerprint, 1, .03, (), 1e-9, "fixture")
        self._calibration_checked = True

    def reference_outputs(self):
        self.baseline_calls += 1
        if self.baseline_error:
            raise self.baseline_error
        self.reference = "synthetic reference fixture; no execution"
        self.baseline_trace = feedback.unavailable_trace("scripted evaluator; no runtime observations")

    def correctness(self, source):
        self.evaluated.append(source)
        category = self.reports.pop(0) if self.reports else None
        if isinstance(category, BaseException):
            raise category
        cases = tuple(benchmark.CaseResult(name, category is None, category, (), category)
                      for name in self.case_names)
        return benchmark.CorrectnessReport(category is None, category, self.policy_hash, cases), {}, None

    def session(self, a, b, *, trusted, repeats=None):
        self.measured.append((a, b, trusted))
        ratio = self.ratios.pop(0) if self.ratios else 1.
        if isinstance(ratio, BaseException):
            raise ratio
        blocks = tuple(benchmark.Block(i, "AB" if i % 2 == 0 else "BA", 1,
            1_000_000, round(1_000_000 * ratio), 4*i, 4*i+1) for i in range(P.n_blocks))
        session = benchmark.PairedSession(protocol_hash(P), 1, blocks)
        return session, benchmark.analyze_session(session, P)

    def profile(self, *args, **kwargs):
        return feedback.unavailable_trace("scripted evaluator; no profile evidence")


@pytest.fixture(autouse=True)
def fixture_environment(monkeypatch):
    monkeypatch.setattr("environment.collect_environment", lambda *args, **kwargs: {
        "schema_version": "environment/v1", "devices": [], "unavailable": {"capture": "synthetic unit fixture"}})


def build(tmp_path, script=(), *, budgets=None, limit=3, reports=(), ratios=(), baseline_error=None, clock=None):
    evaluator = ScriptedEvaluator(tmp_path / "unused", budgets=budgets, reports=reports, ratios=ratios,
                                  baseline_error=baseline_error)
    client = FakeAnthropic(script)
    controller = Controller("fixture", "Return one proposal", evaluator, client=client,
        function_name="softmax", max_iterations=limit, output_dir=tmp_path / "run", clock=clock or Clock())
    return controller, client


def call(source=JAX_NN, **kwargs):
    return [tool_use("fixture", "submit_proposal", proposal(source, **kwargs))]


def test_initialization_is_explicit_and_zero_candidates_returns_verified_baseline(tmp_path):
    controller, client = build(tmp_path, [call()], limit=0)
    assert not controller.output_dir.exists() and controller.evaluator.baseline_calls == 0
    result = controller.optimize()
    assert controller.evaluator.baseline_calls == 1 and not client.messages.calls
    assert result["status"] == "CANDIDATE_LIMIT" and result["best"]["baseline_fallback"]
    assert (controller.output_dir / "best.py").read_text() == controller.initial_code
    assert controller.optimize() == result and controller.evaluator.baseline_calls == 1


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_candidate_limits_fail_before_artifacts(tmp_path, limit):
    with pytest.raises(ConfigError, match="nonnegative integer"):
        build(tmp_path, limit=limit)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field,value", [("max_candidates", -1), ("run_wall_s", -1),
    ("compile_timeout_s", float("nan")), ("max_source_bytes", True), ("max_artifact_bytes", -2)])
def test_invalid_budgets_fail_at_direct_evaluator_boundary(tmp_path, field, value):
    with pytest.raises(ConfigError):
        build(tmp_path, budgets=dataclasses.replace(Budgets(), **{field: value}))
    assert not (tmp_path / "run").exists()


def test_failed_baseline_is_terminal_and_never_creates_a_best_or_calls_model(tmp_path):
    controller, client = build(tmp_path, [call()], baseline_error=BaselineFailed("invalid reference"))
    with pytest.raises(BaselineFailed, match="invalid reference"):
        controller.optimize()
    result = json.loads((controller.output_dir / "run_result.json").read_bytes())
    assert result["status"] == "BASELINE_FAILED" and result["best"] is None
    assert (controller.output_dir / "baseline_failure.json").exists()
    assert not (controller.output_dir / "best.py").exists() and not controller.history
    assert controller.optimize() == result
    assert not client.messages.calls and controller.evaluator.baseline_calls == 1


def test_invalid_proposal_and_policy_rejection_count_once_without_evaluation(tmp_path):
    invalid = [tool_use("invalid", "submit_proposal", {"code": JAX_NN})]
    forbidden = call("import os\ndef softmax(x):\n    return x\n")
    controller, client = build(tmp_path, [invalid, forbidden, call()], limit=3)
    result = controller.optimize()
    assert [row["state"] for row in controller.history] == [ACCEPTED, PROPOSAL_INVALID, POLICY_REJECTED, VALID_NO_CHANGE]
    assert controller.evaluator.evaluated == [JAX_NN] and result["candidate_attempts"] == 3
    assert len(client.messages.calls) == 3 and controller._provider.proposal_count == 3


def test_exact_and_semantic_duplicate_hashes_never_reach_the_evaluator(tmp_path):
    controller, client = build(tmp_path, limit=2)
    client.messages.script = [call(controller.initial_code), call("# formatting only\n" + controller.initial_code)]
    assert controller.optimize()["candidate_attempts"] == 2
    assert [row["state"] for row in controller.history] == [ACCEPTED, DUPLICATE, DUPLICATE]
    assert not controller.evaluator.evaluated
    with pytest.raises(RunStopped):
        controller.submit_proposal(proposal(JAX_NN))


def test_protocol_malformed_response_is_a_request_failure_without_an_invented_candidate(tmp_path):
    controller, client = build(tmp_path, [[tool_use("a", "unknown_tool", {})], call()])
    result = controller.optimize()
    assert result["status"] == "MALFORMED" and result["candidate_attempts"] == 0
    assert len(client.messages.calls) == 1 and result["best"]["baseline_fallback"]


@pytest.mark.parametrize("ratio,state", [(1., VALID_NO_CHANGE), (1.5, VALID_SLOWER), (.5, ACCEPTED)])
def test_real_host_rule_selects_terminal_outcome_and_only_confirmed_win_replaces_best(tmp_path, ratio, state):
    controller, _ = build(tmp_path, [call()], limit=1, ratios=[ratio, ratio])
    result = controller.optimize()
    assert controller.history[-1]["state"] == state
    assert result["candidate_attempts"] == 1 and len(controller.store.committed_records()) == 2
    assert result["best"]["attempt_id"] == (1 if state == ACCEPTED else 0)
    assert len(controller.evaluator.measured) == (2 if state == ACCEPTED else 1)
    assert controller.best_code == (JAX_NN if state == ACCEPTED else controller.initial_code)


def test_unconfirmed_win_is_not_promoted(tmp_path):
    controller, _ = build(tmp_path, [call()], limit=1, ratios=[.5, 1.])
    assert controller.optimize()["best"]["baseline_fallback"]
    assert controller.history[-1]["state"] == VALID_NO_CHANGE


def test_unstable_measurements_exhaust_the_fixed_retry_without_promotion(tmp_path):
    controller, _ = build(tmp_path, [call()], limit=1)
    observations = []
    def noisy(*args, **kwargs):
        observations.append(True)
        blocks = tuple(benchmark.Block(i, "AB", 1, 1_000_000, 1_000_000 if i % 2 else 2_000_000,
                                        4*i, 4*i+1) for i in range(P.n_blocks))
        session = benchmark.PairedSession(protocol_hash(P), 1, blocks)
        return session, benchmark.analyze_session(session, P)
    controller.evaluator.session = noisy
    assert controller.optimize()["best"]["baseline_fallback"]
    assert controller.history[-1]["state"] == MEASUREMENT_UNSTABLE and len(observations) == 2


def test_resubmitted_unstable_candidate_is_measured_once_more(tmp_path):
    # P33: an unstable but correct candidate resubmitted unchanged gets one more measurement, not two.
    controller, _ = build(tmp_path, [call(), call(), call()], limit=3)
    calls = []
    def session(*args, **kwargs):
        calls.append(True)
        noisy = len(calls) <= 2                    # the first attempt and its fixed retry are unstable
        blocks = tuple(benchmark.Block(i, "AB", 1, 1_000_000,
                                        (1_000_000 if i % 2 else 2_000_000) if noisy else 500_000,
                                        4*i, 4*i+1) for i in range(P.n_blocks))
        session = benchmark.PairedSession(protocol_hash(P), 1, blocks)
        return session, benchmark.analyze_session(session, P)
    controller.evaluator.session = session
    controller.optimize()
    assert [row["state"] for row in controller.history[1:]] == [MEASUREMENT_UNSTABLE, ACCEPTED, DUPLICATE]


def test_optional_profile_expiry_preserves_an_already_confirmed_incumbent(tmp_path):
    clock = Clock()
    controller, _ = build(tmp_path, [call()], limit=1, clock=clock, ratios=[.5, .5], budgets=Budgets(run_wall_s=1.))
    def profile(*args, **kwargs):
        clock.now = 2.
        return feedback.unavailable_trace("profile exhausted the remaining budget")
    controller.evaluator.profile = profile
    result = controller.optimize()
    assert result["status"] == "TIME_LIMIT" and result["best"]["attempt_id"] == 1
    assert controller.history[-1]["state"] == ACCEPTED


def test_stale_parent_is_rejected_after_promotion(tmp_path):
    controller, _ = build(tmp_path, [call(), call(LOGSUMEXP, parent_id=0)], limit=2, ratios=[.5, .5])
    result = controller.optimize()
    assert result["best"]["attempt_id"] == 1 and controller.history[-1]["state"] == PROPOSAL_INVALID
    assert controller.evaluator.evaluated == [JAX_NN]


@pytest.mark.parametrize("category,state", [("COMPILE_FAILED", COMPILE_FAILED), ("NUMERIC_MISMATCH", CORRECTNESS_FAILED)])
def test_failures_have_one_terminal_attempt_and_feed_the_second_context(tmp_path, category, state):
    controller, client = build(tmp_path, [call(), call(LOGSUMEXP)], limit=2, reports=[category, None])
    result = controller.optimize()
    assert [entry["state"] for entry in controller.history] == [ACCEPTED, state, VALID_NO_CHANGE]
    second = json.loads(client.messages.calls[1]["messages"][0]["content"])
    assert second["history"][-1]["state"] == state
    assert result["candidate_attempts"] == 2 and result["best"]["baseline_fallback"]


@pytest.mark.parametrize("kind,state", [("schema", PROPOSAL_INVALID), ("policy", POLICY_REJECTED),
    ("duplicate", DUPLICATE), ("parent", PROPOSAL_INVALID)])
def test_rejected_candidates_count_without_evaluation(tmp_path, kind, state):
    controller, _ = build(tmp_path, limit=1)
    controller.start()
    payload = {"code": JAX_NN} if kind == "schema" else proposal(
        "import os\ndef softmax(x):\n    return x\n" if kind == "policy" else
        controller.initial_code if kind == "duplicate" else JAX_NN, **({"parent_id": 99} if kind == "parent" else {}))
    result = controller.submit_proposal(payload)
    assert result["state"] == state and not controller.evaluator.evaluated
    assert controller.result["candidate_attempts"] == 1
    with pytest.raises(RunStopped):
        controller.submit_proposal(proposal(LOGSUMEXP))


def test_schema_invalid_provider_submission_counts_and_feeds_next_request(tmp_path):
    controller, client = build(tmp_path, [[tool_use("bad", "submit_proposal", {"code": JAX_NN})], call()], limit=2)
    result = controller.optimize()
    assert [entry["state"] for entry in controller.history] == [ACCEPTED, PROPOSAL_INVALID, VALID_NO_CHANGE]
    assert result["candidate_attempts"] == 2
    assert json.loads(client.messages.calls[1]["messages"][0]["content"])["history"][-1]["state"] == PROPOSAL_INVALID


@pytest.mark.parametrize("script,status", [([[]], "EMPTY"), ([[tool_use("x", "unknown", {})]], "MALFORMED")])
def test_no_unique_submission_is_a_request_failure_not_a_fabricated_candidate(tmp_path, script, status):
    controller, _ = build(tmp_path, script)
    result = controller.optimize()
    assert result["status"] == status and result["candidate_attempts"] == 0
    assert result["best"]["baseline_fallback"] and len(controller.requests) == 1


def test_second_candidate_measures_current_best_and_preserves_original_baseline(tmp_path):
    original = SOFTMAX.reference_path.read_bytes()
    controller, client = build(tmp_path, [call(), call(LOGSUMEXP, parent_id=1)], limit=2, ratios=[.5, .5, 1.])
    result = controller.optimize()
    assert controller.evaluator.measured[-1][:2] == (JAX_NN, LOGSUMEXP)
    second = json.loads(client.messages.calls[1]["messages"][0]["content"])
    assert second["current_source"] == JAX_NN and second["current_parent_id"] == 1
    assert second["baseline"]["source_sha256"] != second["incumbent"]["source_sha256"]
    assert result["best"]["attempt_id"] == 1 and SOFTMAX.reference_path.read_bytes() == original


@pytest.mark.parametrize("state", [TIMEOUT, RESOURCE_LIMIT, INTERRUPTED])
def test_worker_failure_stops_with_verified_fallback_and_one_result(tmp_path, state):
    controller, client = build(tmp_path, [call(), call(LOGSUMEXP)], reports=[EvaluationError(state, "fixture failure")])
    result = controller.optimize()
    assert result["status"] == state and result["candidate_attempts"] == 1
    assert result["best"]["baseline_fallback"] and len(client.messages.calls) == 1


def test_auxiliary_baseline_measurement_failure_is_recorded_and_not_lost(tmp_path):
    controller, _ = build(tmp_path, [call(), call(LOGSUMEXP)], limit=2,
        ratios=[.5, .5, .5, .5, EvaluationError(TIMEOUT, "baseline comparison timeout")])
    result = controller.optimize()
    assert result["status"] == TIMEOUT and result["best"]["attempt_id"] == 1
    assert controller.history[-1]["state"] == TIMEOUT


@pytest.mark.parametrize("failure", [RuntimeError("unexpected evaluator failure"), KeyboardInterrupt()])
def test_evaluator_exception_or_interrupt_commits_a_terminal_result(tmp_path, failure):
    controller, client = build(tmp_path, [call(), call(LOGSUMEXP)], reports=[failure])
    assert controller.optimize()["status"] == INTERRUPTED
    assert len(client.messages.calls) == 1 and controller.history[-1]["state"] == INTERRUPTED


@pytest.mark.parametrize("budget", [Budgets(max_provider_requests=0), Budgets(max_total_tokens=0)])
def test_exhausted_request_or_token_budget_makes_no_model_call(tmp_path, budget):
    controller, client = build(tmp_path, [call()], budgets=budget)
    result = controller.optimize()
    assert result["status"] in ("BUDGET_BLOCKED", "CONTEXT_BLOCKED") and result["best"]["baseline_fallback"]
    assert not client.messages.calls


def test_wall_limit_counts_baseline_and_never_dispatches_after_expiry(tmp_path):
    clock = Clock()
    controller, client = build(tmp_path, [call()], clock=clock, budgets=Budgets(run_wall_s=1))
    reference = controller.evaluator.reference_outputs
    def baseline():
        reference()
        clock.now = 2.
    controller.evaluator.reference_outputs = baseline
    result = controller.optimize()
    assert result["status"] == "TIME_LIMIT" and result["best"]["baseline_fallback"]
    assert not client.messages.calls


def test_late_confirmed_observations_do_not_promote(tmp_path):
    clock = Clock()
    controller, _ = build(tmp_path, [call()], limit=1, ratios=[.5, .5], clock=clock, budgets=Budgets(run_wall_s=1))
    evaluate = controller.evaluator.evaluate
    def late(*args):
        clock.now = 2.
        return evaluate(*args)
    controller.evaluator.evaluate = late
    result = controller.optimize()
    assert result["status"] == TIMEOUT and result["best"]["baseline_fallback"]


def test_request_count_cap_stops_before_a_second_dispatch(tmp_path):
    controller, client = build(tmp_path, [call(), call(LOGSUMEXP)], budgets=Budgets(max_provider_requests=1))
    result = controller.optimize()
    assert result["status"] == "BUDGET_BLOCKED" and len(client.messages.calls) == 1
    assert result["candidate_attempts"] == 1


def test_explicit_money_cap_blocks_even_an_offline_accounting_fixture(tmp_path):
    controller, client = build(tmp_path, [call()], budgets=Budgets(money_cap=.000001, currency="USD"))
    rates = ModelPrices("anthropic", ModelSettings().model_id, "USD", 1., 5., "2026-01-01",
                        "https://example.invalid/synthetic-controller-prices")
    controller.model_settings = ModelSettings(prices=rates)
    result = controller.optimize()
    assert result["status"] == "CONTEXT_BLOCKED" and not client.messages.calls
    assert result["best"]["baseline_fallback"]


def test_unavailable_isolation_stops_before_provider_dispatch(tmp_path):
    controller, client = build(tmp_path, [call()])
    controller.evaluator.runner = runner.TrustedSubprocessRunner()
    assert controller.optimize()["status"] == INTERRUPTED
    assert not client.messages.calls


def test_promotion_claim_without_complete_confirmation_is_rejected(tmp_path):
    controller, _ = build(tmp_path, [call()], limit=1)
    controller.evaluator.evaluate = lambda source, incumbent: Verdict(ACCEPTED, "invented win",
        Evaluation(storage.sha256(source.encode()), {}), True)
    assert controller.optimize()["best"]["baseline_fallback"]
    assert controller.history[-1]["state"] == INTERRUPTED


def test_commit_failure_never_promotes_or_dispatches_again(tmp_path, monkeypatch):
    controller, client = build(tmp_path, [call()], limit=1, ratios=[.5, .5])
    controller.start()
    monkeypatch.setattr(controller.store, "commit_attempt", lambda *args, **kwargs: (_ for _ in ()).throw(storage.StorageError("disk full")))
    with pytest.raises(storage.StorageError, match="disk full"):
        controller.optimize()
    assert controller.best_iteration == 0 and len(client.messages.calls) == 1
    with pytest.raises(storage.StorageError, match="cannot resume"):
        controller.optimize()


def test_host_provider_timeout_is_uncertain_and_never_retried(tmp_path):
    client = OfflineClient([])
    release, finished = threading.Event(), threading.Event()
    def slow(**kwargs):
        try:
            release.wait(5)
            return {}
        finally:
            finished.set()
    client.messages.create = slow
    provider = ProposalProvider(client, ModelSettings(request_timeout_s=.02, max_retries=2))
    try:
        with pytest.raises(ProposalFailure, match="deadline"):
            provider.propose({})
        assert len(provider.events) == len(provider.reservations) == 1
        assert provider.events[0]["usage_uncertain"] and provider.stopped
    finally:
        release.set()
        assert finished.wait(2)


def test_durable_reservation_expiring_before_dispatch_is_cancelled_without_usage():
    clock = Clock()
    client = OfflineClient([])
    def reserve(record):
        clock.now = 2.
    provider = ProposalProvider(client, budgets=Budgets(run_wall_s=1), clock=clock, reserve=reserve)
    with pytest.raises(ProposalFailure, match="deadline"):
        provider.propose({})
    assert not client.calls and provider.events[0]["status"] == "NOT_DISPATCHED"
    assert provider.events[0]["dispatched"] is False and provider.accounted_tokens == 0


@pytest.mark.slow
def test_parent_phase_deadline_kills_hung_case_and_case_switching_cannot_refill(tmp_path):
    code = "import pathlib,json,time; pathlib.Path('phase.json').write_text(json.dumps({'schema_version':'worker_phase/v1','stage':'correctness','case':'dev'})); time.sleep(30)"
    limits = {"compile": 2., "correctness": .05, "measure": 1., "profile": 1.}
    status, _, _, elapsed = runner.execute([sys.executable, "-c", code], cwd=tmp_path,
        env=runner.worker_env(tmp_path), timeout_s=5, max_log_bytes=1024, stage_limits=limits, stage_cases=["dev"])
    assert status == runner.TIMEOUT and elapsed < 3
    # Controlled clock proves phase changes accumulate compilation, rather than renewing its allowance.
    monitor = runner._StageClock(tmp_path, {**limits, "compile": 1.}, ["dev"], 0.)
    assert monitor.remaining(.2) > 0
    (tmp_path / "phase.json").write_text(json.dumps({"schema_version": "worker_phase/v1", "stage": "compile", "case": None}))
    assert monitor.remaining(.21) > 0
    assert monitor.remaining(1.1) <= 0


def test_no_legacy_generated_execution_tool_is_available():
    assert not hasattr(Controller, "benchmark_code") and not hasattr(Controller, "_dispatch_tool")


@pytest.mark.parametrize("damage", ["policy", "cases", "threshold", "sources", "blocks", "analysis"])
def test_corrupted_promotion_evidence_cannot_replace_the_incumbent(tmp_path, damage):
    controller, _ = build(tmp_path, [call()], limit=1, ratios=[.5, .5])
    evaluate = controller.evaluator.evaluate
    def corrupt(*args):
        verdict = evaluate(*args)
        evidence = verdict.evaluation
        if damage == "policy":
            evidence.correctness["policy_hash"] = "wrong policy"
        elif damage == "cases":
            evidence.correctness["cases"].pop()
        elif damage == "threshold":
            evidence.decision["delta"] = 0.
        elif damage == "sources":
            evidence.sessions[0]["a_sha256"] = "wrong incumbent"
        elif damage == "blocks":
            evidence.sessions[0]["raw"]["blocks"].pop()
        else:
            evidence.sessions[0]["analysis"]["speedup"]["speedup"] = 1000.
        return verdict
    controller.evaluator.evaluate = corrupt
    assert controller.optimize()["best"]["baseline_fallback"]
    assert controller.history[-1]["state"] == INTERRUPTED


@pytest.mark.parametrize("field,mode", [("compile_timeout_s", "outputs"),
    ("correctness_case_timeout_s", "outputs"), ("measurement_session_timeout_s", "measure")])
def test_zero_worker_stage_limit_refuses_launch(tmp_path, field, mode):
    controller, _ = build(tmp_path, budgets=dataclasses.replace(Budgets(), **{field: 0.}))
    evaluator = controller.evaluator
    with pytest.raises(EvaluationError) as caught:
        evaluator._run("blocked", mode, {"candidate": JAX_NN}, trusted=False, cases=evaluator.cases, timeout_s=10)
    assert caught.value.state == TIMEOUT and evaluator._requests == 0
    assert not controller.output_dir.exists()


def test_actual_worker_launch_is_clipped_to_remaining_wall_and_late_results_are_discarded(tmp_path):
    clock = Clock()
    controller, _ = build(tmp_path, clock=clock, budgets=Budgets(run_wall_s=2.))
    controller.start()
    evaluator = controller.evaluator
    evaluator.fingerprint = None
    observed = []
    def run(work, request, *, trusted, timeout_s):
        observed.append((request, timeout_s))
        clock.now = 3.
        return runner.RunOutcome(runner.OK, {"status": "ok", "device": {}}, "", "", 3.)
    evaluator.runner.run = run
    with pytest.raises(EvaluationError) as caught:
        evaluator._run("late", "outputs", {"candidate": JAX_NN}, trusted=False, cases=evaluator.cases, timeout_s=100)
    assert caught.value.state == TIMEOUT and observed[0][1] == 2.
    assert observed[0][0]["stage_limits"]["correctness"] == 10.
    assert observed[0][0]["stage_cases"] == evaluator.case_names


def test_artifact_and_source_caps_are_checked_before_worker_side_effects(tmp_path, monkeypatch):
    controller, _ = build(tmp_path)
    controller.start()
    evaluator = controller.evaluator
    ran = []
    evaluator.runner.run = lambda *args, **kwargs: ran.append(True)
    with pytest.raises(EvaluationError) as caught:
        evaluator._run("source", "outputs", {"candidate": "x" * (65 << 10)},
                       trusted=False, cases=evaluator.cases, timeout_s=10)
    assert caught.value.state == RESOURCE_LIMIT and not ran
    monkeypatch.setattr(controller.store, "check_capacity", lambda: (_ for _ in ()).throw(storage.StorageError("artifact cap")))
    with pytest.raises(storage.StorageError, match="artifact cap"):
        evaluator._run("artifact", "outputs", {"candidate": JAX_NN}, trusted=False, cases=evaluator.cases, timeout_s=10)
    assert not ran and evaluator._requests == 0


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["compile", "measure"])
def test_parent_compile_and_measurement_deadlines_are_enforced(tmp_path, stage):
    code = ("import pathlib,json,time; pathlib.Path('phase.json').write_text(json.dumps("
            + repr({"schema_version": "worker_phase/v1", "stage": stage, "case": None}) + ")); time.sleep(30)")
    limits = {"compile": 2., "correctness": 1., "measure": 2., "profile": 1.}
    limits[stage] = .05
    status, _, _, elapsed = runner.execute([sys.executable, "-c", code], cwd=tmp_path,
        env=runner.worker_env(tmp_path), timeout_s=5, max_log_bytes=0, stage_limits=limits)
    assert status == runner.TIMEOUT and elapsed < 3


def test_expired_deadline_is_rechecked_after_docker_preflight_without_starting_a_container(tmp_path, monkeypatch):
    docker = runner.DockerRunner(selected_backend="cpu", image="synthetic")
    monkeypatch.setattr(docker, "preflight", lambda: None)
    monkeypatch.setattr(runner, "execute", lambda *args, **kwargs: pytest.fail("unexpected process launch"))
    outcome = docker.run(tmp_path, {"host_deadline_monotonic": 0.}, trusted=False, timeout_s=10.)
    assert outcome.status == runner.TIMEOUT


def test_docker_cleanup_runs_when_launch_or_inspection_raises(tmp_path, monkeypatch):
    docker = runner.DockerRunner(selected_backend="cpu", image="synthetic")
    monkeypatch.setattr(docker, "preflight", lambda: None)
    monkeypatch.setattr(runner, "execute", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    commands = []
    def command(*args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(runner, "_docker", command)
    with pytest.raises(KeyboardInterrupt):
        docker.run(tmp_path, {}, trusted=False, timeout_s=10.)
    assert commands[0][:2] == ("rm", "--force") and commands[1][0] == "ps"


@pytest.mark.parametrize("expression", ["jnp.full_like(x, jnp.nan)", "jnp.reshape(x, (3,))"])
def test_native_invalid_baseline_aborts_before_any_model_request(tmp_path, expression):
    import agent
    source = tmp_path / "reference.py"
    source.write_text("import jax.numpy as jnp\nTEST_ARGS=(jnp.ones(4, dtype=jnp.float32),)\n"
                      f"def kernel(x):\n    return {expression}\n")
    before = source.read_bytes()
    client = FakeAnthropic([call()])
    controller = agent.build_controller(client, source_path=source, function_name="kernel",
        output_dir=tmp_path / "run", selected_backend="cpu", runner=runner.TrustedSubprocessRunner())
    with pytest.raises(BaselineFailed):
        controller.optimize()
    saved = json.loads((controller.output_dir / "run_result.json").read_bytes())
    assert saved["status"] == "BASELINE_FAILED" and saved["best"] is None
    assert not (controller.output_dir / "best.py").exists() and not client.messages.calls
    assert source.read_bytes() == before and saved["baseline_failure"]
