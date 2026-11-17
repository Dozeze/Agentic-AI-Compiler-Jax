"""AIC-018 finalization: native correctness, synthetic timing decisions and artifact-only replay.

Synthetic parent selections and timing fixtures are explicitly non-scientific. They never prove a
selected-target speedup. Native final correctness uses reviewed reference/shape-overfit sources.
"""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import benchmark
import heldout
import finalization
from context_builder import ContextError, check_search_access, evaluation_commitment, workload_contract
from contracts import (ACCEPTED, Attempt, Budgets, Evaluation, Proposal, RunConfig, config_hash,
                       protocol_hash, to_dict)
from runner import TrustedSubprocessRunner
from storage import RunStore, StorageError, atomic_write_json, sha256, verified_best, verified_final_result
from workloads.registry import SOFTMAX, LAYER_NORM, MATMUL_GELU


@pytest.fixture(autouse=True)
def private_evaluation_state(monkeypatch, tmp_path):
    monkeypatch.setattr(heldout, "PUBLICATIONS", tmp_path / "publications")
    monkeypatch.setattr(heldout, "LEDGER", tmp_path / "legacy.jsonl")


def parent_run(path, spec=SOFTMAX, candidate=None, *, freeze=True, budgets=None, synthetic_environment=False):
    """Synthetic selection metadata; subsequent native workers, where requested, are real."""
    runner = TrustedSubprocessRunner()
    cfg = RunConfig(backend="cpu", runner="subprocess", workload_id=spec.workload_id,
                    function_name=spec.function_name, output_dir=str(path), budgets=budgets or Budgets())
    if synthetic_environment:
        environment = {"devices": [], "unavailable": {"capture": "synthetic timing fixture"}}
    else:
        from environment import collect_environment
        environment = collect_environment(Path(__file__).resolve().parents[1])
    store = RunStore(path, budgets=cfg.budgets)
    store.initialize(spec.reference_source, config=to_dict(cfg), manifest={
        "evaluation_split": heldout.commitment(spec) if freeze else None,
        "scientific": False, "non_scientific_reasons": ["synthetic parent selection fixture"],
        "runner": runner.identity(), "environment": environment, "run_config_sha256": config_hash(cfg),
        "workload": {"id": spec.workload_id, "reference_sha256": spec.reference_sha256, "split": "development"}})
    for number, source in enumerate([spec.reference_source, *([candidate] if candidate is not None else [])]):
        evaluation = Evaluation(candidate_sha256=sha256(source.encode()),
            stages={"reference": "ok"} if number == 0 else {"correctness": "ok", "measurement": "ok"},
            correctness={"passed": True} if number else None,
            sessions=({"fixture": "synthetic selection observation"},) if number else ())
        attempt = Attempt(number, ACCEPTED, "synthetic selection fixture", evaluation=evaluation,
            parent_id=0 if number else None,
            proposal=Proposal("fixture", "fixture", "fixture", "unclaimed", source) if number else None)
        store.commit_attempt({"iteration": number, "state": ACCEPTED, "reason": attempt.reason,
            "code": source, "output_correct": True, "accepted": True, "function_name": spec.function_name,
            "attempt": attempt.to_json()})
    store.finish_run("CANDIDATE_LIMIT", "synthetic parent fixture complete", {})
    return path


@pytest.fixture(scope="module")
def native_releases(tmp_path_factory):
    root = (Path(os.environ["AIC_FINAL_EVIDENCE_DIR"]).resolve()
            if os.environ.get("AIC_FINAL_EVIDENCE_DIR") else tmp_path_factory.mktemp("final-native"))
    if os.environ.get("AIC_FINAL_EVIDENCE_DIR"):
        root.mkdir(parents=True, exist_ok=False)
    outputs = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(heldout, "LEDGER", root / "legacy.jsonl")
        for spec in (SOFTMAX, LAYER_NORM, MATMUL_GELU):
            patch.setattr(heldout, "PUBLICATIONS", root / spec.workload_id / "private-state")
            parent = parent_run(root / spec.workload_id / "search", spec)
            before = (parent / "best.py").read_bytes()
            out, report = finalization.finalize_run(parent, output_dir=root / spec.workload_id / "final",
                                                    correctness_only=True)
            assert (parent / "best.py").read_bytes() == before
            outputs[spec.workload_id] = out, report
        patch.setattr(heldout, "PUBLICATIONS", root / "overfit-state")
        source = SOFTMAX.reference_source.replace("    shifted =", "    if x.shape not in ((32, 128), (128, 1024)):\n        return jnp.zeros_like(x)\n    shifted =")
        assert benchmark.check_correctness(benchmark.load_function(source, "softmax"), SOFTMAX,
                                            benchmark.build_cases(SOFTMAX)).passed
        parent = parent_run(root / "overfit-search", candidate=source)
        out, report = finalization.finalize_run(parent, output_dir=root / "overfit-final", correctness_only=True,
                                                runner=TrustedSubprocessRunner(reviewed_fixtures=True))
        outputs["overfit"] = out, report
    return outputs


@pytest.mark.parametrize("workload", ["softmax", "layer_norm", "matmul_gelu"])
def test_native_final_correctness_retains_every_case_and_leaves_search_best_unchanged(native_releases, workload):
    out, report = native_releases[workload]
    assert report["correctness_passed"] and report["status"] == "COMPLETE"
    assert not report["performance_verified"] and report["provider_requests"] == 0
    assert report["correctness_only"] and report["selection"]["baseline_fallback"]
    assert len(report["shapes"]) == 2
    for shape in report["shapes"]:
        assert shape["correctness"]["n_cases"] == 20 and shape["correctness"]["passed"]
        assert shape["timing"]["status"] == "NOT_REQUESTED"
    assert verified_final_result(out) == report
    parent_best = verified_best(report["selection"]["parent_run_path"])
    assert "not performed" in parent_best["scope"]["heldout_verification"]


def test_reviewed_shape_overfit_fails_on_both_unseen_shapes_and_blocks_timing(native_releases):
    out, report = native_releases["overfit"]
    assert report["status"] == "CORRECTNESS_FAILED" and not report["correctness_passed"]
    assert not report["performance_verified"]
    assert all(s["correctness"]["n_failed"] == 20 for s in report["shapes"])
    assert all(s["timing"]["status"] == "CORRECTNESS_BLOCKED" for s in report["shapes"])
    assert verified_final_result(out) == report


@pytest.mark.parametrize("artifact", ["winner.py", "split.json", "selection.json", "final_report.json",
                                     "release.json", "parent_best.json", "outputs"])
def test_replay_refuses_tampered_final_evidence(native_releases, artifact):
    out, _ = native_releases["softmax"]
    path = next(out.glob("work/correctness/*-candidate/outputs.npz")) if artifact == "outputs" else out / artifact
    original = path.read_bytes()
    try:
        if artifact == "release.json":
            data = json.loads(original)
            data["correctness_passed"] = False
            path.write_text(json.dumps(data))
        else:
            path.write_bytes(original + b"\n")
        with pytest.raises(StorageError):
            verified_final_result(out)
    finally:
        path.write_bytes(original)


def test_replay_does_not_load_candidate_code(native_releases, monkeypatch):
    monkeypatch.setattr(benchmark, "load_function", lambda *a, **k: pytest.fail("replay imported candidate source"))
    assert verified_final_result(native_releases["softmax"][0])["correctness_passed"]


def test_commitment_changes_with_split_tolerance_and_reference_but_exposes_no_split_values():
    public = evaluation_commitment(SOFTMAX)
    assert public["n_cases"] == 40 and len(public["sha256"]) == 64
    assert not set(public) & {"seeds", "shapes", "heldout_seeds", "heldout_shapes"}
    for change in ({"heldout_seeds": tuple(range(2000, 2020))}, {"heldout_version": 2},
                   {"tolerance": dataclasses.replace(SOFTMAX.tolerance, atol=1e-4)}):
        assert evaluation_commitment(dataclasses.replace(SOFTMAX, **change)) != public
    contract = workload_contract(SOFTMAX)
    assert "seeds" not in json.dumps(contract) and "heldout" not in json.dumps(contract)


@pytest.mark.parametrize("workload", [SOFTMAX, LAYER_NORM, MATMUL_GELU], ids=lambda w: w.workload_id)
def test_reservation_blocks_research_and_relabeling_a_used_split(workload, tmp_path):
    heldout.reserve(workload, run_id="fixture", candidate_sha256="fixture", output_dir=tmp_path / "final")
    with pytest.raises(ContextError, match="sealed"):
        check_search_access(workload)
    with pytest.raises(heldout.HeldoutError, match="sealed"):
        heldout.reserve(workload, run_id="second", candidate_sha256="fixture", output_dir=tmp_path / "other")
    with pytest.raises(heldout.HeldoutError, match="fresh seeds and shapes"):
        heldout.check_search_allowed(dataclasses.replace(workload, heldout_version=2))
    fresh = dataclasses.replace(workload, heldout_version=2, heldout_seeds=tuple(range(2000, 2020)),
        heldout_shapes=tuple(tuple(n + 1 for n in shape) for shape in workload.heldout_shapes))
    heldout.check_search_allowed(fresh)


@pytest.mark.parametrize("change", [{"heldout_version": 0}, {"heldout_seeds": tuple(range(19)) + (-1,)},
    {"heldout_shapes": ((64, 257), (64, 257))}, {"heldout_shapes": ((64, 0), (256, 2048))}])
def test_invalid_final_contracts_are_rejected(change):
    with pytest.raises(heldout.HeldoutError):
        heldout.commitment(dataclasses.replace(SOFTMAX, **change))


def test_missing_or_changed_prefreeze_and_tampered_parent_refuse_before_evaluation(tmp_path, monkeypatch):
    parent = parent_run(tmp_path / "search", freeze=False, synthetic_environment=True)
    with pytest.raises(heldout.HeldoutError, match="pre-search"):
        finalization.finalize_run(parent, output_dir=tmp_path / "final")
    assert not (tmp_path / "final").exists()
    parent = parent_run(tmp_path / "search2", synthetic_environment=True)
    changed = dataclasses.replace(SOFTMAX, heldout_seeds=tuple(range(2000, 2020)))
    monkeypatch.setitem(finalization.registry.REGISTRY, "softmax", changed)
    with pytest.raises(heldout.HeldoutError, match="pre-search"):
        finalization.finalize_run(parent, output_dir=tmp_path / "final")
    monkeypatch.setitem(finalization.registry.REGISTRY, "softmax", SOFTMAX)
    (parent / "best.py").write_text("raise RuntimeError('must never execute')\n")
    with pytest.raises(StorageError):
        finalization.finalize_run(parent, output_dir=tmp_path / "final")
    assert not (tmp_path / "final").exists()


def test_zero_compile_budget_and_failed_publication_leave_split_sealed(tmp_path):
    parent = parent_run(tmp_path / "search", budgets=dataclasses.replace(Budgets(), compile_timeout_s=0),
                        synthetic_environment=True)
    out, report = finalization.finalize_run(parent, output_dir=tmp_path / "final", correctness_only=True)
    assert report["status"] == "INTERRUPTED" and not report["correctness_passed"]
    assert not list((out / "work").rglob("result.json"))
    with pytest.raises(heldout.HeldoutError, match="sealed"):
        finalization.finalize_run(parent, output_dir=tmp_path / "retry", correctness_only=True)


class SyntheticFinalEvaluator(finalization.Evaluator):
    """No code execution. Only explicit synthetic observations for report/decision mechanics."""
    noise = False
    calibration_error = False
    invalid_blocks = False

    def reference_outputs(self):
        self.reference = "synthetic fixture"
        self.fingerprint = "synthetic final timing fixture"

    def correctness(self, source, **kwargs):
        cases = tuple(benchmark.CaseResult(name, True, None, (), None) for name in self.case_names)
        return benchmark.CorrectnessReport(True, None, self.policy_hash, cases), {}, None

    def calibrate(self):
        if self.calibration_error:
            raise benchmark.CalibrationError("synthetic noise refusal")
        self.fingerprint = "synthetic final timing fixture"
        self.calibration = benchmark.Calibration(benchmark.CALIBRATION_SCHEMA, "SYNTHETIC_FIXTURE", self.spec.workload_id,
            self.measure_shape, "float32", protocol_hash(self.protocol), self.fingerprint, 1, .03, (), 1e-9, "fixture")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.calibration.save(self.workdir / "calibration.json")
        return self.calibration

    def session(self, a, b, **kwargs):
        number = len(list(self.workdir.glob("session-*")))
        work = self.workdir / f"session-{number}"
        work.mkdir()
        blocks = tuple(benchmark.Block(i, "AB" if i % 2 else "BA", 1, 1_000_000,
            (100_000 if i % 2 else 1_000_000) if self.noise else 500_000, 4*i, 4*i+1)
            for i in range(self.protocol.n_blocks))
        session = benchmark.PairedSession(protocol_hash(self.protocol), 1, blocks)
        if self.invalid_blocks:
            session = dataclasses.replace(session, blocks=(dataclasses.replace(blocks[0], repeats=0), *blocks[1:]))
        atomic_write_json(work / "result.json", {"mode": "measure", "status": "ok",
            "session": session.to_json(), "synthetic_fixture": True})
        return session, None


@pytest.mark.parametrize("mode", ["stable", "noisy", "calibration_failure"])
def test_per_shape_final_sessions_and_no_gain_results_are_retained_as_synthetic_fixtures(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(finalization, "Evaluator", SyntheticFinalEvaluator)
    monkeypatch.setattr(SyntheticFinalEvaluator, "noise", mode == "noisy")
    monkeypatch.setattr(SyntheticFinalEvaluator, "calibration_error", mode == "calibration_failure")
    parent = parent_run(tmp_path / "search", synthetic_environment=True)
    out, report = finalization.finalize_run(parent, output_dir=tmp_path / "final")
    assert report["correctness_passed"] and not report["performance_verified"]
    assert verified_final_result(out) == report
    for shape in report["shapes"]:
        timing = shape["timing"]
        if mode == "calibration_failure":
            assert timing["status"] == "UNAVAILABLE" and report["status"] == "TIMING_INCOMPLETE"
        else:
            assert timing["status"] == "MEASURED" and len(timing["sessions"]) == 3
            assert timing["confirmed_improvement"] == (mode == "stable")


def test_cli_search_refuses_sealed_split_before_client_construction(tmp_path, monkeypatch):
    import agent
    monkeypatch.chdir(tmp_path)
    heldout.reserve(SOFTMAX, run_id="fixture", candidate_sha256="fixture", output_dir=tmp_path / "final")
    monkeypatch.setattr(agent, "client_factory", lambda *a: pytest.fail("constructed a provider after final exposure"))
    with pytest.raises(SystemExit) as error:
        agent.main(["--backend", "cpu", "--workload", "softmax", "--output-dir", str(tmp_path / "search")])
    assert error.value.code == 2 and not (tmp_path / "search").exists()


def test_final_cli_never_constructs_provider_and_rejects_search_overrides(tmp_path, monkeypatch):
    import agent
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent, "client_factory", lambda *a: pytest.fail("final mode constructed a provider"))
    parent = parent_run(tmp_path / "search", synthetic_environment=True)
    monkeypatch.setattr(finalization, "Evaluator", SyntheticFinalEvaluator)
    out, report = agent.main(["--finalize-run", str(parent), "--output-dir", str(tmp_path / "final"),
                              "--final-correctness-only"])
    assert report["provider_requests"] == 0 and verified_final_result(out) == report
    with pytest.raises(SystemExit):
        agent.main(["--finalize-run", str(parent), "--iterations", "2"])


def test_storage_failure_does_not_allow_another_final_attempt(tmp_path, monkeypatch):
    parent = parent_run(tmp_path / "search", synthetic_environment=True)
    monkeypatch.setattr(finalization, "Evaluator", SyntheticFinalEvaluator)
    def fail(*args, **kwargs):
        raise StorageError("injected final publication failure")
    monkeypatch.setattr(RunStore, "commit_final_result", fail)
    with pytest.raises(StorageError, match="injected"):
        finalization.finalize_run(parent, output_dir=tmp_path / "final", correctness_only=True)
    with pytest.raises(StorageError):
        verified_final_result(tmp_path / "final")
    with pytest.raises(ContextError, match="sealed"):
        check_search_access(SOFTMAX)


def test_invalid_raw_final_blocks_cannot_produce_a_performance_claim(tmp_path, monkeypatch):
    parent = parent_run(tmp_path / "search", synthetic_environment=True)
    monkeypatch.setattr(finalization, "Evaluator", SyntheticFinalEvaluator)
    monkeypatch.setattr(SyntheticFinalEvaluator, "invalid_blocks", True)
    out, report = finalization.finalize_run(parent, output_dir=tmp_path / "final")
    assert report["status"] == "INTERRUPTED" and not report["performance_verified"]
    assert "invalid raw" in report["reason"]
    assert verified_final_result(out) == report


def test_unreadable_or_interrupted_state_cannot_be_bypassed_by_a_new_version(tmp_path, monkeypatch):
    import storage
    heldout.LEDGER.write_text("{")
    with pytest.raises(heldout.HeldoutError, match="burn ledger"):
        heldout.check_search_allowed(SOFTMAX)
    heldout.LEDGER.unlink()
    def fail(*a, **kw):
        raise OSError("injected reservation failure")
    monkeypatch.setattr(storage, "atomic_write_json", fail)
    with pytest.raises(heldout.HeldoutError, match="reserve"):
        heldout.reserve(SOFTMAX, run_id="fixture", candidate_sha256="fixture", output_dir=tmp_path / "final")
    fresh = dataclasses.replace(SOFTMAX, heldout_version=2, heldout_seeds=tuple(range(2000, 2020)),
                                heldout_shapes=((65, 258), (257, 2049)))
    with pytest.raises(heldout.HeldoutError, match="reconciliation"):
        heldout.check_search_allowed(fresh)


def test_replay_cli_reads_a_committed_release_without_importing_source(native_releases, monkeypatch, capsys):
    from scripts import heldout as cli
    monkeypatch.setattr(benchmark, "load_function", lambda *a, **kw: pytest.fail("candidate imported in replay CLI"))
    assert cli.main(["replay", "--run", str(native_releases["softmax"][0])]) == 0
    assert json.loads(capsys.readouterr().out)["correctness_passed"]


def test_prefreeze_is_present_before_any_direct_controller_provider_request(tmp_path):
    from test_controller import ScriptedEvaluator
    from controller import Controller
    from fakes import FakeAnthropic, proposal, tool_use
    from fixtures.softmax_candidates import COMPILE_ERROR
    client = FakeAnthropic([[tool_use("one", "submit_proposal", proposal(COMPILE_ERROR))]])
    evaluator = ScriptedEvaluator(tmp_path / "unused", reports=["COMPILE_FAILED"])
    controller = Controller("fixture", "propose", evaluator, client=client, max_iterations=1,
                             output_dir=tmp_path / "search", function_name="softmax")
    create = client.messages.create
    def checked(**kwargs):
        manifest = json.loads((controller.output_dir / "manifest.json").read_bytes())
        assert manifest["evaluation_split"] == heldout.commitment(SOFTMAX)
        assert "heldout" not in kwargs["messages"][0]["content"]
        return create(**kwargs)
    client.messages.create = checked
    controller.optimize()
    assert len(client.messages.calls) == 1


@pytest.mark.target
@pytest.mark.needs_gpu
@pytest.mark.needs_docker
@pytest.mark.skipif(not os.environ.get("AIC_TARGET_CONFIG"), reason="selected-target acceptance is explicitly invoked")
@pytest.mark.parametrize("workload", ["softmax", "layer_norm", "matmul_gelu"])
def test_final_correctness_and_fresh_sessions_on_selected_target(tmp_path, workload):
    from backends import gpu_env
    repo = Path(__file__).resolve().parents[1]
    config = Path(os.environ["AIC_TARGET_CONFIG"]).resolve()
    driver = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import heldout, agent
heldout.PUBLICATIONS = Path(sys.argv[2]) / 'private-fixture-state'
heldout.LEDGER = Path(sys.argv[2]) / 'private-legacy.jsonl'
parent = Path(sys.argv[2]) / 'search'
agent.main(['--config', sys.argv[3], '--workload', sys.argv[4], '--iterations', '0', '--require-gpu', '--output-dir', str(parent)])
path, report = agent.main(['--finalize-run', str(parent), '--require-gpu', '--output-dir', str(Path(sys.argv[2]) / 'final')])
assert report['status'] == 'COMPLETE' and report['correctness_passed']
assert all(s['timing']['status'] == 'MEASURED' and len(s['timing']['sessions']) >= 3 for s in report['shapes'])
print(json.dumps({'final_run': str(path), 'provider_requests': report['provider_requests']}))
"""
    proc = subprocess.run([sys.executable, "-c", driver, str(repo), str(tmp_path), str(config), workload], cwd=tmp_path,
                          env=gpu_env(), capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = verified_final_result(tmp_path / "final")
    assert report["backend"] == "cuda" and report["correctness_passed"] and report["provider_requests"] == 0
