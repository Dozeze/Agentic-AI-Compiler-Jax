"""AIC-016: reviewed proposals through real workers, with no paid provider calls.

The native CPU run proves failure/feedback/storage mechanics only. The selected-target test requires
CUDA and Docker and measures reviewed rewrites without prescribing a performance outcome.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backends import gpu_env
from contracts import resolve_run_config
from fixtures.softmax_candidates import COMPILE_ERROR, JAX_NN, LOGSUMEXP, NAIVE_UNSTABLE
from scripts.verify_mvp import inspect_run, require_target_run
from storage import StorageError

REPO = Path(__file__).resolve().parents[1]


def replay_file(path, sources):
    """The replay contains reviewed source strings, not simulated compiler or timing results."""
    responses = [{"content": [{"type": "tool_use", "id": f"reviewed-{i}", "name": "submit_proposal",
                              "input": {"hypothesis": "reviewed fixture", "transformation": "reviewed fixture",
                                        "rationale": "exercise the real worker", "expected_effect": "unclaimed",
                                        "candidate_source": source}}], "stop_reason": "tool_use",
                  "usage": {"input_tokens": 100, "output_tokens": 100}} for i, source in enumerate(sources)]
    path.write_text(json.dumps(responses))
    return path


@pytest.fixture(scope="module")
def native_run(tmp_path_factory):
    root = (Path(os.environ["AIC_MVP_EVIDENCE_DIR"]).resolve() / "native"
            if os.environ.get("AIC_MVP_EVIDENCE_DIR") else tmp_path_factory.mktemp("native-mvp"))
    if os.environ.get("AIC_MVP_EVIDENCE_DIR"):
        root.mkdir(parents=True, exist_ok=False)
    replay = replay_file(root / "replay.json", [COMPILE_ERROR, NAIVE_UNSTABLE,
                                              "import os\ndef softmax(x):\n    return x\n"])
    config = root / "config.json"
    config.write_text(json.dumps({"context_condition": "C4"}))
    # Only these fixed, reviewed fixtures may use the trusted subprocess escape hatch. No CLI flag
    # exposes it to a model. Running the real CLI preserves the resolved configuration and manifest.
    driver = """
import sys
sys.path.insert(0, sys.argv[1])
import runner
reviewed = runner.TrustedSubprocessRunner(reviewed_fixtures=True)
runner.TrustedSubprocessRunner = lambda: reviewed
import agent
agent.main(sys.argv[2:])
"""
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CLAUDE_MODEL")}
    proc = subprocess.run([sys.executable, "-c", driver, str(REPO), "--backend", "cpu", "--runner", "subprocess",
                           "--workload", "softmax", "--config", str(config), "--iterations", "3",
                           "--offline-responses", str(replay), "--output-dir", str(root / "run")],
                          cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (root / "cli.stdout.log").write_text(proc.stdout)
    (root / "cli.stderr.log").write_text(proc.stderr)
    return root / "run"


def test_native_compile_correctness_and_policy_paths_feed_actual_next_request(native_run):
    report = inspect_run(native_run)
    assert report["provider_modes"] == ["offline"]
    states = [json.loads((native_run / f"attempts/{i:04d}/record.json").read_bytes())["entry"]["state"]
              for i in range(4)]
    assert states == ["ACCEPTED", "COMPILE_FAILED", "CORRECTNESS_FAILED", "POLICY_REJECTED"]
    assert report["evaluated_attempt_ids"] == [1, 2]
    assert report["measured_attempt_ids"] == []
    assert report["best"]["baseline_fallback"]
    assert (native_run / "best.py").read_bytes() == (REPO / "workloads/softmax.py").read_bytes()
    context = json.loads((native_run / "contexts/0003.json").read_bytes())
    failure = next(item for item in context["history"] if item["iteration"] == 2)
    assert failure["correctness"]["category"] == "NONFINITE"
    assert failure["correctness"]["n_cases"] == 16
    assert report["profiles"] and report["workers"]
    assert all(w["device"]["platform"] == "cpu" for w in report["workers"] if w["device"])
    manifest = json.loads((native_run / "manifest.json").read_bytes())
    assert manifest["scientific"] is False and report["runner"]["isolated"] is False
    with pytest.raises(StorageError, match="CUDA"):
        require_target_run(report, mode="offline")
    with pytest.raises(StorageError, match="live provider"):
        require_target_run(report, mode="live")


@pytest.mark.parametrize("artifact", ["best.py", "contexts/0002.json", "work/0001-reference/result.json"])
def test_evidence_reader_refuses_tampering_without_loading_source(native_run, artifact):
    path = native_run / artifact
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\n")
        with pytest.raises(StorageError):
            inspect_run(native_run)
    finally:
        path.write_bytes(before)


def test_target_contract_selects_softmax_cuda_containers_without_live_funding():
    cfg = resolve_run_config({}, REPO / "configs/target.json")
    assert (cfg.backend, cfg.runner, cfg.workload_id, cfg.context_condition) == ("cuda", "docker", "softmax", "C4")
    assert cfg.budgets.max_candidates == cfg.budgets.max_provider_requests == 3
    assert cfg.budgets.money_cap is None and cfg.model.prices is None


@pytest.mark.parametrize("missing_proof", ["native_execution", "actual_cuda", "live_funding"])
def test_target_review_rejects_declared_capabilities_without_matching_proof(missing_proof):
    # Synthetic documents test the review gate only; they are never written as target evidence.
    report = {"provider_modes": ["offline"], "backend": "cuda", "workload": "softmax", "scientific": True,
              "runner": {"isolated": True, "gpus": True, "image_id": "synthetic-contract-fixture"},
              "budgets": {"max_candidates": 3, "money_cap": None}, "model": {"prices": None},
              "evaluated_attempt_ids": [1, 2], "candidate_attempts": 2,
              "evaluated_sources": {"1": "fixture-a", "2": "fixture-b"},
              "workers": [{"status": "OK", "device": {"platform": "gpu"}, "backend": "cuda",
                           "mode": "outputs", "candidate_sha256": key} for key in ("fixture-a", "fixture-b")]}
    mode, message = "offline", "matching native worker"
    if missing_proof == "native_execution":
        report["workers"].pop()
    elif missing_proof == "actual_cuda":
        report["workers"][1]["device"]["platform"] = "cpu"
        message = "CUDA execution"
    else:
        report["provider_modes"] = ["live"]
        mode, message = "live", "money cap"
    with pytest.raises(StorageError, match=message):
        require_target_run(report, mode=mode)


@pytest.mark.target
@pytest.mark.needs_gpu
@pytest.mark.needs_docker
@pytest.mark.skipif(not os.environ.get("AIC_TARGET_CONFIG"), reason="selected-target acceptance is explicitly invoked")
def test_selected_target_replay_compiles_checks_and_measures_real_proposals(tmp_path):
    target = Path(os.environ["AIC_TARGET_CONFIG"]).resolve()
    cfg = resolve_run_config({}, target)
    assert (cfg.backend, cfg.runner, cfg.workload_id, cfg.context_condition) == ("cuda", "docker", "softmax", "C4")
    assert cfg.source_path is None and cfg.proposal_only is False
    assert 2 <= cfg.budgets.max_candidates <= 3
    replay = replay_file(tmp_path / "replay.json", [COMPILE_ERROR, JAX_NN, LOGSUMEXP])
    source_before = (REPO / "workloads/softmax.py").read_bytes()
    if os.environ.get("AIC_MVP_EVIDENCE_DIR"):
        tmp_path = Path(os.environ["AIC_MVP_EVIDENCE_DIR"]).resolve() / "selected-target"
        tmp_path.mkdir(parents=True, exist_ok=False)
        replay = replay_file(tmp_path / "replay.json", [COMPILE_ERROR, JAX_NN, LOGSUMEXP])
    out = tmp_path / "run"
    env = gpu_env()
    env.pop("CLAUDE_MODEL", None)
    proc = subprocess.run([sys.executable, str(REPO / "agent.py"), "--config", str(target), "--require-gpu",
                           "--iterations", "3", "--offline-responses", str(replay), "--output-dir", str(out)],
                          cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=cfg.budgets.run_wall_s + 180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (tmp_path / "cli.stdout.log").write_text(proc.stdout)
    (tmp_path / "cli.stderr.log").write_text(proc.stderr)
    report = inspect_run(out)
    require_target_run(report, mode="offline")
    assert report["evaluated_attempt_ids"] == [1, 2, 3]
    assert report["measured_attempt_ids"] == [2, 3], report
    assert report["profiles"], "capture coverage or an explicit unavailable reason must be recorded"
    assert (REPO / "workloads/softmax.py").read_bytes() == source_before
    # No speedup or acceptance is prescribed. Correct/no-change/slower/unstable are all real outcomes.
    for i in (2, 3):
        evaluation = json.loads((out / f"attempts/{i:04d}/evaluation.json").read_bytes())
        assert evaluation["correctness"]["passed"] and evaluation["correctness"]["n_cases"] == 16
        assert evaluation["stages"]["measurement"] == "ok" and evaluation["sessions"]
    print(json.dumps(report, indent=2))
