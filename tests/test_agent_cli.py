"""agent.py CLI: --backend, --config and run records (AIC-002).

Every test runs agent.py in a subprocess (selecting a backend is process-wide), with the API key removed
and cwd = tmp_path. Where a run must reach the LLM step, a driver script swaps in the fake client.
"""
import hashlib
import json
import subprocess
import sys

import pytest

LEGACY = ("--iterations", "--model", "--source", "--function", "--output-dir", "--plot", "--require-gpu")
FAKE_KEY = "sk-fake-MUST-NOT-LEAK-0123456789"

DRIVER = """
import json, os, sys
sys.path.insert(0, {repo!r}); sys.path.insert(0, {tests!r})
os.environ["ANTHROPIC_API_KEY"] = {key!r}
import anthropic
from fakes import FakeAnthropic
calls = []

class Recording(FakeAnthropic):
    def __init__(self, **kwargs):
        super().__init__([[]])               # the loop ends after one empty turn
        create = self.messages.create
        def wrapped(**kw):
            calls.append(sorted(p.name for p in __import__("pathlib").Path({out!r}).iterdir()))
            return create(**kw)
        self.messages.create = wrapped

anthropic.Anthropic = Recording
import agent
agent.main({argv!r})
print(json.dumps({{"files_at_first_llm_call": calls[0] if calls else None, "n_calls": len(calls)}}))
"""


def _agent(repo_root, env, cwd, *args):
    return subprocess.run([sys.executable, str(repo_root / "agent.py"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=180)


def _drive(repo_root, env, tmp_path, *extra):
    out = tmp_path / "run"
    argv = ["--backend", "cpu", "--source", str(repo_root / "workloads/example.py"),
            "--function", "bad_func", "--output-dir", str(out), "--runner", "subprocess", *extra]
    code = DRIVER.format(repo=str(repo_root), tests=str(repo_root / "tests"), key=FAKE_KEY,
                         out=str(out), argv=argv)
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    return out, json.loads(proc.stdout.strip().splitlines()[-1]), proc


def test_help_unchanged(repo_root, clean_env, tmp_path):
    proc = _agent(repo_root, clean_env, tmp_path, "--help")
    assert proc.returncode == 0, proc.stderr
    for option in (*LEGACY, "--backend", "--config", "--runner", "--calibration"):
        assert option in proc.stdout
    assert not any(tmp_path.iterdir()), "--help must not write anything"


def test_backend_unavailable_exits_2_without_traceback(repo_root, clean_env, tmp_path):
    proc = _agent(repo_root, clean_env, tmp_path, "--backend", "tpu")
    if proc.returncode != 2 and "ANTHROPIC_API_KEY" in proc.stderr:
        pytest.skip("a TPU is present (selection succeeded and the run stopped at the key check)")
    assert proc.returncode == 2
    assert "unavailable" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_missing_key_writes_nothing(repo_root, clean_env, tmp_path):
    proc = _agent(repo_root, clean_env, tmp_path, "--backend", "cpu")
    assert proc.returncode != 0
    assert "ANTHROPIC_API_KEY" in proc.stderr
    assert not (tmp_path / "runs").exists()


def test_manifest_written_before_llm(repo_root, clean_env, tmp_path):
    out, result, proc = _drive(repo_root, clean_env, tmp_path)
    assert result["n_calls"] == 1
    assert {"run_config.json", "manifest.json"} <= set(result["files_at_first_llm_call"])
    run_config_text = (out / "run_config.json").read_text()
    manifest_text = (out / "manifest.json").read_text()
    run_config = json.loads(run_config_text)
    manifest = json.loads(manifest_text)
    canonical = json.dumps(run_config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert manifest["run_config_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert manifest["schema_version"] == "manifest/v1"
    # --source workload, trusted subprocess runner: recorded honestly as not scientific, with every reason.
    assert manifest["scientific"] is False
    assert manifest["non_scientific_reasons"] == ["workload 'source:example' is not a scientific workload",
                                                  "runner trusted-subprocess/v1 is not isolated"]
    assert run_config["correctness_policy_hash"] == manifest["workload"]["correctness_policy_hash"]
    assert len(run_config["correctness_policy_hash"]) == 64
    assert manifest["backend_requested"] == "cpu"
    assert manifest["environment"]["devices"][0]["platform"] == "cpu"
    assert manifest["environment"]["repo"]["uv_lock_sha256"]
    for path in (p for p in out.rglob("*") if p.is_file()):
        assert FAKE_KEY.encode() not in path.read_bytes(), path.name
    assert FAKE_KEY not in proc.stdout + proc.stderr


def test_iterations_maps_to_max_candidates(repo_root, clean_env, tmp_path):
    out, _, _ = _drive(repo_root, clean_env, tmp_path, "--iterations", "2")
    assert json.loads((out / "run_config.json").read_text())["budgets"]["max_candidates"] == 2


def test_config_file_is_used(repo_root, clean_env, tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"budgets": {"max_candidates": 4}, "context_condition": "C0"}))
    out, _, _ = _drive(repo_root, clean_env, tmp_path, "--config", str(config))
    run_config = json.loads((out / "run_config.json").read_text())
    assert run_config["budgets"]["max_candidates"] == 4
    assert run_config["context_condition"] == "C0"


def test_zero_candidates_makes_no_provider_request(repo_root, clean_env, tmp_path):
    _, result, _ = _drive(repo_root, clean_env, tmp_path, "--iterations", "0")
    assert result["n_calls"] == 0


def test_bad_config_exits_2(repo_root, clean_env, tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"budgets": {"max_candidatez": 1}}))
    proc = _agent(repo_root, clean_env, tmp_path, "--config", str(config))
    assert proc.returncode == 2
    assert "budgets.max_candidatez" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_require_gpu_conflicts_with_cpu_backend(repo_root, clean_env, tmp_path):
    proc = _agent(repo_root, clean_env, tmp_path, "--backend", "cpu", "--require-gpu")
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr


def test_unavailable_isolation_stops_before_any_llm_call(repo_root, clean_env, tmp_path):
    """With --runner docker and no usable Docker (here: an image that cannot exist), nothing is sent."""
    code = DRIVER.format(repo=str(repo_root), tests=str(repo_root / "tests"), key=FAKE_KEY,
                         out=str(tmp_path / "run"),
                         argv=["--backend", "cpu", "--runner", "docker", "--output-dir", str(tmp_path / "run")])
    code = code.replace("import agent\n", "import agent, runner\nrunner.image_reference = lambda v: 'aic-worker:none'\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=clean_env, capture_output=True,
                          text=True, timeout=180)
    assert proc.returncode == 2
    assert "ISOLATION_UNAVAILABLE" in proc.stderr
    assert not (tmp_path / "run").exists()


def test_scientific_needs_backend_scientific_workload_and_isolation():
    from types import SimpleNamespace

    import agent
    from workloads.registry import EXAMPLE_ADD, SOFTMAX
    isolated = SimpleNamespace(isolated=True, name="docker/v1")
    assert agent.scientific_status(SimpleNamespace(backend="cpu"), SOFTMAX, isolated) == (True, [])
    ok, reasons = agent.scientific_status(SimpleNamespace(backend=None), EXAMPLE_ADD,
                                          SimpleNamespace(isolated=False, name="trusted-subprocess/v1"))
    assert not ok and len(reasons) == 3
