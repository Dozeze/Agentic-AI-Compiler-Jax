"""Execution boundary (AIC-008): fresh processes, deadlines, cleanup, bounded logs, trust levels."""
import dataclasses
import json
import os
import sys
import textwrap
import time

import numpy as np
import pytest

import array_io
import runner
import worker
from contracts import MeasurementProtocol
from fixtures.softmax_candidates import REFERENCE

INFINITE = "def softmax(x):\n    while True:\n        pass\n"


def _prepare(work, source):
    work.mkdir(parents=True, exist_ok=True)
    (work / "cand.py").write_text(source)
    array_io.save(work / "inputs.npz", {"a": array_io.flatten((np.ones((4, 8), np.float32),))})
    return {"schema_version": worker.REQUEST_SCHEMA, "mode": "outputs", "backend": "cpu",
            "function_name": "softmax", "sources": {"candidate": "cand.py"}, "inputs": "inputs.npz"}


def test_trusted_code_runs_in_a_fresh_process(tmp_path):
    request = _prepare(tmp_path / "w", REFERENCE)
    outcome = runner.TrustedSubprocessRunner().run(tmp_path / "w", request, trusted=True, timeout_s=120)
    assert outcome.status == runner.OK, outcome.detail
    assert outcome.result["cases"][0]["status"] == "ok"
    assert outcome.result["device"]["platform"] == "cpu"


def test_model_code_is_refused_without_isolation(tmp_path):
    request = _prepare(tmp_path / "w", REFERENCE)
    outcome = runner.TrustedSubprocessRunner().run(tmp_path / "w", request, trusted=False, timeout_s=120)
    assert outcome.status == runner.ISOLATION_UNAVAILABLE
    assert not (tmp_path / "w" / "request.json").exists()          # nothing was even prepared to run


def test_deadline_kills_the_worker(tmp_path):
    request = _prepare(tmp_path / "w", INFINITE)
    outcome = runner.TrustedSubprocessRunner(reviewed_fixtures=True).run(tmp_path / "w", request,
                                                                         trusted=False, timeout_s=8)
    assert outcome.status == runner.TIMEOUT and outcome.result is None
    assert "deadline" in outcome.detail


def test_descendants_are_killed_after_exit(tmp_path):
    marker = tmp_path / "grandchild.pid"
    script = textwrap.dedent(f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
    """)
    status, *_ = runner.execute([sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ),
                                timeout_s=30, max_log_bytes=1000)
    assert status == runner.OK
    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild survived the request")


def test_logs_are_bounded(tmp_path):
    status, out, err, _ = runner.execute([sys.executable, "-c", "print('x' * 100000)"], cwd=tmp_path,
                                         env=dict(os.environ), timeout_s=30, max_log_bytes=1000)
    assert status == runner.OK and len(out) == 1000


def test_missing_or_forged_result_is_a_crash(tmp_path):
    work = tmp_path / "w"
    request = _prepare(work, REFERENCE)
    request["mode"] = "measure"                                    # missing fields: the worker reports an error
    request.update(protocol=dataclasses.asdict(MeasurementProtocol()), case="missing")
    outcome = runner.TrustedSubprocessRunner().run(work, request, trusted=True, timeout_s=120)
    assert outcome.status == runner.OK and outcome.result["status"] == "error"

    (work / "result.json").write_text(json.dumps({"schema_version": "forged"}))
    assert runner.read_result(work) is None
    os.remove(work / "result.json")
    os.symlink(tmp_path / "elsewhere.json", work / "result.json")
    assert runner.read_result(work) is None


def test_worker_env_has_no_credentials_or_backend_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/secret.json")
    env = runner.worker_env(tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert "JAX_PLATFORMS" not in env
    assert env["HOME"] == str(tmp_path)


def test_image_reference_is_a_content_hash(monkeypatch, tmp_path):
    cpu, gpu = runner.image_reference("cpu"), runner.image_reference("cuda12")
    assert cpu.startswith("aic-worker:cpu-") and gpu.startswith("aic-worker:cuda12-") and cpu != gpu
    assert runner.image_reference("cpu") == cpu
    changed = tmp_path / "worker.Dockerfile"
    changed.write_text(runner.DOCKERFILE.read_text() + "\n# changed\n")
    monkeypatch.setattr(runner, "DOCKERFILE", changed)
    assert runner.image_reference("cpu") != cpu
    with pytest.raises(ValueError, match="variant"):
        runner.image_reference("rocm")


def test_image_contains_no_secrets_or_repo_state():
    names = {p.name for p in runner.image_files()}
    assert "worker.py" in names and "uv.lock" in names
    assert not any(n.startswith(".env") or n in ("controller.py", "agent.py") for n in names)


def test_docker_argv_applies_every_control(tmp_path):
    argv = runner.DockerRunner(selected_backend="cpu", image="img:tag").argv(tmp_path, "c1")
    joined = " ".join(argv)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--user 65534:65534", "--pids-limit 128", "--memory 2g", "--memory-swap 2g", "--cpus 2",
                 "--env HOME=/tmp", f"source={tmp_path.resolve()},target=/work"):
        assert flag in joined, flag
    assert "--privileged" not in joined and "--gpus" not in joined
    assert argv[-2:] == ["img:tag", "/work"]
    assert "--gpus all" in " ".join(runner.DockerRunner(selected_backend="cuda", image="i").argv(tmp_path, "c2"))


def test_docker_variant_follows_the_backend():
    assert runner.DockerRunner(selected_backend="cuda").variant == "cuda12"
    assert runner.DockerRunner(selected_backend="cpu").variant == "cpu"
    assert runner.DockerRunner(selected_backend=None).variant == "cpu"


@pytest.mark.slow
def test_calibrate_aa_script_writes_or_refuses(repo_root, clean_env, tmp_path):
    """Mechanics only: on a quiet host it writes a calibration, on a noisy one it refuses (exit 1)."""
    import subprocess
    proc = subprocess.run([sys.executable, str(repo_root / "scripts" / "calibrate_aa.py"), "--workload", "softmax",
                           "--backend", "cpu", "--runner", "subprocess", "--label", "test", "--out", str(tmp_path)],
                          cwd=tmp_path, env=clean_env, capture_output=True, text=True, timeout=600)
    assert proc.returncode in (0, 1), proc.stderr
    if proc.returncode == 0:
        written = list(tmp_path.rglob("softmax-128x1024.json"))
        assert len(written) == 1 and json.loads(written[0].read_text())["label"] == "test"
    else:
        assert "not calibrated" in proc.stderr
