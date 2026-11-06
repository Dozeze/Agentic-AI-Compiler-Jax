"""scripts/probe_target.py writes valid `target_probe/v1` evidence (AIC-031)."""
import json
import subprocess
import sys

import pytest

COVERAGE = ("DEVICE_EXECUTION_EVENTS", "CPU_EXECUTION_EVENTS", "HOST_ONLY", "UNAVAILABLE")
SECTIONS = ("environment", "ir", "profiler", "kernel", "isolation", "resources", "dispatch_probe")


def _probe(repo_root, env, tmp_path, *args):
    cmd = [sys.executable, str(repo_root / "scripts/probe_target.py"), *args, "--out", str(tmp_path)]
    return subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)


def test_probe_cpu_writes_valid_report(repo_root, clean_env, tmp_path):
    proc = _probe(repo_root, clean_env, tmp_path, "--backend", "cpu", "--label", "t-cpu")
    assert proc.returncode == 0, proc.stderr
    report = json.loads((tmp_path / "t-cpu.json").read_text())
    assert report["schema_version"] == "target_probe/v1"
    assert report["backend"]["status"] == "AVAILABLE"
    assert report["ir"]["stablehlo"]["ok"] is True
    assert report["profiler"]["coverage"] in COVERAGE
    assert report["kernel"]["pallas_native"]["status"] != "ok"  # CPU: interpret mode only
    assert "informative only" in report["dispatch_probe"]["note"]
    assert not list(tmp_path.glob("*.gz")), "raw traces are kept only with --keep-trace"


def _tpu_report(repo_root, clean_env, tmp_path, *extra):
    proc = _probe(repo_root, clean_env, tmp_path, "--backend", "tpu", "--label", "t-tpu", *extra)
    report = json.loads((tmp_path / "t-tpu.json").read_text())
    if report["backend"]["status"] == "AVAILABLE":
        pytest.skip("a TPU is present")
    return proc, report


def test_probe_unavailable_backend_is_a_result(repo_root, clean_env, tmp_path):
    proc, report = _tpu_report(repo_root, clean_env, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert report["backend"]["status"] == "UNAVAILABLE"
    assert report["backend"]["error_type"]
    for key in SECTIONS:
        assert report[key] == {"status": "SKIPPED"}


def test_require_available_exit_code(repo_root, clean_env, tmp_path):
    proc, _ = _tpu_report(repo_root, clean_env, tmp_path, "--require-available")
    assert proc.returncode == 2


def _probe_module(repo_root):
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        import probe_target
    finally:
        sys.path.remove(str(repo_root / "scripts"))
    return probe_target


def test_classify_trace_rules(repo_root):
    classify = _probe_module(repo_root).classify_trace
    device = [{"ph": "M", "name": "process_name", "pid": 7, "args": {"name": "/device:GPU:0"}},
              {"ph": "X", "name": "fusion", "pid": 7}]
    assert classify(device)["coverage"] == "DEVICE_EXECUTION_EVENTS"
    cpu = [{"ph": "X", "name": "TfrtCpuExecutable::Execute", "pid": 1}]
    assert classify(cpu)["coverage"] == "CPU_EXECUTION_EVENTS"
    assert classify(cpu)["granularity"] == "executable"
    lane_without_events = device[:1] + [{"ph": "X", "name": "PjitFunction", "pid": 1}]
    assert classify(lane_without_events)["coverage"] == "HOST_ONLY"
