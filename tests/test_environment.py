"""Backend selection and environment capture (AIC-031).

Backend selection is process-wide and irreversible, so every test that calls `select_backend`
runs in its own subprocess and never pollutes the pytest process.
"""
import json
import subprocess
import sys

import pytest

PRELUDE = "import json, sys; sys.path.insert(0, {repo!r}); from environment import *\n"


def _run(repo_root, env, body, timeout=120):
    code = PRELUDE.format(repo=str(repo_root)) + body
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env,
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


SELECT = """
try:
    devices = select_backend({name!r})
    print(json.dumps({{"ok": True, "platforms": sorted({{d.platform for d in devices}})}}))
except BackendUnavailable as exc:
    print(json.dumps({{"ok": False, "error_type": exc.error_type, "detail": exc.detail}}))
"""


def test_select_cpu_returns_cpu_devices(repo_root, clean_env):
    result = _run(repo_root, clean_env, SELECT.format(name="cpu"))
    assert result == {"ok": True, "platforms": ["cpu"]}


@pytest.mark.parametrize("name", ["cuda", "rocm", "tpu"])
def test_unavailable_backend_raises_clean_error(repo_root, clean_env, name):
    result = _run(repo_root, clean_env, SELECT.format(name=name))
    if result["ok"]:
        pytest.skip(f"{name} is actually available here")
    assert result["error_type"]
    assert result["detail"]


def test_late_selection_is_detected(repo_root, clean_env):
    body = "import jax\njax.devices()\n" + SELECT.format(name="cuda")
    result = _run(repo_root, clean_env, body)
    if result["ok"] and "cpu" not in result["platforms"]:
        pytest.skip("CUDA is available and was initialised first")
    assert result["ok"] is False, "late selection must never silently return CPU devices"
    assert result["error_type"] in ("AssertionError", "PlatformMismatch")


def test_invalid_backend_name():
    from environment import select_backend
    with pytest.raises(ValueError):
        select_backend("gpu")  # ambiguous in JAX (blames ROCm); rejected before JAX is touched


def test_environment_has_no_secrets(repo_root, clean_env):
    env = {**clean_env, "ANTHROPIC_API_KEY": "sk-test-SHOULD-NOT-APPEAR"}
    body = "print(json.dumps(collect_environment(__import__('pathlib').Path({repo!r}))))".format(repo=str(repo_root))
    proc = subprocess.run([sys.executable, "-c", PRELUDE.format(repo=str(repo_root)) + body], cwd=repo_root,
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "sk-test-SHOULD-NOT-APPEAR" not in proc.stdout
    assert "ANTHROPIC_API_KEY" not in proc.stdout


def test_environment_required_keys(repo_root, clean_env):
    body = "print(json.dumps(collect_environment(__import__('pathlib').Path({repo!r}))))".format(repo=str(repo_root))
    env = _run(repo_root, clean_env, body)
    assert env["schema_version"] == "environment/v1"
    assert env["versions"]["jax"] == "0.4.38"
    assert env["repo"]["uv_lock_sha256"]
    assert env["timer"]["perf_counter_resolution_s"] > 0
    assert isinstance(env["platform"]["wsl"], bool)
    assert set(env["env_allowlist"]) == set(__import__("environment").ENV_ALLOWLIST)


@pytest.mark.needs_gpu
def test_cuda_on_host(repo_root, clean_env):
    env = {k: v for k, v in clean_env.items() if k != "JAX_PLATFORMS"}
    result = _run(repo_root, env, SELECT.format(name="cuda"))
    if not result["ok"]:
        pytest.skip(f"no CUDA here: {result['error_type']}")
    print(f"CUDA device platforms: {result['platforms']}")
    assert result["platforms"]
