"""Test-side backend helpers: is CUDA usable on this machine?

The pytest process itself is pinned to CPU (conftest sets JAX_PLATFORMS=cpu), so CUDA can only be
checked in a child process that has JAX_PLATFORMS removed.
"""
import functools
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_PROBE = """
import sys
sys.path.insert(0, {repo!r})
from environment import BackendUnavailable, select_backend
try:
    select_backend("cuda")
    print("ok")
except BackendUnavailable as exc:
    print(f"{{exc.error_type}}: {{exc.detail}}"[:200])
"""


def gpu_env() -> dict:
    """Subprocess environment that lets JAX see a GPU: no JAX_PLATFORMS pin and no API key."""
    return {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "ANTHROPIC_API_KEY")}


@functools.lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    """Return (available, reason). Runs one child process per pytest session."""
    try:
        proc = subprocess.run([sys.executable, "-c", _PROBE.format(repo=str(REPO))], cwd=REPO,
                              env=gpu_env(), capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "CUDA probe timed out"
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = proc.stderr.strip().splitlines()[-1:] or ["<no stderr>"]
        return False, f"CUDA probe crashed (exit {proc.returncode}): {tail[0][:200]}"
    if lines[-1] == "ok":
        return True, "CUDA available"
    return False, f"CUDA unavailable ({lines[-1]})"


@functools.lru_cache(maxsize=1)
def docker_status() -> tuple[bool, str]:
    """Return (available, reason): a reachable daemon and a CPU worker image for the current harness.

    The image is built on first use (a content-hashed tag, so an existing one is current).
    """
    sys.path.insert(0, str(REPO))
    import runner
    reason = runner.DockerRunner(selected_backend="cpu").preflight()
    if reason is not None and "not built" in reason:
        build = subprocess.run([sys.executable, str(REPO / "scripts" / "build_worker_image.py")],
                               capture_output=True, text=True, timeout=1800)
        if build.returncode != 0:
            return False, "worker image build failed: " + (build.stderr.strip().splitlines() or ["?"])[-1][:200]
        reason = runner.DockerRunner(selected_backend="cpu").preflight()
    return (True, "Docker available") if reason is None else (False, reason)
