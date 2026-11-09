"""Shared pytest setup: flat-module imports, CPU-only JAX, needs_gpu skipping, and key isolation."""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:               # flat modules: make agent/controller/benchmark importable
    sys.path.insert(0, str(REPO))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:              # make `fakes` and `fixtures` importable from subprocess drivers too
    sys.path.insert(0, str(TESTS))
# Forced, not setdefault: a shell `export JAX_PLATFORMS=cuda` must not move the test process to the GPU.
os.environ["JAX_PLATFORMS"] = "cpu"   # must happen before any jax import

from backends import cuda_status, docker_status  # noqa: E402  (importable only after the sys.path setup)

_REQUIREMENTS = {"needs_gpu": (cuda_status, "--require-gpu"), "needs_docker": (docker_status, "--require-docker")}


def pytest_addoption(parser):
    parser.addoption("--require-gpu", action="store_true",
                     help="fail needs_gpu tests instead of skipping them when CUDA is unavailable")
    parser.addoption("--require-docker", action="store_true",
                     help="fail needs_docker tests instead of skipping them when Docker is unavailable")


def pytest_runtest_setup(item):
    """needs_gpu / needs_docker tests are skipped without CUDA / Docker, or failed when the matching
    --require-* option is given (NV host, CI)."""
    for marker, (status, option) in _REQUIREMENTS.items():
        if item.get_closest_marker(marker) is None:
            continue
        available, reason = status()        # probed lazily, once per session
        if available:
            continue
        if item.config.getoption(option.lstrip("-").replace("-", "_")):
            pytest.fail(f"{option}: {reason}", pytrace=False)
        pytest.skip(f"{marker}: {reason}")


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    """No API key and no developer .env.local can leak into a test."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def repo_root():
    return REPO


@pytest.fixture
def clean_env():
    """Environment for subprocesses: no API key, CPU-only JAX unless a test overrides it."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["JAX_PLATFORMS"] = "cpu"
    return env
