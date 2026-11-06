"""Shared pytest setup: flat-module imports, CPU-only JAX, and key isolation."""
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
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # must happen before any jax import


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
    env.setdefault("JAX_PLATFORMS", "cpu")
    return env
