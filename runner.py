"""Execution boundary (AIC-008, AIC-009): how one worker request is run, and how far it is trusted.

Every request runs in a fresh process with a fixed argument vector, a wall-clock deadline, bounded
captured logs and verified cleanup of everything it started. The host validates the returned result
file before reading it, and the worker's report never sets a verdict.

Two runners exist:
  TrustedSubprocessRunner  a local process. Lifecycle management, NOT a sandbox: it runs trusted code
                           only (the frozen reference, the baseline and reviewed fixtures) and refuses
                           model-written code with ISOLATION_UNAVAILABLE.
  DockerRunner             a locked-down container (AIC-009): the only path for model-written code.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from array_io import ArrayIOError, checked_path

REPO = Path(__file__).resolve().parent
WORKER = REPO / "worker.py"

OK = "OK"
TIMEOUT = "TIMEOUT"
RESOURCE_LIMIT = "RESOURCE_LIMIT"
CRASHED = "CRASHED"
ISOLATION_UNAVAILABLE = "ISOLATION_UNAVAILABLE"

# The only host variables a worker inherits. No credentials, no JAX_PLATFORMS: the backend travels in
# the request and the worker selects it through environment.select_backend.
_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL")
MAX_RESULT_BYTES = 16 << 20


@dataclass(frozen=True)
class RunOutcome:
    status: str                 # OK, TIMEOUT, RESOURCE_LIMIT, CRASHED or ISOLATION_UNAVAILABLE
    result: dict | None         # the validated worker_result/v1 document when status is OK
    stdout: str                 # bounded tails
    stderr: str
    wall_s: float
    detail: str | None = None


def _tail(data: bytes | None, limit: int) -> str:
    data = data or b""
    return data[-limit:].decode("utf-8", errors="replace")


def read_result(workdir: Path) -> dict | None:
    """The worker's result.json if it is a regular file in `workdir`, bounded and well-formed."""
    try:
        path = checked_path(Path(workdir) / "result.json", workdir, MAX_RESULT_BYTES)
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ArrayIOError, ValueError, OSError):
        return None
    if not isinstance(result, dict) or result.get("schema_version") != "worker_result/v1":
        return None
    return result


def execute(argv: list[str], *, cwd: Path, env: dict, timeout_s: float, max_log_bytes: int,
            on_timeout=None) -> tuple[str, bytes, bytes, float]:
    """Run `argv` in its own session; (status, stdout tail, stderr tail, wall_s).

    Only the main process is waited for. Logs go to temporary files rather than pipes, so a descendant
    that inherited them cannot hold the request open. On timeout the whole process group is killed (and
    `on_timeout` runs, e.g. to kill a container); after any exit the group is killed again, so no
    descendant outlives the request.
    """
    start = time.monotonic()
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        status = OK
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            status = TIMEOUT
            if on_timeout is not None:
                on_timeout()
            _kill_group(proc.pid)
            proc.wait()
        _kill_group(proc.pid)
        return status, _read_tail(out, max_log_bytes), _read_tail(err, max_log_bytes), time.monotonic() - start


def _read_tail(handle, limit: int) -> bytes:
    size = handle.seek(0, os.SEEK_END)
    handle.seek(max(0, size - limit))
    return handle.read()


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def worker_env(workdir: Path) -> dict:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update({"HOME": str(workdir), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    return env


def _write_request(workdir: Path, request: dict) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "request.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
    stale = workdir / "result.json"
    if stale.exists():
        stale.unlink()


def _outcome(status, out, err, wall_s, workdir, max_log_bytes, detail=None) -> RunOutcome:
    stdout, stderr = _tail(out, max_log_bytes), _tail(err, max_log_bytes)
    if status != OK:
        return RunOutcome(status, None, stdout, stderr, wall_s, detail)
    result = read_result(workdir)
    if result is None:
        last = (stderr.strip().splitlines() or ["<no stderr>"])[-1][:300]
        return RunOutcome(CRASHED, None, stdout, stderr, wall_s, f"worker wrote no valid result: {last}")
    return RunOutcome(OK, result, stdout, stderr, wall_s)


class TrustedSubprocessRunner:
    """A fresh local Python process per request. Not a sandbox: trusted code only.

    `reviewed_fixtures=True` lets it run code the team wrote and reviewed (test fixtures) as if it were
    model output. agent.py never sets it.
    """
    name = "trusted-subprocess/v1"
    isolated = False

    def __init__(self, *, reviewed_fixtures: bool = False, max_log_bytes: int = 1 << 20):
        self.reviewed_fixtures = reviewed_fixtures
        self.max_log_bytes = max_log_bytes

    def identity(self) -> dict:
        return {"runner": self.name, "isolated": False, "python": platform.python_version()}

    def preflight(self) -> str | None:
        return None

    def run(self, workdir: Path, request: dict, *, trusted: bool, timeout_s: float) -> RunOutcome:
        if not trusted and not self.reviewed_fixtures:
            return RunOutcome(ISOLATION_UNAVAILABLE, None, "", "", 0.0,
                              "model-written code runs only in an isolated runner (--runner docker)")
        workdir = Path(workdir)
        _write_request(workdir, request)
        status, out, err, wall_s = execute([sys.executable, str(WORKER), str(workdir)], cwd=workdir,
                                           env=worker_env(workdir), timeout_s=timeout_s,
                                           max_log_bytes=self.max_log_bytes)
        detail = f"exceeded the {timeout_s:g} s deadline" if status == TIMEOUT else None
        return _outcome(status, out, err, wall_s, workdir, self.max_log_bytes, detail)
