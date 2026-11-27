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

# Everything the worker image is built from. The image tag is a hash of exactly these files, so a change
# to any of them yields a new tag and the runner never runs a stale harness.
DOCKERFILE = REPO / "docker" / "worker.Dockerfile"
IMAGE_FILES = ("pyproject.toml", "uv.lock", "worker.py", "benchmark.py", "contracts.py", "environment.py",
               "array_io.py", "feedback.py", "workloads/__init__.py", "workloads/registry.py", "workloads/oracles.py",
               "workloads/softmax.py", "workloads/layer_norm.py", "workloads/matmul_gelu.py",
               "workloads/example.py")
IMAGE_REPOSITORY = "aic-worker"
VARIANTS = {"cpu": "", "cuda12": "cuda12"}       # variant -> uv extra


def image_files() -> list[Path]:
    return [REPO / name for name in IMAGE_FILES if (REPO / name).exists()]


def image_reference(variant: str) -> str:
    """`aic-worker:<variant>-<content hash>` for the current harness files."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown worker image variant {variant!r}; choose from {sorted(VARIANTS)}")
    import hashlib
    digest = hashlib.sha256(variant.encode())
    inputs = [("Dockerfile", DOCKERFILE), *((str(p.relative_to(REPO)), p) for p in image_files())]
    for label, path in inputs:
        digest.update(label.encode() + b"\0" + path.read_bytes() + b"\0")
    return f"{IMAGE_REPOSITORY}:{variant}-{digest.hexdigest()[:16]}"


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
    return data[-limit:].decode("utf-8", errors="replace") if limit > 0 else ""


def _launch_timeout(request, timeout_s):
    deadline = request.get("host_deadline_monotonic")
    return min(timeout_s, max(0., deadline - time.monotonic())) if deadline is not None else timeout_s


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


class _StageClock:
    """Parent-observed cumulative phase budgets. Changing a marker never replenishes a budget.

    Startup/imports/IR share compilation's allowance; each declared correctness case has its own
    allowance. No marker means compilation. Observation is quantized by execute's 20 ms poll.
    """
    def __init__(self, workdir, limits, cases, now):
        self.path, self.limits, self.cases = Path(workdir) / "phase.json", limits, set(cases)
        self.last, self.key, self.spent = now, ("compile", None), {}

    def remaining(self, now):
        self.spent[self.key] = self.spent.get(self.key, 0) + now - self.last
        self.last = now
        remaining = self.limits[self.key[0]] - self.spent[self.key]
        if remaining <= 0:
            return 0
        if self.path.exists() or self.path.is_symlink():
            path = checked_path(self.path, self.path.parent, 4096)
            marker = json.loads(path.read_bytes())
            stage, case = marker.get("stage"), marker.get("case")
            if (marker.get("schema_version") != "worker_phase/v1" or stage not in self.limits
                    or (stage == "correctness" and case not in self.cases)
                    or (stage != "correctness" and case is not None)):
                raise ValueError("invalid worker phase marker")
            self.key = stage, case
            remaining = min(remaining, self.limits[stage] - self.spent.get(self.key, 0))
        return remaining


def execute(argv: list[str], *, cwd: Path, env: dict, timeout_s: float, max_log_bytes: int,
            on_timeout=None, stage_limits=None, stage_cases=()) -> tuple[str, bytes, bytes, float]:
    """Run `argv` in its own session; (status, stdout tail, stderr tail, wall_s).

    Only the main process is waited for. Logs go to temporary files rather than pipes, so a descendant
    that inherited them cannot hold the request open. On timeout the whole process group is killed (and
    `on_timeout` runs, e.g. to kill a container); after any exit the group is killed again, so no
    descendant outlives the request.
    """
    start = time.monotonic()
    if timeout_s <= 0 or (stage_limits and stage_limits["compile"] <= 0):
        return TIMEOUT, b"", b"launch refused: exhausted deadline", 0.0
    stages = _StageClock(cwd, stage_limits, stage_cases, start) if stage_limits else None
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
        status = OK
        interruption = None
        try:
            while proc.poll() is None:
                remaining = timeout_s - (time.monotonic() - start)
                if stages is not None:
                    remaining = min(remaining, stages.remaining(time.monotonic()))
                if remaining <= 0:
                    status = TIMEOUT
                    break
                try:
                    proc.wait(timeout=min(remaining, .020) if stages else remaining)
                except subprocess.TimeoutExpired:
                    continue
        except (OSError, ValueError, ArrayIOError, TypeError, KeyError):
            status = CRASHED
        except BaseException as error:
            status, interruption = CRASHED, error
        if status != OK:
            try:
                if on_timeout is not None:
                    on_timeout()
            except (OSError, subprocess.SubprocessError):
                status = CRASHED
            finally:
                _kill_group(proc.pid)
                proc.wait()
        _kill_group(proc.pid)
        if interruption is not None:
            raise interruption
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
    phase = workdir / "phase.json"
    if phase.exists() or phase.is_symlink():
        phase.unlink()


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
        workdir = Path(workdir).resolve()  # the worker runs with cwd=workdir, so a relative path would double
        _write_request(workdir, request)
        timeout_s = _launch_timeout(request, timeout_s)
        status, out, err, wall_s = execute([sys.executable, str(WORKER), str(workdir)], cwd=workdir,
                                           env=worker_env(workdir), timeout_s=timeout_s,
                                           max_log_bytes=self.max_log_bytes,
                                           stage_limits=request.get("stage_limits"), stage_cases=request.get("stage_cases", ()))
        detail = f"worker stage or request deadline exceeded (request ceiling {timeout_s:g} s)" if status == TIMEOUT else None
        return _outcome(status, out, err, wall_s, workdir, self.max_log_bytes, detail)


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


class DockerRunner:
    """One locked-down container per request: the only path for model-written code (AIC-009).

    Controls (AIC-1.1 §7): no network; read-only root filesystem; all capabilities dropped and
    no-new-privileges (Docker's default seccomp profile stays on); a non-root user; memory (no swap)
    and PID limits; a size-limited tmpfs as the only scratch space; only the request's own work directory
    mounted; no inherited environment, so no credentials. A GPU is exposed only for backend `cuda`
    (`--gpus all`, the only form WSL2 supports). The parent owns the deadline, kills the container on
    timeout, inspects it for an out-of-memory kill, removes it and verifies it is gone.

    CPU: no CPU-time quota by default. A `--cpus` quota throttles the whole container once XLA's thread
    pool (sized from all visible cores) has used the period's share, which made timing bimodal (P29:
    matmul_gelu blocks of about 375 and 1,130 µs). CPU time stays bounded by the parent's wall-clock
    deadline. `cpuset` pins the container to fixed cores (`--cpuset-cpus`, the plan's preferred Linux
    control); `cpus` restores a quota. The choice is part of the measurement population (`cpu_control`).
    """
    name = "docker/v1"
    isolated = True

    def __init__(self, *, selected_backend: str | None, image: str | None = None, memory: str = "2g", cpus: float | None = None,
                 cpuset: str | None = None, pids: int = 128, scratch_mb: int = 256, max_log_bytes: int = 1 << 20):
        self.variant = "cuda12" if selected_backend == "cuda" else "cpu"
        self.image = image or image_reference(self.variant)
        self.gpus = selected_backend == "cuda"
        if cpus is not None and cpuset is not None:
            raise ValueError("choose either a CPU quota (cpus) or fixed cores (cpuset), not both")
        self.limits = {"memory": memory, "cpus": cpus, "cpuset": cpuset, "pids": pids, "scratch_mb": scratch_mb}
        self.max_log_bytes = max_log_bytes

    def identity(self) -> dict:
        image_id = None
        try:
            inspected = _docker("image", "inspect", "--format", "{{.Id}}", self.image, timeout=30)
            image_id = inspected.stdout.strip() or None if inspected.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {"runner": self.name, "isolated": True, "image": self.image, "image_id": image_id,
                "gpus": self.gpus, "limits": self.limits, "cpu_control": self.cpu_control()}

    def cpu_control(self) -> str:
        """How CPU is constrained: "none", "cpus=<quota>" or "cpuset=<cores>". Timings differ across these."""
        if self.limits["cpuset"] is not None:
            return f"cpuset={self.limits['cpuset']}"
        if self.limits["cpus"] is not None:
            return f"cpus={self.limits['cpus']}"
        return "none"

    def preflight(self) -> str | None:
        """None when containers can run; otherwise why not (the run is then ISOLATION_UNAVAILABLE)."""
        try:
            if _docker("info", "--format", "{{.ServerVersion}}", timeout=30).returncode != 0:
                return "the Docker daemon is not reachable (start Docker Desktop or the docker service)"
            if _docker("image", "inspect", self.image, timeout=30).returncode != 0:
                return (f"worker image {self.image} is not built: "
                        f"uv run --locked python scripts/build_worker_image.py --variant {self.variant}")
        except FileNotFoundError:
            return "the docker CLI is not installed"
        except subprocess.TimeoutExpired:
            return "the Docker daemon did not answer"
        return None

    def argv(self, workdir: Path, container: str) -> list[str]:
        limits = self.limits
        argv = ["docker", "run", "--name", container, "--label", "aic.worker=1",
                "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--user", "65534:65534", "--pids-limit", str(limits["pids"]),
                "--memory", limits["memory"], "--memory-swap", limits["memory"],
                "--tmpfs", f"/tmp:rw,nosuid,nodev,size={limits['scratch_mb']}m,mode=1777",
                "--env", "HOME=/tmp", "--workdir", "/work",
                "--mount", f"type=bind,source={Path(workdir).resolve()},target=/work"]
        if limits["cpuset"] is not None:
            argv += ["--cpuset-cpus", str(limits["cpuset"])]
        elif limits["cpus"] is not None:
            argv += ["--cpus", str(limits["cpus"])]
        if self.gpus:
            argv += ["--gpus", "all"]
        return argv + [self.image, "/work"]

    def run(self, workdir: Path, request: dict, *, trusted: bool, timeout_s: float) -> RunOutcome:
        del trusted                                      # everything is contained here
        reason = self.preflight()
        if reason is not None:
            return RunOutcome(ISOLATION_UNAVAILABLE, None, "", "", 0.0, reason)
        timeout_s = _launch_timeout(request, timeout_s)
        if timeout_s <= 0:
            return RunOutcome(TIMEOUT, None, "", "", 0., "run deadline expired during runner verification")
        import uuid
        workdir = Path(workdir)
        _write_request(workdir, request)
        os.chmod(workdir, 0o777)                         # the container user (nobody) writes its result here
        container = f"aic-worker-{uuid.uuid4().hex[:12]}"
        host_env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT",
                                               "DOCKER_CONFIG") if k in os.environ}
        try:
            status, out, err, wall_s = execute(self.argv(workdir, container), cwd=workdir, env=host_env,
                                           timeout_s=timeout_s, max_log_bytes=self.max_log_bytes,
                                           on_timeout=lambda: _docker("kill", container, timeout=30),
                                           stage_limits=request.get("stage_limits"), stage_cases=request.get("stage_cases", ()))
            oom = _docker("inspect", "--format", "{{.State.OOMKilled}}", container, timeout=30).stdout.strip()
        finally:
            _docker("rm", "--force", container, timeout=60)
            alive = _docker("ps", "--all", "--quiet", "--filter", f"name=^{container}$", timeout=30)
            if alive.returncode != 0 or alive.stdout.strip():
                raise RuntimeError(f"container {container} cleanup could not be verified")
        if status == OK and oom == "true":
            return RunOutcome(RESOURCE_LIMIT, None, _tail(out, self.max_log_bytes), _tail(err, self.max_log_bytes),
                              wall_s, f"killed for exceeding the {self.limits['memory']} memory limit")
        detail = f"worker stage or request deadline exceeded (request ceiling {timeout_s:g} s)" if status == TIMEOUT else None
        return _outcome(status, out, err, wall_s, workdir, self.max_log_bytes, detail)
