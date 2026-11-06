"""Backend selection and environment capture (AIC-031, reused by AIC-002).

JAX is imported lazily so that the backend can be selected before any JAX computation.
JAX 0.4.38 silently ignores a backend selected after its backends were initialised, so
`select_backend` always verifies the devices it actually got.
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = ("cpu", "cuda", "rocm", "tpu")
# In JAX 0.4.x, Device.platform reports "gpu" for both CUDA and ROCm; client.platform may distinguish them.
# To be confirmed on the NV host (AIC-031 phase B).
_ACCEPTED_PLATFORMS = {"cpu": {"cpu"}, "cuda": {"gpu", "cuda"}, "rocm": {"gpu", "rocm"}, "tpu": {"tpu"}}
_GPU_PLATFORMS = _ACCEPTED_PLATFORMS["cuda"] | _ACCEPTED_PLATFORMS["rocm"]

# The only environment variables ever recorded. Never dump os.environ.
ENV_ALLOWLIST = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                 "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "OMP_NUM_THREADS", "TF_CPP_MIN_LOG_LEVEL", "WSL_DISTRO_NAME")

# On WSL2, nvidia-smi lives here and is often not on PATH (NVIDIA CUDA on WSL user guide).
_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def has_gpu(devices) -> bool:
    """True if any device is a CUDA or ROCm GPU, whichever platform name JAX reports for it."""
    return any(device.platform in _GPU_PLATFORMS for device in devices)


class BackendUnavailable(RuntimeError):
    """The requested JAX backend cannot be used in this process."""

    def __init__(self, requested: str, error_type: str, detail: str):
        super().__init__(f"requested backend {requested!r} is unavailable ({error_type}): {detail}")
        self.requested, self.error_type, self.detail = requested, error_type, detail


def select_backend(name: str):
    """Select `name` and return its devices, or raise BackendUnavailable.

    Must run before any JAX computation. Never falls back silently: if JAX ends up on a
    different platform than requested, this raises instead of returning those devices.
    """
    if name not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {name!r}")
    import jax
    try:
        jax.config.update("jax_platforms", name)
        devices = jax.devices()
    except Exception as exc:  # JAX 0.4.38 raises a bare AssertionError for an unavailable "cuda"
        raise BackendUnavailable(name, type(exc).__name__, str(exc) or "<no message>") from exc
    actual = {d.platform for d in devices} | {getattr(d.client, "platform", d.platform) for d in devices}
    if not actual & _ACCEPTED_PLATFORMS[name]:
        # Typical cause: JAX initialised its backends earlier, so the update was silently ignored.
        raise BackendUnavailable(name, "PlatformMismatch",
                                 f"JAX reports {sorted(actual)}; backends were probably initialised earlier")
    return devices


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 10) -> str | None:
    """Run a command; return stripped stdout on success, otherwise None."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi") or (_WSL_NVIDIA_SMI if Path(_WSL_NVIDIA_SMI).exists() else None)
    if exe is None:
        return None
    raw = _run([exe, "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader"])
    return {"path": exe, "raw": raw.splitlines() if raw else None}


def collect_environment(repo: Path, devices=None) -> dict:
    """Everything AIC-1.1 §5.1 asks to record.

    Unavailable values are null and listed in `unavailable` with a reason; never guessed, never zero.
    Only variables in ENV_ALLOWLIST are recorded.
    """
    import jax
    import jaxlib
    import numpy
    repo = Path(repo).resolve()
    unavailable: dict[str, str] = {}
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
        unavailable["cpu_affinity"] = "os.sched_getaffinity is not available on this OS (e.g. macOS)"
    devices = devices if devices is not None else jax.devices()
    gpu = _nvidia_smi()
    if gpu is None:
        unavailable["nvidia_smi"] = "nvidia-smi not found"
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=5)
    if commit is None:
        unavailable["git_commit"] = "git rev-parse HEAD failed (not a git checkout?)"
    dirty = _run(["git", "status", "--porcelain"], cwd=repo, timeout=5)
    return {
        "schema_version": "environment/v1",
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "node": platform.node(), "wsl": is_wsl()},
        "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": numpy.__version__},
        "jax": {"default_backend": jax.default_backend(), "x64": bool(jax.config.jax_enable_x64)},
        "devices": [{"id": d.id, "platform": d.platform, "device_kind": d.device_kind,
                     "client_platform": getattr(d.client, "platform", None),
                     "platform_version": getattr(d.client, "platform_version", None)} for d in devices],
        "device_count": len(devices),
        "cpu": {"count": os.cpu_count(), "affinity": affinity},
        "timer": {"perf_counter_resolution_s": time.get_clock_info("perf_counter").resolution},
        "gpu_tools": {"nvidia_smi": gpu},
        "env_allowlist": {k: os.environ.get(k) for k in ENV_ALLOWLIST},
        "repo": {"git_commit": commit,
                 "git_dirty": None if dirty is None else bool(dirty),
                 "on_windows_mount": str(repo).startswith("/mnt/"),  # slow cross-filesystem I/O on WSL2
                 "uv_lock_sha256": sha256_file(repo / "uv.lock"),
                 "pyproject_sha256": sha256_file(repo / "pyproject.toml")},
        "unavailable": unavailable,
    }
