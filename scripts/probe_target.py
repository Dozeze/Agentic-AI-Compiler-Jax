"""Probe what a JAX backend on this machine can provide and write `target_probe/v1` JSON evidence (AIC-031).

Usage:
    uv run --locked python scripts/probe_target.py --backend cpu  --label mac-cpu          --out evidence/targets
    uv run --locked python scripts/probe_target.py --backend cuda --label nvhost-wsl2-cuda --out evidence/targets
    uv run --locked python scripts/probe_target.py --all --label <machine> --out evidence/targets

Backend selection is process-wide and irreversible, so every backend is probed in its own process
(`--all` spawns one child per backend). "Unavailable" is a valid result: the exit code is 0 whenever the
report was written, unless `--require-available` is given (then unavailable exits 2).
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from environment import BACKENDS, BackendUnavailable, collect_environment, is_wsl, select_backend  # noqa: E402

SCHEMA = "target_probe/v1"
COVERAGE_STATES = ("DEVICE_EXECUTION_EVENTS", "CPU_EXECUTION_EVENTS", "HOST_ONLY", "UNAVAILABLE")
SKIPPED = {"status": "SKIPPED"}
_DEVICE_LANE_MARKERS = ("/device:GPU", "/device:TPU", "Stream")
IR_SHAPE = (128, 1024)


def _softmax(x):
    import jax.numpy as jnp
    shifted = x - jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(shifted)
    return e / jnp.sum(e, axis=-1, keepdims=True)


def _err(exc: BaseException) -> dict:
    return {"ok": False, "error_type": type(exc).__name__, "detail": (str(exc) or "<no message>")[:500]}


def _lines(text: str) -> int:
    return len(text.splitlines())


def probe_ir() -> dict:
    """Which IR and compiler-analysis APIs work; each step is attempted independently."""
    import jax
    import jax.numpy as jnp
    x = jnp.ones(IR_SHAPE, jnp.float32)
    f = jax.jit(_softmax)
    out: dict = {"shape": list(IR_SHAPE)}
    try:
        out["jaxpr"] = {"ok": True, "lines": _lines(str(jax.make_jaxpr(_softmax)(x)))}
    except Exception as exc:
        out["jaxpr"] = _err(exc)
    try:
        lowered = f.lower(x)
        out["lower"] = {"ok": True}
    except Exception as exc:
        out["lower"] = _err(exc)
        for key in ("stablehlo", "hlo", "compiled_text", "cost_analysis", "memory_analysis"):
            out[key] = SKIPPED
        return out
    try:
        out["stablehlo"] = {"ok": True, "lines": _lines(str(lowered.compiler_ir("stablehlo")))}
    except Exception as exc:
        out["stablehlo"] = _err(exc)
    try:
        out["hlo"] = {"ok": True, "lines": _lines(lowered.compiler_ir("hlo").as_hlo_text())}
    except Exception as exc:
        out["hlo"] = _err(exc)
    try:
        compiled = lowered.compile()
        out["compiled_text"] = {"ok": True, "lines": _lines(compiled.as_text())}
    except Exception as exc:
        out["compiled_text"] = _err(exc)
        out["cost_analysis"] = out["memory_analysis"] = SKIPPED
        return out
    try:
        cost = compiled.cost_analysis()
        raw_type = type(cost).__name__
        if isinstance(cost, (list, tuple)):  # JAX 0.4.38 returns a list of dicts
            cost = cost[0] if cost else {}
        flops = cost.get("flops") if isinstance(cost, dict) else None
        out["cost_analysis"] = {"ok": True, "type": raw_type, "has_flops": flops is not None,
                                "flops": flops, "n_keys": len(cost) if isinstance(cost, dict) else None}
    except Exception as exc:
        out["cost_analysis"] = _err(exc)
    try:
        mem = compiled.memory_analysis()
        fields = {k: getattr(mem, k) for k in dir(mem)
                  if k.endswith("_in_bytes") and isinstance(getattr(mem, k), int)}
        out["memory_analysis"] = {"ok": mem is not None, "type": type(mem).__name__, "fields": fields}
    except Exception as exc:
        out["memory_analysis"] = _err(exc)
    return out


def classify_trace(events: list[dict]) -> dict:
    """Classify a Perfetto trace into one of COVERAGE_STATES (AIC-1.1 §6). Pure function; unit-tested."""
    lane_pids = set()
    lane_names = set()
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("process_name", "thread_name"):
            label = str((e.get("args") or {}).get("name", ""))
            if any(marker in label for marker in _DEVICE_LANE_MARKERS):
                lane_pids.add(e.get("pid"))
                lane_names.add(label)
    complete = [e for e in events if e.get("ph") == "X"]
    names = {}
    for e in complete:
        names[e.get("name", "")] = names.get(e.get("name", ""), 0) + 1
    top = [n for n, _ in sorted(names.items(), key=lambda kv: (-kv[1], kv[0]))[:10]]
    result = {"n_complete_events": len(complete), "top_event_names": top, "device_lanes": sorted(lane_names)}
    if lane_pids and any(e.get("pid") in lane_pids for e in complete):
        return {"coverage": "DEVICE_EXECUTION_EVENTS", "granularity": "device", **result}
    if any("Executable::Execute" in str(e.get("name", "")) for e in complete):
        return {"coverage": "CPU_EXECUTION_EVENTS", "granularity": "executable",
                "note": "executable-level runtime events; not per-HLO-op device time", **result}
    return {"coverage": "HOST_ONLY", **result}


def probe_profiler(keep_trace_to: Path | None) -> dict:
    import jax
    import jax.numpy as jnp
    f = jax.jit(_softmax)
    x = jnp.ones(IR_SHAPE, jnp.float32)
    f(x).block_until_ready()
    with tempfile.TemporaryDirectory(prefix="aic-probe-") as tmp:
        try:
            with jax.profiler.trace(tmp, create_perfetto_link=False, create_perfetto_trace=True):
                for _ in range(5):
                    f(x).block_until_ready()
        except Exception as exc:
            return {"status": "UNAVAILABLE", "coverage": "UNAVAILABLE", **_err(exc)}
        traces = sorted(Path(tmp).rglob("perfetto_trace.json.gz"))
        if not traces:
            return {"status": "UNAVAILABLE", "coverage": "UNAVAILABLE", "detail": "no perfetto_trace.json.gz"}
        try:
            with gzip.open(traces[0], "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            return {"status": "UNAVAILABLE", "coverage": "UNAVAILABLE", **_err(exc)}
        events = data.get("traceEvents", []) if isinstance(data, dict) else data
        result = {"status": "ok", **classify_trace(events)}
        if keep_trace_to is not None:
            keep_trace_to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(traces[0], keep_trace_to)
            result["kept_trace"] = keep_trace_to.name
        return result


def probe_kernel() -> dict:
    import jax
    import jax.numpy as jnp
    out: dict = {}
    try:
        from jax.experimental import pallas as pl
        out["pallas_import"] = {"status": "ok"}
    except Exception as exc:
        out["pallas_import"] = {"status": "error", **_err(exc)}
        out["pallas_native"] = out["pallas_interpret"] = SKIPPED
        pl = None

    if pl is not None:
        def add_kernel(x_ref, y_ref, o_ref):
            o_ref[...] = x_ref[...] + y_ref[...]

        x = jnp.arange(8, dtype=jnp.float32)
        for key, interpret in (("pallas_native", False), ("pallas_interpret", True)):
            try:
                call = pl.pallas_call(add_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                                      interpret=interpret)
                got = jax.block_until_ready(call(x, x))
                out[key] = {"status": "ok", "correct": bool(jnp.array_equal(got, x + x))}
            except Exception as exc:
                # On CPU, native Pallas raises "Only interpret mode is supported": a valid result, not a failure.
                out[key] = {"status": "error", **_err(exc)}
    try:
        from jax.extend import ffi
        attrs = sorted(a for a in dir(ffi) if not a.startswith("_"))
        include = None
        try:
            include = Path(ffi.include_dir())
        except Exception:
            pass
        out["ffi"] = {"status": "ok", "attributes": attrs,
                      "include_dir_exists": bool(include and include.is_dir())}
    except Exception as exc:
        out["ffi"] = {"status": "error", **_err(exc)}
    cc = None
    for compiler in ("cc", "gcc", "clang"):
        if shutil.which(compiler):
            try:
                proc = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=10)
                cc = {"compiler": compiler, "version": (proc.stdout or proc.stderr).splitlines()[0]}
            except Exception:
                cc = None
            break
    out["c_compiler"] = cc if cc else {"status": "unavailable"}
    return out


def _ok(cmd: list[str], timeout: float = 15) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def probe_isolation(try_docker_gpu: bool) -> dict:
    """Observations only; real containment is verified in AIC-009."""
    docker = shutil.which("docker") is not None and _ok(["docker", "info"])
    if try_docker_gpu and docker:
        nvidia_container = _ok(["docker", "run", "--rm", "--gpus", "all", "ubuntu", "nvidia-smi"], timeout=300)
    else:
        nvidia_container = None  # not attempted
    unshare = shutil.which("unshare")
    return {
        "docker": docker,
        "nvidia_container": nvidia_container,
        "nvidia_container_note": None if try_docker_gpu else "not attempted (use --try-docker-gpu)",
        "cgroup_v2": Path("/sys/fs/cgroup/cgroup.controllers").exists(),
        "user_namespaces": bool(unshare) and _ok([unshare, "--user", "--map-root-user", "true"]),
        "os": platform.system(),
        "wsl": is_wsl(),
        "cwd_on_windows_mount": str(Path.cwd()).startswith("/mnt/"),
    }


def _host_ram_bytes() -> int | None:
    import os
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:
        proc = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(proc.stdout.strip())
    except Exception:
        return None


def probe_resources(devices) -> dict:
    stats = []
    for d in devices:
        try:
            s = d.memory_stats()
        except Exception:
            s = None
        stats.append({"id": d.id, "memory_stats": {k: v for k, v in s.items() if isinstance(v, int)} if s else None})
    return {"devices": stats, "host_ram_bytes": _host_ram_bytes()}


def probe_dispatch(n: int = 1000) -> dict:
    import jax
    import jax.numpy as jnp
    f = jax.jit(lambda x, y: x + y)
    x = jnp.arange(32, dtype=jnp.float32)
    y = jnp.ones(32, dtype=jnp.float32)
    f(x, y).block_until_ready()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        f(x, y).block_until_ready()
        samples.append(time.perf_counter() - t0)
    return {"calls": n, "median_us": statistics.median(samples) * 1e6,
            "note": "informative only; not a benchmark (see AIC-005/006)"}


def run_probe(backend: str, label: str, out_dir: Path, keep_trace: bool, try_docker_gpu: bool) -> dict:
    report: dict = {"schema_version": SCHEMA, "label": label}
    try:
        devices = select_backend(backend)
    except BackendUnavailable as exc:
        report["backend"] = {"requested": backend, "status": "UNAVAILABLE",
                             "error_type": exc.error_type, "detail": exc.detail[:1000]}
        for key in ("environment", "ir", "profiler", "kernel", "isolation", "resources", "dispatch_probe"):
            report[key] = SKIPPED
        return report
    report["backend"] = {"requested": backend, "status": "AVAILABLE", "error_type": None, "detail": None}
    report["environment"] = collect_environment(REPO, devices)
    report["ir"] = probe_ir()
    report["profiler"] = probe_profiler(out_dir / f"{label}.perfetto_trace.json.gz" if keep_trace else None)
    report["kernel"] = probe_kernel()
    report["isolation"] = probe_isolation(try_docker_gpu)
    report["resources"] = probe_resources(devices)
    report["dispatch_probe"] = probe_dispatch()
    return report


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backend", choices=BACKENDS)
    group.add_argument("--all", action="store_true", help="probe every backend, one child process each")
    parser.add_argument("--label", required=True, help="evidence name, e.g. mac-cpu or nvhost-wsl2-cuda")
    parser.add_argument("--out", type=Path, default=REPO / "evidence" / "targets")
    parser.add_argument("--require-available", action="store_true", help="exit 2 if the backend is unavailable")
    parser.add_argument("--keep-trace", action="store_true", help="keep the raw perfetto trace (*.gz, git-ignored)")
    parser.add_argument("--try-docker-gpu", action="store_true",
                        help="also run `docker run --gpus all ... nvidia-smi` (may pull an image)")
    args = parser.parse_args(argv)

    if args.all:
        worst = 0
        for backend in BACKENDS:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--backend", backend,
                   "--label", f"{args.label}-{backend}", "--out", str(args.out)]
            for flag in ("require_available", "keep_trace", "try_docker_gpu"):
                if getattr(args, flag):
                    cmd.append("--" + flag.replace("_", "-"))
            worst = max(worst, subprocess.run(cmd).returncode)
        return worst

    report = run_probe(args.backend, args.label, args.out, args.keep_trace, args.try_docker_gpu)
    path = args.out / f"{args.label}.json"
    _write(path, report)
    status = report["backend"]["status"]
    print(f"{args.backend}: {status} -> {path}")
    if status != "AVAILABLE" and args.require_available:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
