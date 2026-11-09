"""Evaluation worker (AIC-008): runs ONE request in its own process and reports what happened.

    python worker.py <workdir>

reads <workdir>/request.json and writes <workdir>/result.json (and outputs.npz in `outputs` mode). For
model-written code this process runs inside the container (AIC-009). The worker never judges: it
returns raw outputs and raw timing blocks, and the host compares, analyses and decides.

Modes
  outputs  compile one source per input signature and run it on every case; outputs go to outputs.npz
  measure  compile sources `a` and `b` and run one paired-block session on one case
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REQUEST_SCHEMA = "worker_request/v1"
RESULT_SCHEMA = "worker_result/v1"
MAX_DETAIL_CHARS = 500
MAX_INPUT_BYTES = 256 << 20


def _clip(text: str) -> str:
    return text if len(text) <= MAX_DETAIL_CHARS else text[:MAX_DETAIL_CHARS - 3] + "..."


def _error(exc: BaseException) -> str:
    return _clip(f"{type(exc).__name__}: {exc}")


def _device_info(jax) -> dict:
    device = jax.devices()[0]
    return {"platform": device.platform, "device_kind": device.device_kind,
            "device_count": jax.device_count(), "jax": jax.__version__,
            "default_backend": jax.default_backend(),
            "timer_resolution_s": __import__("time").get_clock_info("perf_counter").resolution}


def _load(workdir: Path, name: str, function_name: str):
    from benchmark import load_function
    path = workdir / name
    return load_function(path.read_text(encoding="utf-8"), function_name, filename=name)


def _signature(args) -> str:
    return ";".join(f"{a.dtype}{list(a.shape)}" for a in args)


def _outputs(request: dict, workdir: Path, jax) -> dict:
    import array_io
    import benchmark
    cases = array_io.load(workdir / request["inputs"], base=workdir, max_bytes=MAX_INPUT_BYTES)
    try:
        fn = _load(workdir, request["sources"]["candidate"], request["function_name"])
    except Exception as exc:   # noqa: BLE001 - a module that does not load fails every case
        detail = "candidate module failed to load: " + _error(exc)
        return {"cases": [{"name": n, "status": "COMPILE_FAILED", "detail": detail} for n in cases],
                "compile": {}}
    results, compiled, failed, outputs = [], {}, {}, {}
    for name, tree in cases.items():
        args = tuple(jax.device_put(a) for a in tree.arrays)
        jax.block_until_ready(args)
        sig = _signature(tree.arrays)
        if sig in failed:
            results.append({"name": name, "status": "COMPILE_FAILED",
                            "detail": f"not retried: same shape/dtype failed to compile in {failed[sig]}"})
            continue
        if sig not in compiled:
            try:
                compiled[sig] = benchmark.compile_timed(fn, args)
            except Exception as exc:   # noqa: BLE001
                failed[sig] = name
                results.append({"name": name, "status": "COMPILE_FAILED", "detail": _error(exc)})
                continue
        try:
            out = jax.block_until_ready(compiled[sig].executable(*args))
        except Exception as exc:   # noqa: BLE001
            results.append({"name": name, "status": "EXECUTION_FAILED", "detail": _error(exc)})
            continue
        outputs[name] = array_io.flatten(out)          # host transfer: correctness phase only
        results.append({"name": name, "status": "ok"})
    array_io.save(workdir / "outputs.npz", outputs)
    return {"cases": results, "outputs": "outputs.npz",
            "compile": {sig: {"lower_s": c.lower_s, "compile_s": c.compile_s} for sig, c in compiled.items()}}


def _measure(request: dict, workdir: Path, jax) -> dict:
    import array_io
    import benchmark
    from contracts import MeasurementProtocol
    protocol = MeasurementProtocol(**request["protocol"])
    case = array_io.load(workdir / request["inputs"], base=workdir, max_bytes=MAX_INPUT_BYTES)[request["case"]]
    args = tuple(jax.device_put(a) for a in case.arrays)
    jax.block_until_ready(args)
    sides = {}
    for side in ("a", "b"):
        fn = _load(workdir, request["sources"][side], request["function_name"])
        sides[side] = benchmark.compile_timed(fn, args)
    session = benchmark.measure_paired(sides["a"].executable, sides["b"].executable, args, protocol,
                                       repeats=request.get("repeats"))
    return {"session": session.to_json(),
            "compile": {side: {"lower_s": c.lower_s, "compile_s": c.compile_s} for side, c in sides.items()}}


MODES = {"outputs": _outputs, "measure": _measure}


def handle(request: dict, workdir: Path) -> dict:
    """Run one request. Selects the backend first, before JAX initialises any backend."""
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"not a {REQUEST_SCHEMA} request")
    if request.get("mode") not in MODES:
        raise ValueError(f"unknown mode {request.get('mode')!r}")
    if request.get("backend") is not None:
        from environment import select_backend
        select_backend(request["backend"])
    import jax
    device = _device_info(jax)
    body = MODES[request["mode"]](request, Path(workdir), jax)
    return {"schema_version": RESULT_SCHEMA, "mode": request["mode"], "status": "ok", "device": device, **body}


def main(argv: list[str]) -> int:
    workdir = Path(argv[1])
    try:
        request = json.loads((workdir / "request.json").read_text(encoding="utf-8"))
        result = handle(request, workdir)
    except Exception as exc:   # noqa: BLE001 - every failure comes back as data
        result = {"schema_version": RESULT_SCHEMA, "status": "error", "error": _error(exc),
                  "traceback": _clip(traceback.format_exc(limit=8)[-MAX_DETAIL_CHARS:])}
    tmp = workdir / "result.json.tmp"
    tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    tmp.replace(workdir / "result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
