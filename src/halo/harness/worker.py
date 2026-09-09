"""Runs inside the measurement subprocess. Never import this from the controller.

Isolation is the point: JAX caches compilations process-wide and holds device
state, and LLM-written candidate code can hang, exhaust memory, or abort the
interpreter outright. Keeping all of that behind a process boundary means a bad
proposal costs one measurement rather than the whole run.

Invoked as ``python -m halo.harness.worker <request.json>``; writes a serialised
:class:`~halo.types.Measurement` to the path named in the request.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from halo.config import RunConfig
from halo.harness import buffers, correctness, hlo, ir, timing
from halo.tasks import base
from halo.types import DeviceInfo, Measurement, ProgramSummary, to_json


def _device_info(jax) -> DeviceInfo:
    device = jax.devices()[0]
    return DeviceInfo(
        platform=device.platform,
        device_kind=device.device_kind,
        device_count=jax.device_count(),
        jax_version=jax.__version__,
        jaxlib_version=getattr(jax.lib, "__version__", "unknown"),
    )


def _analyze(jax, fn, name, args, dump_dir):
    """Compile one program and collect every level of the lowering pipeline.

    ``fn.__name__`` is set first because XLA names its dump files after the jitted
    function; without distinct names the baseline's and the candidate's buffer
    reports are indistinguishable on disk.

    Compile time stays lower + compile, excluding the analysis work, so it remains
    the actual cost of obtaining an executable.
    """
    fn.__name__ = name

    jaxpr = ir.from_jaxpr(jax.make_jaxpr(fn)(*args))

    start = time.perf_counter()
    lowered = jax.jit(fn).lower(*args)
    lower_s = time.perf_counter() - start

    stablehlo = ir.from_stablehlo(lowered.as_text())

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_s = lower_s + (time.perf_counter() - start)

    text = compiled.as_text()
    summary = ProgramSummary(
        jaxpr=jaxpr,
        stablehlo=stablehlo,
        hlo=hlo.summarize(compiled, text),
        buffers=buffers.read(dump_dir, name) if dump_dir else None,
    )
    return compiled, compile_s, text, summary


def run(request: dict) -> Measurement:
    import jax

    cfg = RunConfig.from_dict(request["config"])
    spec = base.load_spec(request["task"])
    if spec.reference_sha256 != request["reference_sha256"]:
        return Measurement.failure(
            request["task"], "task.py hash changed between planning and measurement"
        )
    task = spec.load()

    baseline_fn = base.load_module(Path(request["baseline_path"]), "halo_baseline").candidate
    candidate_fn = base.load_module(Path(request["candidate_path"]), "halo_candidate").candidate

    import numpy as np

    device = _device_info(jax)
    to_device = lambda arrays: tuple(jax.device_put(a) for a in arrays)

    args = to_device(task.make_inputs(np.random.default_rng(cfg.seed)))
    jax.block_until_ready(args)

    # Compile before checking correctness. The correctness pass would otherwise
    # populate jit's cache and the candidate's compile time would read as 0.00s
    # while the baseline's did not - two numbers that are not comparable.
    dump_dir = Path(request["dump_dir"]) if request.get("dump_dir") else None
    try:
        baseline_compiled, baseline_compile_s, _, baseline_program = _analyze(
            jax, baseline_fn, "baseline", args, dump_dir
        )
        candidate_compiled, candidate_compile_s, candidate_text, candidate_program = _analyze(
            jax, candidate_fn, "candidate", args, dump_dir
        )
    except Exception:
        return Measurement.failure(
            spec.name, "candidate failed to trace or compile:\n" + traceback.format_exc(limit=20)
        )

    report = correctness.check(
        candidate_fn, task, cfg.correctness, jit=jax.jit, to_device=to_device
    )
    if not report.passed:
        # Do not spend minutes benchmarking something that computes the wrong answer.
        return Measurement(
            task=spec.name,
            ok=True,
            device=device,
            correctness=report,
            compile_s=candidate_compile_s,
            baseline_compile_s=baseline_compile_s,
            candidate_program=candidate_program,
            baseline_program=baseline_program,
        )

    baseline_stats, candidate_stats, speedup = timing.measure_ab(
        baseline_compiled,
        candidate_compiled,
        args,
        cfg.timing,
        block=jax.block_until_ready,
    )

    if (dump := request.get("hlo_dump")) is not None:
        Path(dump).write_text(candidate_text)

    return Measurement(
        task=spec.name,
        ok=True,
        device=device,
        correctness=report,
        baseline=baseline_stats,
        candidate=candidate_stats,
        speedup=speedup,
        compile_s=candidate_compile_s,
        baseline_compile_s=baseline_compile_s,
        candidate_program=candidate_program,
        baseline_program=baseline_program,
    )


def main(argv: list[str]) -> int:
    request = json.loads(Path(argv[1]).read_text())
    try:
        measurement = run(request)
    except Exception:  # noqa: BLE001 - any failure must come back as data
        measurement = Measurement.failure(
            request.get("task", "?"), traceback.format_exc(limit=20)
        )
    Path(request["out"]).write_text(to_json(measurement))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
