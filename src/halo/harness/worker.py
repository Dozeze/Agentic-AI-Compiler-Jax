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
from halo.harness import correctness, hlo, timing
from halo.tasks import base
from halo.types import DeviceInfo, Measurement, to_json


def _device_info(jax) -> DeviceInfo:
    device = jax.devices()[0]
    return DeviceInfo(
        platform=device.platform,
        device_kind=device.device_kind,
        device_count=jax.device_count(),
        jax_version=jax.__version__,
        jaxlib_version=getattr(jax.lib, "__version__", "unknown"),
    )


def _compile(jax, fn, args):
    """Compile and report how long it took.

    Timed as lower + compile rather than "first call minus steady state", so the
    number is the actual cost of obtaining an executable and contains no dispatch
    or execution time.
    """
    start = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    return compiled, time.perf_counter() - start


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
    try:
        baseline_compiled, baseline_compile_s = _compile(jax, baseline_fn, args)
        candidate_compiled, candidate_compile_s = _compile(jax, candidate_fn, args)
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
        )

    baseline_stats, candidate_stats, speedup = timing.measure_ab(
        baseline_compiled,
        candidate_compiled,
        args,
        cfg.timing,
        block=jax.block_until_ready,
    )

    candidate_text = candidate_compiled.as_text()
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
        hlo=hlo.summarize(candidate_compiled, candidate_text),
        baseline_hlo=hlo.summarize(baseline_compiled, baseline_compiled.as_text()),
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
