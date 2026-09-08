"""Correctness gate. Runs before any timing: a fast wrong answer is not a result.

The oracle is the task's NumPy float64 ``reference``. Cases come from the task
(named, adversarial, fixed) plus a few random draws. The seeds used here are
deliberately *not* included in the agent's context, so a proposal cannot be
tailored to the specific inputs it will be graded on.
"""

from __future__ import annotations

import numpy as np

from halo.config import CorrectnessConfig
from halo.tasks.base import Case
from halo.types import CorrectnessCase, CorrectnessReport

#: Withheld from every prompt. Changing it must not change any verdict.
CORRECTNESS_SEED = 0xC0FFEE


def _compare(
    got: np.ndarray, expected: np.ndarray, rtol: float, atol: float
) -> tuple[bool, float, float, str | None]:
    if got.shape != expected.shape:
        return False, float("inf"), float("inf"), (
            f"shape mismatch: got {got.shape}, expected {expected.shape}"
        )
    if not np.isfinite(got).all():
        n_bad = int((~np.isfinite(got)).sum())
        return False, float("inf"), float("inf"), (
            f"{n_bad} non-finite values (inf/nan) in the output"
        )
    diff = np.abs(got - expected)
    # Relative error is measured against the array's own scale, not against each
    # element. Attention and softmax outputs contain values many orders of
    # magnitude below the signal; dividing by those reports a 178% "error" on a
    # result that is correct to 6e-05, which is worse than useless in a report.
    floor = 1e-3 * float(np.abs(expected).max()) if expected.size else 0.0
    scale = np.maximum(np.abs(expected), max(floor, np.finfo(np.float64).tiny))
    max_abs = float(diff.max())
    max_rel = float((diff / scale).max())
    passed = bool((diff <= atol + rtol * np.abs(expected)).all())
    return passed, max_abs, max_rel, None


def check(
    candidate_fn,
    task,
    cfg: CorrectnessConfig,
    *,
    jit,
    to_device,
) -> CorrectnessReport:
    """Evaluate ``candidate_fn`` against the task oracle on every case.

    ``jit``/``to_device`` are injected so this module stays free of JAX imports and
    is testable with plain NumPy stand-ins.
    """
    rng = np.random.default_rng(CORRECTNESS_SEED)
    cases: list[Case] = list(task.correctness_cases(rng))
    cases += [
        Case(f"random_{i}", task.make_inputs(rng))
        for i in range(cfg.n_random_cases)
    ]

    compiled = jit(candidate_fn)
    results: list[CorrectnessCase] = []
    for case in cases:
        rtol = cfg.rtol if case.rtol is None else case.rtol
        atol = cfg.atol if case.atol is None else case.atol
        expected = np.asarray(task.reference(*case.inputs), np.float64)
        try:
            raw = compiled(*to_device(case.inputs))
            got = np.asarray(raw, np.float64)
        except Exception as exc:  # noqa: BLE001 - candidate code is untrusted
            results.append(
                CorrectnessCase(case.name, False, float("inf"), float("inf"),
                                f"{type(exc).__name__}: {exc}")
            )
            continue
        passed, max_abs, max_rel, detail = _compare(got, expected, rtol, atol)
        if detail is None and not passed:
            detail = (
                f"exceeds tolerance (rtol={rtol:g}, atol={atol:g})"
                + (f"; case note: {case.note}" if case.note else "")
            )
        results.append(
            CorrectnessCase(case.name, passed, max_abs, max_rel, detail)
        )

    return CorrectnessReport(
        passed=all(r.passed for r in results),
        rtol=cfg.rtol,
        atol=cfg.atol,
        cases=tuple(results),
    )
