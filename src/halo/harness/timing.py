"""Timing methodology.

Three things make these numbers trustworthy, and all three are easy to get wrong:

1. **block_until_ready on every timed call.** JAX dispatches asynchronously; without
   this you time the dispatch, not the computation, and every number is fiction.
2. **Interleaving.** Baseline and candidate alternate within each round, and the
   order flips between rounds, so thermal drift and frequency scaling hit both
   equally instead of favouring whichever ran first.
3. **A paired bootstrap CI on the ratio**, not a difference of means. Rounds are
   resampled as pairs because they are paired in time.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from halo.config import TimingConfig
from halo.types import SpeedupEstimate, TimingStats

#: Rounds shorter than this are dominated by dispatch overhead rather than by the
#: computation, so each round repeats the call until it clears the floor.
MIN_ROUND_S = 5e-3


def _stats(samples_ms: list[float]) -> TimingStats:
    ordered = sorted(samples_ms)
    q1, q3 = np.percentile(ordered, [25, 75])
    return TimingStats(
        samples_ms=tuple(samples_ms),
        median_ms=float(statistics.median(ordered)),
        iqr_ms=float(q3 - q1),
        min_ms=float(ordered[0]),
    )


def _time_once(fn: Callable, args, block, reps: int) -> float:
    start = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    block(out)
    return (time.perf_counter() - start) / reps * 1e3


def calibrate_reps(fn: Callable, args, block) -> int:
    """Pick the number of inner repetitions so a round clears ``MIN_ROUND_S``."""
    elapsed_ms = _time_once(fn, args, block, 1)
    if elapsed_ms <= 0:
        return 100
    return max(1, min(1000, int(MIN_ROUND_S * 1e3 / elapsed_ms) + 1))


def measure_ab(
    baseline: Callable,
    candidate: Callable,
    args,
    cfg: TimingConfig,
    *,
    block,
) -> tuple[TimingStats, TimingStats, SpeedupEstimate]:
    """Interleaved A/B measurement of two callables over identical inputs."""
    for _ in range(cfg.warmup):
        block(baseline(*args))
        block(candidate(*args))

    reps = max(calibrate_reps(baseline, args, block),
               calibrate_reps(candidate, args, block))

    base_ms: list[float] = []
    cand_ms: list[float] = []
    for round_index in range(cfg.rounds):
        # Flip the order every round so neither side systematically runs on a
        # warmer or cooler machine.
        if round_index % 2 == 0:
            base_ms.append(_time_once(baseline, args, block, reps))
            cand_ms.append(_time_once(candidate, args, block, reps))
        else:
            cand_ms.append(_time_once(candidate, args, block, reps))
            base_ms.append(_time_once(baseline, args, block, reps))

    return _stats(base_ms), _stats(cand_ms), bootstrap_speedup(base_ms, cand_ms, cfg)


def bootstrap_speedup(
    base_ms: list[float], cand_ms: list[float], cfg: TimingConfig
) -> SpeedupEstimate:
    """Paired bootstrap of median(baseline)/median(candidate)."""
    a = np.asarray(base_ms, float)
    b = np.asarray(cand_ms, float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), size=(cfg.bootstrap_samples, len(a)))
    ratios = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    tail = (1.0 - cfg.confidence) / 2.0 * 100.0
    lo, hi = np.percentile(ratios, [tail, 100.0 - tail])
    return SpeedupEstimate(
        ratio=float(np.median(a) / np.median(b)),
        ci_low=float(lo),
        ci_high=float(hi),
        confidence=cfg.confidence,
        n_rounds=len(a),
    )
