"""The accept/reject policy, with constructed measurements. No JAX, no subprocess."""

from __future__ import annotations

import pytest

from halo.config import AcceptConfig, RunConfig
from halo.controller import decide
from halo.types import (
    CorrectnessCase,
    CorrectnessReport,
    Measurement,
    SpeedupEstimate,
    TimingStats,
)

CFG = RunConfig(task="t", accept=AcceptConfig(min_speedup=1.05))

PASSING = CorrectnessReport(
    passed=True, rtol=1e-5, atol=1e-6,
    cases=(CorrectnessCase("random_0", True, 1e-7, 1e-7),),
)
FAILING = CorrectnessReport(
    passed=False, rtol=1e-5, atol=1e-6,
    cases=(CorrectnessCase("large_magnitude", False, float("inf"), float("inf"),
                           "524288 non-finite values (inf/nan) in the output"),),
)
STATS = TimingStats(samples_ms=(1.0,), median_ms=1.0, iqr_ms=0.1, min_ms=0.9)


def measurement(**kw) -> Measurement:
    base = dict(task="t", ok=True, correctness=PASSING,
                baseline=STATS, candidate=STATS)
    return Measurement(**{**base, **kw})


def speedup(ratio, lo, hi) -> SpeedupEstimate:
    return SpeedupEstimate(ratio=ratio, ci_low=lo, ci_high=hi,
                           confidence=0.95, n_rounds=30)


def test_clear_win_is_accepted():
    d = decide(measurement(speedup=speedup(1.73, 1.68, 1.79)), CFG)
    assert d.accepted and d.rule == "accepted"


def test_wrong_answer_is_rejected_before_performance_is_considered():
    d = decide(measurement(correctness=FAILING, speedup=speedup(9.9, 9.0, 10.0)), CFG)
    assert not d.accepted
    assert d.rule == "correctness"
    assert "non-finite" in d.reason


def test_speedup_inside_the_noise_band_is_rejected():
    """A 1.01x point estimate whose CI straddles 1.0 is not a result."""
    d = decide(measurement(speedup=speedup(1.01, 0.997, 1.025)), CFG)
    assert not d.accepted and d.rule == "performance"
    assert "noise" in d.reason


def test_point_estimate_above_threshold_but_ci_below_is_rejected():
    """The lower bound decides, not the median - this is the whole point."""
    d = decide(measurement(speedup=speedup(1.20, 1.01, 1.40)), CFG)
    assert not d.accepted and d.rule == "performance"


def test_failed_measurement_is_an_error_not_a_slowdown():
    d = decide(Measurement.failure("t", "measurement exceeded the 600s timeout"), CFG)
    assert not d.accepted and d.rule == "error"
    assert "timeout" in d.reason


def test_correct_but_untimed_is_an_error():
    d = decide(measurement(speedup=None), CFG)
    assert not d.accepted and d.rule == "error"
