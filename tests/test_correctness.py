"""The correctness gate, exercised with plain NumPy - no JAX, no device."""

from __future__ import annotations

import numpy as np
import pytest

from halo.config import CorrectnessConfig
from halo.harness import correctness
from halo.tasks.base import Case

IDENTITY = dict(jit=lambda f: f, to_device=lambda args: args)


class FakeTask:
    """A one-line workload with a float64 oracle and one adversarial case."""

    @staticmethod
    def make_inputs(rng):
        return (rng.standard_normal((16, 8), dtype=np.float32),)

    @staticmethod
    def reference(x):
        return np.exp(x.astype(np.float64))

    @staticmethod
    def correctness_cases(rng):
        return [Case("huge", (np.full((16, 8), 90.0, np.float32),), atol=1e30)]


def check(fn, cfg=CorrectnessConfig()):
    return correctness.check(fn, FakeTask, cfg, **IDENTITY)


def test_correct_implementation_passes():
    report = check(lambda x: np.exp(x.astype(np.float64)))
    assert report.passed
    assert [c.name for c in report.cases] == ["huge", "random_0", "random_1", "random_2"]


def test_wrong_answer_is_caught():
    report = check(lambda x: np.exp(x.astype(np.float64)) + 1.0)
    assert not report.passed
    assert "exceeds tolerance" in (report.worst.detail or "")


def test_non_finite_output_is_caught():
    # float32 exp overflows at 90; the oracle in float64 does not. The overflow is
    # the point of the test, so silence the warning it necessarily raises.
    with np.errstate(over="ignore"):
        report = check(lambda x: np.exp(x).astype(np.float64))
    assert not report.passed
    failed = [c for c in report.cases if not c.passed]
    assert failed and "non-finite" in failed[0].detail


def test_shape_mismatch_is_caught():
    report = check(lambda x: np.exp(x.astype(np.float64))[:, :1])
    assert not report.passed
    assert "shape mismatch" in (report.worst.detail or "")


def test_raising_candidate_is_reported_not_propagated():
    def boom(x):
        raise ValueError("bad kernel")

    report = check(boom)
    assert not report.passed
    assert "ValueError: bad kernel" in (report.worst.detail or "")


def test_per_case_tolerance_override_is_honoured():
    """The 'huge' case carries atol=1e30; without it a correct float64 answer,
    whose values are ~1e39, could not satisfy the default 1e-6."""
    report = check(lambda x: np.exp(x.astype(np.float64)))
    huge = next(c for c in report.cases if c.name == "huge")
    assert huge.passed


def test_relative_error_is_scaled_to_the_array_not_to_tiny_elements():
    expected = np.array([[1000.0, 1e-12]])
    got = expected + np.array([[0.0, 1e-9]])
    passed, _, max_rel, _ = correctness._compare(got, expected, rtol=1e-5, atol=1e-6)
    assert passed
    assert max_rel < 1e-3  # not 1000x, which per-element scaling would report
