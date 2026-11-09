"""Paired-block timing protocol (AIC-005).

The protocol logic is tested with a deterministic fake clock: no test here measures real time, so the
numbers are exact and the tests make no speed claims about real code (D-06 rule 5).
"""
import pytest

import benchmark
from contracts import MeasurementProtocol


class FakeClock:
    """A monotonic nanosecond clock that only advances when a fake function 'runs'."""

    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def fn(self, cost_ns, log=None, name=None):
        """A callable that costs `cost_ns` (an int, or a callable returning the next cost) per call."""
        def run(*_args):
            self.now += cost_ns() if callable(cost_ns) else cost_ns
            if log is not None:
                log.append(name)
            return 0.0
        return run


SMALL = MeasurementProtocol(warmup_calls=2, n_blocks=6, bootstrap_samples=500, aa_sessions=5)


def test_calibrate_repeats_targets_the_block_duration():
    clock = FakeClock()
    one_ms = clock.fn(1_000_000)
    # 1 ms per call: doubling stops at 16 calls (>= 10 ms), then 20 ms / 1 ms = 20 calls.
    assert benchmark.calibrate_repeats(one_ms, (), SMALL, clock=clock) == 20


def test_calibrate_repeats_respects_the_cap():
    clock = FakeClock()
    tiny = clock.fn(1)                       # 1 ns per call would need 20 million repeats
    assert benchmark.calibrate_repeats(tiny, (), SMALL, clock=clock) == SMALL.max_repeats


def test_calibrate_repeats_slow_call_gets_one_repeat():
    clock = FakeClock()
    slow = clock.fn(80_000_000)              # 80 ms per call, longer than max_block_s
    assert benchmark.calibrate_repeats(slow, (), SMALL, clock=clock) == 1


def test_warm_up_calls_are_synchronised_and_counted():
    clock, log = FakeClock(), []
    benchmark.warm_up(clock.fn(5, log, "a"), (), 5)
    assert log == ["a"] * 5
