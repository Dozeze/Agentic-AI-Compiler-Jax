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


def test_block_order_is_balanced_and_seeded():
    order = benchmark.block_order(20, seed=0)
    assert order.count("AB") == order.count("BA") == 10
    assert order == benchmark.block_order(20, seed=0)
    assert order != benchmark.block_order(20, seed=1)
    assert benchmark.block_order(5, seed=0).count("AB") == 3


def test_measure_paired_records_raw_blocks_in_the_declared_order():
    clock, log = FakeClock(), []
    a, b = clock.fn(2_000_000, log, "a"), clock.fn(1_000_000, log, "b")
    session = benchmark.measure_paired(a, b, (), SMALL, repeats=3, clock=clock)

    assert session.repeats == 3 and len(session.blocks) == SMALL.n_blocks
    timed = log[2 * SMALL.warmup_calls:]                       # warm-up runs both sides first
    expected = []
    for order in benchmark.block_order(SMALL.n_blocks, SMALL.order_seed):
        expected += [order[0].lower()] * 3 + [order[1].lower()] * 3
    assert timed == expected
    for block in session.blocks:
        assert (block.a_ns, block.b_ns) == (6_000_000, 3_000_000)
        assert (block.a_s, block.b_s) == (0.002, 0.001)
        first, second = (block.a_start_ns, block.b_start_ns) if block.order == "AB" \
            else (block.b_start_ns, block.a_start_ns)
        assert second > first


def test_measure_paired_calibrates_from_a_only():
    clock = FakeClock()
    calls = {"a": 0, "b": 0}

    def counted(name, cost):
        def run(*_):
            calls[name] += 1
            clock.now += cost
        return run

    session = benchmark.measure_paired(counted("a", 1_000_000), counted("b", 7), (), SMALL, clock=clock)
    assert session.repeats == 20                                # from A's 1 ms per call
    assert calls["b"] == SMALL.warmup_calls + SMALL.n_blocks * 20   # B never ran during calibration


def test_session_json_roundtrip():
    clock = FakeClock()
    session = benchmark.measure_paired(clock.fn(10), clock.fn(20), (), SMALL, repeats=2, clock=clock)
    assert benchmark.PairedSession.from_json(session.to_json()) == session
