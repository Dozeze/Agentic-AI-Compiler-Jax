"""Paired-block timing protocol (AIC-005).

The protocol logic is tested with a deterministic fake clock: no test here measures real time, so the
numbers are exact and the tests make no speed claims about real code (D-06 rule 5).
"""
import math

import numpy as np
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


def test_summarize_reports_every_statistic():
    s = benchmark.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (s.n, s.median_s, s.mean_s, s.min_s, s.max_s) == (5, 3.0, 3.0, 1.0, 5.0)
    assert (s.q1_s, s.q3_s) == (2.0, 4.0)
    assert s.iqr_over_median == pytest.approx(2.0 / 3.0)
    assert math.isclose(s.std_s, 1.5811388, rel_tol=1e-6)


def test_paired_speedup_is_the_geometric_mean_of_block_ratios():
    est = benchmark.paired_speedup([2.0, 8.0], [1.0, 2.0], SMALL)    # ratios 2 and 4
    assert math.isclose(est.speedup, 8 ** 0.5)
    assert est.n_blocks == 2 and est.confidence == SMALL.confidence


def test_paired_speedup_interval_is_seeded_and_contains_the_point():
    a = [1.0 + 0.01 * (i % 5) for i in range(20)]
    b = [0.5 + 0.02 * (i % 3) for i in range(20)]
    first, again = benchmark.paired_speedup(a, b, SMALL), benchmark.paired_speedup(a, b, SMALL)
    assert first == again
    bounds = [first.ci_low, first.speedup, first.ci_high]
    assert bounds == sorted(bounds)                   # the interval contains the point estimate


def test_identical_sides_give_a_degenerate_interval_at_one():
    est = benchmark.paired_speedup([0.3] * 20, [0.3] * 20, SMALL)
    assert (est.speedup, est.ci_low, est.ci_high) == (1.0, 1.0, 1.0)


def _session(a_ns, b_ns, protocol=SMALL):
    """A PairedSession with the given per-block elapsed nanoseconds (repeats = 1)."""
    from contracts import protocol_hash
    blocks = tuple(benchmark.Block(i, "AB", 1, a, b, 2 * i, 2 * i + 1)
                   for i, (a, b) in enumerate(zip(a_ns, b_ns)))
    return benchmark.PairedSession(protocol_hash(protocol), 1, blocks)


def test_quiet_session_is_stable():
    s = benchmark.stability(_session([100] * 6, [50] * 6), SMALL)
    assert s.stable and s.reasons == ()
    assert (s.iqr_over_median_a, s.iqr_over_median_b, s.baseline_drift) == (0.0, 0.0, 0.0)


def test_noisy_side_is_unstable():
    s = benchmark.stability(_session([100] * 6, [50, 80, 50, 80, 50, 80]), SMALL)
    assert not s.stable
    assert any(reason.startswith("B IQR/median") for reason in s.reasons)


def test_drifting_baseline_is_unstable():
    s = benchmark.stability(_session([100, 100, 100, 110, 110, 110], [50] * 6), SMALL)
    assert not s.stable
    assert math.isclose(s.baseline_drift, 0.1)
    assert any("drift" in reason for reason in s.reasons)


def test_analyze_session_combines_summary_speedup_and_stability():
    analysis = benchmark.analyze_session(_session([100] * 6, [50] * 6), SMALL)
    assert math.isclose(analysis.speedup.speedup, 2.0)
    assert analysis.stability.stable
    assert analysis.to_json()["a"]["n"] == 6


def test_analyze_session_refuses_a_different_protocol():
    import dataclasses
    other = dataclasses.replace(SMALL, n_blocks=8)
    with pytest.raises(ValueError, match="different protocol"):
        benchmark.analyze_session(_session([100] * 6, [50] * 6), other)


def _aa(scale):
    """An A/A session whose second side runs `scale` times as long as the first."""
    return _session([1_000_000] * 6, [round(1_000_000 * scale)] * 6)


def test_aa_threshold_is_the_95th_percentile_of_aa_differences():
    sessions = [_aa(s) for s in (1.00, 1.01, 0.98, 1.00, 1.02)]
    delta, diffs = benchmark.aa_threshold(sessions, SMALL, timer_resolution_s=1e-9)
    assert len(diffs) == 5
    assert math.isclose(delta, float(np.percentile(diffs, 95)))


def test_aa_threshold_has_a_timer_resolution_floor():
    sessions = [_aa(1.0)] * 5                       # identical sides: zero A/A difference
    delta, _ = benchmark.aa_threshold(sessions, SMALL, timer_resolution_s=1e-6)
    assert math.isclose(delta, 1e-6 / 1e-3)         # 1 us resolution over 1 ms blocks


def test_aa_threshold_needs_enough_stable_sessions():
    with pytest.raises(benchmark.CalibrationError, match="need 5"):
        benchmark.aa_threshold([_aa(1.0)] * 4, SMALL, 1e-9)
    noisy = _session([1_000_000] * 6, [1_000_000, 2_000_000] * 3)
    with pytest.raises(benchmark.CalibrationError, match="unstable"):
        benchmark.aa_threshold([_aa(1.0)] * 4 + [noisy], SMALL, 1e-9)


def _calibration(**overrides):
    from contracts import protocol_hash
    fields = dict(schema_version=benchmark.CALIBRATION_SCHEMA, label="test", workload_id="softmax",
                  shape=(128, 1024), dtype="float32", protocol_hash=protocol_hash(SMALL),
                  fingerprint="cpu|cpu|test", repeats=20, delta=0.03,
                  aa_relative_differences=(0.01, 0.02, 0.0, 0.01, 0.03), timer_resolution_s=1e-9,
                  created_utc="2026-11-09T00:00:00Z")
    return benchmark.Calibration(**{**fields, **overrides})


def test_calibration_roundtrip_and_check(tmp_path):
    cal = _calibration()
    cal.save(tmp_path / "cal.json")
    loaded = benchmark.Calibration.load(tmp_path / "cal.json")
    assert loaded == cal
    loaded.check(workload_id="softmax", shape=(128, 1024), dtype="float32",
                 protocol_hash=cal.protocol_hash, fingerprint="cpu|cpu|test")


@pytest.mark.parametrize("field, value", [("workload_id", "example_add"), ("shape", (32, 128)),
                                          ("fingerprint", "gpu|NVIDIA|test"), ("protocol_hash", "x")])
def test_calibration_refuses_another_measurement(field, value):
    cal = _calibration()
    wanted = dict(workload_id="softmax", shape=(128, 1024), dtype="float32",
                  protocol_hash=cal.protocol_hash, fingerprint="cpu|cpu|test")
    wanted[field] = value
    with pytest.raises(benchmark.CalibrationError, match=field):
        cal.check(**wanted)


def test_compile_timed_returns_a_working_executable_and_stage_times():
    import jax.numpy as jnp
    x = jnp.arange(8, dtype=jnp.float32)
    compiled = benchmark.compile_timed(lambda v: v * 2.0, (x,))
    assert compiled.lower_s >= 0 and compiled.compile_s >= 0
    assert np.array_equal(np.asarray(compiled.executable(x)), np.arange(8, dtype=np.float32) * 2)
    assert "multiply" in compiled.lowered.as_text()


def test_measure_paired_runs_real_executables():
    """Mechanics only on a real backend: synchronised calls, raw blocks, no speed claim."""
    import jax.numpy as jnp
    x = jnp.ones((64, 64), jnp.float32)
    a = benchmark.compile_timed(lambda v: v @ v, (x,)).executable
    b = benchmark.compile_timed(lambda v: jnp.dot(v, v), (x,)).executable
    session = benchmark.measure_paired(a, b, (x,), SMALL, repeats=2)
    assert len(session.blocks) == SMALL.n_blocks
    assert all(block.a_ns > 0 and block.b_ns > 0 for block in session.blocks)
