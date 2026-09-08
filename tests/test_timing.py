from __future__ import annotations

import numpy as np
import pytest

from halo.config import TimingConfig
from halo.harness import timing

CFG = TimingConfig(warmup=1, rounds=8, bootstrap_samples=400)


def test_stats_are_order_independent():
    stats = timing._stats([3.0, 1.0, 2.0, 4.0])
    assert stats.median_ms == 2.5
    assert stats.min_ms == 1.0
    assert stats.iqr_ms == pytest.approx(1.5)


def test_bootstrap_detects_a_real_two_fold_speedup():
    rng = np.random.default_rng(0)
    base = list(rng.normal(10.0, 0.1, 30))
    cand = list(rng.normal(5.0, 0.05, 30))
    est = timing.bootstrap_speedup(base, cand, CFG)
    assert est.ratio == pytest.approx(2.0, rel=0.05)
    assert est.ci_low > 1.9


def test_bootstrap_ci_straddles_one_when_nothing_changed():
    """The control condition: identical distributions must not look like a win."""
    rng = np.random.default_rng(1)
    base = list(rng.normal(10.0, 0.5, 30))
    cand = list(rng.normal(10.0, 0.5, 30))
    est = timing.bootstrap_speedup(base, cand, CFG)
    assert est.ci_low < 1.0 < est.ci_high


def test_measure_ab_blocks_and_interleaves():
    """Both callables run the same number of times, and every call is blocked on."""
    calls = {"a": 0, "b": 0, "blocked": 0}

    def make(key, cost):
        def fn():
            calls[key] += 1
            sum(range(cost))
            return key
        return lambda *args: fn()

    def block(out):
        calls["blocked"] += 1

    base, cand, est = timing.measure_ab(
        make("a", 4000), make("b", 1000), (), CFG, block=block
    )
    assert calls["a"] == calls["b"]
    assert len(base.samples_ms) == len(cand.samples_ms) == CFG.rounds
    assert calls["blocked"] > CFG.rounds
    assert est.ratio > 1.0
