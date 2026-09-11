"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from halo import session as session_mod
from halo.types import (
    CorrectnessCase,
    CorrectnessReport,
    DeviceInfo,
    Measurement,
    SpeedupEstimate,
    TimingStats,
)

DEVICE = DeviceInfo("cpu", "stub", 1, "0.11.1", "0.11.1")
STATS = TimingStats(samples_ms=(1.0,), median_ms=1.0, iqr_ms=0.0, min_ms=1.0)
PASS = CorrectnessReport(True, 1e-5, 1e-6, (CorrectnessCase("r", True, 0.0, 0.0),))
FAIL = CorrectnessReport(False, 1e-5, 1e-6, (CorrectnessCase("r", False, 1.0, 1.0),))

#: Speed of each source relative to the seed; "wrong" fails correctness.
SPEED = {"seed": 1.0, "fast": 3.0, "faster": 4.0, "medium": 1.5, "same": 1.0}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """A session whose measurements are looked up rather than taken.

    The task is a stub whose seed source is the string ``seed``; a candidate's
    speed is ``SPEED[source]``; ``wrong`` fails correctness. Returns the list of
    measurement calls made, each with the baseline and candidate text.
    """
    calls: list[dict] = []

    def fake_measure(spec, baseline_path, candidate_path, cfg, **kw):
        base = Path(baseline_path).read_text().strip()
        cand = Path(candidate_path).read_text().strip()
        calls.append({"baseline": base, "candidate": cand, **kw})
        if cand == "wrong":
            return Measurement(task=spec.name, ok=True, device=DEVICE, correctness=FAIL)
        if kw.get("correctness_only"):
            return Measurement(task=spec.name, ok=True, device=DEVICE, correctness=PASS)
        ratio = SPEED[cand] / SPEED[base]
        return Measurement(
            task=spec.name, ok=True, device=DEVICE, correctness=PASS,
            baseline=STATS, candidate=STATS,
            speedup=SpeedupEstimate(ratio, ratio * 0.98, ratio * 1.02, 0.95, 6),
        )

    monkeypatch.setattr(session_mod.runner, "measure", fake_measure)
    monkeypatch.setattr(
        session_mod.base, "load_spec",
        lambda name: type("Spec", (), {
            "name": name, "seed_source": "seed",
            "directory": tmp_path, "task_path": tmp_path / "task.py",
            "load": lambda self: type("T", (), {"DESCRIPTION": "d"})(),
        })(),
    )
    monkeypatch.setattr(session_mod.base, "reference_excerpt", lambda spec: "")
    (tmp_path / "candidate.py").write_text("seed")
    return calls
