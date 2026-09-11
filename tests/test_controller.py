"""The loop itself: the device guard with stubbed measurements, and the full
pipeline against the real harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from halo import controller
from halo.agent.fake import (
    FAST_ATTENTION,
    UNSTABLE_ATTENTION,
    EchoAgent,
    ScriptedAgent,
)
from halo.config import CorrectnessConfig, RunConfig, TimingConfig
from halo.store import RunStore
from halo.types import (
    CorrectnessCase,
    CorrectnessReport,
    DeviceInfo,
    Measurement,
    SpeedupEstimate,
    TimingStats,
)

#: Small enough to keep the suite usable, large enough to still be a measurement.
FAST = RunConfig(
    task="naive_attention",
    steps=1,
    timing=TimingConfig(warmup=2, rounds=6, bootstrap_samples=300),
    correctness=CorrectnessConfig(n_random_cases=1),
)


def _device(kind: str) -> DeviceInfo:
    return DeviceInfo(
        platform="cpu" if kind == "cpu" else "gpu",
        device_kind=kind,
        device_count=1,
        jax_version="0.11.1",
        jaxlib_version="0.11.1",
    )


def test_measurements_from_a_different_device_are_refused(tmp_path, monkeypatch):
    """A GPU timing is not 'faster' than a CPU baseline; it is incomparable."""
    stats = TimingStats(samples_ms=(1.0,), median_ms=1.0, iqr_ms=0.0, min_ms=1.0)
    ok = CorrectnessReport(True, 1e-5, 1e-6, (CorrectnessCase("r", True, 0.0, 0.0),))
    calls = {"n": 0}

    def fake_measure(spec, baseline_path, candidate_path, cfg, **kw):
        calls["n"] += 1
        return Measurement(
            task=spec.name,
            ok=True,
            device=_device("cpu" if calls["n"] == 1 else "NVIDIA RTX 4070"),
            correctness=ok,
            baseline=stats,
            candidate=stats,
            speedup=SpeedupEstimate(9.0, 8.5, 9.5, 0.95, 6),
        )

    monkeypatch.setattr(controller.runner, "measure", fake_measure)
    store = RunStore(tmp_path, "guard")
    result = controller.run(FAST, ScriptedAgent(FAST_ATTENTION), store)

    decision = result.attempts[1].decision
    assert not decision.accepted, "a 9x speedup from another machine must not be accepted"
    assert decision.rule == "error"
    assert "not comparable" in decision.reason


def test_agent_failure_is_recorded_rather_than_crashing_the_run(tmp_path, monkeypatch):
    class Broken:
        name = "broken"

        def propose(self, context):
            raise RuntimeError("model returned no parseable proposal")

    monkeypatch.setattr(
        controller,
        "baseline_attempt",
        lambda cfg, spec, store: controller.Attempt(
            index=0,
            source=spec.seed_source,
            measurement=Measurement(task=spec.name, ok=True, device=_device("cpu")),
            decision=controller.Decision(True, "accepted", "baseline"),
        ),
    )
    store = RunStore(tmp_path, "broken")
    result = controller.run(FAST, Broken(), store)
    assert result.attempts[1].decision.rule == "error"
    assert not result.improved


@pytest.mark.slow
def test_pipeline_accepts_a_real_speedup(tmp_path):
    store = RunStore(tmp_path, "fast")
    result = controller.run(FAST, ScriptedAgent(FAST_ATTENTION, "fast"), store)
    assert result.improved
    assert result.best.index == 1
    assert result.best.measurement.speedup.ci_low > 1.05
    # Artifacts a reader of the report can re-derive the claim from.
    for name in ("candidate.py", "measurement.json", "decision.json", "hlo.txt"):
        assert (store.attempt_dir(1) / name).exists()


@pytest.mark.slow
def test_pipeline_rejects_a_faster_but_unstable_rewrite(tmp_path):
    store = RunStore(tmp_path, "unstable")
    result = controller.run(FAST, ScriptedAgent(UNSTABLE_ATTENTION, "unstable"), store)
    assert not result.improved
    decision = result.attempts[1].decision
    assert decision.rule == "correctness"
    assert "non-finite" in decision.reason


@pytest.mark.slow
def test_unchanged_source_does_not_clear_the_noise_threshold(tmp_path):
    """The control condition. If this ever passes, the threshold is too low."""
    store = RunStore(tmp_path, "echo")
    result = controller.run(FAST, EchoAgent(), store)
    assert not result.improved
    assert result.attempts[1].decision.rule == "performance"


@pytest.mark.slow
def test_every_pipeline_level_reaches_the_measurement(tmp_path):
    """The wiring most likely to break silently: the buffer report only exists as
    a side effect of an XLA_FLAGS the runner sets on the subprocess, and a missing
    report degrades to None rather than failing."""
    store = RunStore(tmp_path, "levels")
    result = controller.run(FAST, ScriptedAgent(FAST_ATTENTION, "fast"), store)
    seed = result.attempts[0].measurement.baseline_program
    best = result.best.measurement.candidate_program

    for program in (seed, best):
        assert program.jaxpr.total_ops > 0
        assert program.stablehlo.total_ops > 0
        assert program.hlo.instruction_count > 0
        assert program.buffers is not None, "XLA dump not reaching the worker"

    # The seed's Python head-loop leaves the per-head score matrices live; the
    # batched rewrite leaves only the inputs and the output.
    assert seed.buffers.shape_counts["f32[4,256,256]"] > 8
    assert "f32[4,256,256]" not in best.buffers.shape_counts
    assert best.buffers.total_bytes < seed.buffers.total_bytes


class Sequence:
    """An agent that proposes a fixed sequence of sources, one per step."""

    name = "sequence"

    def __init__(self, *sources: str) -> None:
        self._sources = iter(sources)

    def propose(self, context):
        from halo.types import Proposal
        return Proposal(source=next(self._sources))


def test_later_steps_are_judged_against_the_current_best_not_the_seed(tmp_path, monkeypatch):
    """The walk-backwards bug. Step 1 gets 3x. Step 2 is 1.5x over the seed - a
    clear win against the original, but half as fast as what we already have.
    Judged against the seed it would be accepted and the run would end slower
    than it was after step 1."""
    stats = TimingStats(samples_ms=(1.0,), median_ms=1.0, iqr_ms=0.0, min_ms=1.0)
    ok = CorrectnessReport(True, 1e-5, 1e-6, (CorrectnessCase("r", True, 0.0, 0.0),))
    speed_vs_seed = {"seed": 1.0, "fast": 3.0, "medium": 1.5}
    baselines_used: list[str] = []

    def fake_measure(spec, baseline_path, candidate_path, cfg, **kw):
        base = speed_vs_seed[Path(baseline_path).read_text().strip()]
        cand = speed_vs_seed[Path(candidate_path).read_text().strip()]
        baselines_used.append(Path(baseline_path).read_text().strip())
        ratio = cand / base
        return Measurement(
            task=spec.name, ok=True, device=_device("cpu"), correctness=ok,
            baseline=stats, candidate=stats,
            speedup=SpeedupEstimate(ratio, ratio * 0.98, ratio * 1.02, 0.95, 6),
        )

    monkeypatch.setattr(controller.runner, "measure", fake_measure)
    monkeypatch.setattr(
        controller.base, "load_spec",
        lambda name: type("Spec", (), {
            "name": name, "seed_source": "seed",
            "directory": tmp_path, "task_path": tmp_path / "task.py",
            "load": lambda self: type("T", (), {"DESCRIPTION": "d"})(),
        })(),
    )
    monkeypatch.setattr(controller.base, "reference_excerpt", lambda spec: "")
    (tmp_path / "candidate.py").write_text("seed")

    cfg = RunConfig(task="stub", steps=2)
    result = controller.run(cfg, Sequence("fast", "medium"), RunStore(tmp_path, "r"))

    assert result.attempts[1].decision.accepted
    assert not result.attempts[2].decision.accepted, (
        "1.5x over the seed is 0.5x over the current best and must be rejected"
    )
    assert result.best.index == 1
    assert baselines_used[-1] == "fast", "step 2 must be measured against step 1's code"
    assert result.overall.speedup.ratio == pytest.approx(3.0)
