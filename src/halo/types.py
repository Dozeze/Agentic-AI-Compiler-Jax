"""Frozen records exchanged between the harness, the agent and the controller.

Everything here is JSON-serialisable. These objects *are* the run artifacts: the
report's tables and figures come from the files written under ``runs/``, not from
console output. The ``from_dict`` methods exist because measurements cross a
subprocess boundary (see :mod:`halo.harness.runner`).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Literal


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialise any dataclass in this module (tuples become JSON arrays)."""
    return json.dumps(dataclasses.asdict(obj), indent=indent, default=str)


def _opt(cls: type, d: dict | None):
    return cls.from_dict(d) if d is not None else None


@dataclass(frozen=True)
class DeviceInfo:
    """Identifies the machine a measurement was taken on.

    Timings are only ever comparable within one fingerprint; the controller
    refuses to compare across them.
    """

    platform: str
    device_kind: str
    device_count: int
    jax_version: str
    jaxlib_version: str

    @property
    def fingerprint(self) -> str:
        return f"{self.platform}/{self.device_kind}/x{self.device_count}/jax{self.jax_version}"

    @classmethod
    def from_dict(cls, d: dict) -> DeviceInfo:
        return cls(**d)


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> CorrectnessCase:
        return cls(**d)


@dataclass(frozen=True)
class CorrectnessReport:
    passed: bool
    rtol: float
    atol: float
    cases: tuple[CorrectnessCase, ...] = ()

    @property
    def worst(self) -> CorrectnessCase | None:
        failed = [c for c in self.cases if not c.passed]
        pool = failed or list(self.cases)
        return max(pool, key=lambda c: c.max_rel_err) if pool else None

    def summary(self) -> str:
        w = self.worst
        if w is None:
            return "no cases run"
        verdict = "pass" if self.passed else f"FAIL on case '{w.name}'"
        if w.detail:
            return f"{verdict}: {w.detail}"
        return (
            f"{verdict} (worst max_abs_err={w.max_abs_err:.3e}, "
            f"max_rel_err={w.max_rel_err:.3e}, rtol={self.rtol:g}, atol={self.atol:g})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> CorrectnessReport:
        return cls(
            passed=d["passed"],
            rtol=d["rtol"],
            atol=d["atol"],
            cases=tuple(CorrectnessCase.from_dict(c) for c in d.get("cases", ())),
        )


@dataclass(frozen=True)
class HloSummary:
    """Compact view of the *optimized* HLO. The agent never sees the raw dump."""

    instruction_count: int
    fusion_count: int
    op_counts: dict[str, int]
    flops: float | None = None
    bytes_accessed: float | None = None
    temp_bytes: int | None = None
    output_bytes: int | None = None
    argument_bytes: int | None = None

    @property
    def arithmetic_intensity(self) -> float | None:
        """FLOPs per byte moved - the x-axis of a roofline plot."""
        if not self.flops or not self.bytes_accessed:
            return None
        return self.flops / self.bytes_accessed

    def top_ops(self, n: int = 12) -> list[tuple[str, int]]:
        return sorted(self.op_counts.items(), key=lambda kv: -kv[1])[:n]

    @classmethod
    def from_dict(cls, d: dict) -> HloSummary:
        return cls(**d)


@dataclass(frozen=True)
class TimingStats:
    samples_ms: tuple[float, ...]
    median_ms: float
    iqr_ms: float
    min_ms: float

    @classmethod
    def from_dict(cls, d: dict) -> TimingStats:
        return cls(
            samples_ms=tuple(d["samples_ms"]),
            median_ms=d["median_ms"],
            iqr_ms=d["iqr_ms"],
            min_ms=d["min_ms"],
        )


@dataclass(frozen=True)
class SpeedupEstimate:
    """Bootstrap estimate of baseline/candidate, from interleaved A/B rounds."""

    ratio: float
    ci_low: float
    ci_high: float
    confidence: float
    n_rounds: int

    def describe(self) -> str:
        return (
            f"{self.ratio:.3f}x (CI{self.confidence:.0%} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}], n={self.n_rounds})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> SpeedupEstimate:
        return cls(**d)


@dataclass(frozen=True)
class Measurement:
    """One subprocess run: baseline and candidate measured back to back, one device.

    ``baseline`` is always the task's seed implementation, re-measured in the same
    session as the candidate so the pair is directly comparable.
    """

    task: str
    ok: bool
    error: str | None = None
    device: DeviceInfo | None = None
    correctness: CorrectnessReport | None = None
    baseline: TimingStats | None = None
    candidate: TimingStats | None = None
    speedup: SpeedupEstimate | None = None
    compile_s: float | None = None
    baseline_compile_s: float | None = None
    hlo: HloSummary | None = None
    baseline_hlo: HloSummary | None = None

    @classmethod
    def failure(cls, task: str, error: str) -> Measurement:
        return cls(task=task, ok=False, error=error)

    @classmethod
    def from_dict(cls, d: dict) -> Measurement:
        return cls(
            task=d["task"],
            ok=d["ok"],
            error=d.get("error"),
            device=_opt(DeviceInfo, d.get("device")),
            correctness=_opt(CorrectnessReport, d.get("correctness")),
            baseline=_opt(TimingStats, d.get("baseline")),
            candidate=_opt(TimingStats, d.get("candidate")),
            speedup=_opt(SpeedupEstimate, d.get("speedup")),
            compile_s=d.get("compile_s"),
            baseline_compile_s=d.get("baseline_compile_s"),
            hlo=_opt(HloSummary, d.get("hlo")),
            baseline_hlo=_opt(HloSummary, d.get("baseline_hlo")),
        )


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float

    @classmethod
    def from_dict(cls, d: dict) -> Usage:
        return cls(**d)


@dataclass(frozen=True)
class Proposal:
    """What an agent returns: a complete replacement for ``candidate.py``."""

    source: str
    analysis: str = ""
    hypothesis: str = ""
    expected_speedup: float | None = None
    risk: str | None = None
    usage: Usage | None = None
    prompt: str | None = None
    raw_response: str | None = None


DecisionRule = Literal["accepted", "correctness", "performance", "error"]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    rule: DecisionRule
    reason: str


@dataclass(frozen=True)
class Attempt:
    """Attempt 0 is the baseline itself; later attempts carry a proposal."""

    index: int
    source: str
    measurement: Measurement
    decision: Decision
    proposal: Proposal | None = None
