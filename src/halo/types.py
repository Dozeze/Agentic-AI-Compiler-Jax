"""Frozen records exchanged between the harness, the agent and the controller.

Everything here is JSON-serialisable. These objects *are* the run artifacts: the
report's tables and figures come from the files written under ``runs/``, not from
console output. :func:`decode` rebuilds them on the far side of the measurement
subprocess boundary (see :mod:`halo.harness.runner`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from halo.serde import decode, to_json  # re-exported: callers import both from here

__all__ = ["decode", "to_json"]


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



@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str | None = None



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



@dataclass(frozen=True)
class IRStats:
    """Operation counts at one level of the lowering pipeline.

    Comparing these levels is the point: the same program as 139 jaxpr equations,
    then as StableHLO, then as optimized HLO, shows what the compiler managed to
    do on its own - and therefore where a source rewrite still has leverage.
    """

    total_ops: int
    op_counts: dict[str, int]

    def top_ops(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(self.op_counts.items(), key=lambda kv: -kv[1])[:n]



@dataclass(frozen=True)
class BufferEntry:
    """One slot in XLA's buffer assignment. Several values may share a slot."""

    size_bytes: int
    n_values: int
    shapes: tuple[str, ...]



@dataclass(frozen=True)
class BufferReport:
    """What XLA actually materialised, by shape - not just how many bytes."""

    total_bytes: int
    entries: tuple[BufferEntry, ...] = ()
    shape_counts: dict[str, int] = field(default_factory=dict)

    def top_shapes(self, n: int = 6) -> list[tuple[str, int]]:
        return list(self.shape_counts.items())[:n]



@dataclass(frozen=True)
class ProgramSummary:
    """Everything the compiler will tell us about one program, at every level.

    Grouped rather than spread across :class:`Measurement` so that adding a level
    does not add another pair of parallel fields.
    """

    jaxpr: IRStats | None = None
    stablehlo: IRStats | None = None
    hlo: HloSummary | None = None
    buffers: BufferReport | None = None



@dataclass(frozen=True)
class TimingStats:
    samples_ms: tuple[float, ...]
    median_ms: float
    iqr_ms: float
    min_ms: float



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
    candidate_program: ProgramSummary | None = None
    baseline_program: ProgramSummary | None = None

    @classmethod
    def failure(cls, task: str, error: str) -> Measurement:
        return cls(task=task, ok=False, error=error)



@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float



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
