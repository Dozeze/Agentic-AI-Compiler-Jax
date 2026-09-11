"""Run configuration. Everything tunable lives here so a run is reproducible from
its ``config.toml`` alone.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from halo.serde import decode


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 10
    rounds: int = 30
    bootstrap_samples: int = 2000
    confidence: float = 0.95


@dataclass(frozen=True)
class CorrectnessConfig:
    rtol: float = 1e-5
    atol: float = 1e-6
    n_random_cases: int = 3


@dataclass(frozen=True)
class AcceptConfig:
    """A candidate is accepted only if the *lower* bound of the speedup CI clears
    ``min_speedup``. Anything less is indistinguishable from measurement noise."""

    min_speedup: float = 1.05


@dataclass(frozen=True)
class BudgetConfig:
    """What a tool-using agent may spend before the harness stops it.

    ``max_evaluations`` is the comparable quantity across agents: a single-shot
    run with ``steps=k`` and an agentic run with ``max_evaluations=k`` have taken
    the same number of measurements. ``patience`` ends a run after that many
    consecutive rejected evaluations; correctness-only checks never count.
    """

    max_evaluations: int = 6
    max_tool_calls: int = 24
    max_cost_usd: float = 0.25
    patience: int = 3


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "vertex"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.2
    max_output_tokens: int = 16384
    location: str = "global"
    project: str | None = None  # falls back to $GOOGLE_CLOUD_PROJECT


@dataclass(frozen=True)
class RunConfig:
    task: str
    #: Measurements a single-shot agent gets; a tool-using agent is bounded by
    #: ``budget`` instead.
    steps: int = 1
    #: ``single-shot`` (one proposal per step, the controller drives) or
    #: ``agentic`` (the model drives through tools); or a scripted agent.
    agent: str = "single-shot"
    #: What the agent is shown and how the job is framed. See halo.agent.context.
    #: Defaults to `algorithmic` on the evidence in results/framing-n5.md: same
    #: data as `full`, 64% -> 95% of the available speedup, no extra false positives.
    context: str = "algorithmic"
    seed: int = 0
    timeout_s: float = 600.0
    timing: TimingConfig = field(default_factory=TimingConfig)
    correctness: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    accept: AcceptConfig = field(default_factory=AcceptConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        return decode(cls, d)

    def with_overrides(self, **kw: Any) -> RunConfig:
        return replace(self, **{k: v for k, v in kw.items() if v is not None})
