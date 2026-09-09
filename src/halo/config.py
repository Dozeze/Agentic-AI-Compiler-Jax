"""Run configuration. Everything tunable lives here so a run is reproducible from
its ``config.toml`` alone.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


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
    steps: int = 1
    agent: str = "vertex"
    #: How much the agent is shown: code | timing | hlo | full. See halo.agent.context.
    context: str = "full"
    seed: int = 0
    timeout_s: float = 600.0
    timing: TimingConfig = field(default_factory=TimingConfig)
    correctness: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    accept: AcceptConfig = field(default_factory=AcceptConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        sections = {
            "timing": TimingConfig,
            "correctness": CorrectnessConfig,
            "accept": AcceptConfig,
            "llm": LLMConfig,
        }
        kwargs: dict[str, Any] = {
            k: v for k, v in d.items() if k not in sections
        }
        for name, klass in sections.items():
            if name in d:
                kwargs[name] = klass(**d[name])
        return cls(**kwargs)

    def with_overrides(self, **kw: Any) -> RunConfig:
        return replace(self, **{k: v for k, v in kw.items() if v is not None})
