"""What an agent is, and what it gets to see.

In the MVP an agent is a *pure function* ``Context -> Proposal``: it does not
compile, benchmark, or touch the filesystem. The controller owns every side
effect. That keeps the loop trivially loggable and replayable, and it means a
scripted agent is a drop-in substitute for a language model in tests.

The autonomous version of this project replaces ``propose`` with something that
calls tools mid-thought. That is a change *behind* this protocol, not to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from halo.types import Measurement, Proposal


@dataclass(frozen=True)
class AttemptSummary:
    """One earlier attempt, as the agent sees it: what changed and what happened."""

    index: int
    diff: str
    accepted: bool
    rule: str
    reason: str
    speedup: float | None = None


@dataclass(frozen=True)
class Context:
    task_name: str
    task_description: str
    reference_source: str
    current_source: str
    measurement: Measurement
    history: tuple[AttemptSummary, ...] = ()


class Agent(Protocol):
    name: str

    def propose(self, context: Context) -> Proposal: ...
