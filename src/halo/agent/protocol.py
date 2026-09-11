"""What an agent is, and what it gets to see.

A single-shot :class:`Agent` is a *pure function* ``Context -> Proposal``: it
does not compile, benchmark, or touch the filesystem. The controller owns every
side effect, which keeps the loop trivially loggable and replayable, and means a
scripted agent is a drop-in substitute for a language model in tests.

A :class:`ToolAgent` drives the loop itself through a
:class:`~halo.session.Session`'s tools. It still never measures anything: the
session does, and the session decides what is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from halo.session import Session

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


@runtime_checkable
class Agent(Protocol):
    name: str

    def propose(self, context: Context) -> Proposal: ...


@runtime_checkable
class ToolAgent(Protocol):
    name: str

    def run(self, session: Session) -> None: ...
