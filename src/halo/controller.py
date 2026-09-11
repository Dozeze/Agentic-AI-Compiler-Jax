"""Drives one optimization run.

Two kinds of agent exist. A *single-shot* agent is a pure function
``Context -> Proposal``; the controller measures each proposal and shows the agent
what happened. A *tool-using* agent drives the loop itself, calling the session's
tools until it finishes or runs out of budget. Both go through the same
:class:`~halo.session.Session`, which is the only thing that can accept a
candidate, so the two cannot diverge on what counts as an improvement.
"""

from __future__ import annotations

from halo.agent.protocol import Agent, AttemptSummary, Context, ToolAgent
from halo.config import RunConfig
from halo.harness import runner  # noqa: F401 - patched by tests
from halo.session import RunResult, Session, baseline_attempt, decide, diff
from halo.store import RunStore
from halo.tasks import base
from halo.types import Attempt, Decision, Measurement

__all__ = [
    "Attempt", "Decision", "Measurement", "RunResult",
    "baseline_attempt", "build_context", "decide", "run",
]


def build_context(spec, task, best: Attempt, attempts: list[Attempt]) -> Context:
    return Context(
        task_name=spec.name,
        task_description=task.DESCRIPTION,
        reference_source=base.reference_excerpt(spec),
        current_source=best.source,
        measurement=best.measurement,
        history=tuple(
            AttemptSummary(
                index=a.index,
                diff=diff(spec.seed_source, a.source),
                accepted=a.decision.accepted,
                rule=a.decision.rule,
                reason=a.decision.reason,
                speedup=a.measurement.speedup.ratio if a.measurement.speedup else None,
            )
            for a in attempts[1:]
        ),
    )


def single_shot(session: Session, agent: Agent) -> None:
    """``steps`` rounds of propose, measure, decide."""
    for _ in range(session.cfg.steps):
        context = build_context(session.spec, session.task, session.best, session.attempts)
        try:
            proposal = agent.propose(context)
        except Exception as exc:  # noqa: BLE001 - a failed agent is data, not a crash
            session.record_failure(str(exc))
            continue
        if proposal.usage is not None:
            session.charge(proposal.usage)
        session.evaluate(proposal.source, proposal=proposal)
    session.stop(f"{session.cfg.steps} step(s) completed")


def run(
    cfg: RunConfig,
    agent: Agent | ToolAgent,
    store: RunStore,
    baseline: Attempt | None = None,
) -> RunResult:
    session = Session(cfg, store, baseline)
    if isinstance(agent, ToolAgent):
        agent.run(session)
    else:
        single_shot(session, agent)
    return session.result()
