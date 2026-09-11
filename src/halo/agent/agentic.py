"""The tool-using agent: the model drives, the session referees.

The conversation is the model's working memory. Each turn it calls one or more
tools; the session executes them and the results go back verbatim. The loop ends
when the model calls ``finish`` or the session's budget runs out - the model is
told what remains after every call, but the decision to stop on budget is never
its to make.
"""

from __future__ import annotations

from halo.agent import tools
from halo.agent.context import render, system_for
from halo.agent.llm import ToolClient
from halo.agent.protocol import Context
from halo.session import Session
from halo.tasks import base

TOOL_PROTOCOL = """\
You work through tools, and the harness is the referee.

- evaluate: compiles a rewrite, checks it against the oracle on withheld inputs, and
  times it against the current best in interleaved rounds. This is the only way a
  candidate can be accepted, and the only source of truth about speed. Each call
  spends one evaluation from a fixed budget.
- check: correctness only, fast, not counted as an evaluation. Use it to debug a
  rewrite before spending an evaluation on it.
- inspect: the raw jaxpr, StableHLO, optimized HLO or buffer assignment of any
  earlier attempt, or its source or diff, on request.
- finish: stop. The reported result is the best attempt the harness verified.

A candidate is accepted when it is correct on the withheld cases and the 95% lower
bound of its speedup over the *current best* clears the threshold. A rejected
attempt is not lost: build on any earlier attempt by passing its index as `parent`.

Spend evaluations on hypotheses, not on variations. Two rewrites that differ only in
style will measure the same. Stop when you have no further hypothesis worth an
evaluation; on a program with nothing to gain, one evaluation of your best idea and
then finish is the right behaviour, not a failure.
"""


def system_prompt(level: str) -> str:
    return system_for(level) + "\n" + TOOL_PROTOCOL


class AgenticAgent:
    def __init__(self, client: ToolClient, context_level: str, name: str = "agentic") -> None:
        self.name = name
        self._client = client
        self._level = context_level

    def run(self, session: Session) -> None:
        system = system_prompt(self._level)
        transcript: list[dict] = [{"role": "user", "text": opening(session, self._level)}]
        idle_turns = 0

        while session.exhausted is None:
            turn = self._client.converse(system, transcript, tools.TOOLS)
            session.charge(turn.usage)
            # One call per turn. A model that emits six evaluations and a finish
            # at once is guessing at results it has not seen; only the first call
            # is executed, and the discarded ones stay in the transcript for the
            # record but are never shown back to the model as its own history.
            call, discarded = turn.calls[:1], turn.calls[1:]
            if call or turn.text:
                # An empty turn (no text, no call - a truncated or filtered
                # response) is not appended: the API rejects a message with no
                # parts on the next request.
                transcript.append(
                    {
                        "role": "model",
                        "text": turn.text,
                        "calls": [
                            {"name": c.name, "args": c.args, "signature": c.signature}
                            for c in call
                        ],
                        "discarded": [{"name": c.name, "args": c.args} for c in discarded],
                    }
                )
            if not call:
                # Text without a call is a turn spent on nothing. Say so once;
                # a model that keeps talking instead of acting is stopped.
                idle_turns += 1
                if idle_turns >= 2:
                    session.stop("agent stopped calling tools")
                    break
                transcript.append(
                    {"role": "user", "text": "Call a tool, or call finish to stop."}
                )
                continue
            idle_turns = 0

            (call,) = call
            response = tools.dispatch(session, call.name, call.args)
            if discarded:
                response["note"] = (
                    f"{len(discarded)} further call(s) in this turn were discarded. One "
                    "call per turn: read the result before deciding the next."
                )
            transcript.append({"role": "tool", "results": [{"name": call.name, "response": response}]})
            session.store.write_transcript(transcript)

        session.store.write_transcript(transcript)


def opening(session: Session, level: str) -> str:
    """The first message: the task exactly as the single-shot agent sees it,
    closed with tool instructions rather than a request for one answer."""
    context = Context(
        task_name=session.spec.name,
        task_description=session.task.DESCRIPTION,
        reference_source=base.reference_excerpt(session.spec),
        current_source=session.best.source,
        measurement=session.best.measurement,
    )
    budget = session.cfg.budget
    return (
        render(context, min_speedup=session.cfg.accept.min_speedup, level=level, tools=True)
        + f"\nBudget: {budget.max_evaluations} evaluations, {budget.max_tool_calls} tool "
        f"calls in total, and the run stops after {budget.patience} consecutive rejected "
        f"evaluations. Acceptance threshold: {session.cfg.accept.min_speedup:.2f}x "
        "(95% lower bound) over the current best.\n"
    )
