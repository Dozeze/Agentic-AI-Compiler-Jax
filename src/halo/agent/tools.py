"""The tools a model may call, and what each one does to the session.

Four tools, each a thin door onto something the harness already does. There is
deliberately no "run Python" tool: an agent that can execute arbitrary code can
also time itself, and a self-reported speedup is exactly the number this project
refuses to trust.
"""

from __future__ import annotations

from halo.agent.context import feedback_for
from halo.session import INSPECTABLE, Session
from halo.types import Proposal

#: Plain JSON schema; the provider client wraps these in its own types.
TOOLS: list[dict] = [
    {
        "name": "evaluate",
        "description": (
            "Compile a rewrite of candidate.py, check it against the oracle on withheld "
            "inputs, and time it against the current best in interleaved rounds. The only "
            "way a candidate can be accepted. Costs one evaluation from the budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Complete new contents of candidate.py.",
                },
                "hypothesis": {
                    "type": "string",
                    "description": "The single change being made and why it should be faster.",
                },
                "expected_speedup": {
                    "type": "number",
                    "description": "Predicted speedup over the current best, e.g. 1.5.",
                },
                "parent": {
                    "type": "integer",
                    "description": (
                        "Index of the attempt this rewrite builds on. Defaults to the "
                        "current best; pass an earlier one to branch from it."
                    ),
                },
            },
            "required": ["source", "hypothesis"],
        },
    },
    {
        "name": "check",
        "description": (
            "Compile a rewrite and check its correctness against the oracle, without "
            "timing it. Fast; does not count as an evaluation. Use it to debug a rewrite "
            "before spending an evaluation on it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Complete contents of candidate.py."}
            },
            "required": ["source"],
        },
    },
    {
        "name": "inspect",
        "description": (
            "Show one aspect of an earlier attempt: its source, its diff against the seed, "
            "the harness feedback again, or the raw text at one level of the lowering "
            "pipeline (jaxpr, stablehlo, hlo) or the buffer assignment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "attempt": {"type": "integer", "description": "Attempt index; 0 is the seed."},
                "what": {"type": "string", "enum": list(INSPECTABLE)},
                "max_lines": {
                    "type": "integer",
                    "description": "Truncate the text to this many lines (default 120, max 400).",
                },
            },
            "required": ["attempt", "what"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Stop optimizing. The reported result is the best attempt the harness "
            "verified, whatever this summary says."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "What was tried, what worked, and why you are stopping.",
                }
            },
            "required": ["summary"],
        },
    },
]


def dispatch(session: Session, name: str, args: dict) -> dict:
    """Execute one tool call. Every outcome, including a bad call, comes back as a
    result the model can read; nothing raises out of here."""
    session.tool_calls += 1
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool {name!r}; available: {sorted(_HANDLERS)}"}
    try:
        return handler(session, **args)
    except TypeError as exc:  # wrong or missing arguments
        return {"error": f"bad arguments for {name}: {exc}"}
    except ValueError as exc:  # nothing was measured
        return {"error": str(exc), "budget_remaining": session.remaining()}


def _evaluate(
    session: Session,
    source: str,
    hypothesis: str,
    expected_speedup: float | None = None,
    parent: int | None = None,
) -> dict:
    if parent is not None and not 0 <= parent < len(session.attempts):
        raise ValueError(f"no attempt {parent} to build on")
    # Formatting is not a hypothesis. The same program measured twice would spend
    # an evaluation on the noise floor, so it gets the earlier verdict instead.
    if (earlier := session.duplicate_of(source)) is not None:
        raise ValueError(
            f"this is the same program as attempt {earlier.index} "
            f"({earlier.decision.rule}: {earlier.decision.reason}); not measured again. "
            "Comments and formatting do not change what XLA compiles."
        )
    attempt = session.evaluate(
        source,
        proposal=Proposal(
            source=source, hypothesis=hypothesis, expected_speedup=expected_speedup
        ),
        parent=parent,
    )
    m = attempt.measurement
    result = {
        "attempt": attempt.index,
        "parent": attempt.parent,
        "accepted": attempt.decision.accepted,
        "verdict": attempt.decision.rule,
        "reason": attempt.decision.reason,
        "current_best": session.best.index,
        "feedback": feedback_for(m, session.cfg.context) if m.ok else None,
        "budget_remaining": session.remaining(),
    }
    if m.correctness is not None:
        result["correctness"] = m.correctness.summary()
    return result


def _check(session: Session, source: str) -> dict:
    m = session.check(source)
    if not m.ok:
        return {"passed": False, "error": m.error}
    report = m.correctness
    return {
        "passed": report.passed if report else False,
        "correctness": report.summary() if report else "no report",
        "compile_s": m.compile_s,
        "budget_remaining": session.remaining(),
    }


def _inspect(session: Session, attempt: int, what: str, max_lines: int = 120) -> dict:
    return {
        "attempt": attempt,
        "what": what,
        "text": session.inspect(attempt, what, max_lines=max(1, min(int(max_lines), 400))),
    }


def _finish(session: Session, summary: str) -> dict:
    session.finish(summary)
    return {"stopped": True, "best_attempt": session.best.index}


_HANDLERS = {
    "evaluate": _evaluate,
    "check": _check,
    "inspect": _inspect,
    "finish": _finish,
}
