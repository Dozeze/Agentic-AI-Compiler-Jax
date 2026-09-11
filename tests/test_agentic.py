"""The tool loop with a scripted model: transcript shape, idle turns, stop."""

from __future__ import annotations

import json

from halo import controller
from halo.agent.agentic import AgenticAgent
from halo.agent.llm import ToolCall, Turn
from halo.config import BudgetConfig, RunConfig
from halo.store import RunStore
from halo.types import Usage

USAGE = Usage("stub", 100, 10, 0.001, 0.1)


class ScriptedModel:
    """Returns one scripted turn per call and records what it was shown."""

    def __init__(self, *turns):
        self._turns = iter(turns)
        self.seen: list[list[dict]] = []

    def converse(self, system, transcript, tools):
        self.seen.append([dict(t) for t in transcript])
        assert {t["name"] for t in tools} == {"evaluate", "check", "inspect", "lookup", "finish"}
        return next(self._turns)


def turn(*calls, text=""):
    return Turn(calls=tuple(ToolCall(n, a) for n, a in calls), text=text, usage=USAGE)


def test_results_go_back_to_the_model_and_the_transcript_is_saved(tmp_path, stubbed):
    model = ScriptedModel(
        turn(("inspect", {"attempt": 0, "what": "source"})),
        turn(("evaluate", {"source": "fast", "hypothesis": "batch"})),
        turn(("finish", {"summary": "done"})),
    )
    store = RunStore(tmp_path, "r")
    result = controller.run(RunConfig(task="stub"), AgenticAgent(model, "algorithmic"), store)

    assert result.best.index == 1
    assert result.cost_usd == 0.003 and len(result.usages) == 3
    # The second call saw the inspect result; the third saw the evaluation.
    assert model.seen[1][-1]["results"][0]["response"]["text"] == "seed"
    assert model.seen[2][-1]["results"][0]["response"]["accepted"] is True
    transcript = json.loads((store.root / "transcript.json").read_text())
    assert [t["role"] for t in transcript] == ["user"] + ["model", "tool"] * 3
    assert "Budget: 6 evaluations" in transcript[0]["text"]


def test_a_model_that_stops_calling_tools_is_stopped(tmp_path, stubbed):
    model = ScriptedModel(*[turn(text="I think...") for _ in range(4)])
    result = controller.run(
        RunConfig(task="stub"), AgenticAgent(model, "code"), RunStore(tmp_path, "r")
    )
    assert result.stop_reason.startswith("agent stopped calling tools")
    assert "Call a tool" in model.seen[1][-1]["text"]
    assert len(model.seen) == 4


def test_a_malformed_call_is_named_and_retried(tmp_path, stubbed):
    from halo.agent.llm import Turn

    model = ScriptedModel(
        Turn(calls=(), text="", usage=USAGE, finish_reason="FinishReason.MALFORMED_FUNCTION_CALL"),
        turn(("finish", {"summary": "ok"})),
    )
    result = controller.run(RunConfig(task="stub"), AgenticAgent(model, "code"), RunStore(tmp_path, "r"))
    assert result.stop_reason == "agent finished: ok"
    assert "could not be parsed" in model.seen[1][-1]["text"]


def test_only_the_first_call_of_a_turn_runs(tmp_path, stubbed):
    """A model that emits a whole plan at once is guessing at results it has not
    seen. The first call runs; the rest are recorded and the model is told."""
    model = ScriptedModel(
        turn(
            ("evaluate", {"source": "fast", "hypothesis": "a"}),
            ("evaluate", {"source": "faster", "hypothesis": "b"}),
            ("finish", {"summary": "done"}),
        ),
        turn(("finish", {"summary": "ok"})),
    )
    cfg = RunConfig(task="stub", budget=BudgetConfig(max_evaluations=3))
    store = RunStore(tmp_path, "r")
    result = controller.run(cfg, AgenticAgent(model, "code"), store)
    assert len(result.attempts) == 2
    assert result.stop_reason == "agent finished: ok"
    shown = model.seen[1]
    assert len(shown[-2]["calls"]) == 1
    assert "2 further call(s)" in shown[-1]["results"][0]["response"]["note"]
    transcript = json.loads((store.root / "transcript.json").read_text())
    assert [c["name"] for c in transcript[1]["discarded"]] == ["evaluate", "finish"]


def test_an_empty_turn_is_not_sent_back_to_the_model(tmp_path, stubbed):
    """A truncated or filtered response has no parts; echoing it back is a 400."""
    model = ScriptedModel(turn(), turn(("finish", {"summary": "ok"})))
    controller.run(RunConfig(task="stub"), AgenticAgent(model, "code"), RunStore(tmp_path, "r"))
    roles = [t["role"] for t in model.seen[1]]
    assert roles == ["user", "user"], "the empty model turn must be absent, the nudge present"
