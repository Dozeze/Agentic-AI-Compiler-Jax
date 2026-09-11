"""The referee: what a tool-using agent can and cannot make the session do.

Measurements are stubbed (see ``conftest.stubbed``): a source's "speed" is looked
up by its text, so a scripted sequence of tool calls exercises the accept/reject,
budget, patience and lineage rules without JAX in the loop.
"""

from __future__ import annotations

import pytest

from halo import controller
from halo.agent import tools
from halo.agent.fake import ScriptedToolAgent
from halo.config import BudgetConfig, RunConfig
from halo.session import Session
from halo.store import RunStore
from halo.types import Usage


def make_session(tmp_path, **budget) -> Session:
    cfg = RunConfig(task="stub", budget=BudgetConfig(**budget))
    return Session(cfg, RunStore(tmp_path, "s"))


def evaluate(source, **kw):
    return ("evaluate", {"source": source, "hypothesis": "h", **kw})


# --- acceptance is the session's, not the agent's ------------------------------

def test_finish_reports_the_verified_best_whatever_the_agent_claims(tmp_path, stubbed):
    agent = ScriptedToolAgent(
        evaluate("fast"),
        evaluate("wrong"),
        ("finish", {"summary": "attempt 2 is a 10x win, ship it"}),
    )
    result = controller.run(RunConfig(task="stub"), agent, RunStore(tmp_path, "r"))
    assert result.best.index == 1
    assert result.attempts[2].decision.rule == "correctness"
    assert result.stop_reason.startswith("agent finished")


def test_each_evaluation_is_judged_against_the_current_best(tmp_path, stubbed):
    agent = ScriptedToolAgent(evaluate("fast"), evaluate("medium"), evaluate("faster"))
    result = controller.run(RunConfig(task="stub"), agent, RunStore(tmp_path, "r"))
    assert [a.decision.accepted for a in result.attempts[1:]] == [True, False, True]
    assert [c["baseline"] for c in stubbed[1:4]] == ["seed", "fast", "fast"]
    # The headline is re-measured against the seed, with its own interval.
    assert stubbed[-1] == {"baseline": "seed", "candidate": "faster"}
    assert result.overall.speedup.ratio == pytest.approx(4.0)


# --- lineage --------------------------------------------------------------------

def test_branching_from_an_earlier_attempt_records_the_parent(tmp_path, stubbed):
    agent = ScriptedToolAgent(
        evaluate("fast"),                 # 1, accepted
        evaluate("medium"),               # 2, rejected
        evaluate("faster", parent=2),     # 3, built on the rejected one
    )
    result = controller.run(RunConfig(task="stub"), agent, RunStore(tmp_path, "r"))
    assert [a.parent for a in result.attempts] == [None, 0, 1, 2]
    assert result.attempts[3].decision.accepted, (
        "where a candidate came from does not change what it is judged against"
    )
    assert stubbed[3]["baseline"] == "fast"


def test_branching_from_a_nonexistent_attempt_is_an_error_not_a_crash(tmp_path, stubbed):
    agent = ScriptedToolAgent(evaluate("fast", parent=7))
    controller.run(RunConfig(task="stub"), agent, RunStore(tmp_path, "r"))
    assert "no attempt 7" in agent.results[0]["error"]


# --- budget ---------------------------------------------------------------------

def test_the_evaluation_budget_stops_the_run(tmp_path, stubbed):
    s = make_session(tmp_path, max_evaluations=2)
    agent = ScriptedToolAgent(evaluate("same"), evaluate("fast"), evaluate("faster"))
    agent.run(s)
    assert len(agent.results) == 2
    assert "evaluation budget" in s.exhausted


def test_patience_counts_rejections_but_checks_do_not(tmp_path, stubbed):
    s = make_session(tmp_path, patience=2)
    agent = ScriptedToolAgent(
        evaluate("same"),
        ("check", {"source": "wrong"}),
        ("check", {"source": "fast"}),
        evaluate("slow"),
        evaluate("fast"),  # never reached
    )
    agent.run(s)
    assert s.checks == 2
    assert s.evaluations == 2
    assert "2 consecutive" in s.exhausted
    assert agent.results[1]["passed"] is False
    assert agent.results[2]["passed"] is True


def test_an_accepted_evaluation_resets_patience(tmp_path, stubbed):
    s = make_session(tmp_path, patience=2)
    ScriptedToolAgent(
        evaluate("same"), evaluate("fast"), evaluate("slow"), evaluate("faster")
    ).run(s)
    assert s.best.index == 4
    assert s.exhausted is None


def test_the_cost_budget_stops_the_run(tmp_path, stubbed):
    s = make_session(tmp_path, max_cost_usd=0.01)
    s.charge(Usage("m", 1, 1, 0.02, 0.1))
    assert "cost budget" in s.exhausted


def test_the_tool_call_budget_counts_every_call(tmp_path, stubbed):
    s = make_session(tmp_path, max_tool_calls=3)
    agent = ScriptedToolAgent(
        ("inspect", {"attempt": 0, "what": "source"}),
        ("inspect", {"attempt": 0, "what": "diff"}),
        ("inspect", {"attempt": 0, "what": "source"}),
        evaluate("fast"),
    )
    agent.run(s)
    assert len(agent.results) == 3
    assert s.evaluations == 0


# --- inspect --------------------------------------------------------------------

def test_inspect_truncates_and_reports_what_it_dropped(tmp_path, stubbed):
    s = make_session(tmp_path)
    (s.store.attempt_dir(0) / "hlo.txt").write_text("\n".join(f"line {i}" for i in range(50)))
    text = s.inspect(0, "hlo", max_lines=5)
    assert text.startswith("line 0")
    assert "45 more lines" in text


def test_inspect_of_missing_or_unknown_things_is_data_not_an_exception(tmp_path, stubbed):
    s = make_session(tmp_path)
    assert "not available" in s.inspect(0, "stablehlo")
    assert "error" in tools.dispatch(s, "inspect", {"attempt": 0, "what": "profile"})
    assert "error" in tools.dispatch(s, "inspect", {"attempt": 9, "what": "source"})
    assert "error" in tools.dispatch(s, "teleport", {})
    assert "error" in tools.dispatch(s, "evaluate", {"source": "fast"})  # no hypothesis


def test_evaluate_result_tells_the_agent_what_it_needs_to_decide_next(tmp_path, stubbed):
    s = make_session(tmp_path, max_evaluations=4)
    result = tools.dispatch(s, "evaluate", {"source": "fast", "hypothesis": "h"})
    assert result["accepted"] is True
    assert result["attempt"] == 1 and result["current_best"] == 1
    assert result["budget_remaining"]["evaluations"] == 3
    assert "Measurement" in result["feedback"]


# --- artifacts ------------------------------------------------------------------

def test_the_run_leaves_a_result_file_with_the_lineage(tmp_path, stubbed):
    import json

    store = RunStore(tmp_path, "r")
    agent = ScriptedToolAgent(evaluate("fast"), evaluate("faster", parent=0))
    controller.run(RunConfig(task="stub"), agent, store)
    result = json.loads((store.root / "result.json").read_text())
    assert result["best"] == 2
    assert [a["parent"] for a in result["attempts"]] == [None, 0, 0]
    assert result["tool_calls"] == 2


# --- duplicates -------------------------------------------------------------------

def test_the_same_program_is_not_measured_twice(tmp_path, stubbed):
    """Formatting is not a hypothesis. A reformatted seed, or a rewrite already
    tried, gets the earlier verdict back and costs no evaluation. (The single-shot
    driver deliberately still measures an unchanged proposal: that is the echo
    control, and it measures the noise floor.)"""
    s = make_session(tmp_path)
    agent = ScriptedToolAgent(
        evaluate("seed  # a comment\n"),
        evaluate("fast"),
        evaluate("fast\n\n"),
    )
    agent.run(s)
    assert s.evaluations == 1
    assert "same program as attempt 0" in agent.results[0]["error"]
    assert "same program as attempt 1" in agent.results[2]["error"]
    assert len(stubbed) == 2  # the baseline and one real measurement


def test_a_cached_baseline_brings_its_artifacts_along(tmp_path, stubbed):
    from halo.session import baseline_attempt
    from halo.tasks import base

    first_store = RunStore(tmp_path, "first")
    baseline = baseline_attempt(RunConfig(task="stub"), base.load_spec("stub"), first_store)
    (first_store.attempt_dir(0) / "hlo.txt").write_text("HloModule stub")

    s = Session(RunConfig(task="stub"), RunStore(tmp_path, "second"),
                baseline, first_store.attempt_dir(0))
    assert s.inspect(0, "hlo") == "HloModule stub"
