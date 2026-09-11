"""Scoring the ablation, and the context levels it varies."""

from __future__ import annotations

import json

import pytest

from halo import ablation
from halo.agent import context as ctx
from halo.agent.protocol import Context
from halo.tasks import ceilings
from halo.types import (
    CorrectnessCase, CorrectnessReport, DeviceInfo, HloSummary, IRStats,
    Measurement, ProgramSummary, SpeedupEstimate, TimingStats,
)


def row(task, condition, accepted, speedup, replicate=0, error=None):
    return {"task": task, "condition": condition, "replicate": replicate,
            "accepted": accepted, "rule": "accepted" if accepted else "performance",
            "speedup": speedup, "ci_low": speedup, "changed": True,
            "cost_usd": 0.001, "input_tokens": 10, "output_tokens": 5, "error": error}


# --- fraction of ceiling ------------------------------------------------------

def test_reaching_the_ceiling_scores_one():
    ceiling = ceilings.MEASURED_HEADROOM["naive_attention"]
    assert ablation.fraction_of_ceiling("naive_attention", ceiling, True) == 1.0


def test_halfway_to_the_ceiling_scores_a_half():
    ceiling = ceilings.MEASURED_HEADROOM["matmul_chain"]  # 11.31x
    halfway = 1.0 + (ceiling - 1.0) / 2
    assert ablation.fraction_of_ceiling("matmul_chain", halfway, True) == pytest.approx(0.5)


def test_a_rejected_proposal_scores_zero_however_fast_it_measured():
    """What ships is the seed, so a rejected candidate captured nothing."""
    assert ablation.fraction_of_ceiling("matmul_chain", 9.0, False) == 0.0


def test_beating_the_ceiling_is_clamped_to_one():
    assert ablation.fraction_of_ceiling("naive_attention", 99.0, True) == 1.0


def test_a_control_has_no_ceiling_to_be_a_fraction_of():
    with pytest.raises(ValueError, match="no headroom"):
        ablation.fraction_of_ceiling("softmax", 1.02, True)


# --- report -------------------------------------------------------------------

def test_report_separates_improvable_tasks_from_controls():
    rows = [
        row("matmul_chain", "full", True, ceilings.MEASURED_HEADROOM["matmul_chain"]),
        row("softmax", "full", True, 1.2),        # a false positive
        row("gelu", "full", False, 1.0),
    ]
    text = ablation.report(rows)
    assert "1/1" in text          # one improvable task accepted
    assert "100%" in text         # at full ceiling
    assert "## Controls" in text


def test_report_ignores_failed_cells_rather_than_scoring_them():
    rows = [
        row("matmul_chain", "full", True, 11.31),
        row("matmul_chain", "full", False, None, replicate=1, error="boom"),
    ]
    assert "1/1" in ablation.report(rows)


def test_cells_cover_the_whole_matrix():
    cells = list(ablation.cells(["a", "b"], ["code", "full"], 3))
    assert len(cells) == 12
    assert len({c.key for c in cells}) == 12


def test_sweep_skips_cells_already_recorded(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text(json.dumps(row("softmax", "code", False, 1.0)) + "\n")
    logged: list[str] = []
    ablation.sweep(
        __import__("halo.config", fromlist=["RunConfig"]).RunConfig(task=""),
        tasks=["softmax"], conditions=["code"], replicates=1,
        out=out, runs_dir=tmp_path, make_agent=lambda c: None,
        log=logged.append,
    )
    assert "0 cells to run, 1 already done" in logged[0]


# --- context levels -----------------------------------------------------------

def _context() -> Context:
    stats = TimingStats((1.0,), 1.0, 0.1, 0.9)
    program = ProgramSummary(
        jaxpr=IRStats(139, {"dot_general": 16}),
        stablehlo=IRStats(189, {"dot_general": 16}),
        hlo=HloSummary(403, 66, {"fusion": 66, "dot": 16}, temp_bytes=11272192),
    )
    return Context(
        task_name="t", task_description="desc", reference_source="def reference(): ...",
        current_source="def candidate(): ...",
        measurement=Measurement(
            task="t", ok=True,
            device=DeviceInfo("cpu", "cpu", 1, "0.11.1", "0.11.1"),
            correctness=CorrectnessReport(True, 1e-5, 1e-6, (CorrectnessCase("r", True, 0.0, 0.0),)),
            baseline=stats, candidate=stats,
            speedup=SpeedupEstimate(1.0, 0.99, 1.01, 0.95, 30),
            candidate_program=program, baseline_program=program,
        ),
    )


@pytest.mark.parametrize("level", ctx.LEVELS)
def test_every_level_states_the_task_and_the_code(level):
    text = ctx.render(_context(), min_speedup=1.05, level=level)
    assert "## Current implementation" in text
    assert "def candidate(): ..." in text
    assert "## Your task" in text


#: The levels that vary how much the agent is told. `algorithmic` is not one of
#: them: it shows exactly what `full` shows and varies only the framing.
INFORMATION_LEVELS = ("code", "timing", "hlo", "full")


def test_levels_add_information_and_never_remove_it():
    sizes = [len(ctx.render(_context(), min_speedup=1.05, level=l))
             for l in INFORMATION_LEVELS]
    assert sizes == sorted(sizes), "each level should be a superset of the previous"


def test_algorithmic_shows_the_same_evidence_as_full():
    """The comparison that isolates prompt from data: if these two differed in what
    they show, a difference in outcome could not be attributed to the framing."""
    body = {l: ctx.render(_context(), min_speedup=1.05, level=l) for l in ("full", "algorithmic")}
    for marker in ("jaxpr, what you wrote", "StableHLO, handed to XLA",
                   "optimized HLO", "## Measurement"):
        assert marker in body["full"] and marker in body["algorithmic"], marker


def test_only_the_algorithmic_level_reframes_the_job():
    assert ctx.system_for("algorithmic") == ctx.SYSTEM_ALGORITHMIC
    for level in INFORMATION_LEVELS:
        assert ctx.system_for(level) == ctx.SYSTEM


def test_the_algorithmic_prompt_separates_the_two_questions():
    """Guards the specific failure it was written for: the agent reading a clean
    HLO summary as evidence that no better algorithm exists."""
    text = ctx.SYSTEM_ALGORITHMIC
    assert "Never treat a clean HLO summary as evidence against a rewrite" in text
    assert "cannot do is change your algorithm" in text


def test_code_level_reveals_no_measurement():
    text = ctx.render(_context(), min_speedup=1.05, level="code")
    assert "## Measurement" not in text
    assert "## Compiler feedback" not in text


def test_hlo_level_shows_the_compiled_result_but_not_the_levels_above_it():
    text = ctx.render(_context(), min_speedup=1.05, level="hlo")
    assert "optimized HLO" in text
    assert "jaxpr, what you wrote" not in text
    assert "StableHLO, handed to XLA" not in text


def test_full_level_shows_every_stage():
    text = ctx.render(_context(), min_speedup=1.05, level="full")
    assert "jaxpr, what you wrote" in text
    assert "StableHLO, handed to XLA" in text
    assert "optimized HLO" in text


def test_an_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown context level"):
        ctx.render(_context(), min_speedup=1.05, level="everything")


# --- did the agent actually propose anything ---------------------------------

SEED = "import jax.numpy as jnp\n\n\ndef candidate(a, b):\n    return a + b\n"


def test_reformatting_is_not_a_change():
    """The bug this replaced compared raw strings, so a moved blank line counted
    as a proposal. It mislabelled 45 of 120 cells in the first sweep and hid the
    finding: that richer context makes the agent decline to act."""
    reformatted = "import jax.numpy as jnp\n\ndef candidate(a, b):\n    return a + b\n"
    assert not ablation.is_changed(SEED, reformatted)


def test_added_comments_are_not_a_change():
    commented = SEED.replace("    return a + b", "    # sum them\n    return a + b")
    assert not ablation.is_changed(SEED, commented)


def test_a_different_expression_is_a_change():
    assert ablation.is_changed(SEED, SEED.replace("a + b", "jnp.add(a, b)"))


def test_unparseable_source_falls_back_to_text_comparison():
    assert ablation.is_changed(SEED, "def candidate(:")
    assert not ablation.is_changed("def candidate(:", "def candidate(:")


def test_report_covers_every_context_level():
    """The report once hardcoded its own list of conditions and silently omitted a
    newly added level - the one the sweep had just been run to evaluate."""
    rows = [row("matmul_chain", level, True, 11.31) for level in ctx.LEVELS]
    text = ablation.report(rows)
    for level in ctx.LEVELS:
        assert f"| {level} |" in text, f"{level} missing from the report"


# --- conditions name an agent as well as a level ---------------------------------

def test_a_bare_level_is_the_single_shot_agent():
    assert ablation.parse_condition("algorithmic") == ("single-shot", "algorithmic")
    assert ablation.parse_condition("agentic/code") == ("agentic", "code")


def test_report_keeps_agentic_conditions_after_the_single_shot_levels():
    rows = [
        row("matmul_chain", "agentic/algorithmic", True, 5.0),
        row("matmul_chain", "code", True, 5.0),
        row("matmul_chain", "algorithmic", True, 5.0),
    ]
    text = ablation.report(rows)
    lines = [l for l in text.splitlines() if l.startswith("| ") and "|---" not in l]
    order = [l.split("|")[1].strip() for l in lines[1:4]]
    assert order == ["code", "algorithmic", "agentic/algorithmic"]
