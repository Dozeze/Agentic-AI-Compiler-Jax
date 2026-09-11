"""Sweep task x condition x replicate, and score the result.

The experiments the project exists to run: hold the model, the tasks and the
acceptance policy fixed, vary only how much the agent is told - or whether it
answers once or drives the loop itself - and see what changes. A condition is a
context level (``algorithmic``) for the single-shot agent, or ``agent/level``
(``agentic/algorithmic``) to name the agent as well.

Two things make the scoring meaningful. Raw speedup is not comparable across
tasks, so improvable tasks are scored as the *fraction of the known ceiling*
reached (see :mod:`halo.tasks.ceilings`). And a suite of only improvable tasks
would reward a model that proposes a rewrite every time, so the controls are
scored separately, by how often something was wrongly accepted.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from halo import controller
from halo.agent.context import LEVELS
from halo.config import RunConfig
from halo.session import same_program
from halo.store import RunStore
from halo.tasks import base, ceilings
from halo.types import Attempt


@dataclass(frozen=True)
class Cell:
    task: str
    condition: str
    replicate: int

    @property
    def key(self) -> str:
        return f"{self.task}|{self.condition}|{self.replicate}"


def cells(tasks: list[str], conditions: list[str], replicates: int) -> Iterator[Cell]:
    for replicate in range(replicates):
        for task in tasks:
            for condition in conditions:
                yield Cell(task, condition, replicate)


def parse_condition(condition: str) -> tuple[str, str]:
    """``"algorithmic"`` -> ``("single-shot", "algorithmic")``;
    ``"agentic/code"`` -> ``("agentic", "code")``."""
    agent, _, level = condition.rpartition("/")
    return (agent or "single-shot"), level


def _condition_order(condition: str) -> tuple[int, int]:
    agent, level = parse_condition(condition)
    return (agent != "single-shot", LEVELS.index(level) if level in LEVELS else len(LEVELS))


def is_changed(before: str, after: str) -> bool:
    """Did the proposal alter the program, rather than only its formatting?"""
    return not same_program(before, after)


def _row(cell: Cell, result: controller.RunResult) -> dict:
    """One row per cell. Speedup is the best implementation against the seed,
    which for a one-step run is the single attempt and for a longer run is what
    the loop ended with, not merely what it tried last."""
    last = result.attempts[-1]
    speedup = result.overall.speedup
    if result.improved:
        rule = result.best.decision.rule
    elif len(result.attempts) == 1:
        rule = "no_attempt"  # every proposal was the seed again, or none was made
    else:
        rule = last.decision.rule
    return {
        **cell.__dict__,
        "model": result.model,
        "accepted": result.improved,
        "rule": rule,
        "speedup": speedup.ratio if speedup else None,
        "ci_low": speedup.ci_low if speedup else None,
        "changed": any(
            is_changed(result.attempts[0].source, a.source) for a in result.attempts[1:]
        ),
        "attempts": len(result.attempts) - 1,
        "evaluations": result.evaluations,
        "tool_calls": result.tool_calls,
        "checks": result.checks,
        "stop_reason": result.stop_reason,
        "cost_usd": result.cost_usd,
        "input_tokens": sum(u.input_tokens for u in result.usages),
        "output_tokens": sum(u.output_tokens for u in result.usages),
        "error": None,
    }


def _error_row(cell: Cell, model: str) -> dict:
    return {**cell.__dict__, "model": model, "accepted": False, "rule": "error", "speedup": None,
            "ci_low": None, "changed": False, "attempts": 0, "evaluations": 0, "tool_calls": 0,
            "checks": 0, "stop_reason": "error", "cost_usd": 0.0, "input_tokens": 0,
            "output_tokens": 0, "error": traceback.format_exc(limit=5)}


def sweep(
    cfg: RunConfig,
    tasks: list[str],
    conditions: list[str],
    replicates: int,
    out: Path,
    runs_dir: Path,
    make_agent,
    log=print,
) -> list[dict]:
    """Run every cell, appending one JSON row each. Resumable: cells already in
    ``out`` are skipped, so an interrupted sweep continues where it stopped."""
    rows = read(out)
    done = {(r["task"], r["condition"], r["replicate"]) for r in rows}
    baselines: dict[str, tuple[Attempt, Path]] = {}

    todo = [c for c in cells(tasks, conditions, replicates)
            if (c.task, c.condition, c.replicate) not in done]
    log(f"{len(todo)} cells to run, {len(done)} already done")

    for index, cell in enumerate(todo, 1):
        agent, level = parse_condition(cell.condition)
        run_cfg = cfg.with_overrides(task=cell.task, context=level, agent=agent)
        if cell.task not in baselines:
            spec = base.load_spec(cell.task)
            store = RunStore(runs_dir, f"baseline-{cell.task}")
            baselines[cell.task] = (
                controller.baseline_attempt(run_cfg, spec, store), store.attempt_dir(0)
            )

        store = RunStore(
            runs_dir, f"{cell.task}-{cell.condition.replace('/', '-')}-r{cell.replicate}"
        )
        try:
            baseline, artifacts = baselines[cell.task]
            result = controller.run(
                run_cfg, make_agent(run_cfg), store,
                baseline=baseline, baseline_artifacts=artifacts,
            )
            row = _row(cell, result)
        except Exception:  # noqa: BLE001 - one bad cell must not end the sweep
            row = _error_row(cell, run_cfg.llm.model)

        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        verdict = "accept" if row["accepted"] else row["rule"]
        speed = f"{row['speedup']:.3f}x" if row["speedup"] else "-"
        log(f"[{index}/{len(todo)}] {cell.key:52s} {verdict:12s} {speed}")

    return rows


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fraction_of_ceiling(task: str, speedup: float | None, accepted: bool) -> float:
    """How much of the available improvement was captured, in [0, 1].

    A rejected proposal scores 0 because the seed is what ships. Undefined for
    controls, where there is no ceiling to be a fraction of.
    """
    # Keyed on the classification, not on the measured number: a control that
    # happened to measure 1.01x has no headroom, only noise.
    if ceilings.is_null(task):
        raise ValueError(f"{task} is a control and has no headroom to be a fraction of")
    ceiling = ceilings.MEASURED_HEADROOM[task]
    if not accepted or speedup is None:
        return 0.0
    return max(0.0, min(1.0, (speedup - 1.0) / (ceiling - 1.0)))


def _stop_class(row: dict) -> str:
    reason = row.get("stop_reason", "")
    if reason.startswith("agent finished"):
        return "finished"
    if reason.startswith("agent stopped"):
        return "stopped calling tools"
    if "consecutive" in reason:
        return "patience"
    if "budget" in reason:
        return "budget"
    return "steps completed"


def report(rows: list[dict]) -> str:
    # Every condition present is reported, ordered by agent then by LEVELS, so a
    # new level or agent cannot be silently dropped from its own evaluation.
    conditions = sorted({r["condition"] for r in rows}, key=_condition_order)
    improvable = set(ceilings.by_classification("headroom"))
    controls = set(ceilings.by_classification("null"))

    lines = [
        "## Effect of condition",
        "",
        "| condition | improvable: accepted | mean % of ceiling | proposed a change | "
        "controls: false positives | evaluations/cell | cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition and not r["error"]]
        good = [r for r in subset if r["task"] in improvable]
        null = [r for r in subset if r["task"] in controls]
        fractions = [
            fraction_of_ceiling(r["task"], r["speedup"], r["accepted"]) for r in good
        ]
        cost = sum(r["cost_usd"] for r in subset)
        # Each half is reported on its own: a sweep restricted to one class of
        # task still gets the numbers it does have.
        accepted = f"{sum(r['accepted'] for r in good)}/{len(good)}" if good else "-"
        captured = f"{100 * sum(fractions) / len(fractions):.0f}%" if fractions else "-"
        false_positives = f"{sum(r['accepted'] for r in null)}/{len(null)}" if null else "-"
        changed = f"{sum(r['changed'] for r in good)}/{len(good)}" if good else "-"
        evaluations = (
            f"{sum(r.get('evaluations', 1) for r in subset) / len(subset):.1f}"
            if subset else "-"
        )
        lines.append(
            f"| {condition} | {accepted} | {captured} | {changed} | "
            f"{false_positives} | {evaluations} | ${cost:.4f} |"
        )

    lines += ["", "## Per task, mean % of ceiling reached", "",
              "| task | ceiling | " + " | ".join(conditions) + " |",
              "|---|---|" + "---|" * len(conditions)]
    swept = {r["task"] for r in rows}
    for task in sorted(swept & set(ceilings.by_classification("headroom")),
                       key=lambda t: (TIER_ORDER.index(ceilings.tier(t)), t)):
        cells_ = []
        for condition in conditions:
            subset = [r for r in rows if r["task"] == task
                      and r["condition"] == condition and not r["error"]]
            if not subset:
                cells_.append("-")
                continue
            fractions = [fraction_of_ceiling(task, r["speedup"], r["accepted"])
                         for r in subset]
            cells_.append(f"{100 * sum(fractions) / len(fractions):.0f}%")
        lines.append(
            f"| `{task}` ({ceilings.tier(task)}) | {ceilings.MEASURED_HEADROOM[task]:.2f}x | "
            + " | ".join(cells_) + " |"
        )

    lines += ["", "## Controls, times a proposal was wrongly accepted", "",
              "| task | " + " | ".join(conditions) + " |",
              "|---|" + "---|" * len(conditions)]
    for task in sorted(swept & set(ceilings.by_classification("null"))):
        cells_ = []
        for condition in conditions:
            subset = [r for r in rows if r["task"] == task
                      and r["condition"] == condition and not r["error"]]
            cells_.append(
                f"{sum(r['accepted'] for r in subset)}/{len(subset)}" if subset else "-"
            )
        lines.append(f"| `{task}` | " + " | ".join(cells_) + " |")

    # Why each run ended, per condition. For an agentic condition this is the
    # stop decision itself: whether the model finished on its own, ran out of
    # patience, or spent the budget.
    reasons = sorted({_stop_class(r) for r in rows if not r["error"]})
    if any(r.get("stop_reason", "").startswith("agent") for r in rows):
        lines += ["", "## How runs ended", "",
                  "| condition | " + " | ".join(reasons) + " |",
                  "|---|" + "---|" * len(reasons)]
        for condition in conditions:
            subset = [r for r in rows if r["condition"] == condition and not r["error"]]
            counts = [sum(_stop_class(r) == reason for r in subset) for reason in reasons]
            lines.append(f"| {condition} | " + " | ".join(str(c) for c in counts) + " |")

    errors = [r for r in rows if r["error"]]
    if errors:
        lines += ["", f"{len(errors)} cell(s) failed:"]
        lines += [f"- {r['task']}|{r['condition']}|{r['replicate']}" for r in errors[:10]]
    return "\n".join(lines)


TIER_ORDER = ("op", "block", "model")


def suite_table() -> str:
    lines = ["| task | tier | class | ceiling |", "|---|---|---|---|"]
    for name in sorted(ceilings.CLASSIFICATION,
                       key=lambda n: (TIER_ORDER.index(ceilings.tier(n)), n)):
        lines.append(
            f"| `{name}` | {ceilings.tier(name)} | {ceilings.CLASSIFICATION[name]} | "
            f"{ceilings.MEASURED_HEADROOM[name]:.2f}x |"
        )
    return "\n".join(lines)


def index(results_dir: Path) -> str:
    """Every table in ``results/``, regenerated from the rows on disk.

    The prose in the individual result files is written by hand; the numbers it
    quotes must be re-derivable, and this is the derivation. Run ``halo report``
    after any sweep, and before quoting a number.
    """
    sections = [
        "# Results index",
        "",
        "Generated by `halo report` from the `.jsonl` row files in this directory. "
        "Do not edit; edit the sweep or re-run it.",
        "",
        f"## The suite, ceilings measured on {ceilings.REFERENCE_DEVICE}",
        "",
        suite_table(),
    ]
    for path in sorted(results_dir.glob("*.jsonl")):
        rows = read(path)
        models = sorted({r.get("model") or "?" for r in rows})
        errors = sum(1 for r in rows if r["error"])
        cost = sum(r["cost_usd"] for r in rows)
        sections += [
            "",
            f"## `{path.name}`",
            "",
            f"{len(rows)} cells, {errors} failed, model(s): {', '.join(models)}, "
            f"total cost ${cost:.3f}. Narrative: `{path.with_suffix('.md').name}`."
            if path.with_suffix(".md").exists() else
            f"{len(rows)} cells, {errors} failed, model(s): {', '.join(models)}, "
            f"total cost ${cost:.3f}.",
            "",
            "\n".join("#" + l if l.startswith("## ") else l for l in report(rows).splitlines()),
        ]
    return "\n".join(sections) + "\n"
