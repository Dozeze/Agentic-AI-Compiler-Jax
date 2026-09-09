"""Sweep task x context-level x replicate, and score the result.

The experiment the project exists to run: hold the model, the tasks and the
acceptance policy fixed, vary only how much the agent is told, and see whether
compiler feedback changes what it produces.

Two things make the scoring meaningful. Raw speedup is not comparable across
tasks, so improvable tasks are scored as the *fraction of the known ceiling*
reached (see :mod:`halo.tasks.ceilings`). And a suite of only improvable tasks
would reward a model that proposes a rewrite every time, so the controls are
scored separately, by how often something was wrongly accepted.
"""

from __future__ import annotations

import ast
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from halo import controller
from halo.agent.context import LEVELS
from halo.config import RunConfig
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


def is_changed(before: str, after: str) -> bool:
    """Did the proposal alter the program, rather than only its formatting?

    Compared as syntax trees. A model that returns the seed with one blank line
    moved has proposed nothing, and counting that as a change would hide the most
    interesting thing an ablation can find: a condition under which the agent
    declines to act.
    """
    try:
        return ast.dump(ast.parse(before)) != ast.dump(ast.parse(after))
    except SyntaxError:
        return before.strip() != after.strip()


def _row(cell: Cell, result: controller.RunResult) -> dict:
    attempt = result.attempts[-1]
    speedup = attempt.measurement.speedup
    usage = attempt.proposal.usage if attempt.proposal else None
    return {
        **cell.__dict__,
        "accepted": attempt.decision.accepted,
        "rule": attempt.decision.rule,
        "speedup": speedup.ratio if speedup else None,
        "ci_low": speedup.ci_low if speedup else None,
        "changed": attempt.proposal is not None
        and is_changed(result.attempts[0].source, attempt.source),
        "cost_usd": usage.cost_usd if usage else 0.0,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "error": None,
    }


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
    baselines: dict[str, Attempt] = {}

    todo = [c for c in cells(tasks, conditions, replicates)
            if (c.task, c.condition, c.replicate) not in done]
    log(f"{len(todo)} cells to run, {len(done)} already done")

    for index, cell in enumerate(todo, 1):
        run_cfg = cfg.with_overrides(task=cell.task, context=cell.condition)
        if cell.task not in baselines:
            spec = base.load_spec(cell.task)
            store = RunStore(runs_dir, f"baseline-{cell.task}")
            baselines[cell.task] = controller.baseline_attempt(run_cfg, spec, store)

        store = RunStore(runs_dir, f"{cell.task}-{cell.condition}-r{cell.replicate}")
        try:
            result = controller.run(
                run_cfg, make_agent(run_cfg), store, baseline=baselines[cell.task]
            )
            row = _row(cell, result)
        except Exception:  # noqa: BLE001 - one bad cell must not end the sweep
            row = {**cell.__dict__, "accepted": False, "rule": "error",
                   "speedup": None, "ci_low": None, "changed": False,
                   "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                   "error": traceback.format_exc(limit=5)}

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


def report(rows: list[dict]) -> str:
    # Ordered by LEVELS rather than a copy of it, so a new level cannot be
    # silently dropped from the report that is supposed to evaluate it.
    present = {r["condition"] for r in rows}
    conditions = [c for c in LEVELS if c in present]
    improvable = set(ceilings.by_classification("headroom"))
    controls = set(ceilings.by_classification("null"))

    lines = [
        "## Effect of context level",
        "",
        "| context | improvable: accepted | mean % of ceiling | proposed a change | "
        "controls: false positives | cost |",
        "|---|---|---|---|---|---|",
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
        lines.append(
            f"| {condition} | {accepted} | {captured} | {changed} | "
            f"{false_positives} | ${cost:.4f} |"
        )

    lines += ["", "## Per task, mean % of ceiling reached", "",
              "| task | ceiling | " + " | ".join(conditions) + " |",
              "|---|---|" + "---|" * len(conditions)]
    for task in ceilings.by_classification("headroom"):
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
            f"| `{task}` | {ceilings.MEASURED_HEADROOM[task]:.2f}x | "
            + " | ".join(cells_) + " |"
        )

    lines += ["", "## Controls, times a proposal was wrongly accepted", "",
              "| task | " + " | ".join(conditions) + " |",
              "|---|" + "---|" * len(conditions)]
    for task in ceilings.by_classification("null"):
        cells_ = []
        for condition in conditions:
            subset = [r for r in rows if r["task"] == task
                      and r["condition"] == condition and not r["error"]]
            cells_.append(
                f"{sum(r['accepted'] for r in subset)}/{len(subset)}" if subset else "-"
            )
        lines.append(f"| `{task}` | " + " | ".join(cells_) + " |")

    errors = [r for r in rows if r["error"]]
    if errors:
        lines += ["", f"{len(errors)} cell(s) failed:"]
        lines += [f"- {r['task']}|{r['condition']}|{r['replicate']}" for r in errors[:10]]
    return "\n".join(lines)
