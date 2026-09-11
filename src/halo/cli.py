"""Command line entry point: ``halo run``, ``halo ablate``, ``halo ceilings``, ``halo tasks``."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from halo import ablation, controller
from halo.agent import agentic, context, fake
from halo.agent.context import render, system_for
from halo.agent.single_shot import SingleShotAgent
from halo.config import BudgetConfig, RunConfig, TimingConfig
from halo.harness import runner
from halo.session import RunResult, Session
from halo.store import RunStore, new_run_id
from halo.tasks import base, ceilings

#: Agents that need no configuration, used for offline runs and tests.
SCRIPTED = {
    "fast": lambda: fake.ScriptedAgent(fake.FAST_ATTENTION, "fast"),
    "unstable": lambda: fake.ScriptedAgent(fake.UNSTABLE_ATTENTION, "unstable"),
    "echo": fake.EchoAgent,
}
LLM_AGENTS = ("single-shot", "agentic")


def _build_agent(cfg: RunConfig):
    name = cfg.agent
    if name in SCRIPTED:
        return SCRIPTED[name]()
    if name in LLM_AGENTS:
        from halo.agent.vertex import VertexGemini

        client = VertexGemini(cfg.llm)
        if name == "agentic":
            return agentic.AgenticAgent(client, cfg.context, name=f"agentic/{cfg.llm.model}")
        return SingleShotAgent(
            client, cfg.accept.min_speedup, name=cfg.llm.model, context_level=cfg.context
        )
    raise SystemExit(
        f"unknown agent '{name}'; choose from {', '.join(LLM_AGENTS + tuple(sorted(SCRIPTED)))}"
    )


def format_summary(result: RunResult, cfg: RunConfig) -> str:
    rows = ["| # | parent | verdict | speedup vs previous best (95% CI) | correctness |",
            "|---|---|---|---|---|"]
    for attempt in result.attempts:
        speedup = attempt.measurement.speedup
        correctness = attempt.measurement.correctness
        rows.append(
            "| {i} | {parent} | {verdict} | {speed} | {correct} |".format(
                i=attempt.index,
                parent="-" if attempt.parent is None else attempt.parent,
                verdict=("accepted" if attempt.decision.accepted
                         else f"rejected ({attempt.decision.rule})"),
                speed=(f"{speedup.ratio:.3f}x [{speedup.ci_low:.3f}, {speedup.ci_high:.3f}]"
                       if speedup else "-"),
                correct=("pass" if correctness and correctness.passed
                         else "fail" if correctness else "-"),
            )
        )

    best = result.best
    lines = [
        f"# {cfg.task}",
        "",
        f"Device: `{best.measurement.device.fingerprint if best.measurement.device else 'unknown'}`",
        f"Agent: {cfg.agent} · context: {cfg.context} · "
        f"acceptance threshold: {cfg.accept.min_speedup:.2f}x (95% CI lower bound)",
        "",
        *rows,
        "",
        f"Stopped: {result.stop_reason}. LLM calls: {len(result.usages)}, tool calls: "
        f"{result.tool_calls}, correctness checks: {result.checks}, "
        f"cost: ${result.cost_usd:.6f}",
        "",
    ]
    if result.improved and result.overall.speedup:
        lines.append(
            f"**Best: attempt {best.index}, {result.overall.speedup.describe()} "
            f"over the seed implementation.**"
        )
    else:
        lines.append(
            "**No proposal cleared the acceptance threshold; the seed implementation "
            "stands.**"
        )
    for attempt in result.attempts[1:]:
        if attempt.proposal and attempt.proposal.analysis:
            lines += ["", f"## Attempt {attempt.index} rationale",
                      "", f"*Hypothesis:* {attempt.proposal.hypothesis}",
                      "", attempt.proposal.analysis]
    return "\n".join(lines) + "\n"


def _ablate(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        task="", steps=args.steps,
        budget=BudgetConfig(max_evaluations=args.steps, patience=args.patience),
    )
    if args.model:
        cfg = cfg.with_overrides(llm=replace(cfg.llm, model=args.model))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = ablation.sweep(
        cfg,
        tasks=[args.task] if args.task else base.available(),
        conditions=args.conditions.split(","),
        replicates=args.replicates,
        out=out,
        runs_dir=Path(args.runs_dir) / "ablation",
        make_agent=_build_agent,
    )
    summary = ablation.report(rows)
    out.with_suffix(".md").write_text(summary + "\n")
    print()
    print(summary)
    print(f"\nrows: {out}    report: {out.with_suffix('.md')}")
    return 0


def _ceilings(args: argparse.Namespace) -> int:
    """Measure the best known implementation against the seed, per task.

    Calibrates the suite: it says how much room each benchmark actually has, so an
    agent's speedup can be read as a fraction of what was available.
    """
    cfg = RunConfig(
        task="", timing=TimingConfig(warmup=5, rounds=args.rounds, bootstrap_samples=800)
    )
    names = [args.task] if args.task else base.available()
    print(f"{'task':24s} {'tier':6s} {'class':9s} {'seed ms':>9s} {'best ms':>9s} {'headroom':>9s}")
    for name in names:
        spec = base.load_spec(name)
        measurement = runner.measure(
            spec, spec.directory / "candidate.py", ceilings.path(spec),
            cfg.with_overrides(task=name),
        )
        if not measurement.ok:
            print(f"{name:24s} ERROR {measurement.error.splitlines()[-1][:60]}")
            continue
        speedup = measurement.speedup
        print(
            f"{name:24s} {ceilings.tier(name):6s} {ceilings.CLASSIFICATION.get(name, '?'):9s} "
            f"{measurement.baseline.median_ms:9.3f} {measurement.candidate.median_ms:9.3f} "
            f"{speedup.ratio:8.2f}x  [{speedup.ci_low:.2f}, {speedup.ci_high:.2f}]"
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    """Regenerate every results table from the rows on disk."""
    if args.rows:
        print(ablation.report(ablation.read(Path(args.rows))))
        return 0
    results = Path(args.results_dir)
    text = ablation.index(results)
    (results / "README.md").write_text(text)
    print(text)
    return 0


def _tasks(args: argparse.Namespace) -> int:
    print(f"{'task':24s} {'tier':6s} {'class':9s} {'ceiling':>8s}")
    for name in base.available():
        print(
            f"{name:24s} {ceilings.tier(name):6s} {ceilings.CLASSIFICATION.get(name, '?'):9s} "
            f"{ceilings.MEASURED_HEADROOM.get(name, float('nan')):7.2f}x"
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    cfg = RunConfig.from_toml(args.config) if args.config else RunConfig(task=args.task)
    cfg = cfg.with_overrides(
        task=args.task, steps=args.steps, agent=args.agent, context=args.context,
    )
    if args.model:
        cfg = cfg.with_overrides(llm=replace(cfg.llm, model=args.model))
    if args.evaluations or args.patience:
        cfg = cfg.with_overrides(
            budget=replace(
                cfg.budget,
                **{k: v for k, v in
                   {"max_evaluations": args.evaluations, "patience": args.patience}.items()
                   if v},
            )
        )

    if args.dry_run:
        store = RunStore(Path(args.runs_dir), new_run_id(f"dryrun-{cfg.task}"))
        session = Session(cfg, store)
        print("=" * 30, "SYSTEM", "=" * 30)
        if cfg.agent == "agentic":
            print(agentic.system_prompt(cfg.context))
            print("=" * 30, "USER", "=" * 32)
            print(agentic.opening(session, cfg.context))
        else:
            print(system_for(cfg.context))
            print("=" * 30, "USER", "=" * 32)
            ctx = controller.build_context(session.spec, session.task, session.best, session.attempts)
            print(render(ctx, min_speedup=cfg.accept.min_speedup, level=cfg.context))
        return 0

    agent = _build_agent(cfg)
    store = RunStore(Path(args.runs_dir), new_run_id(cfg.task))
    result = controller.run(cfg, agent, store)

    summary = format_summary(result, cfg)
    store.write_summary(summary)
    print(summary)
    print(f"artifacts: {store.root}")
    return 0 if result.improved else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one optimization loop")
    run_parser.add_argument("--task", required=True, choices=base.available())
    run_parser.add_argument("--agent", default=None,
                            help="single-shot (default), agentic, or a scripted agent: "
                                 "fast, unstable, echo")
    run_parser.add_argument("--model", default=None, help="override the LLM model id")
    run_parser.add_argument("--steps", type=int, default=None,
                            help="proposals a single-shot agent gets")
    run_parser.add_argument("--evaluations", type=int, default=None,
                            help="evaluations an agentic agent may spend")
    run_parser.add_argument("--patience", type=int, default=None,
                            help="consecutive rejections before an agentic run stops")
    run_parser.add_argument("--context", default=None, choices=context.LEVELS,
                            help="how much the agent is shown (default: algorithmic)")
    run_parser.add_argument("--config", default=None, help="path to a TOML run config")
    run_parser.add_argument("--runs-dir", default="runs")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="measure the baseline and print the prompt; no LLM call")
    run_parser.set_defaults(func=_run)

    ablate_parser = sub.add_parser(
        "ablate", help="sweep task x condition and score the result"
    )
    ablate_parser.add_argument(
        "--conditions", default=",".join(context.LEVELS),
        help="comma-separated; a context level (single-shot agent) or agent/level, "
             "e.g. code,algorithmic,agentic/algorithmic",
    )
    ablate_parser.add_argument("--replicates", type=int, default=3)
    ablate_parser.add_argument("--task", default=None, choices=base.available())
    ablate_parser.add_argument("--steps", type=int, default=1,
                               help="measurements per cell: steps for single-shot, "
                                    "the evaluation budget for agentic")
    ablate_parser.add_argument("--patience", type=int, default=3)
    ablate_parser.add_argument("--model", default=None)
    ablate_parser.add_argument("--out", default="runs/ablation.jsonl")
    ablate_parser.add_argument("--runs-dir", default="runs")
    ablate_parser.set_defaults(func=_ablate)

    ceilings_parser = sub.add_parser(
        "ceilings", help="measure each task's best known implementation vs its seed"
    )
    ceilings_parser.add_argument("--task", default=None, choices=base.available())
    ceilings_parser.add_argument("--rounds", type=int, default=15)
    ceilings_parser.set_defaults(func=_ceilings)

    report_parser = sub.add_parser(
        "report", help="regenerate results/README.md from the .jsonl rows in results/"
    )
    report_parser.add_argument("--results-dir", default="results")
    report_parser.add_argument("--rows", default=None,
                               help="print the report for one .jsonl file instead")
    report_parser.set_defaults(func=_report)

    tasks_parser = sub.add_parser("tasks", help="list available benchmark tasks")
    tasks_parser.set_defaults(func=_tasks)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
