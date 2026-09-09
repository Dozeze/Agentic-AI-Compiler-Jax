"""Command line entry point: ``halo run``, ``halo tasks``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from halo import ablation, controller
from halo.agent import fake
from halo.agent import context
from halo.agent.context import SYSTEM, render
from halo.agent.single_shot import SingleShotAgent
from halo.config import RunConfig, TimingConfig
from halo.controller import RunResult
from halo.harness import runner
from halo.store import RunStore, new_run_id
from halo.tasks import base, ceilings

AGENTS = {
    "fast": lambda cfg: fake.ScriptedAgent(fake.FAST_ATTENTION, "fast"),
    "unstable": lambda cfg: fake.ScriptedAgent(fake.UNSTABLE_ATTENTION, "unstable"),
    "echo": lambda cfg: fake.EchoAgent(),
}


def _build_agent(name: str, cfg: RunConfig):
    if name in AGENTS:
        return AGENTS[name](cfg)
    if name == "vertex":
        from halo.agent.vertex import VertexGemini

        return SingleShotAgent(
            VertexGemini(cfg.llm),
            cfg.accept.min_speedup,
            name=cfg.llm.model,
            context_level=cfg.context,
        )
    raise SystemExit(
        f"unknown agent '{name}'; choose from {', '.join(sorted(AGENTS))}, vertex"
    )


def format_summary(result: RunResult, cfg: RunConfig) -> str:
    rows = ["| # | agent verdict | speedup (95% CI) | correctness | cost |",
            "|---|---|---|---|---|"]
    total_cost = 0.0
    for attempt in result.attempts:
        speedup = attempt.measurement.speedup
        correctness = attempt.measurement.correctness
        usage = attempt.proposal.usage if attempt.proposal else None
        if usage:
            total_cost += usage.cost_usd
        rows.append(
            "| {i} | {verdict} | {speed} | {correct} | {cost} |".format(
                i=attempt.index,
                verdict=("accepted" if attempt.decision.accepted
                         else f"rejected ({attempt.decision.rule})"),
                speed=(f"{speedup.ratio:.3f}x [{speedup.ci_low:.3f}, {speedup.ci_high:.3f}]"
                       if speedup else "-"),
                correct=("pass" if correctness and correctness.passed
                         else "fail" if correctness else "-"),
                cost=(f"${usage.cost_usd:.6f}" if usage else "-"),
            )
        )

    best = result.best
    lines = [
        f"# {cfg.task}",
        "",
        f"Device: `{best.measurement.device.fingerprint if best.measurement.device else 'unknown'}`",
        f"Acceptance threshold: {cfg.accept.min_speedup:.2f}x (95% CI lower bound)",
        "",
        *rows,
        "",
        f"Total LLM cost: ${total_cost:.6f}",
        "",
    ]
    if result.improved and best.measurement.speedup:
        lines.append(
            f"**Best: attempt {best.index}, {best.measurement.speedup.describe()} "
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
    cfg = RunConfig(task="", steps=1, agent=args.agent)
    if args.model:
        cfg = cfg.with_overrides(llm=type(cfg.llm)(**{**vars(cfg.llm), "model": args.model}))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = ablation.sweep(
        cfg,
        tasks=[args.task] if args.task else base.available(),
        conditions=args.conditions.split(","),
        replicates=args.replicates,
        out=out,
        runs_dir=Path(args.runs_dir) / "ablation",
        make_agent=lambda c: _build_agent(c.agent, c),
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
    print(f"{'task':24s} {'class':9s} {'seed ms':>9s} {'best ms':>9s} {'headroom':>9s}")
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
            f"{name:24s} {ceilings.CLASSIFICATION.get(name, '?'):9s} "
            f"{measurement.baseline.median_ms:9.3f} {measurement.candidate.median_ms:9.3f} "
            f"{speedup.ratio:8.2f}x  [{speedup.ci_low:.2f}, {speedup.ci_high:.2f}]"
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    cfg = RunConfig.from_toml(args.config) if args.config else RunConfig(task=args.task)
    cfg = cfg.with_overrides(
        task=args.task, steps=args.steps, agent=args.agent,
        context=getattr(args, 'context', None),
    )
    if args.model:
        cfg = cfg.with_overrides(llm=type(cfg.llm)(**{**vars(cfg.llm), "model": args.model}))

    if args.dry_run:
        spec = base.load_spec(cfg.task)
        store = RunStore(Path(args.runs_dir), new_run_id(f"dryrun-{cfg.task}"))
        baseline = controller.baseline_attempt(cfg, spec, store)
        ctx = controller.build_context(spec, spec.load(), baseline, [baseline])
        print("=" * 30, "SYSTEM", "=" * 30)
        print(SYSTEM)
        print("=" * 30, "USER", "=" * 32)
        print(render(ctx, min_speedup=cfg.accept.min_speedup, level=cfg.context))
        return 0

    agent = _build_agent(cfg.agent, cfg)
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
    run_parser.add_argument("--agent", default="vertex",
                            help="vertex, or a scripted agent: fast, unstable, echo")
    run_parser.add_argument("--model", default=None, help="override the LLM model id")
    run_parser.add_argument("--steps", type=int, default=1)
    run_parser.add_argument("--context", default=None, choices=context.LEVELS,
                            help="how much the agent is shown (default: full)")
    run_parser.add_argument("--config", default=None, help="path to a TOML run config")
    run_parser.add_argument("--runs-dir", default="runs")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="measure the baseline and print the prompt; no LLM call")
    run_parser.set_defaults(func=_run)

    ablate_parser = sub.add_parser(
        "ablate", help="sweep task x context level and score the result"
    )
    ablate_parser.add_argument("--conditions", default=",".join(context.LEVELS))
    ablate_parser.add_argument("--replicates", type=int, default=3)
    ablate_parser.add_argument("--task", default=None, choices=base.available())
    ablate_parser.add_argument("--agent", default="vertex")
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

    tasks_parser = sub.add_parser("tasks", help="list available benchmark tasks")
    tasks_parser.set_defaults(
        func=lambda args: (print("\n".join(base.available())), 0)[1]
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
