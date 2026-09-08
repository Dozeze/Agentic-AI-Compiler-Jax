"""The optimization loop and the accept/reject policy.

For the MVP ``steps`` is 1, but the loop is written as a loop: making it
autonomous is a matter of raising the step count and giving it a stopping rule,
not restructuring anything.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from halo.agent.protocol import Agent, AttemptSummary, Context
from halo.config import RunConfig
from halo.harness import runner
from halo.store import RunStore
from halo.tasks import base
from halo.types import Attempt, Decision, Measurement


def decide(measurement: Measurement, cfg: RunConfig) -> Decision:
    """Correctness first, then performance, then the noise threshold."""
    if not measurement.ok:
        return Decision(False, "error", measurement.error or "measurement failed")

    report = measurement.correctness
    if report is None:
        return Decision(False, "error", "no correctness report produced")
    if not report.passed:
        return Decision(False, "correctness", report.summary())

    speedup = measurement.speedup
    if speedup is None:
        return Decision(False, "error", "correctness passed but no timing was recorded")

    threshold = cfg.accept.min_speedup
    if speedup.ci_low < threshold:
        return Decision(
            False,
            "performance",
            f"speedup {speedup.describe()}; the {speedup.confidence:.0%} lower bound "
            f"{speedup.ci_low:.3f} does not clear the {threshold:.2f}x threshold, so "
            f"the change is not distinguishable from measurement noise",
        )
    return Decision(True, "accepted", f"speedup {speedup.describe()}")


def _diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="candidate.py (before)",
            tofile="candidate.py (after)",
            n=3,
        )
    ) or "(no textual change)"


@dataclass
class RunResult:
    attempts: list[Attempt]
    best: Attempt

    @property
    def improved(self) -> bool:
        return self.best.index > 0


def baseline_attempt(cfg: RunConfig, spec, store: RunStore) -> Attempt:
    """Measure the seed implementation against itself.

    The speedup is 1.0 by construction; what we want is the baseline timing and
    HLO on *this* machine, in this session, obtained through exactly the code path
    every later attempt takes.
    """
    directory = store.attempt_dir(0)
    (directory / "candidate.py").write_text(spec.seed_source)
    seed_path = spec.directory / "candidate.py"
    measurement = runner.measure(
        spec, seed_path, seed_path, cfg, hlo_dump=directory / "hlo.txt"
    )
    if not measurement.ok:
        raise RuntimeError(f"baseline measurement failed: {measurement.error}")
    return Attempt(
        index=0,
        source=spec.seed_source,
        measurement=measurement,
        decision=Decision(True, "accepted", "baseline"),
    )


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
                diff=_diff(spec.seed_source, a.source),
                accepted=a.decision.accepted,
                rule=a.decision.rule,
                reason=a.decision.reason,
                speedup=a.measurement.speedup.ratio if a.measurement.speedup else None,
            )
            for a in attempts[1:]
        ),
    )


def run(cfg: RunConfig, agent: Agent, store: RunStore) -> RunResult:
    spec = base.load_spec(cfg.task)
    task = spec.load()
    baseline_path = spec.directory / "candidate.py"

    store.write_config(cfg)
    attempts = [baseline_attempt(cfg, spec, store)]
    baseline_m = attempts[0].measurement
    store.write_env(
        {"device": baseline_m.device.fingerprint if baseline_m.device else None}
    )
    run_fingerprint = baseline_m.device.fingerprint if baseline_m.device else None
    store.save_attempt(attempts[0])
    best = attempts[0]

    for step in range(1, cfg.steps + 1):
        context = build_context(spec, task, best, attempts)
        try:
            proposal = agent.propose(context)
        except Exception as exc:  # noqa: BLE001 - a failed agent is data, not a crash
            attempts.append(
                Attempt(
                    index=step,
                    source=best.source,
                    measurement=Measurement.failure(spec.name, str(exc)),
                    decision=Decision(
                        False, "error", f"agent produced no proposal: {exc}"
                    ),
                )
            )
            store.save_attempt(attempts[-1])
            continue

        directory = store.attempt_dir(step)
        candidate_path = directory / "candidate.py"
        candidate_path.write_text(proposal.source)

        measurement = runner.measure(
            spec, baseline_path, candidate_path, cfg, hlo_dump=directory / "hlo.txt"
        )

        # Timings are only meaningful within one device. A measurement from a
        # different machine is not slower or faster than the baseline; it is
        # incomparable, and saying so is the whole point of the fingerprint.
        if (
            measurement.ok
            and measurement.device is not None
            and run_fingerprint is not None
            and measurement.device.fingerprint != run_fingerprint
        ):
            decision = Decision(
                False,
                "error",
                f"measured on {measurement.device.fingerprint} but the baseline was "
                f"taken on {run_fingerprint}; timings across devices are not comparable",
            )
        else:
            decision = decide(measurement, cfg)

        attempt = Attempt(
            index=step,
            source=proposal.source,
            measurement=measurement,
            decision=decision,
            proposal=proposal,
        )
        attempts.append(attempt)
        store.save_attempt(
            attempt, prompt=proposal.prompt, raw_response=proposal.raw_response
        )
        if decision.accepted:
            best = attempt

    return RunResult(attempts=attempts, best=best)
