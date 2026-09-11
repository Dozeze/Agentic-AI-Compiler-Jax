"""The ledger and the referee.

A :class:`Session` is one optimization run seen from the harness's side: the
attempts made so far, which of them is the verified best, and what the agent has
spent. Both kinds of agent drive it - the single-shot loop in
:mod:`halo.controller` and the tool-using agent in :mod:`halo.agent.agentic` -
and neither can accept a candidate, only the session can. That is what keeps a
model's claim of a speedup from ever becoming a reported one.

Every candidate is judged against the current best, not the seed. Interleaved
timing only cancels noise between the two things that ran back to back, and the
question at step N is "is this better than what we have", which after the first
accepted step is no longer the seed. Comparing to the seed instead would accept a
later candidate slower than an earlier one, as long as it still beat the
original, and the loop would walk backwards reporting progress.
"""

from __future__ import annotations

import ast
import difflib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from halo.agent.context import feedback_for
from halo.config import RunConfig
from halo.harness import runner
from halo.store import RunStore
from halo.tasks import base
from halo.types import Attempt, Decision, Measurement, Proposal, Usage


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


def diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="candidate.py (before)",
            tofile="candidate.py (after)",
            n=3,
        )
    ) or "(no textual change)"


def same_program(a: str, b: str) -> bool:
    """Are two sources the same program, ignoring formatting and comments?

    Compared as syntax trees. A rewrite that moves a blank line has proposed
    nothing; measuring it would spend an evaluation on the noise floor, and
    counting it as a change would hide the most interesting thing a sweep can
    find: a condition under which the agent declines to act.
    """
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except SyntaxError:
        return a.strip() == b.strip()


#: The raw text the worker leaves next to an attempt, for ``inspect``.
ARTIFACTS = ("jaxpr.txt", "stablehlo.txt", "hlo.txt")


def baseline_attempt(cfg: RunConfig, spec, store: RunStore) -> Attempt:
    """Measure the seed implementation against itself.

    The speedup is 1.0 by construction; what we want is the baseline timing and
    HLO on *this* machine, in this session, obtained through exactly the code path
    every later attempt takes.
    """
    directory = store.attempt_dir(0)
    (directory / "candidate.py").write_text(spec.seed_source)
    seed_path = spec.directory / "candidate.py"
    measurement = runner.measure(spec, seed_path, seed_path, cfg, artifacts=directory)
    if not measurement.ok:
        raise RuntimeError(f"baseline measurement failed: {measurement.error}")
    return Attempt(
        index=0,
        source=spec.seed_source,
        measurement=measurement,
        decision=Decision(True, "accepted", "baseline"),
    )


@dataclass
class RunResult:
    attempts: list[Attempt]
    best: Attempt
    #: The best implementation measured against the seed, for the headline. Each
    #: step's own measurement is against the version it replaced, which is what
    #: the accept/reject decision needs; this is what the report needs.
    overall: Measurement
    stop_reason: str = ""
    usages: list[Usage] = field(default_factory=list)
    tool_calls: int = 0
    checks: int = 0

    @property
    def improved(self) -> bool:
        return self.best.index > 0

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)


#: What ``inspect`` can show of an attempt. The three raw levels are the files
#: the worker wrote; the rest is derived from the ledger.
INSPECTABLE = ("source", "diff", "feedback", "jaxpr", "stablehlo", "hlo", "buffers")


class Session:
    def __init__(
        self,
        cfg: RunConfig,
        store: RunStore,
        baseline: Attempt | None = None,
        baseline_artifacts: Path | None = None,
    ) -> None:
        """``baseline`` may be a measurement of the seed taken earlier for the same
        task on the same machine, which halves the cost of a sweep; its artifacts
        directory comes along so the seed's HLO can still be inspected. It only
        feeds the agent's context; every accept/reject decision still rests on a
        fresh interleaved measurement."""
        self.cfg = cfg
        self.store = store
        self.spec = base.load_spec(cfg.task)
        self.task = self.spec.load()
        self._seed_path = self.spec.directory / "candidate.py"
        self._best_path = self._seed_path

        store.write_config(cfg)
        first = baseline if baseline is not None else baseline_attempt(cfg, self.spec, store)
        device = first.measurement.device
        self._fingerprint = device.fingerprint if device else None
        store.write_env({"device": self._fingerprint})
        store.save_attempt(first)
        if baseline_artifacts is not None:
            for name in ARTIFACTS:
                if (baseline_artifacts / name).exists():
                    shutil.copy(baseline_artifacts / name, store.attempt_dir(0) / name)

        self.attempts: list[Attempt] = [first]
        self.best: Attempt = first
        self.usages: list[Usage] = []
        self.tool_calls = 0
        self.checks = 0
        self.stop_reason: str | None = None
        self._rejected_in_a_row = 0

    # -- what has been spent ---------------------------------------------------

    @property
    def evaluations(self) -> int:
        return len(self.attempts) - 1

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)

    def charge(self, usage: Usage) -> None:
        self.usages.append(usage)

    @property
    def exhausted(self) -> str | None:
        """Why the run must stop now, or None. The budget belongs to the harness:
        an agent is told how much is left but never asked whether it agrees."""
        if self.stop_reason:
            return self.stop_reason
        budget = self.cfg.budget
        if self.evaluations >= budget.max_evaluations:
            return f"evaluation budget of {budget.max_evaluations} spent"
        if self.tool_calls >= budget.max_tool_calls:
            return f"tool-call budget of {budget.max_tool_calls} spent"
        if self.cost_usd >= budget.max_cost_usd:
            return f"cost budget of ${budget.max_cost_usd:.2f} spent"
        if self._rejected_in_a_row >= budget.patience:
            return f"{budget.patience} consecutive evaluations rejected"
        return None

    def remaining(self) -> dict:
        budget = self.cfg.budget
        return {
            "evaluations": budget.max_evaluations - self.evaluations,
            "tool_calls": budget.max_tool_calls - self.tool_calls,
            "cost_usd": round(budget.max_cost_usd - self.cost_usd, 4),
            "rejections_before_stop": budget.patience - self._rejected_in_a_row,
        }

    # -- the tools -------------------------------------------------------------

    def evaluate(
        self,
        source: str,
        *,
        proposal: Proposal | None = None,
        parent: int | None = None,
    ) -> Attempt:
        """Measure ``source`` against the current best and rule on it."""
        index = len(self.attempts)
        directory = self.store.attempt_dir(index)
        candidate_path = directory / "candidate.py"
        candidate_path.write_text(source)

        measurement = runner.measure(
            self.spec, self._best_path, candidate_path, self.cfg, artifacts=directory
        )
        attempt = Attempt(
            index=index,
            source=source,
            measurement=measurement,
            decision=self._decide(measurement),
            proposal=proposal,
            parent=self.best.index if parent is None else parent,
        )
        self._record(attempt)
        if attempt.decision.accepted:
            self.best = attempt
            self._best_path = candidate_path
            self._rejected_in_a_row = 0
        else:
            self._rejected_in_a_row += 1
        return attempt

    def duplicate_of(self, source: str) -> Attempt | None:
        """The earlier attempt ``source`` is the same program as, if any."""
        return next((a for a in self.attempts if same_program(a.source, source)), None)

    def record_failure(self, error: str) -> Attempt:
        """The agent produced nothing usable. That is data, not a crash."""
        attempt = Attempt(
            index=len(self.attempts),
            source=self.best.source,
            measurement=Measurement.failure(self.spec.name, error),
            decision=Decision(False, "error", f"agent produced no proposal: {error}"),
            parent=self.best.index,
        )
        self._record(attempt)
        self._rejected_in_a_row += 1
        return attempt

    def check(self, source: str) -> Measurement:
        """Correctness only. Cheap, and never counts against patience."""
        self.checks += 1
        path = self.store.root / "checks" / f"{self.checks:03d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return runner.measure(
            self.spec, self._best_path, path, self.cfg, correctness_only=True
        )

    def inspect(self, index: int, what: str, max_lines: int = 120) -> str:
        if what not in INSPECTABLE:
            raise ValueError(f"cannot inspect {what!r}; choose from {INSPECTABLE}")
        if not 0 <= index < len(self.attempts):
            raise ValueError(f"no attempt {index}; attempts so far: 0..{len(self.attempts) - 1}")
        attempt = self.attempts[index]
        if what == "source":
            text = attempt.source
        elif what == "diff":
            text = diff(self.attempts[0].source, attempt.source)
        elif what == "feedback":
            text = feedback_for(attempt.measurement, "full")
        elif what == "buffers":
            program = attempt.measurement.candidate_program
            report = program.buffers if program else None
            if report is None:
                return "buffer report not available for this attempt"
            text = "\n".join(
                f"{entry.size_bytes:>12,} B  x{entry.n_values:<3} {', '.join(entry.shapes)}"
                for entry in report.entries
            )
        else:
            path = self.store.attempt_dir(index) / f"{what}.txt"
            if not path.exists():
                return f"{what} text not available for attempt {index}"
            text = path.read_text()
        return _truncate(text, max_lines)

    def finish(self, summary: str) -> None:
        self.stop(f"agent finished: {summary}")

    def stop(self, reason: str) -> None:
        self.stop_reason = reason

    # -- the verdict -----------------------------------------------------------

    def result(self) -> RunResult:
        # Attempt 1's measurement is already against the seed. After a later
        # accepted step the best was measured against an intermediate, so take one
        # more interleaved measurement for a headline number with a real interval.
        overall = self.best.measurement
        if self.best.index > 1:
            overall = runner.measure(self.spec, self._seed_path, self._best_path, self.cfg)
        result = RunResult(
            attempts=self.attempts,
            best=self.best,
            overall=overall,
            stop_reason=self.exhausted or "not stopped",
            usages=self.usages,
            tool_calls=self.tool_calls,
            checks=self.checks,
        )
        self.store.write_result(result)
        return result

    # -- internals -------------------------------------------------------------

    def _decide(self, measurement: Measurement) -> Decision:
        # Timings are only meaningful within one device. A measurement from a
        # different machine is not slower or faster than the baseline; it is
        # incomparable, and saying so is the whole point of the fingerprint.
        device = measurement.device
        if (
            measurement.ok
            and device is not None
            and self._fingerprint is not None
            and device.fingerprint != self._fingerprint
        ):
            return Decision(
                False,
                "error",
                f"measured on {device.fingerprint} but the baseline was taken on "
                f"{self._fingerprint}; timings across devices are not comparable",
            )
        return decide(measurement, self.cfg)

    def _record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        proposal = attempt.proposal
        self.store.save_attempt(
            attempt,
            prompt=proposal.prompt if proposal else None,
            raw_response=proposal.raw_response if proposal else None,
        )


def _truncate(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
