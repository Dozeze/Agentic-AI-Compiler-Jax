"""On-disk run artifacts.

Every attempt is written out in full: the exact source measured, the measurement,
the optimized HLO, and the prompt and raw response that produced it. Two reasons
this is not optional. The project plan's success criterion is a *reproducible*
system, and a claim you cannot re-derive from files is not reproducible. And the
report's tables and figures come from these files - not from a terminal buffer
that is gone by November.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from halo.config import RunConfig
from halo.types import Attempt, to_json

if TYPE_CHECKING:
    from halo.session import RunResult


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def new_run_id(task: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{task}"


class RunStore:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root) / run_id
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)

    def attempt_dir(self, index: int) -> Path:
        path = self.root / "attempts" / f"{index:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_config(self, cfg: RunConfig) -> None:
        (self.root / "config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2)
        )

    def write_env(self, extra: dict) -> None:
        env = {
            "python": sys.version,
            "executable": sys.executable,
            "git_commit": _git_commit(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        (self.root / "env.json").write_text(json.dumps(env, indent=2, default=str))

    def save_attempt(
        self,
        attempt: Attempt,
        *,
        prompt: str | None = None,
        raw_response: str | None = None,
    ) -> Path:
        directory = self.attempt_dir(attempt.index)
        (directory / "candidate.py").write_text(attempt.source)
        (directory / "measurement.json").write_text(to_json(attempt.measurement))
        (directory / "decision.json").write_text(to_json(attempt.decision))
        if attempt.proposal is not None:
            (directory / "proposal.json").write_text(
                to_json(
                    dataclasses.replace(
                        attempt.proposal, raw_response=None, prompt=None
                    )
                )
            )
        if prompt is not None:
            (directory / "prompt.txt").write_text(prompt)
        if raw_response is not None:
            (directory / "response.txt").write_text(raw_response)
        return directory

    def write_transcript(self, turns: list[dict]) -> None:
        """A tool-using agent's whole conversation: every call and every result."""
        (self.root / "transcript.json").write_text(json.dumps(turns, indent=2))

    def write_result(self, result: RunResult) -> None:
        (self.root / "result.json").write_text(
            to_json(
                {
                    "best": result.best.index,
                    "improved": result.improved,
                    "overall": result.overall,
                    "stop_reason": result.stop_reason,
                    "cost_usd": result.cost_usd,
                    "usages": result.usages,
                    "tool_calls": result.tool_calls,
                    "checks": result.checks,
                    "attempts": [
                        {"index": a.index, "parent": a.parent,
                         "accepted": a.decision.accepted, "rule": a.decision.rule}
                        for a in result.attempts
                    ],
                }
            )
        )

    def write_summary(self, text: str) -> None:
        (self.root / "summary.md").write_text(text)
