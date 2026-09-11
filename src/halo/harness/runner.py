"""The subprocess boundary: source files in, a :class:`Measurement` out."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from halo.config import RunConfig
from halo.tasks.base import TaskSpec
from halo.types import Measurement, decode


def measure(
    spec: TaskSpec,
    baseline_path: Path,
    candidate_path: Path,
    cfg: RunConfig,
    *,
    artifacts: Path | None = None,
    correctness_only: bool = False,
) -> Measurement:
    """Measure ``candidate_path`` against ``baseline_path`` in a fresh process.

    A crash, a hang or an out-of-memory kill in the candidate comes back as a
    failed measurement rather than taking down the controller. ``artifacts``
    receives the candidate's raw jaxpr, StableHLO and optimized HLO.
    ``correctness_only`` skips the timing: it is the cheap check an agent uses to
    debug a rewrite before spending a full evaluation on it.
    """
    with tempfile.TemporaryDirectory(prefix="halo-measure-") as tmp:
        request_path = Path(tmp) / "request.json"
        result_path = Path(tmp) / "measurement.json"
        # XLA writes its buffer-assignment report only as a side effect of
        # --xla_dump_to. Measured cost of enabling it: none.
        dump_dir = Path(tmp) / "xla-dump"
        dump_dir.mkdir()
        env = dict(os.environ)
        env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} --xla_dump_to={dump_dir}".strip()
        request_path.write_text(
            json.dumps(
                {
                    "task": spec.name,
                    "reference_sha256": spec.reference_sha256,
                    "baseline_path": str(baseline_path),
                    "candidate_path": str(candidate_path),
                    "config": dataclasses.asdict(cfg),
                    "out": str(result_path),
                    "artifacts": str(artifacts) if artifacts else None,
                    "dump_dir": str(dump_dir),
                    "mode": "correctness" if correctness_only else "measure",
                }
            )
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "halo.harness.worker", str(request_path)],
                capture_output=True,
                text=True,
                timeout=cfg.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return Measurement.failure(
                spec.name, f"measurement exceeded the {cfg.timeout_s:g}s timeout"
            )

        if not result_path.exists():
            tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            return Measurement.failure(
                spec.name,
                f"worker exited with code {completed.returncode} and wrote no "
                f"result:\n{tail}",
            )
        return decode(Measurement, json.loads(result_path.read_text()))
