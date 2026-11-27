"""Measure the available speedup ("ceiling") of each workload before any search.

For every workload, the team-written rewrite in workloads/ceilings/<id>.py goes through the same
pipeline as an agent candidate: correctness gate on every development and edge case, a fresh A/A
calibration of the reference, then paired sessions and the decision rule. The result says how much an
agent could gain on this host, so a search result can be reported as a fraction of it. Controls
(softmax, layer_norm, matmul_gelu) are expected to show no distinguishable gain.

    uv run --locked python scripts/measure_ceilings.py --backend cpu --runner docker --out evidence/ceilings/mac
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark import Calibration  # noqa: E402
from contracts import MeasurementProtocol  # noqa: E402
from controller import Evaluator  # noqa: E402
from environment import select_backend  # noqa: E402
from runner import DockerRunner, TrustedSubprocessRunner  # noqa: E402
from workloads import registry  # noqa: E402

CEILINGS = REPO / "workloads" / "ceilings"


def calibration_file(directory, spec):
    """The calibrate_aa.py record for this workload's measurement shape, or None."""
    if directory is None:
        return None
    path = Path(directory) / f"{spec.workload_id}-{'x'.join(map(str, spec.development_shapes[-1]))}.json"
    return path if path.exists() else None


def measure(spec, runner, workdir, backend, calibration=None):
    """One workload: correctness, calibration and judged sessions of ceiling vs reference."""
    frozen = Calibration.load(calibration) if calibration else None
    evaluator = Evaluator(spec, runner, MeasurementProtocol(), workdir, selected_backend=backend,
                          calibration=frozen)
    evaluator.reference_outputs()
    ceiling = (CEILINGS / f"{spec.workload_id}.py").read_text(encoding="utf-8")
    verdict = evaluator.evaluate(ceiling, evaluator.baseline_source)
    sessions = [s["analysis"] for s in verdict.evaluation.sessions if s["against"] == "incumbent"]
    return {
        "workload": spec.workload_id,
        "class": "headroom" if spec in registry.HEADROOM else "control",
        "shape": list(evaluator.measure_shape),
        "state": verdict.state,
        "reason": verdict.reason,
        "delta": (verdict.evaluation.decision or {}).get("delta"),
        "speedup": [s["speedup"]["speedup"] for s in sessions],
        "ci": [[s["speedup"]["ci_low"], s["speedup"]["ci_high"]] for s in sessions],
        "reference_median_s": [s["a"]["median_s"] for s in sessions],
        "ceiling_median_s": [s["b"]["median_s"] for s in sessions],
        "fingerprint": evaluator.fingerprint,
        "calibration": str(calibration) if calibration else "lazy",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--runner", choices=("docker", "subprocess"), default="docker")
    parser.add_argument("--out", type=Path, required=True, help="new directory for raw evidence and results.json")
    parser.add_argument("--workload", action="append", help="limit to these workload ids (repeatable)")
    parser.add_argument("--calibrations", type=Path, default=None,
                        help="directory of frozen calibrate_aa.py records; missing ones are taken lazily")
    args = parser.parse_args(argv)
    select_backend(args.backend)
    args.out.mkdir(parents=True, exist_ok=False)
    specs = [registry.get(w) for w in args.workload] if args.workload else [*registry.HEADROOM, *registry.CONTROLS]
    rows = []
    for spec in specs:
        runner = (DockerRunner(selected_backend=args.backend) if args.runner == "docker"
                  else TrustedSubprocessRunner(reviewed_fixtures=True))
        row = measure(spec, runner, args.out / spec.workload_id, args.backend,
                      calibration_file(args.calibrations, spec))
        rows.append(row)
        speed = ", ".join(f"{s:.2f}x [{lo:.2f}, {hi:.2f}]" for s, (lo, hi) in zip(row["speedup"], row["ci"]))
        print(f"{row['workload']:20s} {row['class']:8s} {row['state']:12s} {speed or '-'}  ({row['reason'][:80]})",
              flush=True)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
