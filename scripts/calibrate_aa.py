"""Freeze an A/A calibration for one workload on this measurement population (AIC-005, AIC-1.1 §5.6).

    uv run --locked python scripts/calibrate_aa.py --workload softmax --backend cpu --runner docker \
        --label mac-cpu-docker [--config run.json] [--out evidence/calibration]

Runs the frozen reference, then `aa_sessions` independent A/A sessions (each in a fresh worker) of the
baseline against itself, and writes evidence/calibration/<label>/<workload>-<shape>.json. Pass that file
to agent.py with --calibration. Exit code 1, with the reasons, when the host is too noisy: then lengthen
the blocks or pick a larger declared shape before any search, never loosen the limits after the fact.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark import CalibrationError  # noqa: E402
from contracts import BACKENDS, ConfigError, resolve_run_config  # noqa: E402
from controller import BaselineFailed, EvaluationError, Evaluator  # noqa: E402
from runner import DockerRunner, TrustedSubprocessRunner  # noqa: E402
from workloads import registry  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", required=True, choices=sorted(registry.REGISTRY))
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--runner", choices=("docker", "subprocess"), default="docker")
    parser.add_argument("--label", required=True, help="the measurement population, e.g. nvhost-wsl2-cuda-docker")
    parser.add_argument("--config", type=Path, default=None, help="RunConfig v1 file: its `measurement` protocol is used")
    parser.add_argument("--out", type=Path, default=REPO / "evidence" / "calibration")
    args = parser.parse_args(argv)
    try:
        protocol = resolve_run_config({}, args.config).measurement
    except ConfigError as error:
        parser.error(str(error))

    runner = DockerRunner(selected_backend=args.backend) if args.runner == "docker" else TrustedSubprocessRunner()
    reason = runner.preflight()
    if reason is not None:
        print(f"error: {reason}", file=sys.stderr)
        return 2
    spec = registry.get(args.workload)
    with tempfile.TemporaryDirectory(prefix="aic-calibrate-") as tmp:
        evaluator = Evaluator(spec, runner, protocol, tmp, selected_backend=args.backend)
        try:
            evaluator.reference_outputs()
            calibration = evaluator.calibrate()
        except (BaselineFailed, EvaluationError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        except CalibrationError as error:
            print(f"not calibrated: {error}", file=sys.stderr)
            return 1
    calibration = dataclasses.replace(calibration, label=args.label)
    shape = "x".join(str(n) for n in calibration.shape)
    path = args.out / args.label / f"{args.workload}-{shape}.json"
    calibration.save(path)
    print(f"delta = {calibration.delta:.4f} from {len(calibration.aa_relative_differences)} A/A sessions "
          f"({calibration.repeats} calls per block) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
