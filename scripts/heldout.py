"""Held-out split (AIC-018): check it, or run the final evaluation of a frozen winner. See heldout.py.

    uv run --locked python scripts/heldout.py check    --workload softmax
    uv run --locked python scripts/heldout.py evaluate --workload softmax --candidate best.py --backend cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import heldout  # noqa: E402
from environment import BACKENDS  # noqa: E402
from workloads import registry  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("check", "evaluate"))
    parser.add_argument("--workload", required=True, choices=sorted(registry.REGISTRY))
    parser.add_argument("--candidate", type=Path, help="evaluate: the frozen winner's source file")
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--runner", choices=("docker", "subprocess"), default="docker")
    args = parser.parse_args(argv)
    spec = registry.get(args.workload)
    try:
        if args.command == "check":
            n = len(heldout.cases(spec))
            heldout.check_not_burned(spec)
            print(f"{spec.workload_id} held-out split v{spec.heldout_version}: {n} cases "
                  f"({len(spec.heldout_seeds)} seeds x {len(spec.heldout_shapes)} unseen shapes), not burned")
            return 0
        if args.candidate is None:
            parser.error("evaluate needs --candidate")
        from runner import DockerRunner, TrustedSubprocessRunner
        runner = (DockerRunner(selected_backend=args.backend) if args.runner == "docker"
                  else TrustedSubprocessRunner())
        reason = runner.preflight()
        if reason:
            print(f"error: ISOLATION_UNAVAILABLE: {reason}", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="aic-heldout-") as tmp:
            report, entry = heldout.evaluate(spec, args.candidate.read_text(encoding="utf-8"), runner, tmp,
                                             selected_backend=args.backend)
        print(json.dumps(entry, indent=2))
        return 0 if report.passed else 1
    except heldout.HeldoutError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
