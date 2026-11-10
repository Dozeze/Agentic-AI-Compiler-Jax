"""Held-out split management (AIC-018). See heldout.py for the commit-reveal scheme.

    uv run --locked python scripts/heldout.py init     --workload softmax
    uv run --locked python scripts/heldout.py verify   --workload softmax
    uv run --locked python scripts/heldout.py evaluate --workload softmax --candidate best.py --backend cuda
    uv run --locked python scripts/heldout.py reveal   --workload softmax      # only with the final results
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
    parser.add_argument("command", choices=("init", "verify", "evaluate", "reveal"))
    parser.add_argument("--workload", required=True, choices=sorted(registry.REGISTRY))
    parser.add_argument("--candidate", type=Path, help="evaluate: the frozen winner's source file")
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--runner", choices=("docker", "subprocess"), default="docker")
    args = parser.parse_args(argv)
    spec = registry.get(args.workload)
    try:
        if args.command == "init":
            print(f"salt written to {heldout.salt_path(spec)} (keep it private; share it only with whoever runs "
                  f"the final evaluation)\nset heldout_commitment=\"{heldout.init(spec)}\" for {spec.workload_id} "
                  "in workloads/registry.py and commit it before any search")
        elif args.command == "verify":
            salt = heldout.load_salt(spec)
            heldout.check_not_burned(spec)
            print(f"{spec.workload_id} v{spec.heldout_version}: salt matches the commitment; "
                  f"{len(heldout.cases(spec, salt))} held-out cases, split not burned")
        elif args.command == "evaluate":
            if args.candidate is None:
                parser.error("evaluate needs --candidate")
            from runner import DockerRunner, TrustedSubprocessRunner
            runner = (DockerRunner(selected_backend=args.backend) if args.runner == "docker"
                      else TrustedSubprocessRunner(reviewed_fixtures=False))
            reason = runner.preflight()
            if reason:
                print(f"error: ISOLATION_UNAVAILABLE: {reason}", file=sys.stderr)
                return 2
            with tempfile.TemporaryDirectory(prefix="aic-heldout-") as tmp:
                report, entry = heldout.evaluate(spec, args.candidate.read_text(encoding="utf-8"), runner, tmp,
                                                 selected_backend=args.backend)
            print(json.dumps(entry, indent=2))
            return 0 if report.passed else 1
        else:
            print(f"salt published to {heldout.reveal(spec)}: commit it with the final results")
    except heldout.HeldoutError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
