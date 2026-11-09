"""Build the evaluation worker image for the current harness files (AIC-009).

    uv run --locked python scripts/build_worker_image.py [--variant cpu|cuda12] [--force]

The build context is a temporary directory holding only the Dockerfile and runner.IMAGE_FILES, so
nothing else from the repository (no .env*, no runs/, no .git) can reach the image. The tag is
runner.image_reference(variant): a content hash, so an existing image with that tag is up to date.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402


def image_exists(reference: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", reference], capture_output=True).returncode == 0


def build(variant: str, force: bool = False) -> str:
    reference = runner.image_reference(variant)
    if image_exists(reference) and not force:
        print(f"{reference} is up to date")
        return reference
    with tempfile.TemporaryDirectory(prefix="aic-worker-context-") as tmp:
        context = Path(tmp)
        for path in runner.image_files():
            target = context / path.relative_to(runner.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(runner.DOCKERFILE, context / "Dockerfile")
        subprocess.run(["docker", "build", "--build-arg", f"EXTRA={runner.VARIANTS[variant]}",
                        "--label", "aic.worker.variant=" + variant, "-t", reference, str(context)], check=True)
    print(f"built {reference}")
    return reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(runner.VARIANTS), default="cpu")
    parser.add_argument("--force", action="store_true", help="rebuild even if the tag exists")
    args = parser.parse_args(argv)
    build(args.variant, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
