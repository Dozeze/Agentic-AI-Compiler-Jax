#!/usr/bin/env bash
# Everything that needs the NV host, in one command (AIC-031 phase B, AIC-009 GPU containment, AIC-005 GPU
# calibration). Run inside WSL2 Ubuntu from the repository root, with Docker + NVIDIA Container Toolkit set
# up (docs/WSL2_CUDA_SETUP.md §5). Stops at the first failure. Copy the results into
# docs/TARGET_DECISION.md and docs/IMPLEMENTATION_PROGRESS.md.
set -euo pipefail

echo "== 1/6 locked CUDA environment"
uv sync --locked --extra cuda12

echo "== 2/6 full suite on the host, GPU tests required"
uv run --locked --extra cuda12 pytest -q --require-gpu -m "not needs_docker"

echo "== 3/6 target probe (AIC-031)"
uv run --locked --extra cuda12 python scripts/probe_target.py --backend cuda --label nvhost-wsl2-cuda \
    --out evidence/targets --try-docker-gpu

echo "== 4/6 worker images"
uv run --locked python scripts/build_worker_image.py --variant cpu
uv run --locked python scripts/build_worker_image.py --variant cuda12

echo "== 5/6 containment, CPU and GPU containers (AIC-009)"
uv run --locked --extra cuda12 pytest -q -m needs_docker --require-docker --require-gpu

echo "== 6/6 A/A calibration of the official population (AIC-005)"
uv run --locked python scripts/calibrate_aa.py --workload softmax --backend cuda --runner docker \
    --label nvhost-wsl2-cuda-docker

echo "phase B complete: record evidence/targets/nvhost-wsl2-cuda.json and evidence/calibration/ in the docs"
