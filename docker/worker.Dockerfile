# Evaluation worker image (AIC-009).
#
# Dependencies come from uv.lock, so the container runs exactly the locked JAX/jaxlib. Only the harness
# modules the worker needs are copied in: no repository, no tests, no credentials. The image runs as
# `nobody`; the runner adds the rest (no network, read-only root, no capabilities, resource limits).
#
# Build with scripts/build_worker_image.py, never by hand: the tag is a hash of exactly these inputs.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# EXTRA=cuda12 builds the GPU variant (jax[cuda12] wheels; the driver comes from --gpus at run time).
ARG EXTRA=""
RUN uv sync --locked --no-dev --no-install-project ${EXTRA:+--extra "$EXTRA"} && rm -rf /root/.cache

COPY worker.py benchmark.py contracts.py environment.py array_io.py ./
COPY workloads/ ./workloads/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 65534:65534
ENTRYPOINT ["python", "/app/worker.py"]
