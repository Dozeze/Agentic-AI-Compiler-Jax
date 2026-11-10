For Jonathan L:

Install the locked CPU development environment and run a bounded optimization loop:

```bash
uv sync --locked
uv run --locked python agent.py --iterations 1
```

The command reads `ANTHROPIC_API_KEY` from `.env.local`. Increase `--iterations` after
the first smoke run. Add `--plot` only when a graphical runtime plot is wanted.

On Linux with a compatible NVIDIA GPU, use the already locked CUDA 12 extra:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 python -c "import jax; print(jax.devices())"
uv run --locked --extra cuda12 python agent.py --iterations 1 --require-gpu
```

`--require-gpu` stops before calling Claude if JAX only finds a CPU. There is
no need to edit `pyproject.toml` manually. `uv run --no-project --with anthropic`
skips this project's declared dependencies and lockfile, so it is not the
recommended way to run this project.
The old `jax[cuda13]==0.4.38` line does not install CUDA 13 support: JAX
0.4.38 does not provide that extra. Check `jax.devices()` before benchmarking.

To optimize a Python workload file, give it a callable function and either a
`TEST_ARGS` tuple or a `make_test_args()` function:

```python
import jax.numpy as jnp

TEST_ARGS = (jnp.arange(32, dtype=jnp.float32), jnp.ones(32, dtype=jnp.float32))

def bad_func(x, y):
    return x + y
```

Run that file with:

```bash
uv run --locked python agent.py --source workloads/example.py \
  --function bad_func --iterations 3 --output-dir runs/example
```

On the GPU machine, add `--extra cuda12` after `--locked` and `--require-gpu`
after `agent.py` to this command.

Each evaluation writes `iteration_0000.json`, `iteration_0001.json`, and so on,
plus a cumulative `history.json`. The JSON records include the complete candidate
source, unified source diff, runtime mean/std, correctness, and accept/reject decision.

## Backend and run configuration

Select the JAX backend explicitly; the run stops with exit code 2 if it is unavailable instead of silently
falling back to CPU:

```bash
uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1
uv run --locked --extra cuda12 python agent.py --backend cuda --require-gpu --iterations 1   # inside WSL2 on the NV host
```

`--config run.json` loads a JSON `RunConfig v1` (fields in `contracts.py`), for example:

```json
{"backend": "cpu", "budgets": {"max_candidates": 2}, "model": {"model_id": "claude-haiku-4-5-20251001"}}
```

Precedence is CLI > config file > defaults. `--iterations` defaults to 3 (changed from 10 to match the
AIC-1.1 §7 development ceiling). Each run directory also contains `run_config.json` and `manifest.json`.

## Isolated evaluation, workloads and providers (batches 02–03)

Model-written code only ever runs in the worker container. Start Docker, build the image once per harness
change (the tag is a content hash), then run:

```bash
uv run --locked python scripts/build_worker_image.py                 # --variant cuda12 on the NV host
uv run --locked python agent.py --backend cpu --workload softmax --iterations 2
```

- `--workload softmax|example_add` picks a registered workload; `--source/--function` still works (not scientific).
- `--runner docker` (default) or `--runner subprocess` (no Docker: the baseline runs, model candidates are refused).
- `--calibration FILE` reuses a frozen A/A calibration from `scripts/calibrate_aa.py`; otherwise the run
  calibrates δ before its first measurement and stops if the host is too noisy.
- `--provider` chooses the LLM:
  - `anthropic` (default): `ANTHROPIC_API_KEY` in `.env.local`, as before.
  - `anthropic-vertex`: Claude on Vertex AI, billed to the GCP project.
  - `gemini-vertex`: Gemini on Vertex AI (default `gemini-2.5-flash`), billed to the GCP project.

  For both Vertex providers: `gcloud auth application-default login` once, then
  `export GOOGLE_CLOUD_PROJECT=<project>` (optionally `CLOUD_ML_REGION`, default `global`).

```bash
uv run --locked python agent.py --backend cpu --workload softmax --provider gemini-vertex --iterations 1
```

Each run directory holds `run_config.json`, `manifest.json` (with `scientific` and its reasons), `history.json`
(Attempt v1 records with raw timing blocks and compiler feedback), `requests.json` (tokens per request) and
`work/` (every worker request, the calibration and the raw IR text). The final held-out check of a frozen
winner is `scripts/heldout.py evaluate` (see `heldout.py`); its salt is shared privately, never committed.
