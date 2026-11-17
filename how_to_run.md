For Jonathan L:

Install the locked CPU development environment and run an offline proposal smoke check:

```bash
uv sync --locked
uv run --locked python agent.py --backend cpu --proposal-only \
  --offline-responses tests/fixtures/provider_responses.json
```

This uses no credential or network and never imports the reference or evaluates the proposed source.
The output is explicitly unverified. For a live optimizer run, a key alone is insufficient: supply a
reviewed `--config live_run.json` with a money cap, currency and matching dated model prices/provenance.
See README's AIC-013 section for the required fields and conservative accounting policy. No cap or real
prices are configured by default. Add `--plot` only when a graphical runtime plot is wanted.

On Linux with a compatible NVIDIA GPU, use the already locked CUDA 12 extra:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 python -c "import jax; print(jax.devices())"
uv run --locked --extra cuda12 python agent.py --backend cuda --require-gpu --runner subprocess --iterations 0
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
  --function bad_func --iterations 3 --output-dir runs/example --config live_run.json
```

On the GPU machine, add `--extra cuda12` after `--locked` and `--require-gpu`
after `agent.py` to this command.

Use a new `--output-dir` each time; existing directories are refused before a provider request.
Omitting it creates a unique `runs/<run-id>`. Each completed attempt saves exact source, proposal,
evaluation and raw-artifact hashes under `attempts/<id>/`, then publishes `record.json` last.
`history.json` and `iteration_0000.json`, `iteration_0001.json`, etc. are derived from committed attempts.
`best.py` contains the latest verified incumbent, or the baseline if no candidate wins; `best.json`
records its source hash and verification scope. The original workload file is preserved. This export
does not perform the later final/held-out checks. See README's durable-artifact section for inspection.

Each proposal also saves `contexts/NNNN.json` and `contexts/NNNN.selection.json` before dispatch.
They contain the exact bounded context, byte accounting, included fields and omitted attempt IDs.
The 24 KiB feedback cap and full-request budget checks preserve constraints and the latest outcome;
see README's AIC-014 section for the selection policy and ablation conditions.

Optimization episodes now finish with `run_result.json`, including the stop status and verified best
artifact reference. A failed baseline instead records `baseline_failure.json` and has no best export.
`--iterations 0 --runner subprocess` verifies the trusted baseline without a provider; positive-candidate
subprocess runs stop before requesting model code. Production evaluation requires Docker. See README's
AIC-015 section for shared run/stage deadlines and the programmatic `controller.optimize()` result.

For AIC-016, `configs/target.json` declares CUDA/Docker softmax with three candidate/request slots and C4
feedback. It deliberately has no live funding. README's selected-target MVP section gives the separate
offline, mandatory target/sandbox and approved live commands, evidence retention and read-only verifier.

After selection is finished, `--finalize-run runs/SEARCH_RUN --output-dir runs/FINAL_RUN` performs a
provider-free final evaluation using the frozen configuration. It consumes the one-use final split
and blocks further search under that version. `--final-correctness-only` also consumes it and leaves
performance unverified. See `docs/FINAL_EVALUATION.md` before using final mode; use the offline tests
for rehearsal. Replay is `python scripts/heldout.py replay --run runs/FINAL_RUN` through locked uv.

## Backend and run configuration

Select the JAX backend explicitly; the run stops with exit code 2 if it is unavailable instead of silently
falling back to CPU:

```bash
uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1 --config live_run.json
uv run --locked --extra cuda12 python agent.py --backend cuda --require-gpu --iterations 1 --config live_run.json   # inside WSL2 on the NV host
```

`--config run.json` loads a JSON `RunConfig v1` (fields in `contracts.py`), for example:

```json
{"backend": "cpu", "budgets": {"max_candidates": 2}, "model": {"model_id": "claude-haiku-4-5-20251001"}}
```

Precedence is CLI > config file > defaults. `--iterations` defaults to 3 (changed from 10 to match the
AIC-1.1 §7 development ceiling). Each run directory also contains `run_config.json` and `manifest.json`.
The small config above illustrates model selection only; live calls also need the money and price fields
described in README. A zero candidate budget needs no provider client, key or prices.

## Isolated evaluation, workloads and providers (batches 02–03)

Model-written code only ever runs in the worker container. Start Docker, build the image once per harness
change (the tag is a content hash), then run:

```bash
uv run --locked python scripts/build_worker_image.py                 # --variant cuda12 on the NV host
uv run --locked python agent.py --backend cpu --workload softmax --iterations 2 --config live_run.json
```

- `--workload softmax|layer_norm|matmul_gelu|example_add` picks a registered workload;
  `--source/--function` still works (not scientific). See `docs/WORKLOAD_SUITE.md` for the new contracts.
- `--runner docker` (default) or `--runner subprocess` (no Docker: the baseline runs, model candidates are refused).
- `--calibration FILE` reuses a frozen A/A calibration from `scripts/calibrate_aa.py`; otherwise the run
  calibrates δ before its first measurement and stops if the host is too noisy.
  The calibration CLI accepts `--work-dir <new-directory>` to retain raw sessions and failed/noisy
  evidence, and uses the configured evaluation budgets.
- `--provider` chooses the LLM:
  - `anthropic` (default): `ANTHROPIC_API_KEY` in `.env.local`, as before.
  - `anthropic-vertex`: Claude on Vertex AI, billed to the GCP project.
  - `gemini-vertex`: Gemini on Vertex AI (default `gemini-2.5-flash`), billed to the GCP project.

  For both Vertex providers: `gcloud auth application-default login` once, then
  `export GOOGLE_CLOUD_PROJECT=<project>` (optionally `CLOUD_ML_REGION`, default `global`).

```bash
uv run --locked python agent.py --backend cpu --workload softmax --provider gemini-vertex --iterations 1 --config live_run.json
```

Each run directory holds `run_config.json`, `manifest.json` (with `scientific` and its reasons), `history.json`
(Attempt v1 records with raw timing blocks and compiler feedback), `requests.json` (tokens per request) and
`work/` (every worker request, the calibration and the raw IR text). The final held-out check of a frozen
winner is `scripts/heldout.py evaluate` (see `heldout.py`); its seeds are in the registry and never reach the search.
