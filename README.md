# Agentic AI compiler för JAX/XLA

En prototyp för Ericsson-projektet: Claude föreslår ändringar i en JAX-funktion.
Programmet kontrollerar resultatet, mäter körtiden och behåller snabbare,
korrekta förslag.

## Kom igång

Install [uv](https://docs.astral.sh/uv/) and run the offline proposal smoke check. It needs no API key,
Docker or money cap and does not execute the proposed code:

```bash
uv sync --locked
uv run --locked python agent.py --backend cpu --proposal-only \
  --offline-responses tests/fixtures/provider_responses.json
```

Kör på Linux med NVIDIA-GPU:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 python agent.py --backend cuda --require-gpu --runner subprocess --iterations 0
```

`--require-gpu` avbryter om JAX bara hittar CPU. Projektet använder JAX 0.4.38,
som har ett CUDA 12-tillägg men inget CUDA 13-tillägg. Använd därför inte
`jax[cuda13]==0.4.38`. En nyare NVIDIA-drivrutin kan normalt köra CUDA 12-kod.
`uv run --no-project --with anthropic` använder inte projektets beroenden och
låsfil; använd kommandona ovan.

## Egen Python-fil

Ange fil och funktionsnamn med `--source` och `--function`. Filen kan definiera
testargument som `TEST_ARGS` eller med `make_test_args()`:

```python
import jax.numpy as jnp

TEST_ARGS = (jnp.arange(32), jnp.ones(32))

def bad_func(x, y):
    return x + y
```

Use `--output-dir runs/my_test` to choose a new artifact directory. It must not already exist.
By default, each run gets a unique `runs/<run-id>` directory.

## Vad som finns

- Inläsning av en valfri `.py`-fil och en namngiven JAX-funktion.
- Claude-loop som testar förslag mot referensfunktionen och mäter JIT-körd tid.
- Körtid och beslut (`accepted`/`rejected`) i terminalen efter varje iteration.
- `iteration_0000.json`, `iteration_0001.json` och `history.json` i
  resultatkatalogen. De innehåller källkod, diff, körtid och fel/status.

The original file is preserved. Baseline and candidate sources are also saved as exact `.py` files;
`best.py` exports the latest verified incumbent, with its identity and verification scope in `best.json`.

The implementation status and next tasks are maintained in `AI_CONTEXT.md` and
`docs/IMPLEMENTATION_TASKS.md`.

## Backend selection and run configuration (AIC-002)

- `--backend {cpu,cuda,rocm,tpu}` selects the JAX backend before any JAX computation and exits with code 2
  (one error line, no traceback) if it is unavailable; it never falls back silently. Without `--backend`, JAX
  chooses itself (legacy behaviour) and the run is recorded as non-scientific.
- `--config PATH` reads a JSON `RunConfig v1` file (see `contracts.py`). Precedence: CLI > config file > defaults.
  Unknown keys and wrong types are rejected.
- **`--iterations` now defaults to 3** (was 10), the AIC-1.1 §7 development ceiling (team decision P4).
- Every run writes `run_config.json` (resolved configuration) and `manifest.json` (config hash, argv, versions,
  devices, lock-file hash, git commit and dirty flag; only allowlisted environment variables) into the output
  directory, after backend selection and before any LLM call.

```bash
uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1 --config live_run.json
```

## Structured proposals and provider budgets (AIC-013)

The provider returns one `Proposal v1` through `submit_proposal`; it has no inspection or evaluation tools.
The host validates the entire response and source policy before evaluating anything. Multiple tool calls,
unknown tools, refusals, truncated/oversize responses, unexpected model identities and invalid billing data
are recorded and stop the provider loop. Invalid proposal fields become a `PROPOSAL_INVALID` attempt.
The next request receives bounded context selected by AIC-014, including prior measured outcomes.

Live mode requires `budgets.money_cap`, `budgets.currency`, and `model.prices` in a reviewed `--config` file.
The prices object requires `provider`, `model_id`, `currency`, positive numeric `input_per_million` and
`output_per_million` ceilings, an ISO `checked_date` (not in the future), and an HTTPS `source_url`.
Provider/model/currency must match the request. No real price, cap or exchange rate is supplied automatically.
An API key alone cannot enable a paid call. Put credentials in the environment or `.env.local`, never JSON.

Each SDK attempt first saves `request_reservations/NNNN.json`: exact bounded/redacted structured parameters,
local ID, input/output reservation, price provenance and timeout. Input reservation uses serialized UTF-8
bytes plus a 4,096-token framing allowance; output uses the configured ceiling. This is a conservative
project policy, not a vendor-certified token count. Reported usage beyond the reservation is charged and stops
the loop. Unknown usage keeps its reserved debit; partial usage only releases known components. Cached,
priority or server-tool billing is unsupported and stops the loop rather than assuming a zero price.

SDK retries are disabled. `model.max_retries` (default 0) allows individually reserved application retries;
request/token/money limits and remaining time are checked before each attempt. The default transport timeout
is 30 seconds, shortened to remaining time. Transport timeouts are per I/O operation; AIC-015 still owns the
complete optimizer deadline and worker budget integration. A timed-out request can still be billed remotely,
so its debit is retained. This ledger does not promise cancellation of remote work or certify provider billing.

Outcomes in `requests/NNNN.json` include server IDs (or explicit unavailability), bounded redacted payloads,
usage, accounting and refusal/error status. A storage failure stops further dispatch. An interrupted reservation
without an outcome remains uncertain; safe execution recovery is AIC-019.

`--offline-responses FILE` replays a JSON array of Messages-shaped response objects through `OfflineClient`.
The checked-in fixture uses synthetic usage and makes no correctness/performance claim. `--proposal-only`
saves `proposed.py`, `proposal.json` and `proposal_only_result.json` with `verified: false`, without importing
the reference or running a worker. It never creates `best.py`. The same live-money gate applies when this mode
uses a real provider. Existing provider/model defaults are unchanged; live service proof remains AIC-016.

## Bounded proposal context (AIC-014)

Context includes the public development workload contract, fixed correctness rules, selected environment,
separate baseline/incumbent identities, current source and remaining budgets. It retains the newest five
failed or nonpromoted attempts and up to three accepted rewrites, with baseline, incumbent and latest outcome
always present. Paired medians, uncertainty and source identities point to immutable evaluation evidence;
missing measurements remain unavailable. Compiler estimates are labeled separately from measurements.
Compiler summaries appear only under C3/C4/K, and runtime coverage/hotspots only under C4.

Serialized metadata is capped at 24 KiB; current source has the separate `budgets.max_source_bytes` ceiling
(64 KiB by default). The complete encoded request must also fit the provider payload limit and remaining
conservative token/money reservation. Historical source snippets are limited to 2 KiB each and removed first;
older optional outcomes follow. If required context cannot fit, `CONTEXT_BLOCKED` stops before dispatch.
Diagnostics are bounded summaries; raw logs, IR, timing blocks, input arrays, split seeds and credentials
stay outside the prompt. Source comments and diagnostics are explicitly untrusted data.

Before each SDK attempt, `contexts/NNNN.json` saves the exact user context bytes, and
`contexts/NNNN.selection.json` records its hash, byte accounting, included field paths, omitted attempt IDs
and truncations. The request reservation references both. Committed attempts link these files so artifact
verification detects tampering. Proposal-only mode uses the same context policy and remains unverified.

The locked Anthropic SDK is 1.3.0: its `Messages.create` does not accept `temperature`. Anthropic sampling
uses provider defaults; an explicit `model.temperature` is supported only for the existing Gemini adapter.

## Durable run artifacts (AIC-012)

Every new run reserves its directory exclusively. Reusing an explicit `--output-dir` exits with code 2
before any provider request. `RunConfig.output_dir = null` selects a unique directory; the resolved CLI
configuration records the actual path. Direct Python API runs use `direct_run_config/v1` and record that
they lack a resolved CLI configuration.

```text
runs/<run-id>/
  manifest.json, run_config.json, environment.json, baseline.py
  requests/0001.json                 # immutable provider request metadata
  attempts/0000/source.py            # baseline, then one directory per candidate
  attempts/0000/evaluation.json      # host correctness, timing and feedback evidence
  attempts/0000/feedback.json
  attempts/0000/record.json          # commit marker, published last
  work/0001-reference/...            # raw arrays, IR, worker results and bounded logs
  history.json, iteration_0000.json  # derived compatibility indexes
  best.py, best.json                 # exact source and validated pointer/scope
```

Candidate directories also retain proposals, request references and measurements when available.
Malformed proposals have a bounded submission artifact with an explicit omission reason when needed.
Each commit fingerprints its source, metadata and retained raw artifacts. Incomplete attempt directories
are excluded from history and best selection. A failed publication stops execution; resuming runs remains
AIC-019. History/best files can be regenerated from valid commits through `RunStore.derive_indexes()`.

`best.py` falls back to the verified baseline when no candidate wins. Its export establishes development
correctness and search confirmation only; final and held-out verification are recorded as unperformed.
Inspect an existing run without executing its source:

```bash
uv run --locked python -c 'import json, sys; from storage import verified_best; print(json.dumps(verified_best(sys.argv[1]), indent=2))' runs/REPLACE_WITH_RUN_ID
```

Storage uses strict JSON, atomic publication, credential redaction and configured source/artifact byte
limits. Decision evidence is retained; exceeding the limit stops publication. The byte limit is checked
by the host around publication, and does not impose a filesystem quota on an active worker.

## Isolated evaluation, workloads and providers (batches 02–03)

Model-written code only ever runs in the worker container. Start Docker, build the image once per harness
change (the tag is a content hash), then run:

```bash
uv run --locked python scripts/build_worker_image.py                 # --variant cuda12 on the NV host
uv run --locked python agent.py --backend cpu --workload softmax --iterations 2 --config live_run.json
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
uv run --locked python agent.py --backend cpu --workload softmax --provider gemini-vertex --iterations 1 --config live_run.json
```

Each run directory holds `run_config.json`, `manifest.json` (with `scientific` and its reasons), `history.json`
(Attempt v1 records with raw timing blocks and compiler feedback), `requests.json` (tokens per request) and
`work/` (every worker request, the calibration and the raw IR text). The final held-out check of a frozen
winner is `scripts/heldout.py evaluate` (see `heldout.py`); its seeds are in the registry and never reach the search.

## Runtime profile feedback (AIC-011)

The baseline and each promoted incumbent receive a separate profile worker after compilation and five warmup
calls. Five synchronized calls are annotated and traced. Paired benchmark sessions remain unprofiled.
`history.json` records the result under `attempt.evaluation.runtime_trace`, including coverage, source/device
provenance, and a hashed raw Perfetto gzip artifact under `work/`. Rebuild the worker image after this change.

Set `"context_condition": "C4"` in a run config to expose the observed profile summary to the model. C2/C3/K
continue to exclude profile information. Coverage is `CPU_EXECUTION_EVENTS`, `DEVICE_EXECUTION_EVENTS`,
`HOST_ONLY` or `UNAVAILABLE`, with a reason. CPU events are executable-level observations. Host-only and
unavailable profiles have no hotspots. Reported durations are inclusive interval unions for individual
lanes/names; they may overlap and are never percentages or unprofiled benchmark latency.

Raw traces are capped at 8 MiB compressed / 32 MiB decoded and 100,000 events. Missing profiler permissions,
unsupported traces and profile worker timeouts produce explicit unavailable evidence without changing the
correctness or performance decision. CPU coverage is verified; CUDA and container verification remain pending.

Offline profile tests: `uv run --locked pytest -q tests/test_runtime_profile.py -m "not needs_docker"`.
On the NV host, add `--extra cuda12` to `uv run` and `--require-gpu` to pytest. Container profile tests run with
`-m needs_docker --require-docker` after rebuilding the worker images; add `--require-gpu` on the NV host.

## Tests

```bash
uv sync --locked
uv run --locked pytest -q        # offline: no API key and no network needed
```
