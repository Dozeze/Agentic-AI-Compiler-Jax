# Agentic AI compiler för JAX/XLA

En prototyp för Ericsson-projektet: Claude föreslår ändringar i en JAX-funktion.
Programmet kontrollerar resultatet, mäter körtiden och behåller snabbare,
korrekta förslag.

## Kom igång

Installera [uv](https://docs.astral.sh/uv/) och lägg din Claude-nyckel i
`.env.local` i projektets rot:

```text
ANTHROPIC_API_KEY=din_nyckel
```

Kör på CPU:

```bash
uv sync --locked
uv run --locked python agent.py --source workloads/example.py --function bad_func --iterations 1
```

Kör på Linux med NVIDIA-GPU:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 python agent.py --source workloads/example.py --function bad_func --iterations 1 --require-gpu
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

För att spara resultat på en bestämd plats, lägg till exempelvis
`--output-dir runs/mitt_test`. Standard är `runs/latest`.

## Vad som finns

- Inläsning av en valfri `.py`-fil och en namngiven JAX-funktion.
- Claude-loop som testar förslag mot referensfunktionen och mäter JIT-körd tid.
- Körtid och beslut (`accepted`/`rejected`) i terminalen efter varje iteration.
- `iteration_0000.json`, `iteration_0001.json` och `history.json` i
  resultatkatalogen. De innehåller källkod, diff, körtid och fel/status.

Originalfilen ändras inte. Kandidaterna skickas fortfarande som kodsträngar
och sparas i JSON.

## Viktigast att göra härnäst

- Spara kandidater som riktiga `.py`-filer och kunna välja en vinnare att skriva tillbaka.
- Kontrollera korrekthet med fler testfall och göra tidsmätningarna mindre känsliga för brus.
- Lägga till HLO-inspektion och senare HLO-ändringar.

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
uv run --locked python agent.py --backend cpu --source workloads/example.py --function bad_func --iterations 1
```

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

## Tests

```bash
uv sync --locked
uv run --locked pytest -q        # offline: no API key and no network needed
```
