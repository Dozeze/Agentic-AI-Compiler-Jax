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
