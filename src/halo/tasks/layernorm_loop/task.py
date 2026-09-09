"""Row-wise layer normalisation: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "layernorm_loop"

ROWS, COLS = 32, 1024
EPS = 1e-5

DESCRIPTION = f"""\
Layer normalisation applied independently to each row.

Signature: candidate(x, gamma, beta) -> out
Shapes:    x is float32 ({ROWS}, {COLS}); gamma and beta are float32 ({COLS},);
           out is float32 ({ROWS}, {COLS}).
Semantics: for each row r,
           out[r] = (x[r] - mean(x[r])) / sqrt(var(x[r]) + {EPS}) * gamma + beta
           where var is the uncorrected (population) variance.
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((ROWS, COLS), dtype=np.float32),
        rng.standard_normal((COLS,), dtype=np.float32),
        rng.standard_normal((COLS,), dtype=np.float32),
    )


def reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x64 - mean) / np.sqrt(var + EPS) * gamma.astype(np.float64) + beta.astype(np.float64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    shape = (ROWS, COLS)
    gamma = rng.standard_normal((COLS,), dtype=np.float32)
    beta = rng.standard_normal((COLS,), dtype=np.float32)
    # A constant row has zero variance: only the epsilon keeps this finite.
    constant = (np.full(shape, 3.0, np.float32), gamma, beta)
    # Distinct per-row statistics catch an implementation that normalises over
    # the wrong axis or reuses one row's mean for all of them.
    per_row = (
        rng.standard_normal(shape, dtype=np.float32)
        * np.linspace(1.0, 20.0, ROWS, dtype=np.float32)[:, None],
        gamma,
        beta,
    )
    return [
        Case("constant_row", constant, note="zero variance; epsilon must be inside the sqrt"),
        Case("per_row_scale", per_row),
    ]
