"""Elementwise GELU activation: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "gelu"

ROWS, COLS = 1024, 4096
COEFF = 0.044715
SQRT_2_OVER_PI = 0.7978845608028654

DESCRIPTION = f"""\
The tanh approximation of the GELU activation, applied elementwise.

Signature: candidate(x) -> out
Shapes:    x is float32 ({ROWS}, {COLS}); out has the same shape.
Semantics: out = 0.5 * x * (1 + tanh({SQRT_2_OVER_PI} * (x + {COEFF} * x ** 3)))
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (rng.standard_normal((ROWS, COLS), dtype=np.float32),)


def reference(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    inner = SQRT_2_OVER_PI * (x64 + COEFF * x64 ** 3)
    return 0.5 * x64 * (1.0 + np.tanh(inner))


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    shape = (ROWS, COLS)
    # tanh saturates to -1 and +1 here; the large negative tail must return
    # values near zero rather than overflowing through the cubic term.
    saturated = (rng.standard_normal(shape, dtype=np.float32) * 30.0,)
    return [
        Case("zeros", (np.zeros(shape, np.float32),)),
        Case("saturated_tails", saturated, atol=1e-4,
             note="x ~ +-100 through a cubic; float32 loses absolute precision "
                  "at that magnitude even when correct"),
    ]
