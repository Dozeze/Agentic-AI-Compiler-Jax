"""Root-mean-square normalisation: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "rmsnorm"

ROWS, COLS = 256, 2048
EPS = 1e-6

DESCRIPTION = f"""\
Root-mean-square normalisation with a learned per-feature scale.

Signature: candidate(x, w) -> out
Shapes:    x is float32 ({ROWS}, {COLS}); w is float32 ({COLS},);
           out is float32 ({ROWS}, {COLS}).
Semantics: out[r] = x[r] / sqrt(mean(x[r] ** 2) + {EPS}) * w
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((ROWS, COLS), dtype=np.float32),
        rng.standard_normal((COLS,), dtype=np.float32),
    )


def reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    scale = np.sqrt((x64 ** 2).mean(axis=-1, keepdims=True) + EPS)
    return x64 / scale * w.astype(np.float64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    shape = (ROWS, COLS)
    w = rng.standard_normal((COLS,), dtype=np.float32)
    # An all-zero row has zero mean square: only the epsilon prevents a division
    # by zero, and it must sit inside the square root.
    return [
        Case("zero_rows", (np.zeros(shape, np.float32), w),
             note="epsilon must be inside the sqrt or this divides by zero"),
        Case("per_row_scale",
             (rng.standard_normal(shape, dtype=np.float32)
              * np.linspace(1e-3, 1e3, ROWS, dtype=np.float32)[:, None], w)),
    ]
