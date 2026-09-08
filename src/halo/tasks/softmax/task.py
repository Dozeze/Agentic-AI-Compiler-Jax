"""Row-wise softmax over a large 2-D array: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "softmax"

ROWS, COLS = 1024, 4096

DESCRIPTION = f"""\
Row-wise softmax.

Signature: candidate(x) -> out
Shapes:    x is a float32 array of shape ({ROWS}, {COLS}); out has the same shape.
Semantics: out[i] = exp(x[i] - max(x[i])) / sum(exp(x[i] - max(x[i])))
           Rows must sum to 1 and the result must not overflow for large inputs.
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (rng.standard_normal((ROWS, COLS), dtype=np.float32),)


def reference(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    shifted = x64 - x64.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    shape = (ROWS, COLS)
    # exp(1e4) overflows float32; only a max-subtracting implementation survives.
    # The result is essentially one-hot, so it stays exactly representable and the
    # default tolerance applies.
    overflow = (rng.standard_normal(shape, dtype=np.float32) * 1e4,)
    # A single dominant entry per row - the result is nearly one-hot.
    peaked = rng.standard_normal(shape, dtype=np.float32)
    peaked[:, 0] = 1e3
    return [
        Case("zeros", (np.zeros(shape, np.float32),)),
        Case("large_magnitude", overflow),
        Case("peaked", (peaked,)),
    ]
