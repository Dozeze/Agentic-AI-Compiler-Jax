"""Batch of independent matrix products: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "batched_matmul_loop"

BATCH, ROWS, INNER, COLS = 32, 128, 64, 128

DESCRIPTION = f"""\
A batch of independent matrix products.

Signature: candidate(a, b) -> out
Shapes:    a is float32 ({BATCH}, {ROWS}, {INNER}); b is float32 ({BATCH}, {INNER}, {COLS});
           out is float32 ({BATCH}, {ROWS}, {COLS}).
Semantics: out[i] = a[i] @ b[i] for every i in the batch.
"""


#: Raised from the default 1e-6: 64-term float32 dot products reach a worst measured
#: absolute error of 1.6e-05 here, for both the seed and the best known
#: implementation. Output elements that cancel toward zero still carry the
#: accumulated error of the whole reduction.
ATOL = 2e-4

def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((BATCH, ROWS, INNER), dtype=np.float32),
        rng.standard_normal((BATCH, INNER, COLS), dtype=np.float32),
    )


def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bjk->bik", a.astype(np.float64), b.astype(np.float64))


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    a_shape = (BATCH, ROWS, INNER)
    b_shape = (BATCH, INNER, COLS)
    zeros = (np.zeros(a_shape, np.float32), np.zeros(b_shape, np.float32))
    # Distinct per-batch content: an implementation that broadcasts one batch
    # element over all of them agrees with the oracle on uniform input and
    # nowhere else.
    ramp = (
        (np.arange(BATCH, dtype=np.float32).reshape(BATCH, 1, 1)
         + rng.standard_normal(a_shape, dtype=np.float32)),
        rng.standard_normal(b_shape, dtype=np.float32),
    )
    return [Case("zeros", zeros), Case("per_batch_offset", ramp)]
