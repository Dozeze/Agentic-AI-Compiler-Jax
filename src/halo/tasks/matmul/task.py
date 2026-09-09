"""A single dense matrix product: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "matmul"

SIZE = 512

DESCRIPTION = f"""\
A single dense matrix product.

Signature: candidate(a, b) -> out
Shapes:    a and b are float32 ({SIZE}, {SIZE}); out is float32 ({SIZE}, {SIZE}).
Semantics: out = a @ b, the ordinary matrix product.
"""


#: Raised from the default 1e-6: 512-term float32 dot products reach a worst measured
#: absolute error of 1.3e-04 here, for both the seed and the best known
#: implementation. Output elements that cancel toward zero still carry the
#: accumulated error of the whole reduction.
ATOL = 2e-3

def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((SIZE, SIZE), dtype=np.float32),
        rng.standard_normal((SIZE, SIZE), dtype=np.float32),
    )


def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) @ b.astype(np.float64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    shape = (SIZE, SIZE)
    identity = (np.eye(SIZE, dtype=np.float32),
                rng.standard_normal(shape, dtype=np.float32))
    # a @ b != b @ a: an implementation that transposes the result or swaps the
    # operands passes on symmetric input and fails here.
    return [
        Case("zeros", (np.zeros(shape, np.float32), np.zeros(shape, np.float32))),
        Case("identity_times_random", identity),
    ]
