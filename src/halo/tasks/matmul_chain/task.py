"""Product of three matrices: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "matmul_chain"

BIG, SMALL = 1024, 16

DESCRIPTION = f"""\
The product of three matrices.

Signature: candidate(a, b, c) -> out
Shapes:    a is float32 ({BIG}, {SMALL}); b is float32 ({SMALL}, {BIG});
           c is float32 ({BIG}, {SMALL}); out is float32 ({BIG}, {SMALL}).
Semantics: out = a @ b @ c, the ordinary matrix product, which is associative.
"""


#: Raised from the default 1e-6: two chained 1024-term float32 reductions, and a
#: valid reassociation of the product legitimately differs by more than one reach a worst measured
#: absolute error of 6.3e-04 here, for both the seed and the best known
#: implementation. Output elements that cancel toward zero still carry the
#: accumulated error of the whole reduction.
ATOL = 1e-2

def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((BIG, SMALL), dtype=np.float32),
        rng.standard_normal((SMALL, BIG), dtype=np.float32),
        rng.standard_normal((BIG, SMALL), dtype=np.float32),
    )


def reference(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) @ b.astype(np.float64) @ c.astype(np.float64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    zeros = (
        np.zeros((BIG, SMALL), np.float32),
        np.zeros((SMALL, BIG), np.float32),
        np.zeros((BIG, SMALL), np.float32),
    )
    # b = 0 makes the result exactly zero whatever a and c are: an implementation
    # that drops one operand still passes on random input but fails here.
    middle_zero = (
        rng.standard_normal((BIG, SMALL), dtype=np.float32),
        np.zeros((SMALL, BIG), np.float32),
        rng.standard_normal((BIG, SMALL), dtype=np.float32),
    )
    return [Case("zeros", zeros), Case("zero_middle_factor", middle_zero)]
