"""Squared euclidean distances between two point sets: workload and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "pairwise_distances"

N, M, DIM = 256, 256, 64

DESCRIPTION = f"""\
Squared euclidean distance between every pair of points from two sets.

Signature: candidate(a, b) -> out
Shapes:    a is float32 ({N}, {DIM}); b is float32 ({M}, {DIM});
           out is float32 ({N}, {M}).
Semantics: out[i, j] = sum over d of (a[i, d] - b[j, d]) ** 2.
Inputs are drawn from a standard normal, so points are well separated.
"""


#: Raised from the default 1e-6: 64-term float32 reductions; the squared-norm identity
#: costs a little more accuracy than the direct difference reach a worst measured
#: absolute error of 3.3e-05 here, for both the seed and the best known
#: implementation. Output elements that cancel toward zero still carry the
#: accumulated error of the whole reduction.
ATOL = 5e-4

def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((N, DIM), dtype=np.float32),
        rng.standard_normal((M, DIM), dtype=np.float32),
    )


def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a.astype(np.float64)[:, None, :] - b.astype(np.float64)[None, :, :]
    return (diff ** 2).sum(axis=-1)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    zeros = (np.zeros((N, DIM), np.float32), np.zeros((M, DIM), np.float32))
    # Asymmetric inputs catch an implementation that swaps the two operands or
    # transposes the result; with a == b the error would be invisible.
    asymmetric = (
        rng.standard_normal((N, DIM), dtype=np.float32),
        rng.standard_normal((M, DIM), dtype=np.float32) * 4.0 + 10.0,
    )
    return [Case("zeros", zeros), Case("asymmetric_sets", asymmetric)]
