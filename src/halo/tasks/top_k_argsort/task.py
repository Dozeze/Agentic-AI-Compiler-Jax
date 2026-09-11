"""Top-k values per row: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "top_k_argsort"

ROWS, COLS, K = 64, 16384, 8

DESCRIPTION = f"""\
The {K} largest values of each row, in descending order - the selection step of
top-k sampling and of beam search.

Signature: candidate(x) -> top
Shapes:    x is float32 ({ROWS}, {COLS}); top is float32 ({ROWS}, {K}).
Semantics: top[r] = the {K} largest entries of x[r], sorted from largest to smallest.
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (rng.standard_normal((ROWS, COLS), dtype=np.float32),)


def reference(x: np.ndarray) -> np.ndarray:
    return -np.sort(-x.astype(np.float64), axis=-1)[:, :K]


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    # Ties: many equal maxima, so the answer is K copies of one value.
    tied = rng.standard_normal((ROWS, COLS), dtype=np.float32)
    tied[:, ::100] = 9.0
    # The largest values are all in the last K columns, in ascending order: an
    # implementation that returns them unsorted, or from the wrong end, fails.
    tail = rng.standard_normal((ROWS, COLS), dtype=np.float32)
    tail[:, -K:] = np.arange(10, 10 + K, dtype=np.float32)
    return [Case("ties", (tied,)), Case("ascending_tail", (tail,))]
