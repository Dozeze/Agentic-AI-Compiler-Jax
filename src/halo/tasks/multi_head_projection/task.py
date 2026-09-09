"""One input projected by many weight matrices: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "multi_head_projection"

TOKENS, MODEL, HEADS, HEAD_DIM = 256, 512, 8, 64

DESCRIPTION = f"""\
Project one shared input with a different weight matrix per head.

Signature: candidate(x, w) -> out
Shapes:    x is float32 ({TOKENS}, {MODEL}); w is float32 ({HEADS}, {MODEL}, {HEAD_DIM});
           out is float32 ({HEADS}, {TOKENS}, {HEAD_DIM}).
Semantics: out[h] = x @ w[h] for every head h.
"""


#: Raised from the default 1e-6: 512-term float32 dot products reach a worst measured
#: absolute error of 9.3e-05 here, for both the seed and the best known
#: implementation. Output elements that cancel toward zero still carry the
#: accumulated error of the whole reduction.
ATOL = 1e-3

def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((TOKENS, MODEL), dtype=np.float32),
        rng.standard_normal((HEADS, MODEL, HEAD_DIM), dtype=np.float32),
    )


def reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.einsum("nd,hdk->hnk", x.astype(np.float64), w.astype(np.float64))


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    x_shape = (TOKENS, MODEL)
    w_shape = (HEADS, MODEL, HEAD_DIM)
    zeros = (np.zeros(x_shape, np.float32), np.zeros(w_shape, np.float32))
    # A distinct scale per head catches an implementation that applies one head's
    # weights everywhere, or that stacks the heads in the wrong order.
    per_head = (
        rng.standard_normal(x_shape, dtype=np.float32),
        rng.standard_normal(w_shape, dtype=np.float32)
        * np.linspace(1.0, 8.0, HEADS, dtype=np.float32)[:, None, None],
    )
    return [Case("zeros", zeros), Case("per_head_scale", per_head)]
