"""Token embedding lookup: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "embedding_onehot"

BATCH, SEQ, VOCAB, DIM = 8, 128, 8192, 256

DESCRIPTION = f"""\
Token embedding lookup, the first operation of every language model.

Signature: candidate(ids, table) -> out
Shapes:    ids is int32 ({BATCH}, {SEQ}) with values in [0, {VOCAB});
           table is float32 ({VOCAB}, {DIM}); out is float32 ({BATCH}, {SEQ}, {DIM}).
Semantics: out[b, t] = table[ids[b, t]], the row of the table selected by each id.
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.integers(0, VOCAB, size=(BATCH, SEQ), dtype=np.int32),
        rng.standard_normal((VOCAB, DIM), dtype=np.float32),
    )


def reference(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table.astype(np.float64)[ids]


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    table = rng.standard_normal((VOCAB, DIM), dtype=np.float32)
    # The first and last rows: an off-by-one in the table index shows up here.
    edges = np.where(
        rng.random((BATCH, SEQ)) < 0.5, 0, VOCAB - 1
    ).astype(np.int32)
    # One id everywhere: every output row must be the same table row.
    same = np.full((BATCH, SEQ), 4242, np.int32)
    return [Case("edge_ids", (edges, table)), Case("single_id", (same, table))]
