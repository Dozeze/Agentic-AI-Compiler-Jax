"""One decode step of grouped-query attention: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "gqa_decode"

BATCH, Q_HEADS, KV_HEADS, CACHE, DIM = 2, 32, 4, 2048, 128
GROUP = Q_HEADS // KV_HEADS

DESCRIPTION = f"""\
One autoregressive decoding step of grouped-query attention (as in Llama 2 70B /
Mistral): a single new query position attends over a key/value cache of {CACHE}
positions. {Q_HEADS} query heads share {KV_HEADS} key/value heads, {GROUP} query heads
per key/value head. No mask (the cache holds only past positions).

Signature: candidate(q, k, v) -> out
Shapes:    q is float32 ({BATCH}, {Q_HEADS}, 1, {DIM});
           k and v are float32 ({BATCH}, {KV_HEADS}, {CACHE}, {DIM});
           out is float32 ({BATCH}, {Q_HEADS}, 1, {DIM}).
Semantics: query head h uses key/value head h // {GROUP}:
           out[b, h] = softmax(q[b, h] @ k[b, h // {GROUP}].T / sqrt({DIM})) @ v[b, h // {GROUP}]
           with the standard max-subtraction for numerical stability.
"""

#: Raised from the default 1e-6: a 2048-term softmax-weighted float32 reduction.
#: Worst measured absolute error 1.9e-05 (seed) and 5.1e-05 (best known
#: implementation), both on the large-magnitude case, which carries its own 1e-3.
ATOL = 1e-4


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((BATCH, Q_HEADS, 1, DIM), dtype=np.float32),
        rng.standard_normal((BATCH, KV_HEADS, CACHE, DIM), dtype=np.float32),
        rng.standard_normal((BATCH, KV_HEADS, CACHE, DIM), dtype=np.float32),
    )


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
    k64 = np.repeat(k64, GROUP, axis=1)
    v64 = np.repeat(v64, GROUP, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", q64, k64) / np.sqrt(DIM)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    q_shape = (BATCH, Q_HEADS, 1, DIM)
    kv_shape = (BATCH, KV_HEADS, CACHE, DIM)
    # Scores ~360: exp overflows float32 without max-subtraction.
    offset = np.full((1, 1, 1, DIM), 5.0, np.float32)
    overflow = (
        rng.standard_normal(q_shape, dtype=np.float32) * 0.5 + offset,
        rng.standard_normal(kv_shape, dtype=np.float32) * 0.5 + offset,
        rng.standard_normal(kv_shape, dtype=np.float32),
    )
    # Each kv head constant across the cache with a distinct value: every query
    # head must return exactly the value of the kv head it is grouped with.
    k_const = np.broadcast_to(
        rng.standard_normal((1, KV_HEADS, 1, DIM), dtype=np.float32), kv_shape
    ).copy()
    v_const = np.broadcast_to(
        np.arange(KV_HEADS, dtype=np.float32)[None, :, None, None], kv_shape
    ).copy()
    grouping = (rng.standard_normal(q_shape, dtype=np.float32), k_const, v_const)
    return [
        Case("large_magnitude", overflow, atol=1e-3,
             note="scores ~360; float32 accumulation over 2048 keys limits agreement to ~1e-4"),
        Case("kv_head_identity", grouping, note="each kv head has a distinct constant value"),
    ]
