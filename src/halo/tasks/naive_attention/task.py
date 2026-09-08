"""Multi-head scaled dot-product attention: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "naive_attention"

BATCH, HEADS, SEQ, DIM = 4, 8, 256, 64

DESCRIPTION = f"""\
Multi-head scaled dot-product attention, forward pass only, no mask, no dropout.

Signature: candidate(q, k, v) -> out
Shapes:    q, k, v are float32 arrays of shape ({BATCH}, {HEADS}, {SEQ}, {DIM});
           out has the same shape.
Semantics: out[b,h] = softmax(q[b,h] @ k[b,h].T / sqrt({DIM})) @ v[b,h]
           computed with the standard max-subtraction for numerical stability.
"""


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    shape = (BATCH, HEADS, SEQ, DIM)
    return tuple(
        rng.standard_normal(shape, dtype=np.float32) for _ in range(3)
    )


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy float64 oracle. Deliberately independent of JAX."""
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
    scores = np.einsum("bhqd,bhkd->bhqk", q64, k64) / np.sqrt(q64.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v64)


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    """Named cases including the ones a careless rewrite gets wrong."""
    shape = (BATCH, HEADS, SEQ, DIM)
    zeros = tuple(np.zeros(shape, np.float32) for _ in range(3))

    # A large common component in q and k pushes the pre-softmax scores to ~[180, 220],
    # far past float32's exp overflow at 88, so an implementation that drops the
    # max-subtraction returns inf/nan here and nowhere else. The scores stay in a
    # narrow band, so the softmax remains well conditioned - unlike simply scaling up
    # the random inputs, which makes attention nearly one-hot and then *no* float32
    # implementation can match the float64 oracle.
    offset = np.full((1, 1, 1, DIM), 5.0, np.float32)
    overflow = (
        rng.standard_normal(shape, dtype=np.float32) * 0.5 + offset,
        rng.standard_normal(shape, dtype=np.float32) * 0.5 + offset,
        rng.standard_normal(shape, dtype=np.float32),
    )

    # Identical keys make every attention weight equal - catches axis mix-ups that
    # random data hides.
    k_const = np.broadcast_to(
        rng.standard_normal((1, 1, 1, DIM), dtype=np.float32), shape
    ).copy()
    uniform = (
        rng.standard_normal(shape, dtype=np.float32),
        k_const,
        rng.standard_normal(shape, dtype=np.float32),
    )
    return [
        Case("zeros", zeros),
        Case(
            "large_magnitude",
            overflow,
            atol=1e-3,
            note="scores ~200; float32 accumulation over 256 keys limits agreement "
            "with the float64 oracle to ~1e-4 even when correct",
        ),
        Case("uniform_weights", uniform),
    ]
