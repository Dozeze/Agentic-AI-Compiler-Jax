"""Softmax cross-entropy against integer labels: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "cross_entropy_onehot"

ROWS, VOCAB = 1024, 8192

DESCRIPTION = f"""\
Per-example softmax cross-entropy loss over a vocabulary, as used to train a
language model.

Signature: candidate(logits, labels) -> loss
Shapes:    logits is float32 ({ROWS}, {VOCAB}); labels is int32 ({ROWS},) with
           values in [0, {VOCAB}); loss is float32 ({ROWS},).
Semantics: loss[i] = -log softmax(logits[i])[labels[i]]
                   = logsumexp(logits[i]) - logits[i, labels[i]]
           computed stably for logits of any magnitude.
"""

#: Raised from the default 1e-6: logsumexp over 8192 float32 terms. Worst measured
#: absolute error 1.8e-06 for the seed and 1.6e-05 for the best known
#: implementation (on the large-magnitude case).
ATOL = 1e-4


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((ROWS, VOCAB), dtype=np.float32) * 3.0,
        rng.integers(0, VOCAB, size=(ROWS,), dtype=np.int32),
    )


def reference(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))[:, 0]
    return lse - x[np.arange(x.shape[0]), labels]


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    labels = rng.integers(0, VOCAB, size=(ROWS,), dtype=np.int32)
    # Logits around 500: exp overflows float32 without max-subtraction.
    large = rng.standard_normal((ROWS, VOCAB), dtype=np.float32) + 500.0
    # One logit far above the rest: the loss is ~0 for the right label and ~gap
    # for a wrong one, so a label/index mix-up is off by a large amount.
    peaked = rng.standard_normal((ROWS, VOCAB), dtype=np.float32)
    peaked[np.arange(ROWS), labels] += 30.0
    return [
        Case("large_magnitude", (large, labels), note="exp overflows without max-subtraction"),
        Case("peaked_on_label", (peaked, labels)),
    ]
