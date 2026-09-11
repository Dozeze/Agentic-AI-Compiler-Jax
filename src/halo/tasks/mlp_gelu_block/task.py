"""Transformer MLP block: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "mlp_gelu_block"

TOKENS, MODEL, HIDDEN = 256, 512, 2048
COEFF = 0.044715
SQRT_2_OVER_PI = 0.7978845608028654

DESCRIPTION = f"""\
The feed-forward block of a transformer layer: an up-projection, a tanh-approximate
GELU, and a down-projection.

Signature: candidate(x, w1, b1, w2, b2) -> out
Shapes:    x is float32 ({TOKENS}, {MODEL}); w1 is float32 ({MODEL}, {HIDDEN}); b1 is
           float32 ({HIDDEN},); w2 is float32 ({HIDDEN}, {MODEL}); b2 is float32
           ({MODEL},); out is float32 ({TOKENS}, {MODEL}).
Semantics: out = gelu(x @ w1 + b1) @ w2 + b2 with
           gelu(u) = 0.5 * u * (1 + tanh({SQRT_2_OVER_PI} * (u + {COEFF} * u^3))).
"""

#: Raised from the default 1e-6: two 512/2048-term float32 reductions. Worst
#: measured absolute error 1.1e-04 (on the saturated case).
ATOL = 1e-3


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((TOKENS, MODEL), dtype=np.float32),
        rng.standard_normal((MODEL, HIDDEN), dtype=np.float32) / np.sqrt(MODEL),
        rng.standard_normal((HIDDEN,), dtype=np.float32) * 0.1,
        rng.standard_normal((HIDDEN, MODEL), dtype=np.float32) / np.sqrt(HIDDEN),
        rng.standard_normal((MODEL,), dtype=np.float32) * 0.1,
    )


def _gelu(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * (1.0 + np.tanh(SQRT_2_OVER_PI * (u + COEFF * u ** 3)))


def reference(x, w1, b1, w2, b2) -> np.ndarray:
    x, w1, b1, w2, b2 = (a.astype(np.float64) for a in (x, w1, b1, w2, b2))
    return _gelu(x @ w1 + b1) @ w2 + b2


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    x, w1, b1, w2, b2 = make_inputs(rng)
    # Large pre-activations: tanh saturates, gelu(u) -> u for u >> 0 and -> 0 for
    # u << 0. A rewrite of the activation that is only accurate near zero fails.
    return [Case("saturated", (x * 20.0, w1, b1, w2, b2))]
