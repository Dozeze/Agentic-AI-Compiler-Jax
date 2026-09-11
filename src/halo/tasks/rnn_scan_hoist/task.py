"""Elman recurrent layer over a sequence: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "rnn_scan_hoist"

STEPS, BATCH, INPUT, HIDDEN = 128, 32, 256, 256

DESCRIPTION = f"""\
A recurrent layer (Elman RNN with tanh) run over a whole sequence.

Signature: candidate(x, w_in, w_rec, bias, h0) -> hs
Shapes:    x is float32 ({STEPS}, {BATCH}, {INPUT}); w_in is float32 ({INPUT}, {HIDDEN});
           w_rec is float32 ({HIDDEN}, {HIDDEN}); bias is float32 ({HIDDEN},);
           h0 is float32 ({BATCH}, {HIDDEN}); hs is float32 ({STEPS}, {BATCH}, {HIDDEN}).
Semantics: h_t = tanh(x_t @ w_in + h_(t-1) @ w_rec + bias) with h_(-1) = h0;
           hs[t] = h_t for every step t.
"""

#: Raised from the default 1e-6: a 128-step tanh recurrence in float32. Worst
#: measured absolute error 3.0e-06 for both the seed and the best known
#: implementation, with the weights scaled so the recurrence is contractive.
ATOL = 1e-4

_SCALE = 0.05


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((STEPS, BATCH, INPUT), dtype=np.float32),
        rng.standard_normal((INPUT, HIDDEN), dtype=np.float32) * _SCALE,
        rng.standard_normal((HIDDEN, HIDDEN), dtype=np.float32) * _SCALE,
        rng.standard_normal((HIDDEN,), dtype=np.float32) * 0.1,
        rng.standard_normal((BATCH, HIDDEN), dtype=np.float32),
    )


def reference(x, w_in, w_rec, bias, h0) -> np.ndarray:
    x, w_in, w_rec, bias, h = (a.astype(np.float64) for a in (x, w_in, w_rec, bias, h0))
    out = np.empty((STEPS, BATCH, HIDDEN))
    for t in range(STEPS):
        h = np.tanh(x[t] @ w_in + h @ w_rec + bias)
        out[t] = h
    return out


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    base = make_inputs(rng)
    # No recurrence weight: every step depends on x_t only. A candidate that
    # scrambles the time axis or the initial state still fails on the next case.
    no_rec = (base[0], base[1], np.zeros_like(base[2]), base[3], base[4])
    # No input weight: the output is a pure function of h0 through the recurrence,
    # which a candidate that drops or mis-threads h0 gets wrong at every step.
    no_in = (base[0], np.zeros_like(base[1]), base[2], base[3], base[4] * 5.0)
    return [Case("no_recurrence", no_rec), Case("state_only", no_in)]
