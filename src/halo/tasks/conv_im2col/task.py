"""3x3 same-padded convolution: workload, inputs and float64 oracle."""

from __future__ import annotations

import numpy as np

from halo.tasks.base import Case

NAME = "conv_im2col"

N, H, W, C_IN, C_OUT = 8, 32, 32, 32, 64

DESCRIPTION = f"""\
A 3x3 convolution with stride 1 and one pixel of zero padding on each side, so the
output has the input's spatial size. Channels last.

Signature: candidate(x, w) -> out
Shapes:    x is float32 ({N}, {H}, {W}, {C_IN}); w is float32 (3, 3, {C_IN}, {C_OUT});
           out is float32 ({N}, {H}, {W}, {C_OUT}).
Semantics: out[n, i, j, o] = sum over di, dj in {{0,1,2}} and c of
           x_padded[n, i + di, j + dj, c] * w[di, dj, c, o]
           where x_padded is x with one row/column of zeros on every side.
"""

#: Raised from the default 1e-6: a 288-term float32 reduction per output element.
#: Worst measured absolute error 4.4e-06 for both the seed and the best known
#: implementation.
ATOL = 1e-4


def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    return (
        rng.standard_normal((N, H, W, C_IN), dtype=np.float32),
        rng.standard_normal((3, 3, C_IN, C_OUT), dtype=np.float32) / np.sqrt(9 * C_IN),
    )


def reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x64 = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1), (0, 0)))
    w64 = w.astype(np.float64)
    out = np.zeros((N, H, W, C_OUT))
    for di in range(3):
        for dj in range(3):
            out += x64[:, di:di + H, dj:dj + W, :] @ w64[di, dj]
    return out


def correctness_cases(rng: np.random.Generator) -> list[Case]:
    w = rng.standard_normal((3, 3, C_IN, C_OUT), dtype=np.float32) / np.sqrt(9 * C_IN)
    # A single lit pixel in one corner: the output is the flipped kernel stamped
    # at that corner, and the padding side, the tap order and the flip are all
    # visible in which output pixels are non-zero.
    impulse = np.zeros((N, H, W, C_IN), np.float32)
    impulse[:, 0, 0, 0] = 1.0
    # Only the centre tap is non-zero: the result must equal a 1x1 convolution,
    # so any spatial shift of the input is an error.
    centre_only = np.zeros((3, 3, C_IN, C_OUT), np.float32)
    centre_only[1, 1] = w[1, 1]
    return [
        Case("corner_impulse", (impulse, w)),
        Case("centre_tap_only", (rng.standard_normal((N, H, W, C_IN), dtype=np.float32), centre_only)),
    ]
