"""Trusted NumPy float64 oracles (AIC-003).

They validate the frozen JAX references themselves; candidates are compared with the
jit-compiled reference, never with these oracles. Kept out of agent-visible workload sources.
"""
import math

import numpy as np


def softmax_f64(x) -> np.ndarray:
    """Trusted float64 softmax. Takes the float32 input actually fed to the candidate and upcasts it."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def add_f64(x, y) -> np.ndarray:
    """Trusted float64 elementwise addition for the legacy example workload."""
    return np.asarray(x, dtype=np.float64) + np.asarray(y, dtype=np.float64)


def layer_norm_f64(x, scale, bias) -> np.ndarray:
    """Two-pass population variance with epsilon inside the square root; affine parameters explicit."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(centered ** 2, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + 1e-5) * np.asarray(scale, np.float64) + np.asarray(bias, np.float64)


def matmul_gelu_f64(x, weight, bias) -> np.ndarray:
    """NumPy float64 dot and Python's erfc, independent of JAX/XLA and its GELU implementation."""
    projected = np.asarray(x, np.float64) @ np.asarray(weight, np.float64) + np.asarray(bias, np.float64)
    tail = np.fromiter((math.erfc(float(v)) for v in (-projected / math.sqrt(2.0)).flat),
                       dtype=np.float64, count=projected.size).reshape(projected.shape)
    return 0.5 * projected * tail


# Headroom suite (batch 04): independent float64 NumPy oracles.

def matmul_chain_f64(a, b, c) -> np.ndarray:
    return np.asarray(a, np.float64) @ np.asarray(b, np.float64) @ np.asarray(c, np.float64)


def pairwise_distances_f64(a, b) -> np.ndarray:
    diff = np.asarray(a, np.float64)[:, None, :] - np.asarray(b, np.float64)[None, :, :]
    return (diff ** 2).sum(axis=-1)


def attention_f64(q, k, v) -> np.ndarray:
    q, k, v = (np.asarray(t, np.float64) for t in (q, k, v))
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v)


def batched_matmul_f64(a, b) -> np.ndarray:
    return np.einsum("bij,bjk->bik", np.asarray(a, np.float64), np.asarray(b, np.float64))
