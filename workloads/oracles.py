"""Trusted NumPy float64 oracles (AIC-003).

They validate the frozen JAX references themselves; candidates are compared with the
jit-compiled reference, never with these oracles. Kept out of agent-visible workload sources.
"""
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
