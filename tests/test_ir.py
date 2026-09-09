"""Operation counts for the jaxpr and StableHLO levels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from halo.harness import ir

STABLEHLO = """\
module @jit_softmax {
  func.func public @main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1]
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<4xf32>) -> tensor<4x1xf32>
    %3 = stablehlo.subtract %arg0, %2 : tensor<4x8xf32>
    %4 = stablehlo.exponential %3 : tensor<4x8xf32>
    return %8 : tensor<4x8xf32>
  }
}
"""


def test_stablehlo_counts_assigned_operations():
    stats = ir.from_stablehlo(STABLEHLO)
    assert stats.op_counts == {
        "constant": 1, "reduce": 1, "broadcast_in_dim": 1,
        "subtract": 1, "exponential": 1,
    }
    assert stats.total_ops == 5


def test_jaxpr_counts_primitives():
    def softmax(x):
        shifted = x - jnp.max(x, axis=-1, keepdims=True)
        w = jnp.exp(shifted)
        return w / jnp.sum(w, axis=-1, keepdims=True)

    stats = ir.from_jaxpr(jax.make_jaxpr(softmax)(jnp.ones((4, 8), jnp.float32)))
    assert stats.op_counts["reduce_max"] == 1
    assert stats.op_counts["exp"] == 1
    assert stats.op_counts["reduce_sum"] == 1
    assert stats.total_ops == 7


def test_a_python_loop_shows_up_as_repeated_primitives():
    """The signal that identifies an unrolled loop before XLA hides it in fusions."""
    def looped(x):
        return jnp.stack([jnp.exp(x[i]) for i in range(4)])

    stats = ir.from_jaxpr(jax.make_jaxpr(looped)(jnp.ones((4, 8), jnp.float32)))
    assert stats.op_counts["exp"] == 4


def test_scan_bodies_are_counted_not_skipped():
    """A streaming-softmax rewrite is a scan. Counting only the top level would
    report it as a two-operation program."""
    def scanned(x):
        return jax.lax.scan(lambda c, r: (c + jnp.exp(r).sum(), r), 0.0, x)[0]

    stats = ir.from_jaxpr(jax.make_jaxpr(scanned)(jnp.ones((4, 8), jnp.float32)))
    assert stats.op_counts["scan"] == 1
    assert stats.op_counts["exp"] == 1, "the scan body was not walked"
    assert stats.total_ops > 2
