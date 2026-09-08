import time

import jax
import jax.numpy as jnp


def timeit_version_1(bad_func, X_test, Y_test, num_runs=100):
    """Time a compiled function returning one array; return seconds per call."""
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    _ = bad_func(X_test, Y_test).block_until_ready()

    # Time the compiled function after warm-up.
    start = time.perf_counter()
    for _ in range(num_runs):
        bad_func(X_test, Y_test).block_until_ready()
    end = time.perf_counter()

    avg_time = (end - start) / num_runs
    print(f"Average execution time: {avg_time * 1e6:.2f} µs")
    return avg_time


def evaluate(code, reference_func, X_test, Y_test, num_runs=100, function_name="kernel"):
    """Check a generated kernel against the original, then time it."""
    namespace = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    candidate = jax.jit(namespace[function_name])

    expected = reference_func(X_test, Y_test).block_until_ready()
    actual = candidate(X_test, Y_test).block_until_ready()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, None

    if jnp.issubdtype(expected.dtype, jnp.inexact):
        correct = bool(jnp.allclose(actual, expected, rtol=1e-5, atol=1e-6))
    else:
        correct = bool(jnp.array_equal(actual, expected))
    if not correct:
        return False, None

    return True, timeit_version_1(candidate, X_test, Y_test, num_runs)
