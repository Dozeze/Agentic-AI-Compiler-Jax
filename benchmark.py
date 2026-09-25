import statistics
import timeit

import jax
import jax.numpy as jnp


def timeit_version_1(function, args, num_runs=30, num_tests=1):
    """Return the mean runtime per call across several timing batches."""
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")
    if num_tests <= 0:
        raise ValueError("num_tests must be positive.")

    times = []
    for _ in range(num_tests):
        # värm upp
        function(*args).block_until_ready()

        elapsed = timeit.timeit(
            lambda: function(*args).block_until_ready(),
            number=num_runs,
        )
        times.append(elapsed / num_runs)

    mean_runtime = statistics.mean(times)
    std_runtime = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Average execution time: {mean_runtime * 1e6:.2f} µs "
        f"(std: {std_runtime * 1e6:.2f} µs)"
    )
    return mean_runtime, std_runtime


def evaluate(code, reference_func, args, num_runs=30, num_tests=5, function_name="kernel"):
    """Kolla syntax"""
    namespace = {}

    try:
        compiled_code = compile(code, "<generated>", "exec")
    except SyntaxError as error:
        return {
            "syntax_correct": False,
            "error": str(error),
            "output_correct": False,
            "runtime_mean": None,
            "runtime_std": None,
        }

    try:
        exec(compiled_code, namespace)
        candidate = jax.jit(namespace[function_name])

        expected = reference_func(*args).block_until_ready()
        actual = candidate(*args).block_until_ready()

    except Exception as error:
        return {
            "syntax_correct": True,
            "error": str(error),
            "output_correct": False,
            "runtime_mean": None,
            "runtime_std": None,
        }

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "syntax_correct": True,
            "error": None,
            "output_correct": False,
            "runtime_mean": None,
            "runtime_std": None,
        }

    if jnp.issubdtype(expected.dtype, jnp.inexact):
        correct = bool(
            jnp.allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-6,
            )
        )
    else:
        correct = bool(jnp.array_equal(actual, expected))

    if not correct:
        return {
            "syntax_correct": True,
            "error": None,
            "output_correct": False,
            "runtime_mean": None,
            "runtime_std": None,
        }

    runtime_mean, runtime_std = timeit_version_1(
        candidate,
        args,
        num_runs=num_runs,
        num_tests=num_tests,
    )

    return {
        "syntax_correct": True,
        "error": None,
        "output_correct": True,
        "runtime_mean": runtime_mean,
        "runtime_std": runtime_std,
    }
