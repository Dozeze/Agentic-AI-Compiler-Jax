"""Correctness checking: comparator, correctness matrix and legacy evaluate() (AIC-004)."""
import json

import jax.numpy as jnp
import numpy as np
import pytest

import benchmark
from benchmark import (FAILURE_CATEGORIES, ReferenceInvalid, build_cases, check_correctness,
                       compare_outputs, load_function)
from fixtures.softmax_candidates import EXPECTED, FIXTURES
from workloads.registry import SOFTMAX, Tolerance

TOL = Tolerance(rtol=1e-5, atol=1e-6)
X = np.linspace(-1, 1, 12, dtype=np.float32).reshape(3, 4)


@pytest.fixture(scope="module")
def cases():
    return build_cases(SOFTMAX, include_heldout_shapes=True)


@pytest.fixture(scope="module")
def matrix(cases):
    return {name: check_correctness(load_function(src, "softmax"), SOFTMAX, cases)
            for name, src in FIXTURES.items()}


@pytest.mark.parametrize("tree", [X, (X, X * 2), {"a": X, "b": (X,)}], ids=["array", "tuple", "dict"])
def test_compare_identical_passes(tree):
    result = compare_outputs(tree, tree, TOL)
    assert result.passed and result.category is None


def test_compare_tree_mismatch():
    result = compare_outputs(X, (X,), TOL)
    assert result.category == "STRUCTURE_MISMATCH" and "tree" in result.detail


def test_compare_shape_mismatch():
    assert compare_outputs(X, X[:, :-1], TOL).category == "STRUCTURE_MISMATCH"


def test_compare_dtype_mismatch():
    result = compare_outputs(jnp.asarray(X), jnp.asarray(X).astype(jnp.bfloat16), TOL)
    assert result.category == "STRUCTURE_MISMATCH" and "dtype" in result.detail


def test_compare_int_exact():
    a = np.arange(5, dtype=np.int32)
    assert compare_outputs(a, a, TOL).passed
    assert compare_outputs(a, a + np.int32(1), TOL).category == "NUMERIC_MISMATCH"
    b = np.array([True, False])
    assert compare_outputs(b, b, TOL).passed
    assert compare_outputs(b, ~b, TOL).category == "NUMERIC_MISMATCH"


def test_compare_nonfinite():
    bad = X.copy()
    bad[0, 1] = np.nan
    bad[2, 3] = np.inf
    result = compare_outputs(X, bad, TOL)
    assert result.category == "NONFINITE"
    assert result.leaves[0].n_nonfinite == 2
    assert "(0, 1)" in result.leaves[0].detail


def test_compare_boundary_tolerance():
    tol = Tolerance(rtol=0.25, atol=0.5)           # exactly representable: bound at expected 2.0 is 1.0
    e = np.array([2.0], dtype=np.float32)
    assert compare_outputs(e, np.array([3.0], dtype=np.float32), tol).passed
    just_over = np.array([np.nextafter(np.float32(3.0), np.float32(4.0))], dtype=np.float32)
    assert compare_outputs(e, just_over, tol).category == "NUMERIC_MISMATCH"


def test_relative_error_near_zero():
    e = np.array([1e-30, 0.5], dtype=np.float32)
    a = np.array([2e-30, 0.5], dtype=np.float32)
    result = compare_outputs(e, a, TOL)
    assert result.passed
    assert result.leaves[0].max_rel_error <= 1e3


def test_reference_nonfinite_raises_in_compare():
    inf = np.array([np.inf], dtype=np.float32)
    with pytest.raises(ReferenceInvalid):
        compare_outputs(inf, inf, TOL)


def test_build_cases_counts():
    assert len(build_cases(SOFTMAX)) == 2 * 5 + 6
    assert len(build_cases(SOFTMAX, include_heldout_shapes=True)) == 2 * 5 + 6 + 2 * 5


@pytest.mark.parametrize("name", list(EXPECTED))
def test_correctness_matrix(matrix, name):
    report = matrix[name]
    expected_passed, expected_category = EXPECTED[name]
    assert (report.passed, report.category) == (expected_passed, expected_category), \
        [(c.case, c.category, c.detail) for c in report.cases if not c.passed][:3]
    assert all(c.category in (None, *FAILURE_CATEGORIES) for c in report.cases)


def test_matrix_details(matrix):
    naive = {c.case: c for c in matrix["NAIVE_UNSTABLE"].cases}
    assert all(c.passed for name, c in naive.items() if name.startswith("dev"))
    assert naive["edge/large_positive_offset"].category == "NONFINITE"
    overfit = {c.case: c for c in matrix["OVERFIT_VALUE"].cases}
    assert overfit["dev(32,128)/seed0"].passed
    assert all(not overfit[f"dev(32,128)/seed{s}"].passed for s in (1, 2, 3, 4))
    shape = {c.case: c for c in matrix["OVERFIT_SHAPE"].cases}
    assert shape["dev(32,128)/seed0"].passed and not shape["dev(128,1024)/seed0"].passed
    compile_error = [c for c in matrix["COMPILE_ERROR"].cases]
    assert all(c.category == "COMPILE_FAILED" for c in compile_error)
    assert any("not retried" in (c.detail or "") for c in compile_error)


def test_execution_failure_is_reported_separately(monkeypatch, cases):
    class Exploding:
        def __call__(self, *args):
            raise RuntimeError("device fault")

    monkeypatch.setattr(benchmark, "_compile", lambda jitted, args: Exploding())
    report = check_correctness(load_function(FIXTURES["REFERENCE"], "softmax"), SOFTMAX, cases[:2])
    assert report.category == "EXECUTION_FAILED"
    assert all(c.category == "EXECUTION_FAILED" for c in report.cases)


def test_policy_hash_changes_with_tolerance(cases):
    import dataclasses
    fn = load_function(FIXTURES["REFERENCE"], "softmax")
    base = check_correctness(fn, SOFTMAX, cases[:1]).policy_hash
    looser = dataclasses.replace(SOFTMAX, tolerance=Tolerance(rtol=1e-4, atol=1e-6))
    assert check_correctness(fn, looser, cases[:1]).policy_hash != base
    assert check_correctness(fn, SOFTMAX, cases[:2]).policy_hash != base


def test_report_is_strict_json(matrix):
    for report in matrix.values():
        json.dumps(report.to_json(), allow_nan=False)


def test_report_size_bounded(matrix):
    worst = max(len(json.dumps(r.to_json())) for r in matrix.values())
    assert worst <= 16 * 1024, worst


def test_report_contains_no_inputs(matrix):
    text = json.dumps(matrix["NUMERIC_OFFSET"].to_json())
    assert "seed" not in text.replace("/seed", "")   # only case names mention seeds


def test_reference_failure_raises(cases):
    fn = load_function(FIXTURES["REFERENCE"], "softmax")

    def broken(x):
        raise ValueError("reference bug")

    with pytest.raises(ReferenceInvalid):
        check_correctness(fn, SOFTMAX, cases[:1], reference_fn=broken)
    with pytest.raises(ReferenceInvalid):
        check_correctness(fn, SOFTMAX, cases[:1], reference_fn=lambda x: x + jnp.inf)


ADD = "def bad_func(x, y):\n    return x + y\n"


def _example():
    import jax
    reference = jax.jit(load_function(ADD, "bad_func"))
    return reference, (jnp.arange(32, dtype=jnp.float32), jnp.ones(32, dtype=jnp.float32))


def test_evaluate_backward_compatible():
    reference, args = _example()
    result = benchmark.evaluate(ADD, reference, args, function_name="bad_func")
    assert {"syntax_correct", "error", "output_correct", "runtime_mean", "runtime_std"} <= set(result)
    assert result["output_correct"] is True and result["error"] is None
    assert result["correctness"]["passed"] is True
    tuple_src = "def bad_func(x, y):\n    return (x + y,)\n"
    result = benchmark.evaluate(tuple_src, reference, args, function_name="bad_func")
    assert result["output_correct"] is False
    assert result["runtime_mean"] is None
    assert result["correctness"]["category"] == "STRUCTURE_MISMATCH"
    json.dumps(result, allow_nan=False)


def test_evaluate_supports_pytree_outputs():
    import jax
    src = "def bad_func(x, y):\n    return {'s': x + y, 'd': (x - y,)}\n"
    reference = jax.jit(load_function(src, "bad_func"))
    _, args = _example()
    result = benchmark.evaluate(src, reference, args, num_runs=2, num_tests=2, function_name="bad_func")
    assert result["output_correct"] is True and result["runtime_mean"] > 0


def test_evaluate_rejects_both_inf():
    import jax
    src = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x + y + jnp.inf\n"
    reference = jax.jit(load_function(src, "bad_func"))
    _, args = _example()
    result = benchmark.evaluate(src, reference, args, function_name="bad_func")
    assert result["output_correct"] is False
    assert "non-finite" in result["error"]


# ---------------------------------------------------------------------------
# The same matrix across the process boundary (AIC-008): worker outputs, host comparison
# ---------------------------------------------------------------------------

def _worker_outputs(tmp_path, name, source, cases):
    import array_io
    import worker
    work = tmp_path / name
    work.mkdir()
    (work / "cand.py").write_text(source)
    array_io.save(work / "inputs.npz", {n: array_io.flatten(args) for n, args in cases})
    request = {"schema_version": worker.REQUEST_SCHEMA, "mode": "outputs", "backend": "cpu",
               "function_name": "softmax", "sources": {"candidate": "cand.py"}, "inputs": "inputs.npz"}
    result = worker.handle(request, work)
    outputs = array_io.load(work / "outputs.npz", base=work, max_bytes=256 << 20)
    return result["cases"], outputs


@pytest.fixture(scope="module")
def worker_matrix(cases, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("worker-matrix")
    _, expected = _worker_outputs(tmp, "frozen-reference", FIXTURES["REFERENCE"], cases)
    names = [n for n, _ in cases]
    policy = benchmark.policy_hash(SOFTMAX.tolerance, names, SOFTMAX.workload_id, SOFTMAX.reference_sha256)
    reports = {}
    for name, source in FIXTURES.items():
        statuses, actual = _worker_outputs(tmp, name, source, cases)
        reports[name] = benchmark.report_from_outputs(names, expected, statuses, actual, SOFTMAX.tolerance, policy)
    return reports


@pytest.mark.parametrize("name", list(EXPECTED))
def test_worker_matrix_matches_in_process_matrix(worker_matrix, matrix, name):
    across, inside = worker_matrix[name], matrix[name]
    assert (across.passed, across.category) == EXPECTED[name]
    assert [(c.case, c.passed, c.category) for c in across.cases] == \
        [(c.case, c.passed, c.category) for c in inside.cases]
    assert across.policy_hash == inside.policy_hash


def test_report_from_outputs_missing_output_is_an_execution_failure():
    from array_io import flatten
    expected = {"c": flatten(X)}
    report = benchmark.report_from_outputs(["c"], expected, [{"name": "c", "status": "ok"}], {}, TOL, "h")
    assert report.category == "EXECUTION_FAILED"
    report = benchmark.report_from_outputs(["c"], expected, [], {}, TOL, "h")
    assert "no result" in report.cases[0].detail
