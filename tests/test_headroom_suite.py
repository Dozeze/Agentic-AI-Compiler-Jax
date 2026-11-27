"""Headroom suite (batch 04): references, oracles and ceilings.

Correctness only: the measured ceilings live in evidence/ceilings/ and are never asserted here (rule 5).
"""
from pathlib import Path

import jax
import numpy as np
import pytest

import benchmark
from runner import IMAGE_FILES
from workloads import registry

REPO = Path(__file__).resolve().parents[1]
ALL = (*registry.HEADROOM, *registry.CONTROLS)


def _source(spec, ceiling=False):
    path = REPO / "workloads" / ("ceilings" if ceiling else "") / f"{spec.workload_id}.py"
    return benchmark.load_function(path.read_text(encoding="utf-8"), spec.function_name, str(path))


@pytest.mark.parametrize("spec", registry.HEADROOM, ids=lambda s: s.workload_id)
def test_reference_matches_its_float64_oracle(spec):
    reference = jax.jit(_source(spec))
    for name, args in benchmark.build_cases(spec):
        expected = spec.oracle(*args)
        actual = np.asarray(reference(*args), np.float64)
        assert np.allclose(actual, expected, rtol=spec.tolerance.rtol, atol=spec.tolerance.atol), name


@pytest.mark.parametrize("spec", ALL, ids=lambda s: s.workload_id)
def test_ceiling_passes_the_correctness_gate(spec):
    report = benchmark.check_correctness(_source(spec, ceiling=True), spec, benchmark.build_cases(spec))
    assert report.passed, [case for case in report.to_json()["cases"] if not case["passed"]]


@pytest.mark.parametrize("spec", registry.HEADROOM, ids=lambda s: s.workload_id)
def test_headroom_split_is_disjoint_and_in_the_worker_image(spec):
    assert not set(spec.development_shapes) & set(spec.heldout_shapes)
    assert not set(spec.search_seeds) & set(spec.heldout_seeds) and len(spec.heldout_seeds) >= 20
    assert f"workloads/{spec.workload_id}.py" in IMAGE_FILES
    assert registry.get(spec.workload_id) is spec


def test_edge_cases_have_teeth():
    # A candidate that drops the middle factor or swaps operands must fail on the edge cases.
    wrong = {"matmul_chain": "import jax.numpy as jnp\n\ndef matmul_chain(a, b, c):\n    return jnp.matmul(a, c.T @ c)\n",
             "pairwise_distances": "import jax.numpy as jnp\n\ndef pairwise_distances(a, b):\n"
                                   "    return jnp.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1).T * 1.001\n"}
    for workload, source in wrong.items():
        spec = registry.get(workload)
        report = benchmark.check_correctness(benchmark.load_function(source, spec.function_name), spec,
                                             benchmark.build_cases(spec))
        assert not report.passed, workload
