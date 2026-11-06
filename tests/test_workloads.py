"""Workload registry and the seeded softmax workload (AIC-003).

The frozen reference is itself checked against mathematical invariants and a NumPy float64 oracle,
so that copying a flawed reference could never establish correctness.
"""
import hashlib

import numpy as np
import pytest

from workloads import registry
from workloads.registry import SOFTMAX

# sha256 of SOFTMAX.make_inputs(0, (32, 128))[0].tobytes(), computed on the first run (macOS arm64,
# numpy 2.5.2). NumPy's PCG64 + standard_normal is platform-independent; a change means the inputs changed.
SEED0_32x128_SHA256 = "afb9825a6df7150ed3c35dbaed712813eced3e3fd34175de6ef3db4921f6127a"


def _jit_from_source(source, name):
    import jax
    namespace = {}
    exec(compile(source, "<reference>", "exec"), namespace)
    return jax.jit(namespace[name])


@pytest.fixture(scope="module")
def reference():
    return _jit_from_source(SOFTMAX.reference_source, SOFTMAX.function_name)


def _all_cases():
    cases = [(f"{split}{shape}/seed{seed}", "seed", (seed, shape))
             for split, shapes in (("dev", SOFTMAX.development_shapes), ("heldout", SOFTMAX.heldout_shapes))
             for shape in shapes for seed in SOFTMAX.search_seeds]
    cases += [(f"edge/{e.name}", "edge", e) for e in SOFTMAX.edge_cases]
    return cases


CASES = _all_cases()


def _inputs(kind, payload):
    if kind == "seed":
        seed, shape = payload
        return SOFTMAX.make_inputs(seed, shape)
    return payload.make(SOFTMAX.development_shapes[0])


def _within(got, expected, tol):
    got = np.asarray(got, dtype=np.float64)
    return bool(np.all(np.isfinite(got)) and np.all(np.abs(got - expected) <= tol.atol + tol.rtol * np.abs(expected)))


def test_registry_lists_softmax_and_example():
    assert {"softmax", "example_add"} <= set(registry.REGISTRY)
    assert registry.get("softmax") is SOFTMAX
    assert registry.get("softmax").scientific is True
    assert registry.get("example_add").scientific is False


def test_unknown_id_raises():
    with pytest.raises(KeyError, match="available"):
        registry.get("matmul")


def test_reference_hash_matches_file(tmp_path):
    import dataclasses
    assert SOFTMAX.reference_sha256 == hashlib.sha256(SOFTMAX.reference_path.read_bytes()).hexdigest()
    copy = tmp_path / "softmax.py"
    copy.write_bytes(SOFTMAX.reference_path.read_bytes().replace(b"keepdims", b"keepdims ", 1))
    assert dataclasses.replace(SOFTMAX, reference_path=copy).reference_sha256 != SOFTMAX.reference_sha256


def test_reference_source_has_no_inputs():
    assert "TEST_ARGS" not in SOFTMAX.reference_source
    assert "seed" not in SOFTMAX.reference_source


def test_inputs_deterministic():
    (a,) = SOFTMAX.make_inputs(0, (32, 128))
    (b,) = SOFTMAX.make_inputs(0, (32, 128))
    (c,) = SOFTMAX.make_inputs(1, (32, 128))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    assert a.dtype == np.float32 and a.shape == (32, 128)


def test_input_regression_hash():
    (a,) = SOFTMAX.make_inputs(0, (32, 128))
    assert hashlib.sha256(a.tobytes()).hexdigest() == SEED0_32x128_SHA256


def test_max_input_bytes_enforced():
    with pytest.raises(ValueError, match="B >"):
        SOFTMAX.make_inputs(0, (2048, 2048))   # 16 MiB > 8 MiB
    SOFTMAX.make_inputs(0, (256, 2048))         # the largest declared shape fits


def test_unsupported_dtype_rejected():
    with pytest.raises(ValueError, match="unsupported dtype"):
        SOFTMAX.make_inputs(0, (32, 128), dtype="float64")


@pytest.mark.parametrize("name,kind,payload", CASES, ids=[c[0] for c in CASES])
def test_reference_matches_oracle(reference, name, kind, payload):
    (x,) = _inputs(kind, payload)
    got = np.asarray(reference(x))
    assert got.dtype == np.float32 and got.shape == x.shape
    assert _within(got, SOFTMAX.oracle(x), SOFTMAX.tolerance), name


def test_reference_invariants(reference):
    (x,) = SOFTMAX.make_inputs(0, (32, 128))
    y = np.asarray(reference(x), dtype=np.float64)
    assert np.all(np.abs(y.sum(axis=-1) - 1.0) <= 1e-6)
    assert np.all(y >= 0)
    shifted = np.asarray(reference(x + np.float32(5.0)), dtype=np.float64)
    tol = SOFTMAX.tolerance
    assert np.all(np.abs(shifted - y) <= tol.atol + tol.rtol * np.abs(y))
    equal = next(e for e in SOFTMAX.edge_cases if e.name == "equal_logits")
    (xe,) = equal.make((32, 128))
    assert np.all(np.asarray(reference(xe)) == np.float32(1 / 128))


NAIVE = """
import jax.numpy as jnp

def softmax(x):
    e = jnp.exp(x)
    return e / jnp.sum(e, axis=-1, keepdims=True)
"""

LOGSUMEXP = """
import jax.numpy as jnp
from jax.scipy.special import logsumexp

def softmax(x):
    return jnp.exp(x - logsumexp(x, axis=-1, keepdims=True))
"""


def test_edge_cases_have_teeth():
    naive = _jit_from_source(NAIVE, "softmax")
    edge = next(e for e in SOFTMAX.edge_cases if e.name == "large_positive_offset")
    (x,) = edge.make(SOFTMAX.development_shapes[0])
    assert not np.all(np.isfinite(np.asarray(naive(x))))


def test_logsumexp_rewrite_admitted():
    """At the chosen OFFSET (P5 default 200) the equivalent logsumexp form passes every case.

    If the team picks OFFSET = 1e4 instead, invert this test: at offsets >= 1000, x - logsumexp(x) loses
    float32 precision and fails rtol 1e-5, so only max-subtraction forms would be admitted.
    """
    assert registry.OFFSET == 200.0
    lse = _jit_from_source(LOGSUMEXP, "softmax")
    for name, kind, payload in CASES:
        (x,) = _inputs(kind, payload)
        assert _within(np.asarray(lse(x)), SOFTMAX.oracle(x), SOFTMAX.tolerance), name


def test_example_add_matches_legacy(repo_root):
    import agent
    _, _, legacy_args = agent._load_source_workload(repo_root / "workloads/example.py", "bad_func")
    spec = registry.get("example_add")
    args = spec.make_inputs(0, (32,))
    assert len(args) == len(legacy_args) == 2
    for mine, legacy in zip(args, legacy_args):
        legacy = np.asarray(legacy)
        assert mine.dtype == legacy.dtype == np.float32
        assert np.array_equal(mine, legacy)
    assert spec.reference_path == repo_root / "workloads/example.py"
    assert np.array_equal(spec.oracle(*args), np.asarray(args[0], np.float64) + 1.0)
