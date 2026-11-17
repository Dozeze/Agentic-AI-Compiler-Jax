"""AIC-017 references, frozen numerical contracts and integration with the shared evaluator."""
import dataclasses
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import benchmark
import heldout
from context_builder import workload_contract
from contracts import Budgets, MeasurementProtocol, resolve_run_config
from controller import Evaluator
from runner import DockerRunner, TrustedSubprocessRunner, image_files
from source_policy import check_source
from workloads import registry
from workloads.registry import LAYER_NORM, MATMUL_GELU, SOFTMAX

NEW_SPECS = (LAYER_NORM, MATMUL_GELU)
SPECS = (SOFTMAX, *NEW_SPECS)
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def references():
    functions = {s.workload_id: benchmark.load_function(s.reference_source, s.function_name) for s in SPECS}
    return {key: {"eager": fn, "jit": jax.jit(fn)} for key, fn in functions.items()}


def _reference_cases():
    for spec in NEW_SPECS:
        for shape in spec.development_shapes:
            for seed in spec.search_seeds:
                yield spec, f"dev{shape}/seed{seed}", (seed, shape)
        for edge in spec.edge_cases:
            yield spec, f"edge/{edge.name}", edge
        # Trusted reference shape smoke only, using search seed 0 rather than the final split seeds.
        for shape in spec.heldout_shapes:
            yield spec, f"shape-smoke{shape}", (0, shape)


REFERENCE_CASES = list(_reference_cases())


def within(got, expected, tolerance):
    got = np.asarray(got, np.float64)
    return np.all(np.isfinite(got)) and np.all(
        np.abs(got - expected) <= tolerance.atol + tolerance.rtol * np.abs(expected))


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("spec,name,case", REFERENCE_CASES,
                         ids=[f"{s.workload_id}/{name}" for s, name, _ in REFERENCE_CASES])
def test_reference_matches_independent_oracle(references, mode, spec, name, case):
    args = (spec.make_inputs(*case) if isinstance(case, tuple) else case.make(spec.development_shapes[0]))
    before = [a.tobytes() for a in args]
    got = np.asarray(references[spec.workload_id][mode](*args))
    expected = spec.oracle(*args)
    shape = args[0].shape if spec is LAYER_NORM else (args[0].shape[0], args[1].shape[-1])
    assert got.dtype == np.float32 and got.shape == expected.shape == shape
    assert within(got, expected, spec.tolerance), name
    assert [a.tobytes() for a in args] == before


@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
def test_inputs_contract_is_reproducible_bounded_and_disjoint(spec, monkeypatch):
    heldout.check_split(spec)
    assert registry.get(spec.workload_id) is spec and spec.scientific
    assert "seed" not in spec.reference_source and "TEST_ARGS" not in spec.reference_source
    assert check_source(spec.reference_source, spec.function_name, max_source_bytes=65536).ok
    for shape in (*spec.development_shapes, *spec.heldout_shapes):
        a, b, c = (spec.make_inputs(seed, shape) for seed in (0, 0, 1))
        assert len(a) == 3 and all(x.dtype == np.float32 and np.isfinite(x).all() for x in a)
        assert all(x.tobytes() == y.tobytes() and not np.shares_memory(x, y) for x, y in zip(a, b))
        assert any(x.tobytes() != y.tobytes() for x, y in zip(a, c))
        assert sum(x.nbytes for x in a) <= spec.max_input_bytes
    for dtype in ("float16", "bfloat16", "float64", "int32"):
        with pytest.raises(ValueError, match="unsupported dtype"):
            spec.make_inputs(0, spec.development_shapes[0], dtype)
    monkeypatch.setattr(np.random, "default_rng", lambda *a: pytest.fail("allocated before checking the cap"))
    with pytest.raises(ValueError, match="before allocation"):
        spec.make_inputs(0, (10**8, 128) if spec is LAYER_NORM else (10**8, 256, 256))


@pytest.mark.parametrize("spec,shape", [(LAYER_NORM, (32,)), (LAYER_NORM, (32, 0)),
    (MATMUL_GELU, (2, 3)), (MATMUL_GELU, (32, -1, 64)), (MATMUL_GELU, (32, True, 64)),
    (LAYER_NORM, (2, 3.5))])
def test_invalid_shapes_are_rejected(spec, shape):
    with pytest.raises(ValueError, match="positive integer dimensions"):
        spec.make_inputs(0, shape)


@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_reference_rejects_precision_changes(references, spec, dtype):
    args = [jnp.asarray(a, dtype=dtype) for a in spec.make_inputs(0, spec.development_shapes[0])]
    with pytest.raises(ValueError, match="float32 inputs only"):
        references[spec.workload_id]["jit"](*args)


def test_layer_norm_known_values_and_affine_semantics(references):
    fn = references["layer_norm"]["jit"]
    x = np.array([[-1, 0, 1], [7, 7, 7]], dtype=np.float32)
    scale = np.array([2, -1, 0], dtype=np.float32)
    bias = np.array([0.5, -0.5, 1], dtype=np.float32)
    expected = np.array([[-1, 0, 1], [0, 0, 0]], np.float64) / np.sqrt(2 / 3 + 1e-5)
    expected = expected * scale + bias
    assert within(fn(x, scale, bias), expected, LAYER_NORM.tolerance)
    assert np.array_equal(np.asarray(fn(x, scale, bias))[1], bias)
    # Normalized variance is var/(var+epsilon), not exactly one for nearly constant rows.
    edge = next(e for e in LAYER_NORM.edge_cases if e.name == "nearly_constant_rows")
    (near, _, _) = edge.make(LAYER_NORM.development_shapes[0])
    y = np.asarray(fn(near, np.ones(near.shape[-1], np.float32), np.zeros(near.shape[-1], np.float32)))
    var = np.var(near.astype(np.float64), axis=-1)
    assert np.allclose(np.var(y.astype(np.float64), axis=-1), var / (var + 1e-5), rtol=1e-3, atol=1e-6)


def test_dense_known_values_and_highest_precision(references):
    x = np.array([[1, -1]], np.float32)
    weight = np.array([[1, 1, 0], [0, 2, 0]], np.float32)
    bias = np.zeros(3, np.float32)
    got = references["matmul_gelu"]["jit"](x, weight, bias)
    assert np.allclose(got, [[0.8413447460685429, -0.15865525393145707, 0]], rtol=1e-6, atol=1e-7)
    assert np.allclose(MATMUL_GELU.oracle(x, weight, bias), got, rtol=1e-6, atol=1e-7)
    graph = jax.make_jaxpr(references["matmul_gelu"]["eager"])(x, weight, bias)
    # jnp.matmul is itself jitted in JAX 0.4.38: inspect nested jaxprs as well as the outer function.
    def dots(graph):
        for equation in graph.jaxpr.eqns:
            if equation.primitive.name == "dot_general":
                yield equation
            for value in equation.params.values():
                if hasattr(value, "jaxpr"):
                    yield from dots(value)
    dot, = dots(graph)
    assert dot.params["precision"] == (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    assert dot.params["preferred_element_type"] == np.dtype("float32")


BAD_REWRITES = [
    (LAYER_NORM, "wrong_epsilon", LAYER_NORM.reference_source.replace("variance + 1e-5", "variance + 1e-3")),
    (LAYER_NORM, "missing_affine", LAYER_NORM.reference_source.replace("normalized * scale + bias", "normalized")),
    (LAYER_NORM, "unstable_variance", LAYER_NORM.reference_source.replace(
        "jnp.mean(centered * centered, axis=-1, keepdims=True)", "jnp.mean(x * x, axis=-1, keepdims=True) - mean * mean")),
    (MATMUL_GELU, "approximate_gelu", MATMUL_GELU.reference_source.replace("approximate=False", "approximate=True")),
    (MATMUL_GELU, "missing_bias", MATMUL_GELU.reference_source.replace("projected + bias", "projected")),
    (MATMUL_GELU, "reduced_output_precision", MATMUL_GELU.reference_source.replace(
        "approximate=False)", "approximate=False).astype(jnp.bfloat16)")),
]


@pytest.mark.parametrize("spec,name,source", BAD_REWRITES, ids=[name for _, name, _ in BAD_REWRITES])
def test_edge_population_rejects_semantic_changes(spec, name, source):
    candidate = benchmark.load_function(source, spec.function_name)
    report = benchmark.check_correctness(candidate, spec, benchmark.build_cases(spec))
    assert not report.passed, name
    assert report.category in ("NUMERIC_MISMATCH", "NONFINITE", "STRUCTURE_MISMATCH")


@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
def test_new_workloads_reach_context_cli_and_worker_image(spec):
    contract = workload_contract(spec)
    assert contract["static_params"] == spec.static_params
    assert contract["tolerance"] == dataclasses.asdict(spec.tolerance)
    assert "heldout_shapes" not in contract and "search_seeds" not in contract
    assert spec.reference_path in image_files()
    cfg = resolve_run_config({"workload_id": spec.workload_id}, REPO / "configs/target.json")
    assert cfg.workload_id == spec.workload_id and cfg.backend == "cuda"


@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
def test_calibration_retains_configuration_honors_caps_and_refuses_reuse(spec, tmp_path, monkeypatch, capsys):
    from scripts import calibrate_aa
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budgets": {"compile_timeout_s": 0}}))
    monkeypatch.setattr(TrustedSubprocessRunner, "run", lambda *a, **kw: pytest.fail("launched despite zero compile cap"))
    raw = tmp_path / "raw"
    argv = ["--workload", spec.workload_id, "--backend", "cpu", "--runner", "subprocess", "--label", "bounded",
            "--config", str(config), "--work-dir", str(raw), "--out", str(tmp_path / "calibration")]
    assert calibrate_aa.main(argv) == 2
    assert "reference run failed" in capsys.readouterr().err
    context = json.loads((raw / "calibration_context.json").read_bytes())
    assert context["config"]["budgets"]["compile_timeout_s"] == 0
    assert context["config"]["workload_id"] == spec.workload_id
    assert context["reference_sha256"] == spec.reference_sha256
    assert not list(raw.glob("*-reference"))
    before = (raw / "calibration_context.json").read_bytes()
    assert calibrate_aa.main(argv) == 2
    assert "must be new" in capsys.readouterr().err
    assert (raw / "calibration_context.json").read_bytes() == before


def check_evaluator(spec, runner, selected_backend, workdir, budgets=None):
    evaluator = Evaluator(spec, runner, MeasurementProtocol(), workdir,
                          selected_backend=selected_backend, budgets=budgets)
    before = spec.reference_path.read_bytes()
    evaluator.reference_outputs()
    report, compiled, ir = evaluator.correctness(spec.reference_source, trusted=True)
    assert report.passed and len(report.cases) == len(benchmark.build_cases(spec))
    assert compiled and ir["provenance"]["backend"] in ("cpu", "gpu", "cuda")
    for name, args in benchmark.build_cases(spec):
        tree = evaluator.reference[name]
        assert within(tree.arrays[0], spec.oracle(*args), spec.tolerance)
    assert spec.reference_path.read_bytes() == before
    return evaluator


@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
def test_native_workers_support_multi_input_workloads(spec, tmp_path):
    evaluator = check_evaluator(spec, TrustedSubprocessRunner(), "cpu", tmp_path)
    assert evaluator.measure_shape == spec.development_shapes[-1]
    assert evaluator.baseline_trace["coverage"] in ("CPU_EXECUTION_EVENTS", "HOST_ONLY", "UNAVAILABLE")


@pytest.mark.target
@pytest.mark.needs_gpu
@pytest.mark.needs_docker
@pytest.mark.skipif(not os.environ.get("AIC_TARGET_CONFIG"), reason="selected-target acceptance is explicitly invoked")
@pytest.mark.parametrize("spec", NEW_SPECS, ids=lambda s: s.workload_id)
def test_new_references_on_selected_target(spec, tmp_path):
    cfg = resolve_run_config({}, Path(os.environ["AIC_TARGET_CONFIG"]))
    assert cfg.backend == "cuda" and cfg.runner == "docker"
    evaluator = check_evaluator(spec, DockerRunner(selected_backend=cfg.backend), cfg.backend, tmp_path, cfg.budgets)
    assert evaluator.fingerprint.startswith(("gpu|", "cuda|"))
