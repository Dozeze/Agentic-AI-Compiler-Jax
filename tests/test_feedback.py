"""Compiler feedback (AIC-010): every lowering level, separate estimate namespaces, bounded artifacts."""
import jax
import jax.numpy as jnp
import numpy as np

import benchmark
import feedback
from fixtures.softmax_candidates import REFERENCE

X = jnp.asarray(np.random.default_rng(0).standard_normal((32, 128)).astype(np.float32))


def _analysis(source=REFERENCE, artifacts=None):
    fn = benchmark.load_function(source, "softmax")
    compiled = benchmark.compile_timed(fn, (X,))
    return feedback.analyze(jax, fn, (X,), compiled, artifacts=artifacts)


def test_every_level_is_summarised():
    out = _analysis()
    assert out["parser_version"] == feedback.PARSER_VERSION
    assert set(out["levels"]) == {"jaxpr", "stablehlo", "hlo", "compiled"}
    assert out["unavailable"] == {}
    jaxpr = out["levels"]["jaxpr"]
    assert jaxpr["groups"]["reductions"] == 2            # max and sum over the last axis
    assert jaxpr["groups"]["dots"] == 0
    assert out["levels"]["stablehlo"]["groups"]["reductions"] == 2
    assert out["levels"]["compiled"]["total_ops"] > 0


def test_estimates_are_kept_apart_from_measurements():
    out = _analysis()
    assert set(out["compiler_cost_estimates"]) == set(feedback.COST_KEYS)
    assert out["compiler_cost_estimates"]["flops"] > 0
    assert out["compiler_memory_estimates"]["argument_size_in_bytes"] == 32 * 128 * 4
    assert out["runtime_trace"]["coverage"] == "UNAVAILABLE"
    assert "runtime_measurements" not in out


def test_a_loop_shows_up_as_repeated_ops_before_compilation():
    looped = ("import jax.numpy as jnp\n\ndef softmax(x):\n    rows = []\n    for i in range(4):\n"
              "        r = x[i * 8:(i + 1) * 8]\n        e = jnp.exp(r - jnp.max(r, axis=-1, keepdims=True))\n"
              "        rows.append(e / jnp.sum(e, axis=-1, keepdims=True))\n    return jnp.concatenate(rows)\n")
    assert _analysis(looped)["levels"]["jaxpr"]["groups"]["reductions"] == 8


def test_raw_artifacts_are_written_and_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "MAX_ARTIFACT_BYTES", 100)
    out = _analysis(artifacts=tmp_path / "ir")
    for level in ("jaxpr", "stablehlo", "hlo", "compiled"):
        assert (tmp_path / "ir" / f"{level}.txt").stat().st_size <= 100
        assert out["levels"][level]["truncated"] is True and out["levels"][level]["artifact"] == f"ir/{level}.txt"


def test_unavailable_values_are_null_with_a_reason():
    class NoEstimates:
        def cost_analysis(self):
            raise NotImplementedError("not on this backend")

        def memory_analysis(self):
            return None

        def as_text(self):
            return "ROOT %r = f32[2]{0} add(f32[2]{0} %a, f32[2]{0} %b)"

    fn = benchmark.load_function(REFERENCE, "softmax")
    real = benchmark.compile_timed(fn, (X,))
    fake = benchmark.Compiled(NoEstimates(), real.lowered, 0.0, 0.0)
    out = feedback.analyze(jax, fn, (X,), fake)
    assert out["compiler_cost_estimates"] is None and "NotImplementedError" in out["unavailable"]["compiler_cost_estimates"]
    assert out["compiler_memory_estimates"] is None
    assert out["unavailable"]["compiler_memory_estimates"] == "not reported by this backend"
    assert out["levels"]["compiled"]["top_ops"] == {"add": 1}


def test_hlo_parser_handles_root_names_and_tuple_shapes():
    text = "\n".join([
        "HloModule m",
        "  %p = f32[4,8]{1,0} parameter(0)",
        "  fusion.1 = f32[4]{0} fusion(f32[4,8]{1,0} %p), kind=kLoop, calls=%f",
        "  %t = (f32[4]{0}, s32[]) tuple(f32[4]{0} %fusion.1, s32[] %c)",
        "  ROOT %d = f32[4,4]{1,0} dot(f32[4,8]{1,0} %p, f32[8,4]{1,0} %q), lhs_contracting_dims={1}",
    ])
    counts = feedback.hlo_counts(text)
    assert counts == {"parameter": 1, "fusion": 1, "tuple": 1, "dot": 1}
    groups = feedback.level_summary(text, counts)["groups"]
    assert (groups["dots"], groups["fusions"]) == (1, 1)
