from __future__ import annotations

from halo.harness import hlo

SAMPLE = """\
HloModule jit_candidate, is_scheduled=true

%fused_computation.3 (param_0: f32[4,256,64]) -> f32[4,256,256] {
  %param_0 = f32[4,256,64]{2,1,0} parameter(0)
  %broadcast.1 = f32[4,256,256]{2,1,0} broadcast(f32[] %constant.2), dimensions={}
  ROOT %multiply.4 = f32[4,256,256]{2,1,0} multiply(%param_0, %broadcast.1)
}

ENTRY %main (p0: f32[4,8,256,64]) -> f32[4,8,256,64] {
  %p0 = f32[4,8,256,64]{3,2,1,0} parameter(0)
  %dot.1 = f32[4,8,256,256]{3,2,1,0} dot(%p0, %p0), lhs_contracting_dims={3}
  %fusion.1 = f32[4,8,256,256]{3,2,1,0} fusion(%dot.1), kind=kLoop
  ROOT %fusion.2 = f32[4,8,256,64]{3,2,1,0} fusion(%fusion.1), kind=kInput
}
"""


def test_parses_opcodes_including_root_and_nested_computations():
    counts = hlo.parse_op_counts(SAMPLE)
    assert counts["parameter"] == 2
    assert counts["fusion"] == 2  # the fused_computation header is not an instruction
    assert counts["dot"] == 1
    assert counts["multiply"] == 1
    assert counts["broadcast"] == 1


def test_summarize_tolerates_a_backend_without_cost_analysis():
    class NoAnalysis:
        def cost_analysis(self):
            raise NotImplementedError

        def memory_analysis(self):
            raise NotImplementedError

    summary = hlo.summarize(NoAnalysis(), SAMPLE)
    assert summary.fusion_count == 2
    assert summary.instruction_count == 7
    assert summary.flops is None
    assert summary.arithmetic_intensity is None


def test_arithmetic_intensity():
    class Analysis:
        def cost_analysis(self):
            return {"flops": 200.0, "bytes accessed": 50.0}

        def memory_analysis(self):
            return None

    assert hlo.summarize(Analysis(), SAMPLE).arithmetic_intensity == 4.0
