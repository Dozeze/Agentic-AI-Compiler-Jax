"""End to end in a real container (AIC-006/009): the controller evaluates a candidate it cannot trust.

Mechanics only: these tests check that every stage ran and was recorded, never how fast anything was.
"""
import pytest

from contracts import MeasurementProtocol
from fakes import FakeAnthropic, proposal, tool_use
from fixtures.softmax_candidates import JAX_NN, NAIVE_UNSTABLE

pytestmark = pytest.mark.needs_docker

FAST = MeasurementProtocol(warmup_calls=2, min_block_s=0.001, target_block_s=0.002, max_block_s=0.01,
                           n_blocks=6, bootstrap_samples=500, max_iqr_over_median=1.0, max_baseline_drift=1.0)


def test_softmax_candidates_are_judged_inside_the_container(no_key):
    import agent
    from runner import DockerRunner
    script = [[tool_use("a", "benchmark_code", proposal(JAX_NN))],
              [tool_use("b", "benchmark_code", proposal(NAIVE_UNSTABLE))], []]
    controller = agent.build_controller(FakeAnthropic(script), max_iterations=2, output_dir=no_key / "run",
                                        workload_id="softmax", runner=DockerRunner(selected_backend="cpu"),
                                        selected_backend="cpu", protocol=FAST)
    history = controller.agent_loop(model="fake-model")

    rewrite, unstable = history[1], history[2]
    assert rewrite["output_correct"] is True
    assert rewrite["state"] in ("ACCEPTED", "VALID_NO_CHANGE", "VALID_SLOWER", "MEASUREMENT_UNSTABLE")
    evaluation = rewrite["attempt"]["evaluation"]
    assert evaluation["environment"]["runner"].startswith("aic-worker:cpu-")
    assert evaluation["correctness"]["n_cases"] == 16 and evaluation["sessions"]
    assert unstable["state"] == "CORRECTNESS_FAILED" and "NONFINITE" in unstable["reason"]
    calibration = (no_key / "run" / "work" / "calibration.json").read_text()
    assert "aic-worker:cpu-" in calibration
