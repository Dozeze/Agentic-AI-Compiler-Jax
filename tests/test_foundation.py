"""Characterization tests of the prototype on `main` (AIC-001). They pin today's behaviour offline."""
import json
import math
import subprocess
import sys
import tomllib

import pytest

from contracts import MeasurementProtocol
from fakes import FakeAnthropic, proposal, tool_use

ORIGINAL = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x + y\n"
WRONG = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x - y\n"


@pytest.fixture
def example(repo_root):
    import agent
    source, reference, args = agent._load_source_workload(repo_root / "workloads/example.py", "bad_func")
    return source, reference, args


# A short protocol for tests: the mechanics are the same, the sessions are seconds, not minutes. Blocks of
# 2 ms on a shared CPU are noisy, so the instability limits are loosened here; these tests check the
# pipeline, never a speed (the real limits, 10 % and 5 %, are in contracts.MeasurementProtocol).
FAST = MeasurementProtocol(warmup_calls=2, min_block_s=0.001, target_block_s=0.002, max_block_s=0.01,
                           n_blocks=6, bootstrap_samples=500, max_iqr_over_median=1.0, max_baseline_drift=1.0)


def _build(repo_root, script, output_dir, max_iterations=3, runner=None, context_condition="C2"):
    """A controller on the example workload. Fixtures here are reviewed code, so the trusted runner may
    run them as if they were model output (agent.py never does this)."""
    import agent
    from runner import TrustedSubprocessRunner
    client = FakeAnthropic(script)
    controller = agent.build_controller(client, max_iterations=max_iterations,
                                        source_path=repo_root / "workloads/example.py",
                                        function_name="bad_func", output_dir=output_dir, selected_backend="cpu",
                                        runner=runner or TrustedSubprocessRunner(reviewed_fixtures=True),
                                        protocol=FAST, context_condition=context_condition)
    return client, controller


def test_core_modules_import_without_key(no_key):
    import agent  # noqa: F401
    import benchmark  # noqa: F401
    import controller  # noqa: F401


def test_cli_help_needs_no_key(repo_root, tmp_path, clean_env):
    out = subprocess.run([sys.executable, str(repo_root / "agent.py"), "--help"], cwd=tmp_path,
                         env=clean_env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "--source" in out.stdout


def test_missing_key_is_reported(no_key):
    import agent
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        agent.main(["--iterations", "1"])


def test_require_gpu_fails_on_cpu(no_key):
    import jax
    from environment import has_gpu
    if has_gpu(jax.devices()):
        pytest.skip("a GPU is present")
    import agent
    with pytest.raises(SystemExit) as exc:
        agent.main(["--require-gpu"])
    assert exc.value.code == 2


def test_example_workload_loads(example):
    source, reference, args = example
    assert isinstance(source, str) and "def bad_func" in source
    assert callable(reference)
    assert len(args) == 2
    assert all(a.shape == (32,) for a in args)


def test_evaluate_accepts_identical_code(example):
    import benchmark
    source, reference, args = example
    result = benchmark.evaluate(source, reference, args, function_name="bad_func")
    assert result["syntax_correct"] is True
    assert result["output_correct"] is True
    assert result["runtime_mean"] > 0 and math.isfinite(result["runtime_mean"])


def test_evaluate_rejects_syntax_error(example):
    import benchmark
    _, reference, args = example
    result = benchmark.evaluate("def bad_func(x,y) return", reference, args, function_name="bad_func")
    assert result["syntax_correct"] is False
    assert result["runtime_mean"] is None


def test_evaluate_rejects_wrong_output(example):
    import benchmark
    _, reference, args = example
    result = benchmark.evaluate(WRONG, reference, args, function_name="bad_func")
    assert result["output_correct"] is False


def test_evaluate_rejects_missing_function(example):
    import benchmark
    source, reference, args = example
    result = benchmark.evaluate(source, reference, args, function_name="no_such_function")
    assert result["output_correct"] is False
    assert result["error"]


def _history_script():
    return [
        [tool_use("t1", "get_current_code", {})],
        [tool_use("t2", "benchmark_code", proposal(ORIGINAL))],
        [tool_use("t3", "benchmark_code", proposal(WRONG))],
        [],
    ]


def test_fake_loop_records_history(no_key, repo_root):
    out_dir = no_key / "run"
    _, controller = _build(repo_root, _history_script(), out_dir)
    history = controller.agent_loop(model="fake-model")
    assert len(history) == 3
    assert history[0]["state"] == "ACCEPTED" and "baseline" in history[0]["reason"]
    # The same computation as the baseline: correct, measured, and never accepted on noise (AIC-006).
    assert history[1]["output_correct"] is True
    assert history[1]["state"] in ("VALID_NO_CHANGE", "VALID_SLOWER", "MEASUREMENT_UNSTABLE")
    sessions = history[1]["attempt"]["evaluation"]["sessions"]
    assert sessions and len(sessions[0]["raw"]["blocks"]) == FAST.n_blocks
    assert (out_dir / "work" / "calibration.json").is_file()
    assert history[2]["output_correct"] is False
    assert history[2]["accepted"] is False and history[2]["state"] == "CORRECTNESS_FAILED"
    for i in range(3):
        assert (out_dir / f"iteration_{i:04d}.json").is_file()
    assert len(json.loads((out_dir / "history.json").read_text())) == 3


def test_fake_loop_feeds_tool_results_back(no_key, repo_root):
    script = [[tool_use("t1", "get_current_code", {})], [tool_use("t2", "benchmark_code", proposal(WRONG))], []]
    client, controller = _build(repo_root, script, no_key / "run")
    controller.agent_loop(model="fake-model")
    last = client.messages.calls[1]["messages_snapshot"][-1]
    assert last["role"] == "user"
    assert any(block["type"] == "tool_result" for block in last["content"])


def test_one_benchmark_per_turn(no_key, repo_root):
    script = [[tool_use("a", "benchmark_code", proposal(WRONG)),
               tool_use("b", "benchmark_code", proposal(ORIGINAL))], []]
    client, controller = _build(repo_root, script, no_key / "run")
    controller.start()
    before = len(controller.history)
    controller.agent_loop(model="fake-model")
    assert len(controller.history) == before + 1
    tool_results = client.messages.calls[1]["messages_snapshot"][-1]["content"]
    assert "Only one benchmark_code call is allowed per turn" in tool_results[1]["content"]


def test_root_scripts_not_collected(repo_root):
    config = tomllib.loads((repo_root / "pyproject.toml").read_text())
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_invalid_proposal_is_recorded_not_run(no_key, repo_root):
    script = [[tool_use("a", "benchmark_code", {"code": ORIGINAL})], []]
    _, controller = _build(repo_root, script, no_key / "run")
    history = controller.agent_loop(model="fake-model")
    assert history[-1]["state"] == "PROPOSAL_INVALID"
    assert "proposal.code: unknown key" in history[-1]["error"]
    assert history[-1]["code"] is None


def test_policy_violation_is_recorded_not_run(no_key, repo_root):
    evil = "import os\n\ndef bad_func(x, y):\n    os.system('touch /tmp/pwned')\n    return x + y\n"
    script = [[tool_use("a", "benchmark_code", proposal(evil))], []]
    _, controller = _build(repo_root, script, no_key / "run")
    history = controller.agent_loop(model="fake-model")
    assert history[-1]["state"] == "POLICY_REJECTED"
    assert "import of 'os'" in history[-1]["error"]
    assert history[-1]["runtime_mean"] is None


def test_constructor_has_no_side_effects(no_key, repo_root):
    _, controller = _build(repo_root, [], no_key / "run")
    assert controller.history == [] and not (no_key / "run").exists()


def test_model_code_is_refused_without_isolation(no_key, repo_root):
    from runner import TrustedSubprocessRunner
    script = [[tool_use("a", "benchmark_code", proposal(WRONG))], [tool_use("b", "get_history", {})], []]
    client, controller = _build(repo_root, script, no_key / "run", runner=TrustedSubprocessRunner())
    history = controller.agent_loop(model="fake-model")
    assert history[-1]["state"] == "INTERRUPTED"
    assert "isolated runner" in history[-1]["reason"]
    assert controller.stop_reason and len(client.messages.calls) == 1   # the loop stopped right there


def test_duplicate_is_not_measured_again(no_key, repo_root):
    baseline = (repo_root / "workloads/example.py").read_text()
    reformatted = baseline.replace("    return x + y", "    return x+y   # same program")
    script = [[tool_use("a", "benchmark_code", proposal(reformatted))], []]
    _, controller = _build(repo_root, script, no_key / "run")
    history = controller.agent_loop(model="fake-model")
    assert history[-1]["state"] == "DUPLICATE" and "iteration 0" in history[-1]["reason"]
    assert history[-1]["attempt"]["evaluation"] is None


def test_history_shown_to_the_model_is_compact(no_key, repo_root):
    script = [[tool_use("a", "benchmark_code", proposal(WRONG))], []]
    _, controller = _build(repo_root, script, no_key / "run")
    controller.agent_loop(model="fake-model")
    view = controller.provide_hist()[-1]
    assert "attempt" not in view
    assert view["correctness"]["passed"] is False and view["correctness"]["first_failure"]


@pytest.mark.parametrize("condition, shown", [("C2", False), ("C3", True)])
def test_compiler_summary_only_in_compiler_conditions(no_key, repo_root, condition, shown):
    _, controller = _build(repo_root, [], no_key / "run", context_condition=condition)
    controller.start()
    baseline = controller.history[0]["attempt"]["evaluation"]["ir"]
    assert baseline["levels"]["compiled"]["total_ops"] > 0          # always recorded
    view = controller.provide_hist()[0]
    assert ("compiler" in view) is shown
    if shown:
        assert "estimates (not measurements)" in view["compiler"]
        assert "add" not in str(view["compiler"]["levels"])          # groups and totals, not raw op lists


def test_usage_is_recorded_per_request(no_key, repo_root):
    script = [[tool_use("a", "benchmark_code", proposal(WRONG))], []]
    _, controller = _build(repo_root, script, no_key / "run")
    history = controller.agent_loop(model="fake-model")
    assert controller.usage_totals() == {"requests": 2, "input_tokens": 200, "output_tokens": 20}
    assert history[-1]["attempt"]["provider"]["request"] == 1
    recorded = json.loads((no_key / "run" / "requests.json").read_text())
    assert [r["stop_reason"] for r in recorded] == ["tool_use", "end_turn"]
