"""Characterization tests of the prototype on `main` (AIC-001). They pin today's behaviour offline."""
import json
import math
import subprocess
import sys
import tomllib

import pytest

from fakes import FakeAnthropic, proposal, tool_use

ORIGINAL = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x + y\n"
WRONG = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x - y\n"


@pytest.fixture
def example(repo_root):
    import agent
    source, reference, args = agent._load_source_workload(repo_root / "workloads/example.py", "bad_func")
    return source, reference, args


def _build(repo_root, script, output_dir, max_iterations=3):
    import agent
    client = FakeAnthropic(script)
    controller = agent.build_controller(client, max_iterations=max_iterations,
                                        source_path=repo_root / "workloads/example.py",
                                        function_name="bad_func", output_dir=output_dir)
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
    # Never assert `accepted` for identical code: today's acceptance rule is timing noise (AIC-005/006).
    assert history[1]["output_correct"] is True
    assert history[2]["output_correct"] is False
    assert history[2]["accepted"] is False
    for i in range(3):
        assert (out_dir / f"iteration_{i:04d}.json").is_file()
    assert len(json.loads((out_dir / "history.json").read_text())) == 3


def test_fake_loop_feeds_tool_results_back(no_key, repo_root):
    client, controller = _build(repo_root, _history_script(), no_key / "run")
    controller.agent_loop(model="fake-model")
    last = client.messages.calls[1]["messages_snapshot"][-1]
    assert last["role"] == "user"
    assert any(block["type"] == "tool_result" for block in last["content"])


def test_one_benchmark_per_turn(no_key, repo_root):
    script = [[tool_use("a", "benchmark_code", proposal(ORIGINAL)),
               tool_use("b", "benchmark_code", proposal(WRONG))], []]
    client, controller = _build(repo_root, script, no_key / "run")
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
