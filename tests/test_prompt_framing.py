"""Prompt framing (batch 04): validated config field, reaches the system prompt, carries no task hints."""
import json

import pytest

import agent
from contracts import ConfigError, resolve_run_config
from workloads import registry

ALL = (*registry.HEADROOM, *registry.CONTROLS, registry.EXAMPLE_ADD)


def test_prompt_framing_is_validated(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"prompt_framing": "compiler_complement"}))
    assert resolve_run_config({}, path).prompt_framing == "compiler_complement"
    path.write_text(json.dumps({"prompt_framing": "hints"}))
    with pytest.raises(ConfigError):
        resolve_run_config({}, path)


def test_compiler_complement_framing_names_no_workload_or_rewrite():
    text = agent.COMPILER_COMPLEMENT.lower()
    for spec in ALL:
        assert spec.workload_id.replace("_", " ") not in text and spec.workload_id not in text
    for hint in ("einsum", "matmul", "associat", "loop", "||", "norm", "batch", "transpose"):
        assert hint not in text, hint


def test_framing_reaches_the_system_prompt(tmp_path):
    plain = agent.build_controller(None, workload_id="matmul_chain", output_dir=tmp_path / "a",
                                   selected_backend="cpu")
    framed = agent.build_controller(None, workload_id="matmul_chain", output_dir=tmp_path / "b",
                                    selected_backend="cpu", prompt_framing="compiler_complement")
    assert agent.COMPILER_COMPLEMENT not in plain.init_prompt
    assert framed.init_prompt.endswith(agent.COMPILER_COMPLEMENT)
