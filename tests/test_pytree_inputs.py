"""A task written the way JAX models are written: ``candidate(params, x)`` with
``params`` a dict of weights. The harness must move the leaves and keep the
structure, in-process and across the measurement subprocess."""

from __future__ import annotations

import textwrap

import jax
import numpy as np
import pytest

from halo import controller
from halo.agent.fake import ScriptedAgent
from halo.config import CorrectnessConfig, RunConfig, TimingConfig
from halo.harness import worker
from halo.store import RunStore
from halo.tasks import base

TASK = textwrap.dedent('''
    import numpy as np
    from halo.tasks.base import Case

    NAME = "tiny_mlp"
    DESCRIPTION = "A two-layer MLP. candidate(params, x) -> out; params is a dict."
    D, H = 64, 128

    def make_inputs(rng):
        params = {
            "w1": rng.standard_normal((D, H), dtype=np.float32) / 8,
            "b1": np.zeros((H,), np.float32),
            "w2": rng.standard_normal((H, D), dtype=np.float32) / 11,
        }
        return (params, rng.standard_normal((256, D), dtype=np.float32))

    def reference(params, x):
        p = {k: v.astype(np.float64) for k, v in params.items()}
        h = np.tanh(x.astype(np.float64) @ p["w1"] + p["b1"])
        return h @ p["w2"]

    def correctness_cases(rng):
        params, x = make_inputs(rng)
        return [Case("zeros", (params, np.zeros_like(x)))]
''')

SEED = textwrap.dedent('''
    import jax.numpy as jnp

    def candidate(params, x):
        rows = []
        for r in range(x.shape[0]):
            h = jnp.tanh(x[r] @ params["w1"] + params["b1"])
            rows.append(h @ params["w2"])
        return jnp.stack(rows)
''')

FAST = textwrap.dedent('''
    import jax.numpy as jnp

    def candidate(params, x):
        return jnp.tanh(x @ params["w1"] + params["b1"]) @ params["w2"]
''')


@pytest.fixture
def tasks_root(tmp_path, monkeypatch):
    task_dir = tmp_path / "tiny_mlp"
    task_dir.mkdir()
    (task_dir / "task.py").write_text(TASK)
    (task_dir / "candidate.py").write_text(SEED)
    monkeypatch.setenv("HALO_TASKS_ROOT", str(tmp_path))
    monkeypatch.setattr(base, "TASKS_ROOT", tmp_path)
    return tmp_path


def test_to_device_keeps_the_structure_and_moves_the_leaves():
    params = {"w": np.ones((2, 2), np.float32), "nested": (np.zeros(3, np.float32),)}
    moved = worker.to_device(jax, (params, np.ones(4, np.float32)))
    assert isinstance(moved, tuple) and isinstance(moved[0], dict)
    assert isinstance(moved[0]["w"], jax.Array)
    assert isinstance(moved[0]["nested"][0], jax.Array)
    assert isinstance(moved[1], jax.Array)


@pytest.mark.slow
def test_a_dict_of_params_survives_the_subprocess_and_is_measured(tmp_path, tasks_root):
    cfg = RunConfig(
        task="tiny_mlp",
        timing=TimingConfig(warmup=2, rounds=6, bootstrap_samples=300),
        correctness=CorrectnessConfig(n_random_cases=1),
    )
    result = controller.run(cfg, ScriptedAgent(FAST, "fast"), RunStore(tmp_path / "runs", "r"))
    assert result.attempts[0].measurement.correctness.passed
    assert result.improved, result.attempts[1].decision.reason
    assert result.best.measurement.speedup.ratio > 2
