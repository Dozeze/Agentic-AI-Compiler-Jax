"""Integrity of the benchmark suite itself.

The suite is the instrument the ablation is measured with, so its properties are
asserted rather than assumed: every task well-formed, every seed and every ceiling
numerically correct, and enough of both classes for a comparison to mean anything.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from halo.config import CorrectnessConfig
from halo.harness import correctness
from halo.tasks import base, ceilings

TASKS = base.available()
IMPLEMENTATIONS = [(t, impl) for t in TASKS for impl in ("candidate", "ceiling")]


@pytest.mark.parametrize("name", TASKS)
def test_task_is_well_formed(name):
    spec = base.load_spec(name)
    module = spec.load()
    assert module.NAME == name
    assert module.DESCRIPTION.strip()
    for required in ("make_inputs", "reference", "correctness_cases"):
        assert callable(getattr(module, required))
    assert ceilings.path(spec).exists(), "every task needs a best-known implementation"


@pytest.mark.parametrize("name", TASKS)
def test_task_is_classified(name):
    assert name in ceilings.CLASSIFICATION
    assert ceilings.CLASSIFICATION[name] in {"headroom", "null"}
    assert name in ceilings.MEASURED_HEADROOM


def test_suite_has_power_in_both_directions():
    """An ablation needs tasks that can improve and tasks that must not."""
    headroom = ceilings.by_classification("headroom")
    null = ceilings.by_classification("null")
    assert len(headroom) >= 4, "too few improvable tasks to detect an effect"
    assert len(null) >= 4, "too few controls to detect a false-positive rate"


def test_headroom_classification_matches_the_measured_numbers():
    for name, kind in ceilings.CLASSIFICATION.items():
        measured = ceilings.MEASURED_HEADROOM[name]
        if kind == "headroom":
            assert measured > 1.2, f"{name} is classified as improvable but measured {measured}x"
        else:
            assert measured < 1.1, f"{name} is a control but measured {measured}x"


@pytest.mark.parametrize("name", TASKS)
def test_graded_inputs_are_withheld_from_the_prompt(name):
    excerpt = base.reference_excerpt(base.load_spec(name))
    assert "def correctness_cases" not in excerpt
    assert "def reference" in excerpt, "the agent still needs the semantics"


@pytest.mark.slow
@pytest.mark.parametrize("name,impl", IMPLEMENTATIONS)
def test_implementation_matches_its_oracle(name, impl):
    """Both the seed and the best known rewrite must pass. If a ceiling fails, the
    task's tolerance is too tight and its headroom would be reported as zero."""
    spec = base.load_spec(name)
    fn = base.load_module(spec.directory / f"{impl}.py", f"{impl}_{name}").candidate
    report = correctness.check(
        fn, spec.load(), CorrectnessConfig(), jit=jax.jit,
        to_device=lambda arrays: tuple(jnp.asarray(a) for a in arrays),
    )
    assert report.passed, f"{name}/{impl}: {report.summary()}"
