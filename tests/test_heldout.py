"""Held-out evaluation (AIC-018): a frozen split, unreachable from the search, burned on failure."""
import dataclasses

import pytest

import heldout
from fixtures.softmax_candidates import OVERFIT_SHAPE, REFERENCE
from runner import TrustedSubprocessRunner
from workloads.registry import REGISTRY, SOFTMAX


@pytest.mark.parametrize("workload", [w for w, s in REGISTRY.items() if s.scientific])
def test_every_scientific_workload_has_a_valid_split(workload):
    heldout.check_split(REGISTRY[workload])


def test_split_has_enough_disjoint_seeds_and_unseen_shapes():
    cases = heldout.cases(SOFTMAX)
    assert len(cases) == len(SOFTMAX.heldout_seeds) * len(SOFTMAX.heldout_shapes) >= 40
    assert {args[0].shape for _, args in cases} == set(SOFTMAX.heldout_shapes)


@pytest.mark.parametrize("change, message", [
    ({"heldout_seeds": tuple(range(1000, 1019))}, "at least 20"),
    ({"heldout_seeds": tuple(range(0, 20))}, "overlap"),
    ({"heldout_seeds": tuple(range(1000, 1019)) + (1000,)}, "distinct"),
    ({"heldout_shapes": ((64, 257),)}, "never uses"),
    ({"heldout_shapes": ((64, 257), (32, 128))}, "never uses"),
])
def test_bad_splits_are_refused(change, message):
    with pytest.raises(heldout.HeldoutError, match=message):
        heldout.check_split(dataclasses.replace(SOFTMAX, **change))


def test_evaluation_passes_the_reference_and_burns_on_failure(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    runner = TrustedSubprocessRunner(reviewed_fixtures=True)
    report, entry = heldout.evaluate(SOFTMAX, REFERENCE, runner, tmp_path / "w1", selected_backend="cpu",
                                     ledger=ledger)
    assert report.passed and entry["n_cases"] == 40 and entry["passed"]

    report, entry = heldout.evaluate(SOFTMAX, OVERFIT_SHAPE, runner, tmp_path / "w2", selected_backend="cpu",
                                     ledger=ledger)
    assert not report.passed and entry["category"] == "NUMERIC_MISMATCH"
    with pytest.raises(heldout.HeldoutError, match="burned"):
        heldout.evaluate(SOFTMAX, REFERENCE, runner, tmp_path / "w3", selected_backend="cpu", ledger=ledger)
    heldout.check_not_burned(dataclasses.replace(SOFTMAX, heldout_version=2), ledger)   # a new split


def test_search_never_receives_the_heldout_split(no_key, repo_root):
    """The search's cases and everything the controller can send to the model come from development data."""
    import inspect

    import agent
    import controller
    assert "heldout" not in inspect.getsource(controller)
    built = agent.build_controller(None, workload_id="softmax", output_dir=no_key / "run", selected_backend="cpu",
                                   runner=TrustedSubprocessRunner())
    names = built.evaluator.case_names
    assert names and not any(name.startswith("heldout") for name in names)
    assert not any(f"seed{s}" in name for name in names for s in SOFTMAX.heldout_seeds)
