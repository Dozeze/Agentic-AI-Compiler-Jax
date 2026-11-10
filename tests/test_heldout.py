"""Held-out evaluation (AIC-018): commit-reveal, disjoint seeds, host-side verdict, burned splits."""
import dataclasses

import pytest

import heldout
from fixtures.softmax_candidates import OVERFIT_SHAPE, REFERENCE
from runner import TrustedSubprocessRunner
from workloads.registry import SOFTMAX


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """A softmax spec with a freshly frozen split in a temporary salt directory."""
    monkeypatch.setenv("AIC_HELDOUT_DIR", str(tmp_path / "salts"))
    spec = dataclasses.replace(SOFTMAX, heldout_commitment=None)
    spec = dataclasses.replace(spec, heldout_commitment=heldout.init(spec))
    return spec, tmp_path / "ledger.jsonl"


def test_seeds_are_many_disjoint_and_reproducible(frozen):
    spec, _ = frozen
    salt = heldout.load_salt(spec)
    seeds = heldout.derive_seeds(salt, spec)
    assert len(set(seeds)) == heldout.N_SEEDS >= 20
    assert not set(seeds) & (set(spec.search_seeds) | {heldout.EDGE_SEED})
    assert heldout.derive_seeds(salt, spec) == seeds
    cases = heldout.cases(spec, salt)
    assert len(cases) == 2 * heldout.N_SEEDS
    assert all(str(seed) not in name for name, _ in cases for seed in seeds)    # names do not reveal seeds
    assert {args[0].shape for _, args in cases} == set(spec.heldout_shapes)


def test_salt_must_match_the_commitment(frozen):
    spec, _ = frozen
    with pytest.raises(heldout.HeldoutError, match="does not match"):
        heldout.load_salt(dataclasses.replace(spec, heldout_commitment="0" * 64))
    with pytest.raises(heldout.HeldoutError, match="no committed"):
        heldout.load_salt(dataclasses.replace(spec, heldout_commitment=None))
    with pytest.raises(heldout.HeldoutError, match="frozen once"):
        heldout.init(spec)


def test_split_needs_two_unseen_shapes(frozen):
    spec, _ = frozen
    with pytest.raises(heldout.HeldoutError, match="two unseen shapes"):
        heldout.cases(dataclasses.replace(spec, heldout_shapes=((64, 257),)), heldout.load_salt(spec))


def test_evaluation_passes_the_reference_and_burns_on_failure(frozen, tmp_path):
    spec, ledger = frozen
    runner = TrustedSubprocessRunner(reviewed_fixtures=True)
    report, entry = heldout.evaluate(spec, REFERENCE, runner, tmp_path / "w1", selected_backend="cpu", ledger=ledger)
    assert report.passed and entry["n_cases"] == 40 and entry["passed"]

    report, entry = heldout.evaluate(spec, OVERFIT_SHAPE, runner, tmp_path / "w2", selected_backend="cpu",
                                     ledger=ledger)
    assert not report.passed and entry["category"] == "NUMERIC_MISMATCH"
    with pytest.raises(heldout.HeldoutError, match="burned"):
        heldout.evaluate(spec, REFERENCE, runner, tmp_path / "w3", selected_backend="cpu", ledger=ledger)
    newer = dataclasses.replace(spec, heldout_version=2, heldout_commitment=None)
    newer = dataclasses.replace(newer, heldout_commitment=heldout.init(newer))
    heldout.check_not_burned(newer, ledger)                    # a new version is a new split


def test_reveal_publishes_the_salt(frozen, tmp_path):
    spec, _ = frozen
    path = heldout.reveal(spec, tmp_path / "public")
    assert heldout.commitment(path.read_text().strip()) == spec.heldout_commitment


def test_search_tools_cannot_reach_the_heldout_split():
    import controller
    import inspect
    assert "heldout" not in inspect.getsource(controller)
