"""RunConfig v1: strict parsing, precedence and stable hashing (AIC-002)."""
import dataclasses
import json

import pytest

import contracts
from contracts import Budgets, ConfigError, RunConfig, config_hash, resolve_run_config


def _write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


def test_defaults_match_plan():
    assert dataclasses.asdict(Budgets()) == {
        "max_candidates": 3,
        "max_provider_requests": 6,
        "max_output_tokens_per_request": 2048,
        "max_total_tokens": 30_000,
        "run_wall_s": 600.0,
        "compile_timeout_s": 60.0,
        "correctness_case_timeout_s": 10.0,
        "measurement_session_timeout_s": 120.0,
        "max_log_bytes": 1 << 20,
        "max_source_bytes": 64 << 10,
        "max_artifact_bytes": 256 << 20,
        "money_cap": None,
        "currency": None,
    }


def test_precedence_cli_over_file_over_default(tmp_path):
    path = _write(tmp_path, {"budgets": {"max_candidates": 5}})
    assert resolve_run_config({"budgets.max_candidates": 7}, path).budgets.max_candidates == 7
    assert resolve_run_config({"budgets.max_candidates": None}, path).budgets.max_candidates == 5
    assert resolve_run_config({}, None).budgets.max_candidates == 3


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, {"budgets": {"max_candidatez": 1}})
    with pytest.raises(ConfigError, match=r"budgets\.max_candidatez"):
        resolve_run_config({}, path)
    with pytest.raises(ConfigError, match=r"^cli\.nonsense"):
        resolve_run_config({"nonsense": 1}, None)


@pytest.mark.parametrize("override", [
    {"budgets.max_candidates": True},
    {"budgets.max_candidates": "3"},
    {"budgets.max_candidates": 3.0},
    {"seeds": [0, "1"]},
    {"backend": "gpu"},
    {"budgets": 3},
    {"budgets.run_wall_s": "600"},
])
def test_wrong_types_rejected(override):
    with pytest.raises(ConfigError):
        resolve_run_config(override, None)


def test_float_field_accepts_int():
    cfg = resolve_run_config({"budgets.run_wall_s": 30}, None)
    assert cfg.budgets.run_wall_s == 30.0 and isinstance(cfg.budgets.run_wall_s, float)


def test_negative_budget_rejected():
    with pytest.raises(ConfigError, match="negative"):
        resolve_run_config({"budgets.max_total_tokens": -1}, None)


def test_zero_candidates_allowed():
    assert resolve_run_config({"budgets.max_candidates": 0}, None).budgets.max_candidates == 0


def test_money_cap_requires_currency():
    with pytest.raises(ConfigError, match="currency"):
        resolve_run_config({"budgets.money_cap": 5.0}, None)
    cfg = resolve_run_config({"budgets.money_cap": 5.0, "budgets.currency": "SEK"}, None)
    assert cfg.budgets.money_cap == 5.0


def test_nonfinite_rejected():
    with pytest.raises(ConfigError, match="non-finite"):
        resolve_run_config({"budgets.run_wall_s": float("nan")}, None)


@pytest.mark.parametrize("override", [{"seeds": []}, {"seeds": [1, 1]}, {"seeds": [-1]},
                                      {"split": "test"}, {"context_condition": "C9"},
                                      {"schema_version": "run_config/v0"}])
def test_invalid_values_rejected(override):
    with pytest.raises(ConfigError):
        resolve_run_config(override, None)


def test_invalid_json_file_is_a_config_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        resolve_run_config({}, path)


def test_hash_stable_roundtrip():
    cfg = resolve_run_config({"backend": "cpu", "budgets.max_candidates": 2}, None)
    rebuilt = contracts.from_dict(json.loads(contracts.canonical_json(cfg)))
    assert rebuilt == cfg
    assert config_hash(rebuilt) == config_hash(cfg)
    assert isinstance(rebuilt.seeds, tuple)


def test_hash_changes_with_value():
    base = config_hash(RunConfig())
    for f in dataclasses.fields(Budgets):
        value = getattr(Budgets(), f.name)
        if f.name == "currency":
            changed = {"budgets.currency": "SEK"}
        elif f.name == "money_cap":
            changed = {"budgets.money_cap": 1.0, "budgets.currency": "SEK"}
        else:
            changed = {f"budgets.{f.name}": value + 1}
        assert config_hash(resolve_run_config(changed, None)) != base, f.name


def test_measurement_protocol_defaults_meet_plan():
    p = contracts.MeasurementProtocol()
    contracts.validate_protocol(p)
    assert p.warmup_calls >= 5 and p.n_blocks >= 20 and p.aa_sessions >= 5 and p.final_sessions >= 3
    assert (p.min_block_s, p.target_block_s, p.max_block_s) == (0.010, 0.020, 0.050)   # AIC-1.1: 10-50 ms


@pytest.mark.parametrize("override", [
    {"measurement.n_blocks": 19},
    {"measurement.warmup_calls": 4},
    {"measurement.aa_sessions": 4},
    {"measurement.final_sessions": 2},
    {"measurement.min_block_s": 0.1},
    {"measurement.confidence": 1.0},
    {"measurement.protocol": "legacy_timeit/v0"},
])
def test_measurement_protocol_below_plan_minimum_rejected(override):
    with pytest.raises(ConfigError, match="measurement"):
        resolve_run_config(override, None)


def test_protocol_hash_tracks_every_field():
    base = contracts.MeasurementProtocol()
    assert contracts.protocol_hash(base) == contracts.protocol_hash(contracts.MeasurementProtocol())
    assert contracts.protocol_hash(base) != contracts.protocol_hash(dataclasses.replace(base, order_seed=1))


GOOD_PROPOSAL = {"hypothesis": "one batched op", "transformation": "vectorise the loop",
                 "rationale": "XLA sees one op", "expected_effect": "about 2x",
                 "candidate_source": "def f(x):\n    return x\n"}


def test_proposal_parses_and_fills_defaults():
    p = contracts.parse_proposal(GOOD_PROPOSAL, max_source_bytes=1024)
    assert (p.target, p.schema_version, p.parent_id, p.constraints) == ("jax_rewrite", "proposal/v1", None, ())


@pytest.mark.parametrize("change, message", [
    ({"code": "x"}, "proposal.code: unknown key"),
    ({"hypothesis": None}, "proposal.hypothesis"),
    ({"hypothesis": "  "}, "must not be empty"),
    ({"target": "native_kernel"}, "only 'jax_rewrite'"),
    ({"parent_id": True}, "proposal.parent_id"),
    ({"parent_id": -1}, "must be >= 0"),
    ({"candidate_source": "x" * 2000}, "larger than 1024 bytes"),
    ({"rationale": "r" * 5000}, "longer than"),
])
def test_proposal_rejects_bad_fields(change, message):
    with pytest.raises(contracts.ContractError, match=message):
        contracts.parse_proposal({**GOOD_PROPOSAL, **change}, max_source_bytes=1024)


def test_proposal_missing_field_rejected():
    data = dict(GOOD_PROPOSAL)
    del data["candidate_source"]
    with pytest.raises(contracts.ContractError, match="candidate_source: required"):
        contracts.parse_proposal(data, max_source_bytes=1024)


def test_attempt_requires_a_terminal_state_and_serialises():
    evaluation = contracts.Evaluation(candidate_sha256="ab", stages={"policy": "ok"})
    attempt = contracts.Attempt(attempt_id=1, state=contracts.VALID_NO_CHANGE, reason="noise",
                                evaluation=evaluation)
    data = attempt.to_json()
    assert data["schema_version"] == "attempt/v1"
    assert data["evaluation"]["schema_version"] == "evaluation/v1"
    json.dumps(data, allow_nan=False)
    with pytest.raises(contracts.ContractError, match="unknown terminal state"):
        contracts.Attempt(attempt_id=2, state="FASTER", reason="trust me")
