"""Data contracts: RunConfig v1 (AIC-002).

A run's parameters are resolved once into one JSON document, with precedence
CLI > config file > documented defaults, and the resolved values are recorded.
Parsing is strict: unknown keys, wrong types and invalid values raise ConfigError
naming the dotted path of the offending key.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from environment import BACKENDS

RUN_CONFIG_SCHEMA = "run_config/v1"
SPLITS = ("development", "heldout")
CONTEXT_CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "K")
RUNNERS = ("docker", "subprocess")


class ContractError(ValueError):
    """A document does not satisfy its contract. The message names the dotted path of the problem."""


class ConfigError(ContractError):
    """Unknown key, wrong type or invalid value in a run config. The message names the dotted path."""


# Terminal states of a candidate (AIC-1.1 §3). A provider error or an empty response is a request
# event, not a candidate, so it has no state here.
PROPOSAL_INVALID = "PROPOSAL_INVALID"
POLICY_REJECTED = "POLICY_REJECTED"
COMPILE_FAILED = "COMPILE_FAILED"
EXECUTION_FAILED = "EXECUTION_FAILED"
TIMEOUT = "TIMEOUT"
RESOURCE_LIMIT = "RESOURCE_LIMIT"
CORRECTNESS_FAILED = "CORRECTNESS_FAILED"
MEASUREMENT_UNSTABLE = "MEASUREMENT_UNSTABLE"
VALID_SLOWER = "VALID_SLOWER"
VALID_NO_CHANGE = "VALID_NO_CHANGE"
DUPLICATE = "DUPLICATE"
ACCEPTED = "ACCEPTED"
INTERRUPTED = "INTERRUPTED"
TERMINAL_STATES = (PROPOSAL_INVALID, POLICY_REJECTED, COMPILE_FAILED, EXECUTION_FAILED, TIMEOUT, RESOURCE_LIMIT,
                   CORRECTNESS_FAILED, MEASUREMENT_UNSTABLE, VALID_SLOWER, VALID_NO_CHANGE, DUPLICATE, ACCEPTED,
                   INTERRUPTED)


@dataclass(frozen=True)
class Budgets:
    """Development ceilings from AIC-1.1 §7. Recorded only in this batch; enforced by AIC-013/015."""
    max_candidates: int = 3
    max_provider_requests: int = 6
    max_output_tokens_per_request: int = 2048
    max_total_tokens: int = 30_000
    run_wall_s: float = 600.0
    compile_timeout_s: float = 60.0
    correctness_case_timeout_s: float = 10.0
    measurement_session_timeout_s: float = 120.0
    max_log_bytes: int = 1 << 20
    max_source_bytes: int = 64 << 10
    max_artifact_bytes: int = 256 << 20
    money_cap: float | None = None   # without a cap, live mode is disabled (AIC-1.1 §7)
    currency: str | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    """What benchmark.py actually does today. AIC-005/006 will replace this protocol."""
    protocol: str = "legacy_timeit/v0"
    num_runs: int = 30
    num_tests: int = 5


@dataclass(frozen=True)
class MeasurementProtocol:
    """Paired-block timing protocol (AIC-1.1 §5, AIC-005). Frozen per experiment; `protocol_hash` records it.

    Each block runs the same number of synchronised calls of A (incumbent) and B (candidate) back to back,
    in a balanced, seeded AB/BA order. The repeat count is calibrated from the baseline only so that one
    block lasts about `target_block_s`, then frozen for both sides.
    """
    protocol: str = "paired_blocks/v1"
    warmup_calls: int = 5                  # per executable, before any timed block
    min_block_s: float = 0.010             # calibration grows the repeat count until a block reaches this
    target_block_s: float = 0.020          # ... then scales it to this
    max_block_s: float = 0.050             # a single call longer than this is still one repeat (recorded)
    max_repeats: int = 10_000              # hard cap on calls per block
    n_blocks: int = 20                     # paired blocks per session (AIC-1.1: at least 20)
    order_seed: int = 0                    # seeds the AB/BA shuffle
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0
    confidence: float = 0.95
    max_iqr_over_median: float = 0.10      # diagnostic instability limits (AIC-1.1 §5.6)
    max_baseline_drift: float = 0.05
    aa_sessions: int = 5                   # independent A/A sessions before search
    unstable_retries: int = 1              # one fixed retry, never unbounded reruns
    confirmation_sessions: int = 1         # fresh sessions a provisional winner must also pass
    final_sessions: int = 3                # fresh sessions against the original baseline for a final winner


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "anthropic"
    model_id: str = "claude-haiku-4-5-20251001"   # AIC-1.1: "a starting suggestion, not a guarantee"
    max_tokens: int = 2048


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = RUN_CONFIG_SCHEMA
    backend: str | None = None       # cpu|cuda|rocm|tpu; None = legacy auto-selection, marked non-scientific
    workload_id: str = "example_add"
    function_name: str = "bad_func"
    source_path: str | None = None
    split: str = "development"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    correctness_policy_hash: str | None = None   # filled from the workload policy once the registry is wired in
    context_condition: str = "C2"
    runner: str = "docker"           # docker: isolated containers; subprocess: trusted code only (AIC-009)
    calibration_path: str | None = None   # a frozen A/A calibration; without it the run calibrates at start
    output_dir: str = "runs/latest"
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    measurement: MeasurementProtocol = field(default_factory=MeasurementProtocol)
    model: ModelSettings = field(default_factory=ModelSettings)
    budgets: Budgets = field(default_factory=Budgets)


# ---------------------------------------------------------------------------
# Strict parsing (dataclasses do not validate types)
# ---------------------------------------------------------------------------

def _type_name(value: Any) -> str:
    return type(value).__name__


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(hint)
        if value is None and type(None) in options:
            return None
        non_none = [o for o in options if o is not type(None)]
        if len(non_none) != 1:
            raise ConfigError(f"{path}: unsupported union type {hint!r}")
        return _coerce(non_none[0], value, path)
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {_type_name(value)}")
        return _build(hint, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {_type_name(value)}")
        (item_hint, _ellipsis) = typing.get_args(hint)
        return tuple(_coerce(item_hint, v, f"{path}[{i}]") for i, v in enumerate(value))
    if hint is int:
        # bool is a subclass of int: reject it explicitly. Floats (even 3.0) are rejected too.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {_type_name(value)} {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {_type_name(value)} {value!r}")
        if not math.isfinite(value):
            raise ConfigError(f"{path}: non-finite numbers are not allowed")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {_type_name(value)} {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported field type {hint!r}")


def _build(cls, data: dict, path: str):
    """Build dataclass `cls` from a plain dict, strictly. Missing keys take the dataclass default."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object, got {_type_name(data)}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    for key in data:
        if key not in names:
            raise ConfigError(f"{path}.{key}: unknown key")
    kwargs = {key: _coerce(hints[key], value, f"{path}.{key}") for key, value in data.items()}
    for f in dataclasses.fields(cls):
        required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        if required and f.name not in kwargs:
            raise ConfigError(f"{path}.{f.name}: required")
    obj = cls(**kwargs)
    if cls is RunConfig:
        validate(obj)
    return obj


def validate(cfg: RunConfig) -> None:
    """Value checks after type checks. Raises ConfigError."""
    if cfg.schema_version != RUN_CONFIG_SCHEMA:
        raise ConfigError(f"run_config.schema_version: expected {RUN_CONFIG_SCHEMA!r}, got {cfg.schema_version!r}")
    if cfg.backend is not None and cfg.backend not in BACKENDS:
        raise ConfigError(f"run_config.backend: must be one of {BACKENDS} or null, got {cfg.backend!r}")
    if cfg.split not in SPLITS:
        raise ConfigError(f"run_config.split: must be one of {SPLITS}, got {cfg.split!r}")
    if cfg.runner not in RUNNERS:
        raise ConfigError(f"run_config.runner: must be one of {RUNNERS}, got {cfg.runner!r}")
    if cfg.context_condition not in CONTEXT_CONDITIONS:
        raise ConfigError(f"run_config.context_condition: must be one of {CONTEXT_CONDITIONS}, "
                          f"got {cfg.context_condition!r}")
    if not cfg.seeds:
        raise ConfigError("run_config.seeds: must not be empty")
    if len(set(cfg.seeds)) != len(cfg.seeds):
        raise ConfigError("run_config.seeds: must be unique")
    if any(s < 0 for s in cfg.seeds):
        raise ConfigError("run_config.seeds: must be >= 0")
    for f in dataclasses.fields(Budgets):
        value = getattr(cfg.budgets, f.name)
        if isinstance(value, (int, float)) and value < 0:
            raise ConfigError(f"run_config.budgets.{f.name}: negative limits are invalid")
    if cfg.budgets.money_cap is not None:
        if cfg.budgets.money_cap <= 0:
            raise ConfigError("run_config.budgets.money_cap: must be > 0 when set")
        if not cfg.budgets.currency:
            raise ConfigError("run_config.budgets.currency: required when money_cap is set")
    for name in ("num_runs", "num_tests"):
        if getattr(cfg.benchmark, name) <= 0:
            raise ConfigError(f"run_config.benchmark.{name}: must be positive")
    if cfg.model.max_tokens <= 0:
        raise ConfigError("run_config.model.max_tokens: must be positive")
    validate_protocol(cfg.measurement, path="run_config.measurement")


def validate_protocol(p: MeasurementProtocol, path: str = "measurement") -> None:
    """The AIC-1.1 §5 minimums. Tests may build smaller protocols directly; a run config never can."""
    if p.protocol != "paired_blocks/v1":
        raise ConfigError(f"{path}.protocol: expected 'paired_blocks/v1', got {p.protocol!r}")
    if p.warmup_calls < 5:
        raise ConfigError(f"{path}.warmup_calls: AIC-1.1 requires at least 5")
    if p.n_blocks < 20:
        raise ConfigError(f"{path}.n_blocks: AIC-1.1 requires at least 20 paired blocks")
    if not 0 < p.min_block_s <= p.target_block_s <= p.max_block_s:
        raise ConfigError(f"{path}: need 0 < min_block_s <= target_block_s <= max_block_s")
    if p.max_repeats < 1 or p.bootstrap_samples < 100:
        raise ConfigError(f"{path}: max_repeats must be >= 1 and bootstrap_samples >= 100")
    if not 0.5 < p.confidence < 1:
        raise ConfigError(f"{path}.confidence: must be in (0.5, 1)")
    if p.max_iqr_over_median <= 0 or p.max_baseline_drift <= 0:
        raise ConfigError(f"{path}: instability limits must be positive")
    if p.aa_sessions < 5:
        raise ConfigError(f"{path}.aa_sessions: AIC-1.1 requires at least 5 A/A sessions")
    if p.unstable_retries < 0 or p.confirmation_sessions < 1 or p.final_sessions < 3:
        raise ConfigError(f"{path}: need unstable_retries >= 0, confirmation_sessions >= 1, final_sessions >= 3")


def protocol_hash(p: MeasurementProtocol) -> str:
    """sha256 of the canonical protocol: a calibration is only valid for the exact protocol it was made with."""
    return hashlib.sha256(json.dumps(dataclasses.asdict(p), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Merge, serialise, hash
# ---------------------------------------------------------------------------

def _expand_dotted(data: dict) -> dict:
    """{"budgets.max_candidates": 5} -> {"budgets": {"max_candidates": 5}}."""
    out: dict = {}
    for key, value in data.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"{key}: conflicts with a non-object value")
        node[parts[-1]] = value
    return out


def _deep_merge(base: dict, override: dict, path: str) -> None:
    """Merge `override` into `base` in place; an unknown key raises immediately, naming its source."""
    if not isinstance(override, dict):
        raise ConfigError(f"{path}: expected an object, got {_type_name(override)}")
    for key, value in _expand_dotted(override).items():
        here = f"{path}.{key}"
        if key not in base:
            raise ConfigError(f"{here}: unknown key")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{here}: expected an object, got {_type_name(value)}")
            _deep_merge(base[key], value, here)
        else:
            base[key] = value


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)                                   # tuples become lists: JSON friendly


def canonical_json(cfg) -> str:
    return json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(cfg) -> str:
    """sha256 of the canonical JSON of a resolved config."""
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()


def from_dict(data: dict) -> RunConfig:
    """Strictly rebuild a RunConfig from a resolved JSON document (e.g. a run's run_config.json)."""
    return _build(RunConfig, data, path="run_config")


def resolve_run_config(cli: dict[str, Any], config_path: Path | None) -> RunConfig:
    """Resolve once: documented defaults, then the JSON config file, then explicit CLI values (None = not given)."""
    merged = to_dict(RunConfig())                                    # 1. defaults
    if config_path is not None:                                      # 2. file (JSON)
        try:
            file_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"config: cannot read {config_path}: {exc.strerror or exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config: {config_path} is not valid JSON: {exc}") from exc
        _deep_merge(merged, file_data, path="config")
    _deep_merge(merged, {k: v for k, v in cli.items() if v is not None}, path="cli")  # 3. explicit CLI only
    return _build(RunConfig, merged, path="run_config")


# ---------------------------------------------------------------------------
# Proposal v1, Evaluation v1, Attempt v1 (AIC-007, AIC-1.1 §3)
# ---------------------------------------------------------------------------

PROPOSAL_SCHEMA = "proposal/v1"
EVALUATION_SCHEMA = "evaluation/v1"
ATTEMPT_SCHEMA = "attempt/v1"
MAX_PROPOSAL_TEXT_CHARS = 2000          # each free-text field; the source has its own byte limit


@dataclass(frozen=True)
class Proposal:
    """What the model may send: a hypothesis and a complete candidate source, nothing else.

    No file paths, shell commands or evaluation parameters: inputs, tolerances, the timing protocol and
    the decision belong to the harness. `expected_effect` is a hypothesis, never a measurement.
    """
    hypothesis: str
    transformation: str
    rationale: str
    expected_effect: str
    candidate_source: str
    parent_id: int | None = None
    constraints: tuple[str, ...] = ()
    notes: str | None = None
    target: str = "jax_rewrite"
    schema_version: str = PROPOSAL_SCHEMA


def parse_proposal(data: Any, *, max_source_bytes: int) -> Proposal:
    """Strictly validate a model-produced proposal. Unknown fields and wrong types raise ContractError."""
    proposal = _build(Proposal, data, path="proposal")
    if proposal.schema_version != PROPOSAL_SCHEMA:
        raise ContractError(f"proposal.schema_version: expected {PROPOSAL_SCHEMA!r}")
    if proposal.target != "jax_rewrite":
        raise ContractError(f"proposal.target: only 'jax_rewrite' is supported, got {proposal.target!r}")
    for name in ("hypothesis", "transformation", "rationale", "expected_effect"):
        value = getattr(proposal, name)
        if not value.strip():
            raise ContractError(f"proposal.{name}: must not be empty")
        if len(value) > MAX_PROPOSAL_TEXT_CHARS:
            raise ContractError(f"proposal.{name}: longer than {MAX_PROPOSAL_TEXT_CHARS} characters")
    if proposal.notes is not None and len(proposal.notes) > MAX_PROPOSAL_TEXT_CHARS:
        raise ContractError(f"proposal.notes: longer than {MAX_PROPOSAL_TEXT_CHARS} characters")
    if len(proposal.constraints) > 20 or any(len(c) > 200 for c in proposal.constraints):
        raise ContractError("proposal.constraints: at most 20 items of 200 characters")
    if len(proposal.candidate_source.encode("utf-8")) > max_source_bytes:
        raise ContractError(f"proposal.candidate_source: larger than {max_source_bytes} bytes")
    if proposal.parent_id is not None and proposal.parent_id < 0:
        raise ContractError("proposal.parent_id: must be >= 0")
    return proposal


@dataclass(frozen=True)
class Evaluation:
    """Evidence about one candidate, assembled by the host from validated worker results.

    Unavailable values are None with a reason in `unavailable`, never zero. `stages` maps each stage
    (policy, compile, correctness, measurement) to "ok", "failed" or "skipped".
    """
    candidate_sha256: str
    stages: dict
    correctness: dict | None = None        # CorrectnessReport.to_json()
    compile: dict | None = None            # {"lower_s": ..., "compile_s": ...} per input signature
    sessions: tuple = ()                   # [{"raw": PairedSession json, "analysis": SessionAnalysis json}]
    environment: dict | None = None        # worker device info and runner identity
    ir: dict | None = None                 # compiler feedback summary (AIC-010)
    logs: dict | None = None               # bounded stdout/stderr tails
    failure_category: str | None = None
    unavailable: dict = field(default_factory=dict)
    schema_version: str = EVALUATION_SCHEMA


@dataclass(frozen=True)
class Attempt:
    """One candidate's full record. Built by the host only: a worker cannot set its state or reason."""
    attempt_id: int
    state: str
    reason: str
    proposal: Proposal | None = None       # None for the immutable baseline
    parent_id: int | None = None
    evaluation: Evaluation | None = None
    provider: dict | None = None           # request id and token usage of the proposing request
    started_utc: str | None = None
    finished_utc: str | None = None
    schema_version: str = ATTEMPT_SCHEMA

    def __post_init__(self):
        if self.state not in TERMINAL_STATES:
            raise ContractError(f"attempt.state: unknown terminal state {self.state!r}")

    def to_json(self) -> dict:
        """Strict JSON-ready dict; writers must use allow_nan=False."""
        return dataclasses.asdict(self)
