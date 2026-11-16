"""Durable run publication and artifact-only verification (AIC-012)."""
import dataclasses
import json
from pathlib import Path

import pytest

import storage
from contracts import ACCEPTED, CORRECTNESS_FAILED, Attempt, Evaluation, Proposal

BASELINE = "def kernel(x):\n    return x + 1\n"
CANDIDATE = "def kernel(x):\n    return 1 + x\n"


def _store(tmp_path, *, output="run", budgets=None):
    store = storage.RunStore(tmp_path / output, budgets=budgets)
    store.initialize(BASELINE, config={"schema_version": "test_config/v1"}, manifest={
        "workload": {"id": "test", "correctness_policy_hash": "frozen-test-policy", "split": "development"},
        "environment": {"schema_version": "environment/v1", "repo": {"uv_lock_sha256": "test-lock-hash"}}})
    return store


def _entry(number=0, *, source=None, state=ACCEPTED, provider=None):
    source = source or (CANDIDATE if number else BASELINE)
    proposal = Proposal("hypothesis", "transformation", "rationale", "expected", source) if number else None
    evaluation = Evaluation(
        candidate_sha256=storage.sha256(source.encode()),
        stages={"correctness": "ok", "measurement": "ok"} if number else {"reference": "ok"},
        correctness={"passed": state == ACCEPTED, "n_cases": 1} if number else None,
        sessions=({"against": "incumbent", "raw": {"blocks": [1, 2]}, "analysis": {"fixture": True}},)
        if number and state == ACCEPTED else ())
    attempt = Attempt(number, state, "host decision", proposal=proposal, parent_id=0 if number else None,
                      evaluation=evaluation, provider=provider)
    return {"iteration": number, "state": state, "accepted": state == ACCEPTED,
            "output_correct": state == ACCEPTED, "code": source, "function_name": "kernel",
            "runtime_mean": None, "runtime_std": None, "attempt": attempt.to_json()}


def _baseline(store):
    store.commit_attempt(_entry())
    return store.verified_best()


def test_default_runs_are_unique_and_constructors_do_not_write(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    first, second = storage.RunStore(), storage.RunStore()
    assert first.path != second.path
    assert first.run_id != second.run_id
    assert not list(tmp_path.iterdir())
    for store in (first, second):
        store.initialize(BASELINE, config={}, manifest={})
    assert first.path.is_dir() and second.path.is_dir()


@pytest.mark.parametrize("existing", ["empty_directory", "old_run", "file", "symlink"])
def test_explicit_path_collision_never_changes_existing_content(tmp_path, existing):
    target = tmp_path / "run"
    if existing == "file":
        target.write_text("user source")
    elif existing == "symlink":
        original = tmp_path / "original"
        original.mkdir()
        (original / "keep").write_text("unchanged")
        target.symlink_to(original, target_is_directory=True)
    else:
        target.mkdir()
        if existing == "old_run":
            (target / "history.json").write_text("old history")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(storage.StorageError, match="cannot create new run"):
        storage.RunStore(target).initialize(BASELINE, config={}, manifest={})
    assert {p: p.read_bytes() for p in before} == before
    if target.is_dir():
        assert not (target / "baseline.py").exists()
    else:
        assert target.read_text() == "user source"


def test_manifest_fingerprints_config_environment_and_exact_source(tmp_path):
    store = _store(tmp_path)
    manifest = json.loads((store.path / "manifest.json").read_bytes())
    assert manifest["run_id"] == store.run_id
    assert manifest["schema_version"] == "manifest/v1"
    for key in ("baseline", "config", "environment_artifact"):
        reference = manifest[key]
        data = (store.path / reference["path"]).read_bytes()
        assert reference["sha256"] == storage.sha256(data)
        assert reference["bytes"] == len(data)
    config = json.loads((store.path / "run_config.json").read_bytes())
    assert manifest["run_config_sha256"] == storage.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    assert (store.path / "baseline.py").read_bytes() == BASELINE.encode()
    assert manifest["environment"]["repo"]["uv_lock_sha256"] == "test-lock-hash"


def test_baseline_fallback_and_raw_artifact_references(tmp_path):
    store = _store(tmp_path)
    raw = store.path / "work/0001-reference/outputs.npz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw fixture output")
    best = _baseline(store)
    record = store.committed_records()[0]
    assert record["raw_artifacts"] == [store._reference("work/0001-reference/outputs.npz")]
    assert best["baseline_fallback"] is True
    assert best["scope"]["verification"] == "frozen reference"
    assert "not performed" in best["scope"]["final_verification"]
    assert (store.path / "best.py").read_bytes() == BASELINE.encode()
    store.commit_attempt(_entry(1, state=CORRECTNESS_FAILED))
    assert store.verified_best()["attempt_id"] == 0
    assert len(json.loads((store.path / "history.json").read_bytes())) == 2


def test_promotion_has_exact_source_proposal_request_and_measurements(tmp_path):
    store = _store(tmp_path)
    _baseline(store)
    event = {"request": 1, "model": "offline-fixture", "input_tokens": 123, "output_tokens": 45}
    event = store.record_request(event)
    store.commit_attempt(_entry(1, provider=event))
    record = store.committed_records()[1]
    assert set(record["evidence"]) == {"source", "proposal", "evaluation", "measurements", "feedback", "request"}
    assert record["evidence"]["request"]["path"] == "requests/0001.json"
    assert record["evidence"]["source"]["sha256"] == storage.sha256(CANDIDATE.encode())
    assert (store.path / "attempts/0001/source.py").read_bytes() == CANDIDATE.encode()
    assert (store.path / "best.py").read_bytes() == CANDIDATE.encode()
    assert store.verified_best()["baseline_fallback"] is False
    assert storage.verified_best(store.path)["attempt_id"] == 1


def test_atomic_immutable_publication_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    storage.atomic_write_json(path, {"original": True}, exclusive=True)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        storage.atomic_write_json(path, {"new": True}, exclusive=True)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_replace_failure_keeps_old_file_and_cleans_temporary(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    path.write_bytes(b"original history")
    def fail(*args):
        raise OSError("injected replace failure")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        storage.atomic_write_json(path, {"new": True})
    assert path.read_bytes() == b"original history"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_strict_json_failure_never_touches_target(tmp_path, value):
    path = tmp_path / "data.json"
    path.write_bytes(b"unchanged")
    with pytest.raises((TypeError, ValueError)):
        storage.atomic_write_json(path, {"value": value})
    assert path.read_bytes() == b"unchanged"
    assert list(tmp_path.iterdir()) == [path]


def test_failure_before_commit_marker_cannot_publish_candidate(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _baseline(store)
    before = (store.path / "history.json").read_bytes()
    write = storage.atomic_write_bytes
    def fail_marker(path, data, **kwargs):
        if Path(path).as_posix().endswith("attempts/0001/record.json"):
            raise OSError("injected marker failure")
        return write(path, data, **kwargs)
    monkeypatch.setattr(storage, "atomic_write_bytes", fail_marker)
    with pytest.raises(storage.StorageError):
        store.commit_attempt(_entry(1))
    assert (store.path / "attempts/0001/source.py").exists()
    assert not (store.path / "attempts/0001/record.json").exists()
    assert (store.path / "history.json").read_bytes() == before
    assert store.verified_best()["attempt_id"] == 0
    assert len(store.derive_indexes()) == 1
    with pytest.raises(storage.StorageError, match="directory already exists"):
        store.commit_attempt(_entry(1))


@pytest.mark.parametrize("index_name", ["history.json", "best.py", "best.json"])
def test_failure_after_commit_marker_leaves_rebuildable_evidence(monkeypatch, tmp_path, index_name):
    store = _store(tmp_path)
    _baseline(store)
    write = storage.atomic_write_bytes
    def fail_index(path, data, **kwargs):
        if Path(path) == store.path / index_name:
            raise OSError("injected index failure")
        return write(path, data, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(storage, "atomic_write_bytes", fail_index)
        with pytest.raises(storage.StorageError):
            store.commit_attempt(_entry(1))
    assert len(store.committed_records()) == 2
    with pytest.raises(storage.StorageError):
        store.verified_best()
    store.derive_indexes()
    assert store.verified_best()["attempt_id"] == 1


@pytest.mark.parametrize("artifact", ["baseline.py", "run_config.json", "environment.json",
                                     "attempts/0001/source.py", "attempts/0001/evaluation.json",
                                     "work/raw.json", "best.py"])
def test_modified_artifacts_are_rejected(tmp_path, artifact):
    store = _store(tmp_path)
    raw = store.path / "work/raw.json"
    raw.parent.mkdir()
    raw.write_text("{}")
    _baseline(store)
    store.commit_attempt(_entry(1))
    (store.path / artifact).write_bytes(b"modified")
    with pytest.raises(storage.StorageError):
        store.verified_best()


@pytest.mark.parametrize("change", ["source_hash", "output_incorrect", "no_sessions", "parent", "id"])
def test_inconsistent_attempt_never_gets_committed(tmp_path, change):
    store = _store(tmp_path)
    _baseline(store)
    entry = _entry(1)
    if change == "source_hash":
        entry["attempt"]["evaluation"]["candidate_sha256"] = "wrong"
    elif change == "output_incorrect":
        entry["output_correct"] = False
    elif change == "no_sessions":
        entry["attempt"]["evaluation"]["sessions"] = []
    elif change == "parent":
        entry["attempt"]["parent_id"] = 100
    else:
        entry["attempt"]["attempt_id"] = 100
    with pytest.raises(storage.StorageError):
        store.commit_attempt(entry)
    assert not (store.path / "attempts/0001/record.json").exists()
    assert store.verified_best()["attempt_id"] == 0


def test_request_failure_cannot_partially_replace_event(monkeypatch, tmp_path):
    store = _store(tmp_path)
    _baseline(store)
    event = {"request": 1, "input_tokens": 42}
    event = store.record_request(event)
    with pytest.raises(storage.StorageError):
        store.record_request({"request": 1, "input_tokens": 999})
    assert json.loads((store.path / "requests/0001.json").read_bytes()) == event


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside"])
def test_artifact_path_escape_is_refused(tmp_path, relative):
    store = _store(tmp_path)
    with pytest.raises(storage.StorageError, match="inside"):
        store._write(relative, b"data")


def test_symlink_artifact_cannot_be_committed_or_read(tmp_path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("external data")
    (store.path / "work").mkdir()
    (store.path / "work/link").symlink_to(outside)
    with pytest.raises(storage.StorageError, match="symlink"):
        store.commit_attempt(_entry())
    assert not (store.path / "attempts/0000/record.json").exists()
    assert outside.read_text() == "external data"


def test_budget_exhaustion_prevents_commit_and_best_promotion(tmp_path):
    store = _store(tmp_path)
    _baseline(store)
    before = (store.path / "best.py").read_bytes()
    # Leave enough room for a source file but not its complete evidence and commit.
    store.budgets = dataclasses.replace(store.budgets, max_artifact_bytes=store._size() + len(CANDIDATE) + 10)
    with pytest.raises(storage.StorageError, match="budget exceeded"):
        store.commit_attempt(_entry(1))
    assert not (store.path / "attempts/0001/record.json").exists()
    assert (store.path / "best.py").read_bytes() == before
    assert store.verified_best()["attempt_id"] == 0


def test_known_secrets_redacted_and_executable_source_never_rewritten(monkeypatch, tmp_path):
    key = "sk-test-do-not-persist-123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    store = _store(tmp_path)
    _baseline(store)
    saved = store.record_request({"request": 1, "error": f"failure {key}", "api_key": key,
                                  "input_tokens": 12, "output_tokens": 34})
    assert saved["error"] == "failure [REDACTED]"
    assert saved["api_key"] == "[REDACTED]"
    assert saved["input_tokens"] == 12 and saved["output_tokens"] == 34
    with pytest.raises(storage.StorageError, match="credential"):
        store.commit_attempt(_entry(1, source=f"# {key}\n{CANDIDATE}"))
    assert (store.path / "baseline.py").read_bytes() == BASELINE.encode()
    for path in store.path.rglob("*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes()


def test_reporting_never_imports_saved_source(tmp_path):
    store = _store(tmp_path)
    _baseline(store)
    # The storage layer verifies host records and bytes; it must never execute a module while reporting.
    source = "raise RuntimeError('reporting executed source')\n" + CANDIDATE
    store.commit_attempt(_entry(1, source=source))
    assert storage.verified_best(store.path)["source"]["sha256"] == storage.sha256(source.encode())


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_evaluator_freezes_source_when_original_file_changes(tmp_path, newline):
    from contracts import MeasurementProtocol
    from controller import Evaluator
    from runner import TrustedSubprocessRunner
    from workloads.registry import legacy_source_spec
    import numpy as np
    original = tmp_path / "original.py"
    source_bytes = BASELINE.replace("\n", newline).encode()
    original.write_bytes(source_bytes)
    evaluator = Evaluator(legacy_source_spec(original, "kernel", (np.ones(4, dtype=np.float32),)),
                          TrustedSubprocessRunner(), MeasurementProtocol(), tmp_path / "work",
                          selected_backend="cpu")
    original.write_text("user edited source")
    assert evaluator.baseline_source.encode() == source_bytes
    assert evaluator.baseline_sha256 == storage.sha256(source_bytes)
    assert original.read_text() == "user edited source"


def test_controller_storage_failure_cannot_promote_in_memory(monkeypatch, tmp_path):
    from controller import Verdict
    from fakes import proposal
    from test_foundation import _build
    # A real baseline worker exercises its raw outputs and source without provider calls.
    repo = Path(__file__).resolve().parents[1]
    _, controller = _build(repo, [], tmp_path / "run")
    controller.start()
    baseline = controller.best_code
    entry = _entry(1, source=CANDIDATE.replace("kernel", "bad_func"))
    evaluation = Evaluation(**entry["attempt"]["evaluation"])
    monkeypatch.setattr(controller.evaluator, "evaluate", lambda *args: Verdict(ACCEPTED, "fixture", evaluation, True))
    def fail(*args, **kwargs):
        raise storage.StorageError("injected")
    monkeypatch.setattr(controller.store, "commit_attempt", fail)
    with pytest.raises(storage.StorageError, match="injected"):
        controller.submit_proposal(proposal(entry["code"]))
    assert controller.best_code == baseline
    assert controller.best_iteration == 0 and controller.iteration == 0
    assert len(controller.history) == 1
    with pytest.raises(storage.StorageError, match="cannot resume"):
        controller.agent_loop(model="fake")


@pytest.mark.parametrize("payload,omitted", [({"bad_field": "submitted"}, None),
                                           ({"large": "x" * (100 << 10)}, "retention limit"),
                                           ({"bad": float("nan")}, "not strict JSON")])
def test_invalid_submission_is_bounded_and_explicit(tmp_path, payload, omitted):
    from contracts import PROPOSAL_INVALID
    store = _store(tmp_path)
    _baseline(store)
    attempt = Attempt(1, PROPOSAL_INVALID, "invalid proposal", parent_id=0)
    entry = {"iteration": 1, "state": PROPOSAL_INVALID, "accepted": False, "output_correct": False,
             "code": None, "attempt": attempt.to_json()}
    store.commit_attempt(entry, submitted_proposal=payload)
    record = store.committed_records()[1]
    reference = record["evidence"]["submission"]
    saved = json.loads((store.path / reference["path"]).read_bytes())
    if omitted is None:
        assert saved["payload"] == payload and saved["omitted"] is None
    else:
        assert saved["payload"] is None and omitted in saved["omitted"]
    assert store.verified_best()["attempt_id"] == 0


@pytest.mark.parametrize("damage", ["missing", "malformed", "wrong_scope"])
def test_invalid_best_index_is_a_storage_error(tmp_path, damage):
    store = _store(tmp_path)
    _baseline(store)
    path = store.path / "best.json"
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("[]")
    else:
        index = json.loads(path.read_bytes())
        index["scope"]["final_verification"] = "passed"
        path.write_text(json.dumps(index))
    with pytest.raises(storage.StorageError):
        storage.verified_best(store.path)


def test_worker_profile_hash_preserves_crlf_source_bytes(tmp_path):
    import array_io
    import numpy as np
    from runner import TrustedSubprocessRunner
    from worker import REQUEST_SCHEMA
    source = BASELINE.replace("\n", "\r\n").encode()
    (tmp_path / "candidate.py").write_bytes(source)
    array_io.save(tmp_path / "inputs.npz", {"p": array_io.flatten((np.ones(4, np.float32),))})
    request = {"schema_version": REQUEST_SCHEMA, "mode": "profile", "backend": "cpu", "case": "p",
               "function_name": "kernel", "sources": {"candidate": "candidate.py"}, "inputs": "inputs.npz"}
    result = TrustedSubprocessRunner().run(tmp_path, request, trusted=True, timeout_s=120)
    assert result.status == "OK", result.stderr
    trace = result.result["runtime_trace"]
    assert trace["coverage"] == "CPU_EXECUTION_EVENTS", trace
    assert trace["provenance"]["candidate_sha256"] == storage.sha256(source)
    assert (tmp_path / "candidate.py").read_bytes() == source


def test_direct_controller_retains_all_worker_evidence_inside_run(tmp_path):
    import numpy as np
    from contracts import MeasurementProtocol
    from controller import Controller, Evaluator
    from runner import TrustedSubprocessRunner
    from workloads.registry import legacy_source_spec
    original = tmp_path / "original.py"
    original.write_bytes(BASELINE.encode())
    external = tmp_path / "external-work"
    evaluator = Evaluator(legacy_source_spec(original, "kernel", (np.ones(4, np.float32),)),
                          TrustedSubprocessRunner(), MeasurementProtocol(), external, selected_backend="cpu")
    controller = Controller("offline", "test", evaluator, function_name="kernel", output_dir=tmp_path / "run")
    controller.start()
    assert not external.exists()
    assert controller.store.verified_best()["attempt_id"] == 0
    references = controller.store.committed_records()[0]["raw_artifacts"]
    assert references and all((controller.output_dir / reference["path"]).is_file() for reference in references)
    with pytest.raises(storage.StorageError, match="unused evaluator"):
        Controller("offline", "test", evaluator, output_dir=tmp_path / "another-run")
    assert not (tmp_path / "another-run").exists()
