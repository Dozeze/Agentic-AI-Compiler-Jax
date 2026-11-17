"""Durable, bounded run artifacts. Reporting reads bytes and JSON, never imports source.

An attempts/NNNN/record.json is the commit marker. Everything it references is
written first and fingerprinted. History and best exports are replaceable indexes;
an incomplete attempt directory cannot enter either index. This is not a resume API.
"""
from __future__ import annotations

import hashlib
import dataclasses
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from stat import S_ISREG

from contracts import ACCEPTED, Budgets, TERMINAL_STATES


class StorageError(RuntimeError):
    """Artifacts could not be safely published. Stop before another provider call."""


def _storage_boundary(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise StorageError(f"invalid or unavailable run artifact ({type(error).__name__})") from error
    return checked


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    """Strict JSON: unsupported objects and non-finite floats are errors."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sync_directory(path):
    # Directory fsync is supported on Unix, but some filesystems/platforms reject it.
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def atomic_write_bytes(path, data: bytes, *, exclusive=False):
    """Publish a complete file atomically; exclusive publication refuses collisions.

    A hard link publishes immutable files without the check-then-replace race.
    The temporary file is on the same filesystem and is removed on every exit.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path, value, *, exclusive=False):
    atomic_write_bytes(path, json_bytes(value), exclusive=exclusive)


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|private[_-]?key|(?:^|[_-])token$)", re.I)


def credential_values():
    """Read only credential-named variables for redaction; never persist their values."""
    return tuple(value for key, value in os.environ.items() if _SECRET_KEY.search(key) and len(value) >= 8)


def redact(value, secrets=None):
    secrets = credential_values() if secrets is None else secrets
    if isinstance(value, dict):
        return {redact(key, secrets): "[REDACTED]" if isinstance(key, str) and _SECRET_KEY.search(key) else redact(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
    return value


def check_source_credentials(source):
    """Refuse known credential values rather than changing executable source or its hash."""
    if redact(source) != source:
        raise StorageError("source contains a credential value; refusing to persist or execute it")


class RunStore:
    """One new run, exclusively owned by this object. Construction has no disk effects."""

    def __init__(self, output_dir=None, *, budgets=None):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:12]
        self.path = Path(output_dir) if output_dir is not None else Path("runs") / self.run_id
        self.budgets = budgets or Budgets()
        self.initialized = False

    def _safe(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise StorageError("artifact path must stay inside its run")
        target = self.path / relative
        # Do not follow candidate-created links, including links in intermediate directories.
        for part in (target, *target.parents):
            if part.is_symlink():
                raise StorageError(f"symlink artifact refused: {relative}")
            if part == self.path:
                break
        return target

    def _size(self):
        total = 0
        for path in self.path.rglob("*"):
            self._safe(path.relative_to(self.path))
            if path.is_file():
                total += path.stat().st_size
                if total > self.budgets.max_artifact_bytes:
                    raise StorageError("run artifact byte budget exceeded")
            elif not path.is_dir():
                raise StorageError("non-regular artifact refused")
        return total

    def check_capacity(self):
        """Check prepared artifacts before an external request or worker launch."""
        return self._size()

    def _write(self, relative, data, *, immutable=True):
        path = self._safe(relative)
        old_size = path.stat().st_size if path.is_file() and not immutable else 0
        if self._size() - old_size + len(data) > self.budgets.max_artifact_bytes:
            raise StorageError("run artifact byte budget exceeded")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write_bytes(path, data, exclusive=immutable)
        except OSError as error:
            raise StorageError(f"cannot publish {relative}: {error.strerror}") from error
        return self._reference(relative)

    def _json(self, relative, value, *, immutable=True):
        return self._write(relative, json_bytes(redact(value)), immutable=immutable)

    def _reference(self, relative):
        path = self._safe(relative)
        info = path.stat()
        if not S_ISREG(info.st_mode):
            raise StorageError("artifact reference must name a regular file")
        size = info.st_size
        if size > self.budgets.max_artifact_bytes:
            raise StorageError("artifact byte budget exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return {"path": Path(relative).as_posix(), "sha256": digest.hexdigest(), "bytes": size}

    @_storage_boundary
    def initialize(self, baseline_source, *, config, manifest):
        if self.initialized:
            raise StorageError("run store is already initialized")
        check_source_credentials(baseline_source)
        source = baseline_source.encode("utf-8")
        if len(source) > self.budgets.max_source_bytes:
            raise StorageError("baseline exceeds source byte budget")
        # Serialize before reserving a directory. An existing path is never adopted, even if empty.
        config_data = json_bytes(redact(config))
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise StorageError(f"cannot create new run at {self.path}: {error.strerror}") from error
        baseline = self._write("baseline.py", source)
        configuration = self._write("run_config.json", config_data)
        environment = self._json("environment.json", manifest.get("environment", {
            "schema_version": "environment/v1", "unavailable": {"capture": "not supplied by caller"}}))
        manifest = {**manifest, "schema_version": "manifest/v1", "run_id": self.run_id,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "baseline": baseline,
                    "config": configuration, "environment_artifact": environment,
                    "run_config_sha256": sha256(json.dumps(redact(config), sort_keys=True,
                                                           separators=(",", ":"), allow_nan=False).encode())}
        self._json("manifest.json", manifest)
        self.initialized = True

    @_storage_boundary
    def context_inputs(self):
        """Read verified, frozen environment/evaluation references for context provenance."""
        if not self.initialized:
            raise StorageError("run store is not initialized")
        manifest = json.loads(self._safe("manifest.json").read_bytes())
        for key in ("baseline", "environment_artifact"):
            self._verify_reference(manifest[key])
        return {"baseline": manifest["baseline"], "environment_artifact": manifest["environment_artifact"],
                "runner": manifest.get("runner"),
                "environment": json.loads(self._safe(manifest["environment_artifact"]["path"]).read_bytes()),
                "evaluations": {record["attempt_id"]: record["evidence"].get("evaluation")
                                for record in self.committed_records()}}

    @_storage_boundary
    def reserve_request(self, reservation):
        """Persist intent and worst-case debit before dispatch; an unmatched reservation is uncertain."""
        if not self.initialized:
            raise StorageError("run store is not initialized")
        number = reservation.get("request")
        if type(number) is not int or number != len(self.request_events()) + 1:
            raise StorageError("reservation must follow completed request events")
        reservation = dict(reservation)
        selection = reservation.pop("context_selection", None)
        if selection is not None:
            context_bytes = reservation["payload"]["messages"][0]["content"].encode()
            if sha256(context_bytes) != selection["payload_sha256"]:
                raise StorageError("context selection does not match the dispatched payload")
            reservation["context"] = self._write(f"contexts/{number:04d}.json", context_bytes)
            reservation["context_selection"] = self._json(f"contexts/{number:04d}.selection.json", selection)
        return self._json(f"request_reservations/{number:04d}.json", reservation)

    @_storage_boundary
    def record_provider_stop(self, status, reason, budget):
        if not self.initialized:
            raise StorageError("run store is not initialized")
        return self._json("provider_stop.json", {"schema_version": "provider_stop/v1", "run_id": self.run_id,
                                                 "status": status, "reason": reason, "budget": budget})

    @_storage_boundary
    def finish_run(self, status, reason, budget, *, baseline_failure=False):
        """Immutable terminal result. Failed initialization has no invented best artifact."""
        if not self.initialized:
            raise StorageError("run store is not initialized")
        records = self.committed_records()
        best = self.verified_best() if records else None
        result = {"schema_version": "run_result/v1", "run_id": self.run_id,
                  "output_dir": str(self.path),
                  "status": status, "reason": reason, "budget": budget, "best": best,
                  "candidate_attempts": max(0, len(records) - 1),
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        if baseline_failure:
            result["baseline_failure"] = self._json("baseline_failure.json", {
                "schema_version": "baseline_failure/v1", "reason": reason,
                "raw_artifacts": self._work_references()})
        self._json("run_result.json", result)
        return result

    @_storage_boundary
    def save_proposal(self, proposal, request):
        """Proposal-only artifacts are explicitly unverified and never enter history or best."""
        if not self.initialized:
            raise StorageError("run store is not initialized")
        source = proposal.candidate_source
        check_source_credentials(source)
        source_ref = self._write("proposed.py", source.encode())
        proposal_ref = self._json("proposal.json", dataclasses.asdict(proposal))
        return self._json("proposal_only_result.json", {"schema_version": "proposal_only_result/v1",
                          "run_id": self.run_id, "verified": False, "evaluation": "not performed",
                          "source": source_ref, "proposal": proposal_ref,
                          "request": self._reference(f"requests/{request:04d}.json")})

    @_storage_boundary
    def record_request(self, event):
        if not self.initialized:
            raise StorageError("run store is not initialized")
        events = self.request_events()
        if type(event.get("request")) is not int or event["request"] != len(events) + 1:
            raise StorageError("request IDs must be consecutive host integers")
        event = {**redact(event), "schema_version": "provider_request_event/v1", "run_id": self.run_id}
        if event.get("reservation"):
            self._verify_reference(event["reservation"])
        self._json(f"requests/{event['request']:04d}.json", event)
        self._json("requests.json", [*events, event], immutable=False)
        return event

    def request_events(self):
        events = []
        for path in sorted((self.path / "requests").glob("*.json")):
            path = self._safe(path.relative_to(self.path))
            event = json.loads(path.read_bytes())
            number = len(events) + 1
            if (path.name != f"{number:04d}.json" or event.get("request") != number
                    or event.get("schema_version") != "provider_request_event/v1" or event.get("run_id") != self.run_id):
                raise StorageError("invalid saved request identity")
            events.append(event)
        return events

    def _work_references(self):
        self._size()
        references = []
        for path in sorted((self.path / "work").rglob("*")):
            self._safe(path.relative_to(self.path))
            if path.is_file():
                references.append(self._reference(path.relative_to(self.path)))
        return references

    @_storage_boundary
    def commit_attempt(self, entry, *, submitted_proposal=None):
        if not self.initialized:
            raise StorageError("run store is not initialized")
        records = self.committed_records()
        number = entry.get("iteration")
        if type(number) is not int or number != len(records):
            raise StorageError("attempt IDs must be consecutive host integers")
        source = entry.get("code")
        if source is not None:
            check_source_credentials(source)
            if len(source.encode("utf-8")) > self.budgets.max_source_bytes:
                raise StorageError("candidate exceeds source byte budget")
        self._check_entry(entry, records)
        directory = f"attempts/{number:04d}"
        try:
            self._safe(directory).mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise StorageError(f"attempt directory already exists or is unavailable: {number}") from error
        evidence = {}
        if source is not None:
            evidence["source"] = self._write(f"{directory}/source.py", source.encode("utf-8"))
        attempt = entry["attempt"]
        if submitted_proposal is not None:
            # Malformed/oversize inputs still have an attempt record. Do not retain an unbounded payload.
            try:
                data = json_bytes(redact(submitted_proposal))
                submission = {"schema_version": "proposal_submission/v1", "sha256": sha256(data),
                              "bytes": len(data), "payload": json.loads(data), "omitted": None}
                if len(data) > self.budgets.max_source_bytes + (16 << 10):
                    submission.update(payload=None, omitted="submitted proposal exceeds retention limit")
            except (TypeError, ValueError):
                submission = {"schema_version": "proposal_submission/v1", "payload": None,
                              "omitted": "submitted proposal is not strict JSON"}
            evidence["submission"] = self._json(f"{directory}/submission.json", submission)
        for key in ("proposal", "evaluation"):
            if attempt.get(key) is not None:
                evidence[key] = self._json(f"{directory}/{key}.json", attempt[key])
        evaluation = attempt.get("evaluation") or {}
        if evaluation.get("sessions"):
            evidence["measurements"] = self._json(f"{directory}/measurements.json", evaluation["sessions"])
        evidence["feedback"] = self._json(f"{directory}/feedback.json",
                                         {key: evaluation.get(key) for key in ("ir", "runtime_trace", "unavailable")})
        provider = attempt.get("provider")
        if provider:
            request_number = provider.get("request")
            if type(request_number) is not int or request_number < 1:
                raise StorageError("invalid request reference")
            request_path = f"requests/{request_number:04d}.json"
            if json.loads(self._safe(request_path).read_bytes()) != redact(provider):
                raise StorageError("attempt provider does not match its saved request")
            evidence["request"] = self._reference(request_path)
            if provider.get("reservation"):
                self._verify_reference(provider["reservation"])
                evidence["reservation"] = provider["reservation"]
                reservation = json.loads(self._safe(provider["reservation"]["path"]).read_bytes())
                for key in ("context", "context_selection"):
                    if key in reservation:
                        self._verify_reference(reservation[key])
                        evidence[key] = reservation[key]
        # Raw worker measurements, arrays, IR and traces are retained; nothing needed for a decision is pruned.
        record = {"schema_version": "attempt_commit/v1", "run_id": self.run_id, "attempt_id": number,
                  "entry": redact(entry), "evidence": evidence, "raw_artifacts": self._work_references(),
                  "manifest": self._reference("manifest.json")}
        self._json(f"{directory}/record.json", record)  # commit marker MUST be last
        return self.derive_indexes()

    def _check_entry(self, entry, previous):
        number, attempt = entry["iteration"], entry.get("attempt") or {}
        if (type(number) is not int or type(attempt.get("attempt_id")) is not int
                or attempt.get("schema_version") != "attempt/v1" or attempt.get("attempt_id") != number
                or entry.get("state") not in TERMINAL_STATES or attempt.get("state") != entry["state"]
                or entry.get("accepted") is not (entry["state"] == ACCEPTED)):
            raise StorageError("inconsistent attempt identity or state")
        if number and (type(attempt.get("parent_id")) is not int or not 0 <= attempt["parent_id"] < number):
            raise StorageError("attempt parent must reference an earlier committed attempt")
        source = entry.get("code")
        evaluation = attempt.get("evaluation") or {}
        if evaluation and evaluation.get("candidate_sha256") != sha256((source or "").encode("utf-8")):
            raise StorageError("evaluation source hash does not match saved source")
        proposal = attempt.get("proposal")
        if proposal and proposal.get("candidate_source") != source:
            raise StorageError("proposal source does not match saved source")
        if entry["state"] == ACCEPTED:
            if not source or entry.get("output_correct") is not True:
                raise StorageError("best requires a correct source")
            if number == 0:
                if source.encode("utf-8") != self._safe("baseline.py").read_bytes():
                    raise StorageError("baseline source does not match its frozen snapshot")
                if evaluation.get("stages", {}).get("reference") != "ok":
                    raise StorageError("baseline reference was not verified")
            elif (evaluation.get("correctness", {}).get("passed") is not True
                  or evaluation.get("stages", {}).get("measurement") != "ok" or not evaluation.get("sessions")):
                raise StorageError("accepted candidate is missing correctness or measurement evidence")
        elif number == 0:
            raise StorageError("first committed attempt must be the verified baseline")

    def _verify_reference(self, reference):
        if self._reference(reference["path"]) != reference:
            raise StorageError(f"artifact hash or size mismatch: {reference['path']}")

    @_storage_boundary
    def committed_records(self):
        """Read committed artifacts only, checking identities and every referenced hash."""
        self._size()
        manifest = json.loads(self._safe("manifest.json").read_bytes())
        if manifest.get("run_id") != self.run_id or manifest.get("schema_version") != "manifest/v1":
            raise StorageError("invalid manifest identity")
        for key in ("baseline", "config", "environment_artifact"):
            self._verify_reference(manifest[key])
        records = []
        for path in sorted((self.path / "attempts").glob("*/record.json")):
            self._safe(path.relative_to(self.path))
            record = json.loads(path.read_bytes())
            number = len(records)
            if (record.get("schema_version") != "attempt_commit/v1" or record.get("run_id") != self.run_id
                    or type(record.get("attempt_id")) is not int or record.get("attempt_id") != number
                    or path.parent.name != f"{number:04d}"):
                raise StorageError("invalid committed attempt identity")
            self._check_entry(record["entry"], records)
            if record["entry"]["iteration"] != number:
                raise StorageError("committed entry ID mismatch")
            for reference in [record["manifest"], *record["evidence"].values(), *record["raw_artifacts"]]:
                self._verify_reference(reference)
            source = record["evidence"].get("source")
            if record["entry"].get("code") is not None:
                if not source or source["sha256"] != sha256(record["entry"]["code"].encode("utf-8")):
                    raise StorageError("committed source reference mismatch")
            for key in ("proposal", "evaluation"):
                reference = record["evidence"].get(key)
                expected = record["entry"]["attempt"].get(key)
                if (reference is None) != (expected is None):
                    raise StorageError(f"missing {key} evidence")
                if reference and json.loads(self._safe(reference["path"]).read_bytes()) != expected:
                    raise StorageError(f"committed {key} evidence mismatch")
            records.append(record)
        return records

    def _best_index(self, best):
        number = best["attempt_id"]
        source = best["evidence"]["source"]
        manifest = json.loads(self._safe("manifest.json").read_bytes())
        scope = {"workload": manifest.get("workload"), "function_name": best["entry"].get("function_name"),
                 "correctness_policy_hash": manifest.get("workload", {}).get("correctness_policy_hash"),
                 "verification": "frozen reference" if number == 0 else "development correctness and search confirmation",
                 "final_verification": "not performed by this export", "heldout_verification": "not performed by this export"}
        return {"schema_version": "best/v1", "run_id": self.run_id, "attempt_id": number,
                 "baseline_fallback": number == 0, "source": source,
                 "record": self._reference(f"attempts/{number:04d}/record.json"), "scope": scope}

    @_storage_boundary
    def derive_indexes(self):
        """Rebuild disposable indexes from commits. Never evaluate or import saved Python."""
        records = self.committed_records()
        if not records:
            raise StorageError("no verified baseline is committed")
        entries = [record["entry"] for record in records]
        best = next(record for record in reversed(records) if record["entry"]["state"] == ACCEPTED)
        source = best["evidence"]["source"]
        index = self._best_index(best)
        self._json("history.json", entries, immutable=False)
        for entry in entries:
            self._json(f"iteration_{entry['iteration']:04d}.json", entry, immutable=False)
        self._write("best.py", self._safe(source["path"]).read_bytes(), immutable=False)
        self._json("best.json", index, immutable=False)  # pointer published after source
        return entries

    @_storage_boundary
    def verified_best(self):
        """Validate the best pointer against commits and exact export bytes, without executing code."""
        records = self.committed_records()
        index = json.loads(self._safe("best.json").read_bytes())
        accepted = [r for r in records if r["entry"]["state"] == ACCEPTED]
        best = accepted[-1] if accepted else None
        if not best or index != self._best_index(best):
            raise StorageError("best index does not resolve to the latest committed verified source")
        self._verify_reference(index["record"])
        if sha256(self._safe("best.py").read_bytes()) != index["source"]["sha256"]:
            raise StorageError("best export hash does not match its committed source")
        return index

    @_storage_boundary
    def commit_final_result(self, report):
        """Publish a separate final evaluation; never alter the search run or its best pointer."""
        manifest = json.loads(self._safe("manifest.json").read_bytes())
        if not self.initialized or manifest.get("mode") != "final_evaluation":
            raise StorageError("final publication requires a separate final-evaluation store")
        report = {**report, "run_id": self.run_id, "finished_utc": datetime.now(timezone.utc).isoformat()}
        report_ref = self._json("final_report.json", report)
        references = [self._reference(name) for name in (
            "manifest.json", "run_config.json", "environment.json", "baseline.py", "winner.py",
            "selection.json", "split.json", "parent_manifest.json", "parent_best.json", "parent_result.json")]
        references.extend(self._work_references())
        references.append(report_ref)
        # The release is the commit marker; partial directories without it are never final evidence.
        release = {**report, "schema_version": "final_release/v1",
                   "artifacts": references, "report_artifact": report_ref}
        self._json("release.json", release)
        return json.loads(self._safe("release.json").read_bytes())


@_storage_boundary
def verified_best(output_dir):
    """Inspect an existing run without reopening it for execution or appending attempts."""
    store = RunStore(output_dir)
    store._size()
    manifest = json.loads(store._safe("manifest.json").read_bytes())
    store.run_id = manifest["run_id"]
    return store.verified_best()


@_storage_boundary
def verified_final_result(output_dir):
    """Replay final evidence by hashes/JSON only; saved source is never imported."""
    store = RunStore(output_dir)
    store._size()
    report = json.loads(store._safe("release.json").read_bytes())
    manifest = json.loads(store._safe("manifest.json").read_bytes())
    if (report.get("schema_version") != "final_release/v1" or report.get("run_id") != manifest.get("run_id")
            or manifest.get("mode") != "final_evaluation"):
        raise StorageError("invalid final release identity")
    for reference in report["artifacts"]:
        store._verify_reference(reference)
    store._verify_reference(report["report_artifact"])
    payload = json.loads(store._safe(report["report_artifact"]["path"]).read_bytes())
    if (set(report) != set(payload) | {"schema_version", "artifacts", "report_artifact"}
            or any(report.get(key) != value for key, value in payload.items())):
        raise StorageError("final release differs from its committed report")
    selection = json.loads(store._safe("selection.json").read_bytes())
    if (report["selection"] != selection or sha256(store._safe("winner.py").read_bytes()) != selection["source_sha256"]
            or sha256(store._safe("baseline.py").read_bytes()) != selection["baseline_sha256"]):
        raise StorageError("final selection/source mismatch")
    split = json.loads(store._safe("split.json").read_bytes())
    digest = sha256(json.dumps(split, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if digest != report["commitment"]["sha256"]:
        raise StorageError("final split differs from its pre-search commitment")
    parent = json.loads(store._safe("parent_manifest.json").read_bytes())
    best = json.loads(store._safe("parent_best.json").read_bytes())
    if (parent.get("evaluation_split") != report["commitment"] or best["attempt_id"] != selection["attempt_id"]
            or best["source"]["sha256"] != selection["source_sha256"]):
        raise StorageError("final result differs from its frozen parent selection")
    from contracts import from_dict
    from benchmark import PairedSession, analyze_session
    from controller import decide
    cfg = from_dict(json.loads(store._safe("run_config.json").read_bytes()))
    if [item["shape"] for item in report["shapes"]] != split["shapes"]:
        raise StorageError("final result is missing declared shapes")
    native_sessions = set()
    calibrations = []
    for ref in report["artifacts"]:
        if ref["path"].startswith("work/") and ref["path"].endswith("/result.json"):
            worker = json.loads(store._safe(ref["path"]).read_bytes())
            if worker.get("mode") == "measure" and worker.get("status") == "ok":
                native_sessions.add(sha256(json_bytes(worker["session"])))
        if ref["path"].startswith("work/") and ref["path"].endswith("/calibration.json"):
            calibrations.append(json.loads(store._safe(ref["path"]).read_bytes()))
    for item in report["shapes"]:
        correctness = item["correctness"]
        if "cases" in correctness:
            names = [f"heldout{item['shape']}/seed{seed}" for seed in split["seeds"]]
            if ([case["case"] for case in correctness["cases"]] != names
                    or correctness["n_cases"] != len(names)
                    or correctness["passed"] != all(c["passed"] for c in correctness["cases"])):
                raise StorageError("final correctness report does not cover the declared split")
        timing = item["timing"]
        if timing["status"] == "MEASURED":
            calibration = timing["calibration"]
            if calibration not in calibrations or len(timing["sessions"]) != cfg.measurement.final_sessions:
                raise StorageError("final timing is missing frozen calibration or fresh sessions")
            for session in timing["sessions"]:
                analysis = analyze_session(PairedSession.from_json(session["raw"]), cfg.measurement)
                decision = dataclasses.asdict(decide(analysis, calibration["delta"]))
                if (sha256(json_bytes(session["raw"])) not in native_sessions
                        or json_bytes(analysis.to_json()) != json_bytes(session["analysis"])
                        or decision != session["decision"]
                        or session["baseline_sha256"] != selection["baseline_sha256"]
                        or session["candidate_sha256"] != selection["source_sha256"]):
                    raise StorageError("final timing differs from native measurements and host analysis")
            if timing["confirmed_improvement"] != all(s["decision"]["state"] == ACCEPTED for s in timing["sessions"]):
                raise StorageError("final confirmation differs from host decisions")
    correct = report["status"] != "INTERRUPTED" and all(i["correctness"]["passed"] for i in report["shapes"])
    faster = (correct and bool(manifest.get("scientific")) and cfg.backend == "cuda" and not best["baseline_fallback"]
              and all(i["timing"].get("confirmed_improvement") for i in report["shapes"]))
    if report["correctness_passed"] != correct or report["performance_verified"] != bool(faster):
        raise StorageError("final verification flags disagree with per-shape evidence")
    return report
