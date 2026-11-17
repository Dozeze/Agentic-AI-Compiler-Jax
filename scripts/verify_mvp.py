"""Inspect AIC-016 run evidence without importing stored candidate code.

This checks the recorded execution chain, not the truth of arbitrary externally fabricated files.
Target containment probes and a human review of profile limitations remain separate acceptance gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from contracts import config_hash, resolve_run_config, to_dict  # noqa: E402
from storage import RunStore, StorageError, sha256  # noqa: E402


def inspect_run(path):
    """Verify committed hashes and feedback links; inventory native worker observations."""
    # Construct a read-only handle, as storage.verified_best does. Never create a run directory.
    store = RunStore(Path(path).resolve())
    manifest = json.loads(store._safe("manifest.json").read_bytes())
    store.run_id = manifest["run_id"]
    best = store.verified_best()
    records = store.committed_records()
    config = resolve_run_config({}, store.path / "run_config.json")
    if manifest.get("run_config_sha256") != config_hash(config):
        raise StorageError("MVP evidence requires a resolved CLI RunConfig and matching manifest hash")
    result = json.loads(store._safe("run_result.json").read_bytes())
    if (result.get("schema_version") != "run_result/v1" or result.get("run_id") != store.run_id
            or result.get("best") != best or result.get("candidate_attempts") != len(records) - 1):
        raise StorageError("terminal result does not match committed attempts and verified best")

    feedback = []
    previous = records[0]
    events = store.request_events()
    modes = {event["mode"] for event in events}
    evaluated, measured = [], []
    for record in records[1:]:
        entry, evidence = record["entry"], record["evidence"]
        event = entry["attempt"].get("provider")
        if not event or event.get("status") != "PROPOSAL_VALID" or not event.get("dispatched"):
            raise StorageError("MVP candidate lacks a dispatched, validated provider proposal")
        if event["model"] != config.model.model_id or event["provider"] != config.model.provider:
            raise StorageError("provider identity differs from the frozen configuration")
        context = json.loads(store._safe(evidence["context"]["path"]).read_bytes())
        reservation = json.loads(store._safe(evidence["reservation"]["path"]).read_bytes())
        if reservation["payload"]["messages"][0]["content"].encode() != store._safe(evidence["context"]["path"]).read_bytes():
            raise StorageError("saved context differs from the dispatched request")
        prior = previous["entry"]
        summary = next((item for item in context["history"] if item["iteration"] == prior["iteration"]), None)
        if (not summary or summary["state"] != prior["state"]
                or summary["source_sha256"] != (sha256(prior["code"].encode()) if prior.get("code") else None)
                or summary["provenance"]["evaluation_artifact"] != previous["evidence"].get("evaluation")):
            raise StorageError("next request lacks the actual previous committed outcome")
        feedback.append({"attempt_id": record["attempt_id"], "previous_attempt_id": previous["attempt_id"],
                         "context": evidence["context"]})
        evaluation = entry["attempt"].get("evaluation") or {}
        if evaluation.get("stages", {}).get("correctness") in ("ok", "failed"):
            evaluated.append(record["attempt_id"])
        if evaluation.get("sessions"):
            measured.append(record["attempt_id"])
        previous = record

    workers = []
    raw = {ref["path"]: ref for ref in records[-1]["raw_artifacts"]}
    for reference in raw.values():
        if not reference["path"].endswith("/runner_outcome.json"):
            continue
        outcome = json.loads(store._safe(reference["path"]).read_bytes())
        result_path = Path(reference["path"]).with_name("result.json").as_posix()
        worker_ref = raw.get(result_path)
        worker = json.loads(store._safe(result_path).read_bytes()) if worker_ref else {}
        request_path = Path(reference["path"]).with_name("request.json").as_posix()
        request = json.loads(store._safe(request_path).read_bytes()) if request_path in raw else {}
        source_path = Path(reference["path"]).with_name("candidate.py").as_posix()
        workers.append({"outcome": reference, "status": outcome["status"], "mode": worker.get("mode"),
                        "device": worker.get("device"), "result": worker_ref, "backend": request.get("backend"),
                        "candidate_sha256": raw.get(source_path, {}).get("sha256")})
    profiles = []
    for record in records:
        trace = (record["entry"]["attempt"].get("evaluation") or {}).get("runtime_trace")
        if trace:
            profiles.append({"attempt_id": record["attempt_id"], "coverage": trace["coverage"],
                             "reason": trace.get("reason"),
                             "provenance": trace.get("provenance")})
    return {"schema_version": "mvp_evidence_review/v1", "run_id": store.run_id,
            "manifest_sha256": sha256(store._safe("manifest.json").read_bytes()),
            "provider_modes": sorted(modes), "candidate_attempts": len(records) - 1,
            "evaluated_attempt_ids": evaluated, "measured_attempt_ids": measured,
            "feedback_links": feedback, "workers": workers, "profiles": profiles,
            "runner": manifest["runner"], "backend": config.backend, "workload": config.workload_id,
            "scientific": manifest["scientific"], "budgets": to_dict(config.budgets),
            "model": to_dict(config.model),
            "evaluated_sources": {str(record["attempt_id"]): record["entry"]["attempt"]["evaluation"]["candidate_sha256"]
                                  for record in records[1:] if record["attempt_id"] in evaluated},
            "requests": [{key: event.get(key) for key in ("request", "mode", "status", "response_id",
                         "provider_request_id", "input_tokens", "output_tokens", "usage_uncertain", "unavailable")}
                         for event in events],
            "best": best, "stop_status": result["status"],
            "acceptance": "requires selected-target containment evidence and profile/M3 review"}


def require_target_run(report, *, mode):
    """Reject a CPU, synthetic or incomplete run presented as the selected-target MVP."""
    if report["provider_modes"] != [mode]:
        raise StorageError(f"expected exclusively {mode} provider evidence")
    if report["backend"] != "cuda" or report["workload"] != "softmax":
        raise StorageError("selected MVP target requires CUDA softmax")
    identity = report["runner"]
    if (not report["scientific"] or not identity.get("isolated") or not identity.get("gpus")
            or not identity.get("image_id")):
        raise StorageError("selected MVP target requires an identified GPU container image")
    if not 2 <= report["budgets"]["max_candidates"] <= 3:
        raise StorageError("selected MVP requires a frozen two-to-three candidate cap")
    if mode == "live" and (report["model"].get("offline_responses_path") is not None
            or not report["model"].get("prices") or not report["budgets"].get("money_cap")
            or report["model"]["prices"]["currency"] != report["budgets"].get("currency")):
        raise StorageError("live MVP requires matching recorded prices and a money cap")
    if not 2 <= len(report["evaluated_attempt_ids"]) <= report["candidate_attempts"] <= 3:
        raise StorageError("MVP requires two to three genuinely evaluated proposals")
    workers = [worker for worker in report["workers"] if worker["status"] == "OK" and worker["device"]]
    if not workers or any(w["device"]["platform"] not in ("gpu", "cuda") or w["backend"] != "cuda" for w in workers):
        raise StorageError("worker observations do not establish CUDA execution")
    native_sources = {w["candidate_sha256"] for w in workers if w["mode"] == "outputs"}
    if not set(report["evaluated_sources"].values()) <= native_sources:
        raise StorageError("evaluated proposals lack matching native worker source observations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--target-mode", choices=("offline", "live"),
                        help="also require the selected CUDA/container MVP evidence; never dispatches a request")
    args = parser.parse_args(argv)
    try:
        report = inspect_run(args.run)
        if args.target_mode:
            require_target_run(report, mode=args.target_mode)
    except (StorageError, ValueError, KeyError, TypeError, OSError) as error:
        print(f"MVP evidence incomplete: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
