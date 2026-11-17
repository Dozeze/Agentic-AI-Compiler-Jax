"""Trusted, provider-free evaluation of a completed run's frozen best on its final split."""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import benchmark
import heldout
from contracts import from_dict, to_dict, protocol_hash
from controller import BaselineFailed, EvaluationError, Evaluator, decide
from source_policy import check_source
from storage import RunStore, StorageError, atomic_write_json, sha256, verified_final_result
from workloads import registry


def _load_selection(parent):
    store = RunStore(parent)
    store._size()
    manifest = json.loads(store._safe("manifest.json").read_bytes())
    store.run_id = manifest["run_id"]
    best = store.verified_best()
    result = json.loads(store._safe("run_result.json").read_bytes())
    if (result.get("schema_version") != "run_result/v1" or result.get("run_id") != store.run_id
            or result.get("best") != best or result.get("candidate_attempts") != len(store.committed_records()) - 1):
        raise heldout.HeldoutError("final evaluation requires a completed run with a verified terminal best")
    cfg = from_dict(json.loads(store._safe("run_config.json").read_bytes()))
    if cfg.source_path or cfg.proposal_only or cfg.split != "development" or cfg.backend is None:
        raise heldout.HeldoutError("final evaluation requires an explicitly configured registered development run")
    spec = registry.get(cfg.workload_id)
    if not spec.scientific or manifest.get("evaluation_split") != heldout.commitment(spec):
        raise heldout.HeldoutError("missing or changed pre-search split commitment; start a newly frozen run")
    baseline = store._safe("baseline.py").read_bytes()
    if sha256(baseline) != spec.reference_sha256:
        raise heldout.HeldoutError("frozen reference differs from current workload")
    lock = manifest.get("environment", {}).get("repo", {}).get("uv_lock_sha256")
    if lock is not None and lock != sha256((registry.HERE.parent / "uv.lock").read_bytes()):
        raise heldout.HeldoutError("dependency lock differs from the search population")
    source = store._safe(best["source"]["path"]).read_bytes()
    policy = check_source(source.decode("utf-8"), spec.function_name, max_source_bytes=cfg.budgets.max_source_bytes)
    if not policy.ok:
        raise heldout.HeldoutError(f"selected source violates evaluation policy: {policy.summary()}")
    selection = {"parent_run_id": store.run_id, "parent_run_path": str(store.path.resolve()),
                 "parent_manifest_sha256": sha256(store._safe("manifest.json").read_bytes()),
                 "parent_best_sha256": sha256(store._safe("best.json").read_bytes()),
                 "attempt_id": best["attempt_id"], "baseline_fallback": best["baseline_fallback"],
                 "source_sha256": sha256(source), "baseline_sha256": sha256(baseline)}
    return store, manifest, result, best, cfg, spec, baseline, source, selection


def _session_record(evaluator, candidate, calibration, protocol):
    session, _ = evaluator.session(evaluator.baseline_source, candidate,
                                   trusted=candidate == evaluator.baseline_source)
    if (type(session.repeats) is not int or session.repeats < 1 or session.repeats != calibration.repeats
            or len(session.blocks) != protocol.n_blocks
            or type(calibration.delta) not in (int, float) or not math.isfinite(calibration.delta) or calibration.delta < 0
            or any(block.index != i or block.order not in ("AB", "BA") or block.repeats != session.repeats
                   or any(type(value) is not int or value <= 0 for value in (block.a_ns, block.b_ns))
                   for i, block in enumerate(session.blocks))):
        raise heldout.HeldoutError("invalid raw final measurement session")
    analysis = benchmark.analyze_session(session, protocol)
    decision = decide(analysis, calibration.delta)
    return {"raw": session.to_json(), "analysis": analysis.to_json(),
            "decision": dataclasses.asdict(decision), "baseline_sha256": evaluator.baseline_sha256,
            "candidate_sha256": sha256(candidate.encode())}


def finalize_run(parent_run, *, output_dir=None, correctness_only=False, runner=None):
    """Reserve once, evaluate in workers, then publish a separate immutable release.

    Injecting a reviewed-fixture runner is a Python-test capability only. CLI candidates require Docker;
    a verified reference fallback may use the trusted subprocess runner. No provider is constructed.
    """
    from runner import DockerRunner, TrustedSubprocessRunner
    parent, manifest, terminal, best, cfg, spec, baseline, source, selection = _load_selection(parent_run)
    heldout.check_search_allowed(spec)
    runner = runner or (DockerRunner(selected_backend=cfg.backend) if cfg.runner == "docker" else TrustedSubprocessRunner())
    if source != baseline and not (runner.isolated or getattr(runner, "reviewed_fixtures", False)):
        raise heldout.HeldoutError("final candidate requires verified isolation")
    if reason := runner.preflight():
        raise heldout.HeldoutError(f"ISOLATION_UNAVAILABLE: {reason}")
    identity = runner.identity()
    if identity != manifest.get("runner"):
        raise heldout.HeldoutError("final runner/image differs from the search population")
    # Worker observations must also match the frozen host's device/JAX population when available.
    expected_devices = manifest.get("environment", {}).get("devices", [])
    started = time.monotonic()
    store = RunStore(output_dir, budgets=cfg.budgets)
    final_cfg = dataclasses.replace(cfg, output_dir=str(store.path), split="heldout",
                                    budgets=dataclasses.replace(cfg.budgets, max_candidates=0))
    store.initialize(baseline.decode(), config=to_dict(final_cfg), manifest={
        "mode": "final_evaluation", "scientific": manifest.get("scientific", False),
        "workload": manifest["workload"], "runner": identity, "environment": manifest["environment"],
        "evaluation_split": manifest["evaluation_split"], "parent": selection})
    store._write("winner.py", source)
    store._json("selection.json", selection)
    for name, value in (("parent_manifest.json", manifest), ("parent_best.json", best), ("parent_result.json", terminal)):
        store._json(name, value)
    # Reservation is durable before generating final inputs or launching any worker. It is deliberately
    # not cleared on failure/interruption; automatic retry could turn exposed feedback into search data.
    reservation = heldout.reserve(spec, run_id=parent.run_id, candidate_sha256=selection["source_sha256"],
                                  output_dir=store.path.resolve())
    store._json("split.json", heldout.split_contract(spec))
    shape_results = [{"shape": list(shape), "correctness": None,
                      "timing": {"status": "NOT_REQUESTED" if correctness_only else "NOT_RUN", "sessions": []}}
                     for shape in spec.heldout_shapes]
    status, reason, report, fingerprint = "COMPLETE", None, None, None
    deadline = started + cfg.budgets.run_wall_s
    evaluation_spec = dataclasses.replace(spec, reference_path=store.path / "baseline.py")

    def evaluator_for(directory, cases):
        evaluator = Evaluator(evaluation_spec, runner, cfg.measurement, store.path / "work" / directory,
                              selected_backend=cfg.backend, budgets=cfg.budgets, cases=cases)
        evaluator.store, evaluator.deadline = store, deadline
        return evaluator

    try:
        split = heldout.cases(spec)
        evaluator = evaluator_for("correctness", split)
        evaluator.reference_outputs()
        fingerprint = evaluator.fingerprint
        report, _, _ = evaluator.correctness(source.decode(), trusted=source == baseline)
        if expected_devices:
            worker_device = json.loads(next((store.path / "work/correctness").glob("*-reference/result.json")).read_bytes())["device"]
            if not any(d["platform"] == worker_device["platform"] and d["device_kind"] == worker_device["device_kind"]
                       for d in expected_devices):
                raise heldout.HeldoutError("final worker differs from the search device")
            if worker_device["jax"] != manifest["environment"]["versions"]["jax"]:
                raise heldout.HeldoutError("final JAX runtime differs from the search population")
        for entry, shape in zip(shape_results, spec.heldout_shapes):
            prefix = f"heldout{list(shape)}/seed"
            cases = tuple(case for case in report.cases if case.case.startswith(prefix))
            if len(cases) != len(spec.heldout_seeds):
                raise heldout.HeldoutError("final report is missing declared cases")
            failure = next((case for case in cases if not case.passed), None)
            entry["correctness"] = benchmark.CorrectnessReport(not failure, failure.category if failure else None,
                                                               report.policy_hash, cases).to_json()
        if not report.passed:
            status, reason = "CORRECTNESS_FAILED", report.category
            for entry in shape_results:
                entry["timing"]["status"] = "CORRECTNESS_BLOCKED"
        elif not correctness_only:
            for index, (entry, shape) in enumerate(zip(shape_results, spec.heldout_shapes)):
                timing = entry["timing"]
                name = f"heldout{list(shape)}/seed{spec.heldout_seeds[0]}"
                population = evaluator_for(f"shape-{index:02d}", [(name, spec.make_inputs(spec.heldout_seeds[0], shape))])
                population.measure_shape, population.measure_case = tuple(shape), name
                try:
                    calibration = population.calibrate()
                    if population.fingerprint != fingerprint:
                        raise heldout.HeldoutError("measurement population changed during final evaluation")
                    timing["calibration"] = calibration.to_json()
                    timing["protocol_hash"] = protocol_hash(cfg.measurement)
                    for _ in range(cfg.measurement.final_sessions):
                        timing["sessions"].append(_session_record(population, source.decode(), calibration, cfg.measurement))
                    timing["confirmed_improvement"] = all(s["decision"]["state"] == "ACCEPTED" for s in timing["sessions"])
                    timing["status"] = "MEASURED"
                except (benchmark.CalibrationError, EvaluationError) as error:
                    timing.update(status="UNAVAILABLE", reason=str(error))
                    status = "TIMING_INCOMPLETE"
    except (BaselineFailed, EvaluationError, heldout.HeldoutError, KeyboardInterrupt) as error:
        status, reason = "INTERRUPTED", f"{type(error).__name__}: {error}"
        for entry in shape_results:
            if entry["correctness"] is None:
                entry["correctness"] = {"passed": False, "unavailable": reason}
            if entry["timing"]["status"] == "NOT_RUN":
                entry["timing"].update(status="UNAVAILABLE", reason=reason)
    correctness_passed = report is not None and report.passed and status != "INTERRUPTED"
    performance_verified = (correctness_passed and bool(manifest.get("scientific")) and cfg.backend == "cuda"
        and not best["baseline_fallback"] and all(e["timing"].get("confirmed_improvement") for e in shape_results))
    release = store.commit_final_result({
        "selection": selection, "commitment": manifest["evaluation_split"], "status": status, "reason": reason,
        "correctness_only": correctness_only, "correctness_passed": correctness_passed,
        "performance_verified": bool(performance_verified), "shapes": shape_results,
        "fingerprint": fingerprint, "backend": cfg.backend, "runner": identity, "provider_requests": 0,
        "scope": "finite declared final cases; fresh baseline comparison per shape; no universal equivalence",
        "elapsed_s": time.monotonic() - started})
    atomic_write_json(reservation / "release.json", {"output_dir": str(store.path.resolve()),
        "release_sha256": sha256((store.path / "release.json").read_bytes()),
        "correctness_passed": correctness_passed, "performance_verified": bool(performance_verified)}, exclusive=True)
    verified_final_result(store.path)
    return store.path, release
