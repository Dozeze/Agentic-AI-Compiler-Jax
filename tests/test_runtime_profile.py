"""AIC-011: observed coverage, overlap-safe summaries, separate capture, and optional failure handling."""
import contextlib
import dataclasses
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import array_io
import benchmark
import feedback
import worker
from contracts import ACCEPTED, MeasurementProtocol, RuntimeTrace
from controller import Controller, Evaluator, _compact
from fixtures.softmax_candidates import REFERENCE
from runner import TrustedSubprocessRunner
from workloads.registry import SOFTMAX


def event(name, start, duration, pid=1, tid=2):
    return {"ph": "X", "name": name, "ts": start, "dur": duration, "pid": pid, "tid": tid}


def annotation():
    return event(feedback.PROFILE_ANNOTATION, 10, 100)


def metadata(name="/device:GPU:0", pid=7):
    return {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}


def test_nested_and_crossing_events_use_interval_unions():
    events = [annotation(), metadata(), event("kernel", 0, 150, pid=7),
              event("kernel", 20, 20, pid=7), event("kernel", 100, 50, pid=7)]
    out = feedback.summarize_trace({"traceEvents": events}, {"backend": "gpu"})
    assert out["coverage"] == "DEVICE_EXECUTION_EVENTS"
    assert out["hotspots"][0]["inclusive_union_us"] == 100
    assert out["hotspots"][0]["count"] == 3
    lane = out["lanes"][0]
    assert lane["inclusive_union_us"] == 100 and lane["overlapping_intervals"] == 2
    assert "inclusive" in lane["duration_semantics"] and "no percentages" in lane["duration_semantics"]


def test_distinct_names_and_lanes_are_never_summed_into_percentages():
    events = [annotation(), metadata(), event("outer", 10, 80, pid=7),
              event("inner", 20, 20, pid=7), event("parallel", 10, 80, pid=7, tid=3)]
    out = feedback.summarize_trace(events, {"backend": "gpu"})
    assert [lane["inclusive_union_us"] for lane in out["lanes"]] == [80, 80]
    assert {h["name"]: h["inclusive_union_us"] for h in out["hotspots"]} == {
        "outer": 80, "inner": 20, "parallel": 80}
    assert all("percent" not in key for item in out["hotspots"] for key in item)


@pytest.mark.parametrize("events, backend, coverage", [
    ([annotation(), event("TfrtCpuExecutable::Execute", 20, 10)], "cpu", "CPU_EXECUTION_EVENTS"),
    ([annotation(), event("TfrtCpuExecutable::Execute", 20, 10)], "gpu", "HOST_ONLY"),
    ([annotation(), event("PjitFunction(f)", 20, 10), metadata()], "gpu", "HOST_ONLY"),
    ([annotation(), metadata("StreamExecutor"), event("GpuExecutable::Execute", 20, 10, pid=7)], "gpu", "HOST_ONLY"),
    ([annotation(), metadata("/device:TPU:0"), event("execute", 20, 10, pid=7)], "tpu", "DEVICE_EXECUTION_EVENTS"),
    ([annotation(), metadata(), event("kernel", 200, 10, pid=7)], "gpu", "HOST_ONLY"),
])
def test_coverage_requires_actual_execution_in_the_annotated_scope(events, backend, coverage):
    out = feedback.summarize_trace(events, {"backend": backend})
    assert out["coverage"] == coverage
    assert bool(out["hotspots"]) is (coverage != "HOST_ONLY")


def test_metadata_tracks_thread_identity_without_promoting_host_dispatch():
    events = [annotation(), {"ph": "M", "name": "thread_name", "pid": 1, "tid": 2,
                             "args": {"name": "host thread"}}, event("PjitFunction(f)", 20, 10)]
    out = feedback.summarize_trace(events, {"backend": "cpu"})
    assert out["lanes"][0]["thread"] == "host thread"
    assert out["lanes"][0]["kind"] == "host_dispatch" and out["hotspots"] == ()


@pytest.mark.parametrize("provenance", [
    {"backend": "cpu"}, {"backend": "tpu"},
    {"backend": "gpu", "devices": [{"id": 1, "platform": "gpu"}]},
])
def test_foreign_device_lanes_do_not_establish_selected_target_coverage(provenance):
    events = [annotation(), metadata(), event("kernel", 20, 10, pid=7)]
    out = feedback.summarize_trace(events, provenance)
    assert out["coverage"] == "HOST_ONLY" and out["hotspots"] == ()
    assert out["lanes"][0]["kind"] == "unattributed_device"


@pytest.mark.parametrize("change", [{"ts": -1}, {"ts": float("nan")}, {"dur": float("inf")},
                                    {"dur": -1}, {"pid": []}, {"tid": True}, {"ts": 10 ** 1000}])
def test_invalid_complete_events_are_counted_not_used(change):
    out = feedback.summarize_trace([annotation(), {**event("bad", 20, 10), **change}], {"backend": "cpu"})
    assert out["invalid_events"] == 1 and out["coverage"] == "HOST_ONLY"
    json.dumps(out, allow_nan=False)


def test_unknown_phases_and_missing_annotations_do_not_invent_intervals():
    out = feedback.summarize_trace([{"ph": "B", "name": "begin"}, {"ph": "E"},
                                    event("TfrtCpuExecutable::Execute", 20, 10)], {"backend": "cpu"})
    assert out["coverage"] == "UNAVAILABLE" and out["ignored_events"] == 2
    assert out["annotation"] is None and out["hotspots"] == ()


def test_summary_and_event_count_are_bounded(monkeypatch):
    events = [annotation(), metadata()]
    events += [event("k" * 500 + str(i), 20, 10, pid=7, tid=i) for i in range(100)]
    out = feedback.summarize_trace(events, {"backend": "gpu"})
    assert len(out["lanes"]) == feedback.MAX_PROFILE_LANES and out["omitted_lanes"] == 84
    assert len(out["hotspots"]) == feedback.TOP_OPS
    assert len(json.dumps(out)) < 16 << 10
    monkeypatch.setattr(feedback, "MAX_TRACE_EVENTS", 1)
    assert feedback.summarize_trace(events, {})["coverage"] == "UNAVAILABLE"


@pytest.mark.parametrize("raw", [b"not gzip", gzip.compress(b"not json"), gzip.compress(b'{"traceEvents": null}'),
                                 b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x07" + b"\x00" * 8])
def test_corrupt_trace_falls_back_with_a_reason(tmp_path, raw):
    path = tmp_path / "trace.gz"
    path.write_bytes(raw)
    out = feedback.read_runtime_trace(path, {"backend": "cpu"})
    assert out["coverage"] == "UNAVAILABLE" and out["reason"]


def test_compressed_and_decoded_byte_limits(tmp_path, monkeypatch):
    path = tmp_path / "trace.gz"
    path.write_bytes(gzip.compress(b" " * 1024))
    monkeypatch.setattr(feedback, "MAX_TRACE_DECODED_BYTES", 100)
    assert "decoded trace" in feedback.read_runtime_trace(path, {})["reason"]
    monkeypatch.setattr(feedback, "MAX_TRACE_BYTES", 1)
    assert "compressed trace" in feedback.read_runtime_trace(path, {})["reason"]


class FakeJax:
    __version__ = "test"

    def __init__(self):
        self.active = False
        self.calls = []
        self.profiler = SimpleNamespace(trace=self.trace, TraceAnnotation=lambda name: contextlib.nullcontext())

    @contextlib.contextmanager
    def trace(self, directory, **options):
        assert options == {"create_perfetto_link": False, "create_perfetto_trace": True}
        self.active = True
        yield
        self.active = False
        path = Path(directory) / "perfetto_trace.json.gz"
        path.write_bytes(gzip.compress(json.dumps([annotation(), event("TfrtCpuExecutable::Execute", 20, 10)]).encode()))

    def default_backend(self):
        return "cpu"

    def devices(self):
        return [SimpleNamespace(id=0, platform="cpu", device_kind="test")]

    def block_until_ready(self, value):
        return value

    def executable(self, *args):
        self.calls.append(self.active)
        return args


def test_capture_warms_before_tracing_and_retains_hashed_raw_artifact(tmp_path):
    jax = FakeJax()
    out = feedback.capture_runtime_trace(jax, jax.executable, (np.ones(2, np.float32),),
                                         artifacts=tmp_path / "profile", provenance={"candidate_sha256": "source"})
    assert jax.calls == [False] * 5 + [True] * 5
    assert out["coverage"] == "CPU_EXECUTION_EVENTS"
    artifact = tmp_path / out["artifact"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == out["artifact_sha256"]
    assert out["provenance"]["candidate_sha256"] == "source"
    assert out["provenance"]["signature"] == ["float32[2]"]


def test_permission_failure_preserves_optional_evidence_fallback(tmp_path):
    jax = FakeJax()

    def denied(*args, **kwargs):
        raise PermissionError("profiler denied")

    jax.profiler.trace = denied
    out = feedback.capture_runtime_trace(jax, jax.executable, (), artifacts=tmp_path, provenance={})
    assert out["coverage"] == "UNAVAILABLE" and "PermissionError" in out["reason"]
    assert out["artifact"] is None and out["hotspots"] == ()


def test_oversize_capture_omits_the_raw_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "MAX_TRACE_BYTES", 1)
    jax = FakeJax()
    out = feedback.capture_runtime_trace(jax, jax.executable, (), artifacts=tmp_path / "profile", provenance={})
    assert out["coverage"] == "UNAVAILABLE" and out["artifact"] is None
    assert not (tmp_path / "profile").exists()


def test_real_cpu_profile_in_a_fresh_worker(tmp_path):
    work = tmp_path / "worker"
    work.mkdir()
    (work / "candidate.py").write_text(REFERENCE)
    array_io.save(work / "inputs.npz", {"p": array_io.flatten((np.ones((32, 128), np.float32),))})
    request = {"schema_version": worker.REQUEST_SCHEMA, "mode": "profile", "backend": "cpu", "case": "p",
               "function_name": "softmax", "sources": {"candidate": "candidate.py"}, "inputs": "inputs.npz"}
    outcome = TrustedSubprocessRunner().run(work, request, trusted=True, timeout_s=120)
    assert outcome.status == "OK", (outcome.detail, outcome.stderr)
    trace = outcome.result["runtime_trace"]
    assert trace["coverage"] == "CPU_EXECUTION_EVENTS", trace
    assert trace["annotation"]["calls"] == 5 and trace["hotspots"]
    assert trace["provenance"]["candidate_sha256"] == hashlib.sha256(REFERENCE.encode()).hexdigest()
    assert (work / trace["artifact"]).is_file()
    json.dumps(trace, allow_nan=False)


def test_measurement_worker_never_starts_a_profiler(tmp_path, monkeypatch):
    import jax
    monkeypatch.setattr(jax.profiler, "trace", lambda *a, **k: pytest.fail("profile during measurement"))
    for role in ("a", "b"):
        (tmp_path / f"{role}.py").write_text(REFERENCE)
    array_io.save(tmp_path / "inputs.npz", {"m": array_io.flatten((np.ones((4, 8), np.float32),))})
    request = {"schema_version": worker.REQUEST_SCHEMA, "mode": "measure", "backend": "cpu", "case": "m",
               "function_name": "softmax", "sources": {"a": "a.py", "b": "b.py"}, "inputs": "inputs.npz",
               "protocol": dataclasses.asdict(MeasurementProtocol(n_blocks=4)), "repeats": 1}
    assert "session" in worker.handle(request, tmp_path)


@pytest.mark.parametrize("condition, shown", [("C2", False), ("C3", False), ("C4", True), ("K", False)])
def test_only_c4_exposes_observed_profile_context(condition, shown):
    trace = RuntimeTrace("HOST_ONLY", "host events only", {"backend": "gpu"}).to_json()
    entry = {"attempt": {"evaluation": {"runtime_trace": trace}}}
    view = _compact(entry, condition)
    assert ("runtime_trace" in view) is shown
    if shown:
        assert view["runtime_trace"]["coverage"] == "HOST_ONLY" and view["runtime_trace"]["hotspots"] == ()


def test_promoted_candidate_is_profiled_after_confirmation(tmp_path, monkeypatch):
    evaluator = Evaluator(SOFTMAX, TrustedSubprocessRunner(), MeasurementProtocol(), tmp_path, selected_backend="cpu")
    report = SimpleNamespace(passed=True, to_json=lambda: {"passed": True})
    monkeypatch.setattr(evaluator, "correctness", lambda source: (report, {}, None))
    from contracts import protocol_hash
    calibration = benchmark.Calibration(benchmark.CALIBRATION_SCHEMA, "synthetic fixture", SOFTMAX.workload_id,
        evaluator.measure_shape, "float32", protocol_hash(evaluator.protocol), "synthetic fixture", 1, .03, (), 1e-9, "fixture")
    monkeypatch.setattr(evaluator, "calibrate", lambda: calibration)
    blocks = tuple(benchmark.Block(i, "AB", 1, 100, 50, 2*i, 2*i+1) for i in range(20))
    from contracts import protocol_hash
    session = benchmark.PairedSession(protocol_hash(evaluator.protocol), 1, blocks)
    calls = []

    def measure(*args, **kwargs):
        calls.append("measure")
        return session, benchmark.analyze_session(session, evaluator.protocol)

    def profile(*args, **kwargs):
        calls.append("profile")
        return feedback.unavailable_trace("test profiler unavailable")

    monkeypatch.setattr(evaluator, "session", measure)
    monkeypatch.setattr(evaluator, "profile", profile)
    result = evaluator.evaluate("candidate", evaluator.baseline_source)
    assert calls == ["measure", "measure", "profile"]
    assert result.state == ACCEPTED and result.evaluation.runtime_trace["coverage"] == "UNAVAILABLE"
    assert result.evaluation.stages["profile"] == "skipped"
    assert result.evaluation.unavailable["runtime_trace"] == "test profiler unavailable"


def test_baseline_profile_is_persisted_with_a_resolvable_artifact(tmp_path):
    output = tmp_path / "run"
    evaluator = Evaluator(SOFTMAX, TrustedSubprocessRunner(), MeasurementProtocol(), output / "work",
                          selected_backend="cpu")
    controller = Controller("test", "test", evaluator, output_dir=output, context_condition="C4")
    controller.start()
    record = json.loads((output / "history.json").read_text())[0]
    trace = record["attempt"]["evaluation"]["runtime_trace"]
    assert trace["coverage"] == "CPU_EXECUTION_EVENTS"
    assert (output / trace["artifact"]).is_file()
    assert controller.provide_hist()[0]["runtime_trace"]["provenance"]["fingerprint"]
    assert controller.best_code == SOFTMAX.reference_source


@pytest.mark.parametrize("forgery", ["source", "hash", "symlink"])
def test_host_checks_profile_source_and_raw_artifact_identity(tmp_path, monkeypatch, forgery):
    work = tmp_path / "work" / "profile-request"
    work.mkdir(parents=True)
    source = "candidate"
    digest = hashlib.sha256(source.encode()).hexdigest()
    artifact = work / "trace.gz"
    artifact.write_bytes(b"raw")
    trace = RuntimeTrace("HOST_ONLY", "host events", {"candidate_sha256": digest}, artifact="trace.gz",
                          artifact_sha256=hashlib.sha256(b"raw").hexdigest()).to_json()
    if forgery == "source":
        trace["provenance"]["candidate_sha256"] = "wrong"
    elif forgery == "hash":
        trace["artifact_sha256"] = "wrong"
    else:
        artifact.unlink()
        artifact.symlink_to(tmp_path / "outside")
    evaluator = Evaluator(SOFTMAX, TrustedSubprocessRunner(), MeasurementProtocol(), work.parent,
                          selected_backend="cpu")
    monkeypatch.setattr(evaluator, "_run", lambda *args, **kwargs: ({"runtime_trace": trace}, work))
    assert evaluator.profile(source, trusted=True)["coverage"] == "UNAVAILABLE"


def test_profile_deadline_is_reported_without_altering_the_candidate_verdict(tmp_path, monkeypatch):
    from controller import EvaluationError
    evaluator = Evaluator(SOFTMAX, TrustedSubprocessRunner(), MeasurementProtocol(), tmp_path, selected_backend="cpu")

    def timeout(*args, **kwargs):
        raise EvaluationError("TIMEOUT", "profile deadline")

    monkeypatch.setattr(evaluator, "_run", timeout)
    trace = evaluator.profile("candidate", trusted=True)
    assert trace["coverage"] == "UNAVAILABLE" and "profile deadline" in trace["reason"]


def test_c4_reports_missing_evidence_explicitly():
    trace = _compact({"attempt": {"evaluation": None}}, "C4")["runtime_trace"]
    assert trace["coverage"] == "UNAVAILABLE" and trace["hotspots"] == ()


@pytest.mark.needs_docker
@pytest.mark.parametrize("selected_backend", ["cpu", pytest.param("cuda", marks=pytest.mark.needs_gpu)])
def test_runtime_profile_inside_the_container(tmp_path, selected_backend):
    from runner import DockerRunner
    work = tmp_path / "work"
    work.mkdir()
    (work / "candidate.py").write_text(REFERENCE)
    array_io.save(work / "inputs.npz", {"p": array_io.flatten((np.ones((32, 128), np.float32),))})
    request = {"schema_version": worker.REQUEST_SCHEMA, "mode": "profile", "backend": selected_backend, "case": "p",
               "function_name": "softmax", "sources": {"candidate": "candidate.py"}, "inputs": "inputs.npz"}
    outcome = DockerRunner(selected_backend=selected_backend).run(work, request, trusted=False, timeout_s=120)
    assert outcome.status == "OK", (outcome.detail, outcome.stderr)
    trace = outcome.result["runtime_trace"]
    expected = "CPU_EXECUTION_EVENTS" if selected_backend == "cpu" else "DEVICE_EXECUTION_EVENTS"
    assert trace["coverage"] == expected, trace
    assert (work / trace["artifact"]).is_file()


@pytest.mark.needs_gpu
def test_cuda_profile_requires_real_device_execution_events(repo_root, tmp_path):
    from backends import gpu_env
    driver = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from environment import select_backend
select_backend("cuda")
import jax, jax.numpy as jnp
import benchmark, feedback
args = (jnp.ones((128, 1024), jnp.float32),)
compiled = benchmark.compile_timed(lambda x: jax.nn.softmax(x), args)
trace = feedback.capture_runtime_trace(jax, compiled.executable, args, artifacts=Path(sys.argv[2]) / "profile",
                                        provenance={"case": "cuda-profile-smoke"})
print(json.dumps(trace))
"""
    proc = subprocess.run([sys.executable, "-c", driver, str(repo_root), str(tmp_path)],
                          env=gpu_env(), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    trace = json.loads(proc.stdout.strip().splitlines()[-1])
    assert trace["coverage"] == "DEVICE_EXECUTION_EVENTS", trace
