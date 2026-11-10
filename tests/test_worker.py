"""Evaluation worker (AIC-008), called in-process here: its request/response contract and modes.

The process boundary itself is tested in tests/test_runner.py.
"""
import dataclasses

import numpy as np
import pytest

import array_io
import worker
from benchmark import PairedSession
from contracts import MeasurementProtocol
from fixtures.softmax_candidates import COMPILE_ERROR, REFERENCE, WRONG_SHAPE

SMALL = MeasurementProtocol(warmup_calls=2, n_blocks=4, bootstrap_samples=200)


def _workdir(tmp_path, sources, cases):
    for name, text in sources.items():
        (tmp_path / name).write_text(text)
    array_io.save(tmp_path / "inputs.npz", {n: array_io.flatten(args) for n, args in cases.items()})
    return tmp_path


def _request(mode, sources, **extra):
    return {"schema_version": worker.REQUEST_SCHEMA, "mode": mode, "backend": "cpu",
            "function_name": "softmax", "sources": sources, "inputs": "inputs.npz", **extra}


CASES = {"a": (np.ones((4, 8), np.float32),), "b": (np.zeros((4, 8), np.float32),),
         "c": (np.ones((2, 3), np.float32),)}


def test_outputs_mode_runs_every_case(tmp_path):
    work = _workdir(tmp_path, {"cand.py": REFERENCE}, CASES)
    result = worker.handle(_request("outputs", {"candidate": "cand.py"}), work)
    assert result["status"] == "ok" and result["device"]["platform"] == "cpu"
    assert [c["status"] for c in result["cases"]] == ["ok", "ok", "ok"]
    assert len(result["compile"]) == 2                     # two input signatures, compiled once each
    outputs = array_io.load(work / "outputs.npz", base=work, max_bytes=1 << 20)
    assert np.allclose(outputs["a"].arrays[0], 1 / 8)


def test_outputs_mode_reports_compile_failures_once_per_signature(tmp_path):
    work = _workdir(tmp_path, {"cand.py": COMPILE_ERROR}, CASES)
    result = worker.handle(_request("outputs", {"candidate": "cand.py"}), work)
    statuses = [c["status"] for c in result["cases"]]
    assert statuses == ["COMPILE_FAILED"] * 3
    assert "not retried" in result["cases"][1]["detail"]


def test_outputs_mode_module_that_fails_to_load(tmp_path):
    work = _workdir(tmp_path, {"cand.py": "def softmax(x):\n    return x\nraise RuntimeError('at import')\n"},
                    CASES)
    result = worker.handle(_request("outputs", {"candidate": "cand.py"}), work)
    assert {c["status"] for c in result["cases"]} == {"COMPILE_FAILED"}
    assert "failed to load" in result["cases"][0]["detail"]


def test_outputs_keep_the_candidate_structure(tmp_path):
    work = _workdir(tmp_path, {"cand.py": WRONG_SHAPE}, {"a": CASES["a"]})
    worker.handle(_request("outputs", {"candidate": "cand.py"}), work)
    outputs = array_io.load(work / "outputs.npz", base=work, max_bytes=1 << 20)
    assert outputs["a"].arrays[0].shape == (4, 7)


def test_measure_mode_returns_a_raw_session(tmp_path):
    work = _workdir(tmp_path, {"a.py": REFERENCE, "b.py": REFERENCE}, {"m": CASES["a"]})
    result = worker.handle(_request("measure", {"a": "a.py", "b": "b.py"}, case="m",
                                    protocol=dataclasses.asdict(SMALL), repeats=3), work)
    session = PairedSession.from_json(result["session"])
    assert session.repeats == 3 and len(session.blocks) == SMALL.n_blocks
    assert set(result["compile"]) == {"a", "b"}


def test_main_turns_any_failure_into_a_result(tmp_path):
    (tmp_path / "request.json").write_text('{"schema_version": "nope"}')
    assert worker.main(["worker.py", str(tmp_path)]) == 0
    import json
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "error" and "worker_request/v1" in result["error"]


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        worker.handle({"schema_version": worker.REQUEST_SCHEMA, "mode": "shell"}, tmp_path)


def test_outputs_mode_returns_compiler_feedback_for_the_measurement_case(tmp_path):
    work = _workdir(tmp_path, {"cand.py": REFERENCE}, CASES)
    result = worker.handle(_request("outputs", {"candidate": "cand.py"}, analyze_case="a"), work)
    ir = result["ir"]
    assert set(ir["levels"]) == {"jaxpr", "stablehlo", "hlo", "compiled"}
    assert ir["provenance"]["signature"] == ["float32[4, 8]"]
    assert (work / "ir" / "compiled.txt").is_file()


def test_feedback_is_unavailable_when_the_case_did_not_compile(tmp_path):
    work = _workdir(tmp_path, {"cand.py": COMPILE_ERROR}, CASES)
    result = worker.handle(_request("outputs", {"candidate": "cand.py"}, analyze_case="a"), work)
    assert "did not compile" in result["ir"]["unavailable"]["all"]
