"""Containment of model-written code (AIC-009), checked against a real container.

Every probe below is deliberately hostile and deliberately bypasses the source policy: the policy is the
first layer, and these tests prove the second one holds on its own. Each probe runs as a candidate in
the `outputs` mode and records what it could observe into /work/probe.json, the only place it may write.
Run on a machine with Docker; the NV host runs the GPU variant with --require-docker --require-gpu.
"""
import json
import subprocess
import textwrap

import numpy as np
import pytest

import array_io
import runner
import worker

pytestmark = pytest.mark.needs_docker

PROBE = textwrap.dedent('''
    import json, os, socket, subprocess

    def softmax(x):
        seen = {}
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=3)
            seen["network"] = "REACHED"
        except OSError as e:
            seen["network"] = type(e).__name__
        try:
            socket.getaddrinfo("example.com", 80)
            seen["dns"] = "RESOLVED"
        except OSError as e:
            seen["dns"] = type(e).__name__
        seen["uid"] = os.getuid()
        status = open("/proc/self/status").read()
        seen["cap_eff"] = [l.split()[1] for l in status.splitlines() if l.startswith("CapEff")][0]
        seen["no_new_privs"] = [l.split()[1] for l in status.splitlines() if l.startswith("NoNewPrivs")][0]
        seen["env_keys"] = sorted(os.environ)
        seen["home_entries"] = os.listdir("/home") if os.path.isdir("/home") else []
        seen["visible"] = {p: os.path.exists(p) for p in ("/Users", "/root/.ssh", "/app/.env.local",
                                                          "/app/controller.py", "/var/run/docker.sock")}
        for path in ("/app/pwned", "/etc/pwned", "/work/../pwned"):
            try:
                open(path, "w").write("x")
                seen["write:" + path] = "WROTE"
            except OSError as e:
                seen["write:" + path] = type(e).__name__
        try:
            open("/tmp/scratch", "w").write("x" * 1000)
            seen["write:/tmp"] = "ok"
        except OSError as e:
            seen["write:/tmp"] = type(e).__name__
        children = []
        try:
            for _ in range(400):
                children.append(subprocess.Popen(["sleep", "30"]))
        except OSError as e:
            seen["fork_error"] = type(e).__name__
        seen["children_started"] = len(children)
        for child in children:                     # give the slots back so XLA can still start threads
            child.kill()
            child.wait()
        open("/work/probe.json", "w").write(json.dumps(seen))
        return x
''')


def _run(tmp_path, source, timeout_s=120, memory="2g"):
    work = tmp_path / "work"
    work.mkdir()
    (work / "cand.py").write_text(source)
    array_io.save(work / "inputs.npz", {"a": array_io.flatten((np.ones((4, 8), np.float32),))})
    request = {"schema_version": worker.REQUEST_SCHEMA, "mode": "outputs", "backend": "cpu",
               "function_name": "softmax", "sources": {"candidate": "cand.py"}, "inputs": "inputs.npz"}
    docker = runner.DockerRunner(backend="cpu", memory=memory)
    return docker.run(work, request, trusted=False, timeout_s=timeout_s), work


def _no_container_left():
    out = subprocess.run(["docker", "ps", "--all", "--quiet", "--filter", "label=aic.worker=1"],
                         capture_output=True, text=True).stdout
    return out.strip() == ""


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    outcome, work = _run(tmp_path_factory.mktemp("probe"), PROBE)
    assert outcome.status == runner.OK, (outcome.detail, outcome.stderr[-2000:])
    return outcome, json.loads((work / "probe.json").read_text())


def test_candidate_still_produces_its_output(probe):
    outcome, _ = probe
    assert outcome.result["cases"] == [{"name": "a", "status": "ok"}]


def test_no_network(probe):
    _, seen = probe
    assert seen["network"] != "REACHED" and seen["dns"] != "RESOLVED"


def test_unprivileged(probe):
    _, seen = probe
    assert seen["uid"] == 65534
    assert int(seen["cap_eff"], 16) == 0
    assert seen["no_new_privs"] == "1"


def test_no_credentials_or_host_files(probe):
    _, seen = probe
    # GPG_KEY is set by the python base image: the public fingerprint of the CPython release key.
    suspicious = [k for k in seen["env_keys"] if k != "GPG_KEY"
                  and any(s in k for s in ("KEY", "TOKEN", "SECRET", "GOOGLE", "ANTHROPIC", "AWS", "CREDENTIAL"))]
    assert suspicious == [], seen["env_keys"]
    assert not any(seen["visible"].values()), seen["visible"]
    assert seen["home_entries"] == []                  # the image's own empty /home, not the host's


def test_only_scratch_and_workdir_are_writable(probe):
    _, seen = probe
    assert seen["write:/app/pwned"] != "WROTE" and seen["write:/etc/pwned"] != "WROTE"
    assert seen["write:/work/../pwned"] != "WROTE"
    assert seen["write:/tmp"] == "ok"


def test_process_count_is_capped(probe):
    _, seen = probe
    assert seen["children_started"] < 128 and seen["fork_error"]


def test_memory_exhaustion_is_contained(tmp_path):
    hog = "def softmax(x):\n    blocks = [bytearray(64 << 20) for _ in range(64)]\n    return x\n"
    outcome, _ = _run(tmp_path, hog, memory="512m")
    assert outcome.status == runner.RESOURCE_LIMIT or (
        outcome.status == runner.OK and "MemoryError" in json.dumps(outcome.result)), outcome
    assert _no_container_left()


def test_deadline_kills_the_container(tmp_path):
    outcome, _ = _run(tmp_path, "def softmax(x):\n    while True:\n        pass\n", timeout_s=15)
    assert outcome.status == runner.TIMEOUT
    assert _no_container_left()


def test_no_container_survives_its_request(probe):
    assert _no_container_left()


def test_preflight_reports_a_missing_image():
    reason = runner.DockerRunner(backend="cpu", image="aic-worker:does-not-exist").preflight()
    assert reason and "not built" in reason
