"""Held-out evaluation (AIC-018, AIC-1.1 §4): frozen before search, unseen by the search, revealed at the end.

The repository is public, so the held-out seeds cannot simply live in it. They are derived from a secret
salt (commit-reveal):

  init     a random salt is written to heldout/<workload>-v<version>.salt (git-ignored); its sha256 goes into
           the workload's `heldout_commitment` in workloads/registry.py and is committed - the freeze evidence.
  verify   the harness refuses held-out work unless sha256(salt) equals the committed commitment.
  reveal   with the final results, the salt is published so anyone can re-derive the seeds and re-run.

The seeds (at least 20, disjoint from the search seeds and the edge-case seed) never enter a prompt, a
candidate's source or the search's artifacts. Held-out results never go back into the agent's context. A
failed held-out evaluation burns its split version in the ledger: the scope's generalisation claim is
invalid, and a later attempt on the same split is not called held out - debugging needs a new version.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
N_SEEDS = 20
EDGE_SEED = 12345
LEDGER = REPO / "evidence" / "heldout" / "ledger.jsonl"


class HeldoutError(RuntimeError):
    """No usable held-out split: missing or wrong salt, no commitment, or a burned split version."""


def salt_dir() -> Path:
    return Path(os.environ.get("AIC_HELDOUT_DIR", REPO / "heldout"))


def salt_path(spec) -> Path:
    return salt_dir() / f"{spec.workload_id}-v{spec.heldout_version}.salt"


def commitment(salt: str) -> str:
    return hashlib.sha256(salt.encode("ascii")).hexdigest()


def init(spec) -> str:
    """Create the secret salt for `spec` (never overwrites one) and return its commitment."""
    path = salt_path(spec)
    if path.exists():
        raise HeldoutError(f"{path} already exists; a split is frozen once")
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    path.write_text(salt + "\n", encoding="ascii")
    os.chmod(path, 0o600)
    return commitment(salt)


def load_salt(spec) -> str:
    """The salt, checked against the committed commitment."""
    if not spec.heldout_commitment:
        raise HeldoutError(f"{spec.workload_id} has no committed held-out commitment (run scripts/heldout.py init)")
    path = salt_path(spec)
    if not path.is_file():
        raise HeldoutError(f"{path} is missing: get the salt from whoever froze the split")
    salt = path.read_text(encoding="ascii").strip()
    if commitment(salt) != spec.heldout_commitment:
        raise HeldoutError(f"{path} does not match the committed commitment for {spec.workload_id} "
                           f"v{spec.heldout_version}")
    return salt


def derive_seeds(salt: str, spec, n: int = N_SEEDS) -> tuple[int, ...]:
    """`n` distinct seeds from the salt, disjoint from the search seeds and the edge-case seed."""
    excluded = set(spec.search_seeds) | {EDGE_SEED}
    seeds, counter = [], 0
    while len(seeds) < n:
        digest = hashlib.sha256(f"{salt}:{spec.workload_id}:v{spec.heldout_version}:{counter}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
        counter += 1
    return tuple(seeds)


def cases(spec, salt: str):
    """[(name, args)] for every held-out shape x held-out seed. Names do not reveal the seeds."""
    if len(spec.heldout_shapes) < 2:
        raise HeldoutError(f"{spec.workload_id}: AIC-1.1 needs at least two unseen shapes")
    seeds = derive_seeds(salt, spec)
    return [(f"heldout{list(shape)}/#{i}", spec.make_inputs(seed, shape))
            for shape in spec.heldout_shapes for i, seed in enumerate(seeds)]


def _ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_not_burned(spec, ledger: Path = LEDGER) -> None:
    for entry in _ledger(ledger):
        if (entry["workload"], entry["version"]) == (spec.workload_id, spec.heldout_version) and not entry["passed"]:
            raise HeldoutError(f"{spec.workload_id} held-out split v{spec.heldout_version} is burned: it failed for "
                               f"candidate {entry['candidate_sha256'][:12]} on {entry['utc']}. A new evaluation "
                               "needs a new split version (heldout_version + 1, new salt, new commitment)")


def record(spec, candidate_source: str, report, ledger: Path = LEDGER) -> dict:
    entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "workload": spec.workload_id,
             "version": spec.heldout_version, "commitment": spec.heldout_commitment,
             "candidate_sha256": hashlib.sha256(candidate_source.encode()).hexdigest(),
             "passed": report.passed, "category": report.category,
             "n_cases": len(report.cases), "n_failed": sum(not c.passed for c in report.cases)}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def evaluate(spec, candidate_source: str, runner, workdir, *, selected_backend, ledger: Path = LEDGER):
    """Final held-out correctness of a frozen candidate. Never called by the search loop or its tools.

    Reference and candidate both run through `runner` (the candidate untrusted), outputs are compared on
    the host, and the verdict is appended to the ledger. Returns (CorrectnessReport, ledger entry).
    """
    from contracts import MeasurementProtocol
    from controller import Evaluator
    salt = load_salt(spec)
    check_not_burned(spec, ledger)
    evaluator = Evaluator(spec, runner, MeasurementProtocol(), workdir, selected_backend=selected_backend,
                          cases=cases(spec, salt))
    evaluator.reference_outputs()
    report, _, _ = evaluator.correctness(candidate_source)
    return report, record(spec, candidate_source, report, ledger)


def reveal(spec, out_dir: Path = REPO / "evidence" / "heldout") -> Path:
    """Publish the salt with the final results, so the split can be re-derived and re-run by anyone."""
    salt = load_salt(spec)
    path = out_dir / f"{spec.workload_id}-v{spec.heldout_version}.salt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt + "\n", encoding="ascii")
    return path
