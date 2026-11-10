"""Held-out evaluation (AIC-018, AIC-1.1 §4): frozen before search, never seen by the search.

Each research workload declares its held-out split in workloads/registry.py: at least 20 `heldout_seeds`,
disjoint from the search seeds and the edge-case seed, and two `heldout_shapes` the search never uses.
Committing them before any search freezes the split: the commit is the evidence. They are public, which
also satisfies AIC-1.1's "release them with the final results". What keeps them held out is that nothing
in the search ever receives them: the model only sees the prompt and the tool results the controller
builds from the development cases, and this module is not reachable from the search loop or its tools.

A failed held-out evaluation burns its split version in the ledger: the scope's generalisation claim is
invalid, and a later attempt on the same split is not called held out. Debugging needs a new split
(new seeds and `heldout_version + 1`).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
MIN_SEEDS = 20
MIN_SHAPES = 2
EDGE_SEED = 12345
LEDGER = REPO / "evidence" / "heldout" / "ledger.jsonl"


class HeldoutError(RuntimeError):
    """No usable held-out split: too few or overlapping seeds or shapes, or a burned split version."""


def check_split(spec) -> None:
    """The split meets AIC-1.1 §4: enough seeds and shapes, none shared with the search."""
    seeds, shapes = spec.heldout_seeds, spec.heldout_shapes
    if len(set(seeds)) < MIN_SEEDS or len(set(seeds)) != len(seeds):
        raise HeldoutError(f"{spec.workload_id}: needs at least {MIN_SEEDS} distinct held-out seeds")
    if set(seeds) & (set(spec.search_seeds) | {EDGE_SEED}):
        raise HeldoutError(f"{spec.workload_id}: held-out seeds overlap the search or edge-case seeds")
    if len(set(shapes)) < MIN_SHAPES or set(shapes) & set(spec.development_shapes):
        raise HeldoutError(f"{spec.workload_id}: needs at least {MIN_SHAPES} shapes the search never uses")


def cases(spec):
    """[(name, args)] for every held-out shape x held-out seed."""
    check_split(spec)
    return [(f"heldout{list(shape)}/seed{seed}", spec.make_inputs(seed, shape))
            for shape in spec.heldout_shapes for seed in spec.heldout_seeds]


def _ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_not_burned(spec, ledger: Path = LEDGER) -> None:
    for entry in _ledger(ledger):
        if (entry["workload"], entry["version"]) == (spec.workload_id, spec.heldout_version) and not entry["passed"]:
            raise HeldoutError(f"{spec.workload_id} held-out split v{spec.heldout_version} is burned: it failed for "
                               f"candidate {entry['candidate_sha256'][:12]} on {entry['utc']}. A new evaluation "
                               "needs a new split (new seeds, heldout_version + 1)")


def record(spec, candidate_source: str, report, ledger: Path = LEDGER) -> dict:
    entry = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "workload": spec.workload_id,
             "version": spec.heldout_version,
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
    split = cases(spec)
    check_not_burned(spec, ledger)
    evaluator = Evaluator(spec, runner, MeasurementProtocol(), workdir, selected_backend=selected_backend,
                          cases=split)
    evaluator.reference_outputs()
    report, _, _ = evaluator.correctness(candidate_source)
    return report, record(spec, candidate_source, report, ledger)
