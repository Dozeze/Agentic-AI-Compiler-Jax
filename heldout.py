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
import dataclasses
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
MIN_SEEDS = 20
MIN_SHAPES = 2
EDGE_SEED = 12345
LEDGER = REPO / "evidence" / "heldout" / "ledger.jsonl"
PUBLICATIONS = REPO / "evidence" / "heldout" / "publications"


class HeldoutError(RuntimeError):
    """No usable held-out split: too few or overlapping seeds or shapes, or a burned split version."""


def check_split(spec) -> None:
    """The split meets AIC-1.1 §4: enough seeds and shapes, none shared with the search."""
    seeds, shapes = spec.heldout_seeds, spec.heldout_shapes
    if type(spec.heldout_version) is not int or spec.heldout_version < 1:
        raise HeldoutError("split version must be a positive integer")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise HeldoutError("split seeds must be nonnegative integers")
    if len(set(seeds)) < MIN_SEEDS or len(set(seeds)) != len(seeds):
        raise HeldoutError(f"{spec.workload_id}: needs at least {MIN_SEEDS} distinct held-out seeds")
    if set(seeds) & (set(spec.search_seeds) | {EDGE_SEED}):
        raise HeldoutError(f"{spec.workload_id}: held-out seeds overlap the search or edge-case seeds")
    if (len(set(shapes)) < MIN_SHAPES or len(set(shapes)) != len(shapes)
            or set(shapes) & set(spec.development_shapes)):
        raise HeldoutError(f"{spec.workload_id}: needs at least {MIN_SHAPES} shapes the search never uses")
    if any(not shape or any(type(n) is not int or n <= 0 for n in shape) for shape in shapes):
        raise HeldoutError("split shapes need positive integer dimensions")


def split_contract(spec):
    """Trusted split recipe. Only its opaque commitment goes into the pre-search manifest."""
    check_split(spec)
    return {"schema_version": "final_split/v1", "workload_id": spec.workload_id,
            "workload_version": spec.version, "version": spec.heldout_version,
            "seeds": list(spec.heldout_seeds), "shapes": [list(s) for s in spec.heldout_shapes],
            "development_shapes": [list(s) for s in spec.development_shapes],
            "search_seeds": list(spec.search_seeds), "dtypes": list(spec.dtypes),
            "static_params": spec.static_params, "tolerance": dataclasses.asdict(spec.tolerance),
            "reference_sha256": spec.reference_sha256,
            "input_recipe_sha256": hashlib.sha256((REPO / "workloads/registry.py").read_bytes()).hexdigest()}


def commitment(spec):
    """No raw split data in this value, which must be saved before a search request."""
    from contracts import SplitCommitment
    contract = split_contract(spec)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return dataclasses.asdict(SplitCommitment(spec.workload_id, spec.heldout_version, digest,
                                             len(spec.heldout_seeds) * len(spec.heldout_shapes)))


def _publication_root(spec):
    root = PUBLICATIONS / hashlib.sha256(spec.workload_id.encode()).hexdigest()
    for path in (PUBLICATIONS, root):
        if path.is_symlink():
            raise HeldoutError("symlink evaluation state refused")
    return root


def check_search_allowed(spec):
    """Sealed, failed and interrupted final evaluations cannot guide another search on that split."""
    if not spec.scientific:
        return
    check_split(spec)
    try:
        check_not_burned(spec, LEDGER)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HeldoutError("invalid prior burn ledger; reconciliation required") from error
    root = _publication_root(spec)
    if (root / str(spec.heldout_version)).exists():
        raise HeldoutError("evaluation split is sealed; search/final reuse requires a fresh split and version")
    for directory in root.iterdir() if root.exists() else ():
        reservation = directory / "reservation.json"
        if directory.is_symlink() or reservation.is_symlink():
            raise HeldoutError("symlink evaluation reservation refused")
        try:
            previous = json.loads(reservation.read_bytes())["split"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HeldoutError("invalid prior evaluation reservation; reconciliation required") from error
        if (set(previous["seeds"]) & set(spec.heldout_seeds)
                or {tuple(s) for s in previous["shapes"]} & set(spec.heldout_shapes)):
            raise HeldoutError("new evaluation version must use fresh seeds and shapes")


def reserve(spec, *, run_id, candidate_sha256, output_dir):
    """Atomic one-use reservation before any final worker starts. An interrupted reservation stays sealed."""
    from storage import atomic_write_json
    check_search_allowed(spec)
    directory = _publication_root(spec) / str(spec.heldout_version)
    try:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_json(directory / "reservation.json", {
            "schema_version": "final_reservation/v1", "run_id": run_id,
            "candidate_sha256": candidate_sha256, "output_dir": str(output_dir),
            "split": split_contract(spec), "commitment": commitment(spec)}, exclusive=True)
    except OSError as error:
        raise HeldoutError("cannot reserve final split; an existing or partial reservation cannot retry") from error
    return directory


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
