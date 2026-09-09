"""The benchmark task contract.

A task lives in ``tasks/<name>/`` and is exactly two files:

``task.py``       Immutable. Defines the workload, the input distribution and the
                  *oracle*. The oracle is written in NumPy float64, deliberately
                  not in JAX: an independent implementation also catches errors in
                  the compiler itself, not just in the agent's rewrite.
``candidate.py``  The only file the agent may rewrite. Must expose
                  ``candidate(*inputs)`` with the same signature as the reference.

A task module must provide::

    NAME: str
    DESCRIPTION: str                       # shown to the agent
    def make_inputs(rng) -> tuple[np.ndarray, ...]
    def reference(*inputs) -> np.ndarray            # float64 NumPy
    def correctness_cases(rng) -> list[Case]

It may additionally set module-level ``ATOL`` / ``RTOL`` to raise the tolerance
floor for the whole task, which reductions over many float32 terms need. State
the measured error that justifies the value; an unexplained loose tolerance is
how a broken candidate gets accepted.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

TASKS_ROOT = Path(__file__).parent


def load_module(path: Path, name: str) -> ModuleType:
    """Import a module from an explicit file path, bypassing the package system.

    Used both for immutable ``task.py`` files and for per-attempt ``candidate.py``
    files, which live outside the source tree under ``runs/``.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Case:
    """One named correctness check.

    ``atol``/``rtol`` override the run defaults. Overrides are for cases that are
    *ill-conditioned in float32*, where even a correct implementation cannot match
    the float64 oracle tightly - a peaked softmax is the canonical example. Always
    say why in ``note``; a silently loosened tolerance is how a broken candidate
    gets accepted.
    """

    name: str
    inputs: tuple
    atol: float | None = None
    rtol: float | None = None
    note: str = ""


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    directory: Path
    seed_source: str
    reference_sha256: str

    @property
    def task_path(self) -> Path:
        return self.directory / "task.py"

    def load(self) -> ModuleType:
        """Import ``task.py``, verifying it has not been modified.

        The agent is only ever handed ``candidate.py``; this hash check makes that
        structural guarantee explicit, so a run cannot silently "succeed" by having
        weakened its own oracle.
        """
        digest = hashlib.sha256(self.task_path.read_bytes()).hexdigest()
        if digest != self.reference_sha256:
            raise RuntimeError(
                f"task.py for '{self.name}' changed during the run "
                f"(expected {self.reference_sha256[:12]}, got {digest[:12]})"
            )
        return load_module(self.task_path, f"halo_task_{self.name}")


def load_spec(name: str) -> TaskSpec:
    directory = TASKS_ROOT / name
    task_path = directory / "task.py"
    candidate_path = directory / "candidate.py"
    if not task_path.exists() or not candidate_path.exists():
        raise KeyError(f"unknown task '{name}' (looked in {directory})")
    module = load_module(task_path, f"halo_task_probe_{name}")
    return TaskSpec(
        name=getattr(module, "NAME", name),
        description=module.DESCRIPTION,
        directory=directory,
        seed_source=candidate_path.read_text(),
        reference_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
    )


def available() -> list[str]:
    return sorted(
        p.name
        for p in TASKS_ROOT.iterdir()
        if p.is_dir() and (p / "task.py").exists()
    )


#: Definitions withheld from every prompt. ``correctness_cases`` names the exact
#: adversarial inputs a candidate is graded on; showing it would let a proposal be
#: tailored to the test instead of to the problem, and would make the system
#: prompt's claim that those inputs are unseen simply false.
WITHHELD_FROM_AGENT = frozenset({"correctness_cases"})


def reference_excerpt(spec: TaskSpec) -> str:
    """The task source as the agent sees it, with the graded cases removed.

    Line slicing rather than ``ast.unparse`` so comments and formatting survive.
    """
    source = spec.task_path.read_text()
    lines = source.splitlines(keepends=True)
    drop: set[int] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in WITHHELD_FROM_AGENT:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            drop.update(range(start - 1, node.end_lineno))
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return "".join(kept).rstrip() + (
        "\n\n# (correctness cases omitted: the inputs you are graded on are withheld)\n"
    )
