"""The best implementation we know of for each task, and what it is worth.

Each task directory carries a third file, ``ceiling.py``. The agent never sees it:
:func:`halo.tasks.base.load_spec` reads only ``task.py`` and ``candidate.py``.

It exists so the benchmark suite can be calibrated. Raw speedup is a poor score
for an agent, because it conflates "the agent did well" with "this task happened
to have room". Measuring the ceiling separately turns the result into a fraction
of the achievable improvement, and - just as importantly - proves that a task
classified as having no headroom really has none, rather than having headroom the
agent simply failed to find.

Run ``halo ceilings`` to re-measure. The numbers below are a reference point, not
a promise: they are specific to one machine and one XLA version.
"""

from __future__ import annotations

from pathlib import Path

from halo.tasks.base import TaskSpec

#: Whether a source-level rewrite can beat XLA's own output on this task.
#: "null" tasks are controls: the correct outcome is that no proposal is accepted.
CLASSIFICATION: dict[str, str] = {
    "matmul_chain": "headroom",
    "layernorm_loop": "headroom",
    "pairwise_distances": "headroom",
    "batched_matmul_loop": "headroom",
    "naive_attention": "headroom",
    "multi_head_projection": "headroom",
    "softmax": "null",
    "rmsnorm": "null",
    "gelu": "null",
    "matmul": "null",
}

REFERENCE_DEVICE = "Apple M5, CPU backend, JAX 0.11.1"

#: Measured ceiling speedup over the seed on REFERENCE_DEVICE.
MEASURED_HEADROOM: dict[str, float] = {
    "matmul_chain": 11.31,
    "layernorm_loop": 4.88,
    "pairwise_distances": 4.58,
    "batched_matmul_loop": 2.63,
    "naive_attention": 1.65,
    "multi_head_projection": 1.26,
    "rmsnorm": 1.01,
    "softmax": 1.01,
    "gelu": 1.00,
    "matmul": 0.99,
}

# `multi_head_projection` was a control until the agent beat its ceiling. The
# hand-written einsum measured 1.04x over the seed's head loop and the task was
# classified as having no headroom; the tool-using agent then found
# `jnp.matmul(x[None], w)` at 1.26x, three runs out of three, and 1.19x over the
# einsum itself (same HLO op counts, different layout). Its ceiling.py is now the
# agent's program. Ceilings are the best implementation *known*, not a bound.


def path(spec: TaskSpec) -> Path:
    return spec.directory / "ceiling.py"


def is_null(task_name: str) -> bool:
    return CLASSIFICATION.get(task_name) == "null"


def by_classification(kind: str) -> list[str]:
    return sorted(n for n, c in CLASSIFICATION.items() if c == kind)
