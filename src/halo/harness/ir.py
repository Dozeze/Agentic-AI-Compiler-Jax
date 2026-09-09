"""Operation counts for the two levels above optimized HLO.

The harness already summarises the *end* of the pipeline. These two cover the
start and the middle, so the agent can see what it wrote (jaxpr), what JAX handed
the compiler (StableHLO), and what XLA made of it - and therefore judge whether
XLA already solved the problem or left something on the table.
"""

from __future__ import annotations

import re
from collections import Counter

from halo.types import IRStats

_STABLEHLO_OP = re.compile(r"=\s*stablehlo\.([a-z_0-9]+)")


def _count_eqns(jaxpr, counts: Counter[str]) -> None:
    """Walk a jaxpr, descending into the sub-jaxprs of higher-order primitives.

    ``scan``, ``while`` and ``cond`` hide their bodies in equation params. A
    streaming-softmax rewrite is exactly a ``scan``, so counting only the top
    level would report such a candidate as trivially small.
    """
    for eqn in jaxpr.eqns:
        counts[str(eqn.primitive)] += 1
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    _count_eqns(inner, counts)


def from_jaxpr(closed_jaxpr) -> IRStats:
    counts: Counter[str] = Counter()
    _count_eqns(closed_jaxpr.jaxpr, counts)
    return IRStats(total_ops=sum(counts.values()), op_counts=dict(counts))


def from_stablehlo(text: str) -> IRStats:
    counts = Counter(m.group(1) for m in _STABLEHLO_OP.finditer(text))
    return IRStats(total_ops=sum(counts.values()), op_counts=dict(counts))
