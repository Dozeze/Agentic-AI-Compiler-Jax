"""Workload registry: WorkloadSpec v1 (AIC-003).

A workload is an immutable reference source plus everything needed to test candidates against it:
a pure, seeded input factory, declared shapes per split, dtypes, tolerance, non-finite policy and
edge cases. Inputs come from here, so seeds never appear in agent-visible source.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from workloads.oracles import add_f64, softmax_f64

WORKLOAD_SCHEMA = "workload/v1"
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    nonfinite: str = "reject"   # AIC-1.1 §4: finite inputs must give finite outputs; equal_nan is not a default


@dataclass(frozen=True)
class EdgeCase:
    name: str
    make: Callable[[tuple[int, ...]], tuple[np.ndarray, ...]]
    note: str = ""


@dataclass(frozen=True)
class WorkloadSpec:
    schema_version: str
    workload_id: str
    version: int
    reference_path: Path
    function_name: str
    dtypes: tuple[str, ...]
    static_params: dict                   # e.g. {"axis": -1}; recorded, not passed to the function
    development_shapes: tuple[tuple[int, ...], ...]
    heldout_shapes: tuple[tuple[int, ...], ...]
    search_seeds: tuple[int, ...]         # 0-4; held-out seeds are frozen by AIC-018, not defined here
    input_factory: Callable[[int, tuple[int, ...], str], tuple[np.ndarray, ...]]
    oracle: Callable[..., np.ndarray] | None
    output_contract: str                  # human-readable, e.g. "single array, same shape as x, float32"
    tolerance: Tolerance
    edge_cases: tuple[EdgeCase, ...]
    max_input_bytes: int
    scientific: bool                      # False for example_add

    @property
    def reference_source(self) -> str:
        return self.reference_path.read_text(encoding="utf-8")

    @property
    def reference_sha256(self) -> str:
        return hashlib.sha256(self.reference_path.read_bytes()).hexdigest()

    def make_inputs(self, seed: int, shape: tuple[int, ...], dtype: str = "float32"):
        """Deterministic NumPy inputs for one case. Device placement is the harness's job."""
        if dtype not in self.dtypes:
            raise ValueError(f"{self.workload_id}: unsupported dtype {dtype}")
        args = self.input_factory(seed, shape, dtype)
        nbytes = sum(a.nbytes for a in args)
        if nbytes > self.max_input_bytes:
            raise ValueError(f"{self.workload_id}: inputs are {nbytes} B > {self.max_input_bytes} B")
        return args


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

# [TEAM DECISION P5] Large-offset edge cases use +/-200: far above float32's exp overflow point (~88.7), so a
# softmax without max-subtraction must overflow and is rejected, while the mathematically equivalent
# exp(x - logsumexp(x)) still passes rtol 1e-5 (it fails at offsets >= 1000). This fixes the problem domain
# before any search; it is not a tolerance loosened after the fact.
OFFSET = 200.0
EDGE_SEED = 12345   # fixed, and separate from the search seeds 0-4


def _softmax_inputs(seed, shape, dtype):
    rng = np.random.default_rng(seed)                     # PCG64: deterministic across platforms
    return (rng.standard_normal(shape).astype(dtype),)


def _edge(fn):
    return lambda shape: (np.asarray(fn(np.random.default_rng(EDGE_SEED), shape)).astype(np.float32),)


def _dominant(rng, shape):
    a = rng.standard_normal(shape)
    a[..., 7] += 50.0
    return a


SOFTMAX_EDGE_CASES = (
    EdgeCase("zeros", _edge(lambda r, s: np.zeros(s))),
    EdgeCase("equal_logits", _edge(lambda r, s: np.full(s, 3.7))),
    EdgeCase("large_positive_offset", _edge(lambda r, s: r.standard_normal(s) + OFFSET),
             note="exp overflows float32 (> 88.7) without max-subtraction"),
    EdgeCase("large_negative_offset", _edge(lambda r, s: r.standard_normal(s) - OFFSET),
             note="exp underflows to 0 without max-subtraction"),
    EdgeCase("mixed_magnitudes", _edge(lambda r, s: r.standard_normal(s) * 10.0 ** r.integers(-3, 3, s))),
    EdgeCase("one_dominant_logit", _edge(_dominant)),
)

SOFTMAX = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="softmax", version=1,
    reference_path=HERE / "softmax.py", function_name="softmax",
    dtypes=("float32",), static_params={"axis": -1},
    development_shapes=((32, 128), (128, 1024)), heldout_shapes=((64, 257), (256, 2048)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_softmax_inputs, oracle=softmax_f64,
    output_contract="single float32 array with the same shape as x",
    tolerance=Tolerance(rtol=1e-5, atol=1e-6),          # AIC-1.1 §4 starting values; provisional
    edge_cases=SOFTMAX_EDGE_CASES,
    max_input_bytes=8 << 20,                             # largest input (256, 2048) float32 is 2 MiB
    scientific=True,
)


# ---------------------------------------------------------------------------
# example_add: legacy trusted-example adapter for workloads/example.py (not scientific)
# ---------------------------------------------------------------------------

def _example_add_inputs(seed, shape, dtype):
    del seed  # the legacy example is not seeded; its inputs equal workloads/example.py TEST_ARGS at (32,)
    (n,) = shape
    return (np.arange(n, dtype=dtype), np.ones(n, dtype=dtype))


EXAMPLE_ADD = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="example_add", version=1,
    reference_path=HERE / "example.py", function_name="bad_func",
    dtypes=("float32",), static_params={},
    development_shapes=((32,), (4096,)), heldout_shapes=(),
    search_seeds=(0,), input_factory=_example_add_inputs, oracle=add_f64,
    output_contract="single float32 array with the same shape as x",
    tolerance=Tolerance(rtol=0.0, atol=0.0),
    edge_cases=(),
    max_input_bytes=1 << 20,
    scientific=False,
)

REGISTRY = {s.workload_id: s for s in (SOFTMAX, EXAMPLE_ADD)}


def get(workload_id: str) -> WorkloadSpec:
    """Return a workload spec; KeyError lists the available IDs."""
    try:
        return REGISTRY[workload_id]
    except KeyError:
        raise KeyError(f"unknown workload {workload_id!r}; available: {sorted(REGISTRY)}") from None
