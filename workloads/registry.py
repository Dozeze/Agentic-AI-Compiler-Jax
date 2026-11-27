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

from workloads.oracles import (add_f64, attention_f64, batched_matmul_f64, conv3x3_f64, cross_entropy_f64,
                              embedding_lookup_f64, gqa_decode_f64, layer_norm_f64, matmul_chain_f64,
                              matmul_gelu_f64, pairwise_distances_f64, rnn_layer_f64, softmax_f64, top_k_rows_f64)

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
    search_seeds: tuple[int, ...]         # 0-4; the held-out seeds are `heldout_seeds` below
    input_factory: Callable[[int, tuple[int, ...], str], tuple[np.ndarray, ...]]
    oracle: Callable[..., np.ndarray] | None
    output_contract: str                  # human-readable, e.g. "single array, same shape as x, float32"
    tolerance: Tolerance
    edge_cases: tuple[EdgeCase, ...]
    max_input_bytes: int
    scientific: bool                      # False for example_add
    heldout_seeds: tuple[int, ...] = ()   # AIC-018: >= 20 seeds the search never uses (see heldout.py)
    heldout_version: int = 1              # a new split after a held-out failure gets a new version

    @property
    def reference_source(self) -> str:
        return self.reference_path.read_bytes().decode("utf-8")

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
    # AIC-018: held-out split v1, frozen 2026-11-10 before any search. Never passed to the search loop.
    heldout_seeds=tuple(range(1000, 1020)),
)


# ---------------------------------------------------------------------------
# Representative suite (AIC-017). Numerical contracts are declared before any candidate search.
# These byte bounds cover every input, including explicit affine parameters/weights.
# ---------------------------------------------------------------------------

REPRESENTATIVE_MAX_INPUT_BYTES = 8 << 20


def _checked_shape(shape, arity, workload):
    if (len(shape) != arity or any(isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n <= 0 for n in shape)):
        raise ValueError(f"{workload}: expected {arity} positive integer dimensions")
    return tuple(int(n) for n in shape)


def _check_input_size(elements, workload):
    size = elements * np.dtype("float32").itemsize
    if size > REPRESENTATIVE_MAX_INPUT_BYTES:
        raise ValueError(f"{workload}: inputs are {size} B > {REPRESENTATIVE_MAX_INPUT_BYTES} B before allocation")


def _layer_norm_inputs(seed, shape, dtype):
    rows, width = _checked_shape(shape, 2, "layer_norm")
    _check_input_size(rows * width + 2 * width, "layer_norm")
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, width)).astype(dtype),
            rng.uniform(0.5, 1.5, width).astype(dtype),
            (0.1 * rng.standard_normal(width)).astype(dtype))


def _layer_norm_edge(kind, shape):
    x, scale, bias = _layer_norm_inputs(EDGE_SEED, shape, "float32")
    if kind == "zeros":
        x.fill(0)
    elif kind == "constant_rows":
        x.fill(3.25)
    elif kind == "nearly_constant_rows":
        x = np.float32(1) + np.float32(1e-4) * x
    elif kind == "positive_offset":
        x += np.float32(32)
    elif kind == "mixed_magnitudes":
        factors = 10.0 ** np.random.default_rng(EDGE_SEED).integers(-3, 3, x.shape)
        x = (x * factors).astype(np.float32)
    elif kind == "one_dominant_value":
        x[:, -1] += np.float32(50)
    elif kind == "affine_parameters":
        scale = np.linspace(-2, 2, x.shape[-1], dtype=np.float32)
        bias = np.linspace(0.5, -0.5, x.shape[-1], dtype=np.float32)
    return x, scale, bias


LAYER_NORM = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="layer_norm", version=1,
    reference_path=HERE / "layer_norm.py", function_name="layer_norm",
    dtypes=("float32",), static_params={"axis": -1, "epsilon": 1e-5, "variance": "population_two_pass",
        "affine": True, "input_layout": "x[rows,width], scale[width], bias[width]"},
    development_shapes=((32, 128), (128, 1024)), heldout_shapes=((64, 257), (256, 2048)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_layer_norm_inputs, oracle=layer_norm_f64,
    output_contract="single float32 array with the same shape as x; last-axis affine layer normalization",
    # Mean rounding is amplified by 1/sqrt(epsilon) on nearly constant float32 rows near 1.
    # Absolute 1e-4 covers that conditioning; ordinary rows also use relative 1e-4. Target checks pending.
    tolerance=Tolerance(rtol=1e-4, atol=1e-4),
    edge_cases=tuple(EdgeCase(name, lambda shape, kind=name: _layer_norm_edge(kind, shape)) for name in
        ("zeros", "constant_rows", "nearly_constant_rows", "positive_offset", "mixed_magnitudes",
         "one_dominant_value", "affine_parameters")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True,
    heldout_seeds=tuple(range(1000, 1020)),
)


def _matmul_gelu_inputs(seed, shape, dtype):
    m, k, n = _checked_shape(shape, 3, "matmul_gelu")
    _check_input_size(m * k + k * n + n, "matmul_gelu")
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((m, k)).astype(dtype),
            (rng.standard_normal((k, n)) / np.sqrt(k)).astype(dtype),
            (0.1 * rng.standard_normal(n)).astype(dtype))


def _matmul_gelu_edge(kind, shape):
    x, weight, bias = _matmul_gelu_inputs(EDGE_SEED, shape, "float32")
    if kind == "zeros":
        x.fill(0)
        weight.fill(0)
        bias.fill(0)
    elif kind == "zero_inputs":
        x.fill(0)
    elif kind == "zero_weights":
        weight.fill(0)
    elif kind == "bias_activation_sweep":
        x.fill(0)
        weight.fill(0)
        bias = np.linspace(-6, 6, weight.shape[-1], dtype=np.float32)
    elif kind == "cancellation":
        x[:] = np.where(np.arange(x.shape[-1]) % 2, -1, 1).astype(np.float32)
        weight.fill(np.float32(1 / np.sqrt(x.shape[-1])))
    elif kind == "mixed_magnitudes":
        factors = (10.0 ** (np.arange(x.shape[-1]) % 7 - 3)).astype(np.float32)
        x *= factors
        weight /= factors[:, None]
    return x, weight, bias


MATMUL_GELU = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="matmul_gelu", version=1,
    reference_path=HERE / "matmul_gelu.py", function_name="matmul_gelu",
    dtypes=("float32",), static_params={"gelu": "exact_erfc", "matmul_precision": "HIGHEST",
        "accumulation_dtype": "float32", "bias_axis": -1, "shape_order": ["M", "K", "N"],
        "input_layout": "x[M,K], weight[K,N], bias[N]"},
    development_shapes=((32, 64, 64), (128, 256, 256)), heldout_shapes=((63, 127, 95), (256, 512, 384)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_matmul_gelu_inputs, oracle=matmul_gelu_f64,
    output_contract="single float32 array [M,N]; exact GELU(x @ weight + bias)",
    # Float32 dot accumulation/reassociation up to K=512 and GELU near zero need a distinct absolute
    # floor. This policy is fixed before search; HIGHEST precision forbids a TF32 baseline shortcut.
    tolerance=Tolerance(rtol=1e-4, atol=1e-5),
    edge_cases=tuple(EdgeCase(name, lambda shape, kind=name: _matmul_gelu_edge(kind, shape)) for name in
        ("zeros", "zero_inputs", "zero_weights", "bias_activation_sweep", "cancellation", "mixed_magnitudes")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True,
    heldout_seeds=tuple(range(1000, 1020)),
)


# ---------------------------------------------------------------------------
# Headroom suite (batch 04, after HALO on `claude-experiment`). Each reference is written the way a user
# might write it, and a known rewrite that XLA cannot find on its own is faster (workloads/ceilings/).
# softmax, layer_norm and matmul_gelu above are the controls: XLA already compiles them well.
# Tolerances are fixed before any search from the measured error of the reference and of the ceiling.
# ---------------------------------------------------------------------------

def _normal(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _matmul_chain_inputs(seed, shape, dtype):
    big, small = _checked_shape(shape, 2, "matmul_chain")
    rng = np.random.default_rng(seed)
    return _normal(rng, big, small), _normal(rng, small, big), _normal(rng, big, small)


def _matmul_chain_edge(kind, shape):
    a, b, c = _matmul_chain_inputs(EDGE_SEED, shape, "float32")
    if kind == "zeros":
        a, b, c = np.zeros_like(a), np.zeros_like(b), np.zeros_like(c)
    elif kind == "zero_middle_factor":     # result is exactly zero: dropping an operand is caught
        b = np.zeros_like(b)
    elif kind == "asymmetric_scales":      # catches a transposed or swapped operand
        a *= np.float32(4.0)
        c = c * np.float32(0.25) + np.float32(1.0)
    return a, b, c


MATMUL_CHAIN = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="matmul_chain", version=1,
    reference_path=HERE / "matmul_chain.py", function_name="matmul_chain",
    dtypes=("float32",), static_params={"shape_order": ["big", "small"],
        "input_layout": "a[big,small], b[small,big], c[big,small]"},
    development_shapes=((512, 32), (1024, 16)), heldout_shapes=((2048, 8), (768, 24)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_matmul_chain_inputs, oracle=matmul_chain_f64,
    output_contract="single float32 array [big,small] equal to the matrix product a @ b @ c",
    tolerance=Tolerance(rtol=1e-4, atol=1e-3),
    edge_cases=tuple(EdgeCase(n, lambda shape, kind=n: _matmul_chain_edge(kind, shape))
                     for n in ("zeros", "zero_middle_factor", "asymmetric_scales")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)),
)


def _pairwise_inputs(seed, shape, dtype):
    n, m, d = _checked_shape(shape, 3, "pairwise_distances")
    rng = np.random.default_rng(seed)
    return _normal(rng, n, d), _normal(rng, m, d)


def _pairwise_edge(kind, shape):
    a, b = _pairwise_inputs(EDGE_SEED, shape, "float32")
    if kind == "zeros":
        a, b = np.zeros_like(a), np.zeros_like(b)
    elif kind == "asymmetric_sets":        # catches swapped operands or a transposed result
        b = b * np.float32(4.0) + np.float32(10.0)
    elif kind == "shared_points":          # identical rows: exact zeros on matching pairs
        b[: min(len(a), len(b))] = a[: min(len(a), len(b))]
    return a, b


PAIRWISE_DISTANCES = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="pairwise_distances", version=1,
    reference_path=HERE / "pairwise_distances.py", function_name="pairwise_distances",
    dtypes=("float32",), static_params={"shape_order": ["n", "m", "d"], "input_layout": "a[n,d], b[m,d]"},
    development_shapes=((128, 512, 32), (256, 256, 64)), heldout_shapes=((200, 300, 48), (512, 128, 16)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_pairwise_inputs, oracle=pairwise_distances_f64,
    output_contract="single float32 array [n,m]; out[i,j] = sum_d (a[i,d] - b[j,d])**2",
    tolerance=Tolerance(rtol=1e-4, atol=1e-3),
    edge_cases=tuple(EdgeCase(n, lambda shape, kind=n: _pairwise_edge(kind, shape))
                     for n in ("zeros", "asymmetric_sets", "shared_points")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)),
)


def _attention_inputs(seed, shape, dtype):
    _checked_shape(shape, 4, "attention_loop")
    _check_input_size(3 * int(np.prod(shape)), "attention_loop")
    rng = np.random.default_rng(seed)
    return _normal(rng, *shape), _normal(rng, *shape), _normal(rng, *shape)


def _attention_edge(kind, shape):
    q, k, v = _attention_inputs(EDGE_SEED, shape, "float32")
    rng = np.random.default_rng(EDGE_SEED + 1)
    if kind == "zeros":
        q, k, v = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    elif kind == "large_scores":           # scores near 200: overflow without max-subtraction
        offset = np.float32(np.sqrt(200.0 * np.sqrt(shape[-1]) / shape[-1]))
        q = q * np.float32(0.1) + offset
        k = k * np.float32(0.1) + offset
    elif kind == "uniform_keys":           # identical keys: every weight equal, catches axis mix-ups
        k = np.broadcast_to(rng.standard_normal((1, 1, 1, shape[-1])).astype(np.float32), k.shape).copy()
    return q, k, v


ATTENTION_LOOP = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="attention_loop", version=1,
    reference_path=HERE / "attention_loop.py", function_name="attention_loop",
    dtypes=("float32",), static_params={"shape_order": ["batch", "heads", "seq", "dim"], "mask": None,
        "scale": "1/sqrt(dim)", "input_layout": "q, k, v [batch,heads,seq,dim]"},
    development_shapes=((2, 4, 128, 32), (4, 8, 256, 64)), heldout_shapes=((2, 8, 192, 64), (1, 16, 128, 32)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_attention_inputs, oracle=attention_f64,
    output_contract="single float32 array, same shape as q; softmax(q @ k^T / sqrt(dim)) @ v per batch and head",
    tolerance=Tolerance(rtol=1e-4, atol=1e-5),
    edge_cases=tuple(EdgeCase(n, lambda shape, kind=n: _attention_edge(kind, shape))
                     for n in ("zeros", "large_scores", "uniform_keys")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)),
)


LAYER_NORM_LOOP = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="layer_norm_loop", version=1,
    reference_path=HERE / "layer_norm_loop.py", function_name="layer_norm_loop",
    dtypes=("float32",), static_params={**LAYER_NORM.static_params},
    development_shapes=((16, 512), (32, 1024)), heldout_shapes=((24, 768), (48, 256)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_layer_norm_inputs, oracle=layer_norm_f64,
    output_contract=LAYER_NORM.output_contract, tolerance=LAYER_NORM.tolerance,
    edge_cases=LAYER_NORM.edge_cases,
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)),
)


def _batched_matmul_inputs(seed, shape, dtype):
    batch, rows, inner, cols = _checked_shape(shape, 4, "batched_matmul_loop")
    rng = np.random.default_rng(seed)
    return _normal(rng, batch, rows, inner), _normal(rng, batch, inner, cols)


def _batched_matmul_edge(kind, shape):
    a, b = _batched_matmul_inputs(EDGE_SEED, shape, "float32")
    if kind == "zeros":
        a, b = np.zeros_like(a), np.zeros_like(b)
    elif kind == "per_batch_offset":       # catches one batch element broadcast over all of them
        a = a + np.arange(a.shape[0], dtype=np.float32)[:, None, None]
    return a, b


BATCHED_MATMUL_LOOP = WorkloadSpec(
    schema_version=WORKLOAD_SCHEMA, workload_id="batched_matmul_loop", version=1,
    reference_path=HERE / "batched_matmul_loop.py", function_name="batched_matmul_loop",
    dtypes=("float32",), static_params={"shape_order": ["batch", "rows", "inner", "cols"],
        "input_layout": "a[batch,rows,inner], b[batch,inner,cols]"},
    development_shapes=((16, 64, 32, 64), (32, 128, 64, 128)), heldout_shapes=((24, 96, 48, 80), (8, 256, 64, 32)),
    search_seeds=(0, 1, 2, 3, 4), input_factory=_batched_matmul_inputs, oracle=batched_matmul_f64,
    output_contract="single float32 array [batch,rows,cols]; out[i] = a[i] @ b[i]",
    tolerance=Tolerance(rtol=1e-4, atol=1e-4),
    edge_cases=tuple(EdgeCase(n, lambda shape, kind=n: _batched_matmul_edge(kind, shape))
                     for n in ("zeros", "per_batch_offset")),
    max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)),
)

# Block tier: the building blocks of a language model, after HALO's block suite. Harder than the op tier:
# the faster form needs knowing an API or restructuring the data flow, not only removing a loop.

def _spec(workload_id, factory, oracle, contract, tolerance, edges, dev, heldout, static=None):
    return WorkloadSpec(
        schema_version=WORKLOAD_SCHEMA, workload_id=workload_id, version=1,
        reference_path=HERE / f"{workload_id}.py", function_name=workload_id, dtypes=("float32",),
        static_params=static or {}, development_shapes=dev, heldout_shapes=heldout,
        search_seeds=(0, 1, 2, 3, 4), input_factory=factory, oracle=oracle, output_contract=contract,
        tolerance=tolerance, edge_cases=tuple(EdgeCase(name, make) for name, make in edges),
        max_input_bytes=REPRESENTATIVE_MAX_INPUT_BYTES, scientific=True, heldout_seeds=tuple(range(1000, 1020)))


def _gqa_inputs(seed, shape, dtype):
    b, hq, hkv, cache, d = _checked_shape(shape, 5, "gqa_decode")
    if hq % hkv:
        raise ValueError("gqa_decode: q heads must be a multiple of kv heads")
    _check_input_size(b * hq * d + 2 * b * hkv * cache * d, "gqa_decode")
    rng = np.random.default_rng(seed)
    return _normal(rng, b, hq, 1, d), _normal(rng, b, hkv, cache, d), _normal(rng, b, hkv, cache, d)


def _gqa_edge(kind, shape):
    q, k, v = _gqa_inputs(EDGE_SEED, shape, "float32")
    d = shape[-1]
    if kind == "large_scores":             # scores near 200: overflow without max-subtraction
        offset = np.float32(np.sqrt(200.0 * np.sqrt(d) / d))
        q, k = q * np.float32(0.1) + offset, k * np.float32(0.1) + offset
    elif kind == "kv_head_identity":       # each kv head returns its own constant: catches a wrong grouping
        k = np.broadcast_to(k[:, :, :1, :], k.shape).copy()
        v = np.broadcast_to(np.arange(k.shape[1], dtype=np.float32)[None, :, None, None], v.shape).copy()
    return q, k, v


GQA_DECODE = _spec(
    "gqa_decode", _gqa_inputs, gqa_decode_f64,
    "single float32 array [batch,q_heads,1,dim]; q head h attends over k/v head h // (q_heads // kv_heads)",
    Tolerance(rtol=1e-4, atol=2e-5),       # the grouped rewrite needs 5e-6 beyond rtol; keep a 4x margin
    [(n, lambda shape, kind=n: _gqa_edge(kind, shape)) for n in ("large_scores", "kv_head_identity")],
    dev=((2, 16, 4, 256, 64), (1, 32, 4, 1024, 128)), heldout=((1, 16, 2, 768, 128), (2, 8, 4, 512, 64)),
    static={"shape_order": ["batch", "q_heads", "kv_heads", "cache", "dim"], "query_positions": 1,
            "input_layout": "q[batch,q_heads,1,dim], k and v [batch,kv_heads,cache,dim]", "mask": None})

TOP_K = 8


def _top_k_inputs(seed, shape, dtype):
    rows, cols = _checked_shape(shape, 2, "top_k_rows")
    _check_input_size(rows * cols, "top_k_rows")
    return (_normal(np.random.default_rng(seed), rows, cols),)


def _top_k_edge(kind, shape):
    (x,) = _top_k_inputs(EDGE_SEED, shape, "float32")
    if kind == "ties":
        x[:, ::100] = np.float32(9.0)
    elif kind == "ascending_tail":         # the answer sits at the end of every row
        x[:, -TOP_K:] = np.arange(10, 10 + TOP_K, dtype=np.float32)
    return (x,)


TOP_K_ROWS = _spec(
    "top_k_rows", _top_k_inputs, top_k_rows_f64,
    f"single float32 array [rows,{TOP_K}]: the {TOP_K} largest values of each row, largest first",
    Tolerance(rtol=0.0, atol=0.0),
    [(n, lambda shape, kind=n: _top_k_edge(kind, shape)) for n in ("ties", "ascending_tail")],
    dev=((32, 4096), (64, 16384)), heldout=((48, 8192), (16, 32768)), static={"k": TOP_K})


def _cross_entropy_inputs(seed, shape, dtype):
    rows, vocab = _checked_shape(shape, 2, "cross_entropy")
    _check_input_size(rows * vocab + rows, "cross_entropy")
    rng = np.random.default_rng(seed)
    return _normal(rng, rows, vocab) * np.float32(3.0), rng.integers(0, vocab, size=(rows,), dtype=np.int32)


def _cross_entropy_edge(kind, shape):
    logits, labels = _cross_entropy_inputs(EDGE_SEED, shape, "float32")
    if kind == "large_logits":             # exp overflows without max-subtraction
        logits = logits / np.float32(3.0) + np.float32(500.0)
    elif kind == "peaked_on_label":
        logits[np.arange(len(labels)), labels] += np.float32(30.0)
    elif kind == "boundary_labels":
        labels = np.where(np.arange(len(labels)) % 2, logits.shape[1] - 1, 0).astype(np.int32)
    return logits, labels


CROSS_ENTROPY = _spec(
    "cross_entropy", _cross_entropy_inputs, cross_entropy_f64,
    "single float32 array [rows]: logsumexp(logits[i]) - logits[i, labels[i]]; labels are int32 in [0, vocab)",
    Tolerance(rtol=1e-4, atol=1e-4),
    [(n, lambda shape, kind=n: _cross_entropy_edge(kind, shape))
     for n in ("large_logits", "peaked_on_label", "boundary_labels")],
    dev=((128, 1024), (384, 4096)), heldout=((200, 3000), (96, 8192)),
    static={"input_layout": "logits[rows,vocab] float32, labels[rows] int32"})


def _conv_inputs(seed, shape, dtype):
    n, h, w, c_in, c_out = _checked_shape(shape, 5, "conv3x3")
    _check_input_size(n * h * w * c_in + 9 * c_in * c_out, "conv3x3")
    rng = np.random.default_rng(seed)
    return _normal(rng, n, h, w, c_in), _normal(rng, 3, 3, c_in, c_out) / np.float32(np.sqrt(9 * c_in))


def _conv_edge(kind, shape):
    x, w = _conv_inputs(EDGE_SEED, shape, "float32")
    if kind == "corner_impulse":           # catches wrong padding or a flipped kernel
        x = np.zeros_like(x)
        x[:, 0, 0, 0] = np.float32(1.0)
    elif kind == "centre_tap_only":
        centre = np.zeros_like(w)
        centre[1, 1] = w[1, 1]
        w = centre
    return x, w


CONV3X3 = _spec(
    "conv3x3", _conv_inputs, conv3x3_f64,
    "single float32 array [n,h,w,c_out]: 3x3 convolution, stride 1, zero padding 1 (SAME), NHWC/HWIO",
    Tolerance(rtol=1e-4, atol=1e-4),
    [(n, lambda shape, kind=n: _conv_edge(kind, shape)) for n in ("corner_impulse", "centre_tap_only")],
    dev=((2, 16, 16, 16, 32), (8, 32, 32, 32, 64)), heldout=((4, 24, 24, 32, 48), (1, 64, 64, 16, 32)),
    static={"shape_order": ["n", "h", "w", "c_in", "c_out"], "layout": "x NHWC, w HWIO", "padding": "SAME"})


def _embedding_inputs(seed, shape, dtype):
    batch, seq, vocab, dim = _checked_shape(shape, 4, "embedding_lookup")
    _check_input_size(vocab * dim + batch * seq, "embedding_lookup")
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=(batch, seq), dtype=np.int32), _normal(rng, vocab, dim)


def _embedding_edge(kind, shape):
    ids, table = _embedding_inputs(EDGE_SEED, shape, "float32")
    if kind == "boundary_ids":
        ids = np.where(np.random.default_rng(EDGE_SEED).random(ids.shape) < 0.5, 0, table.shape[0] - 1).astype(np.int32)
    elif kind == "single_id":
        ids = np.full_like(ids, table.shape[0] // 2)
    return ids, table


EMBEDDING_LOOKUP = _spec(
    "embedding_lookup", _embedding_inputs, embedding_lookup_f64,
    "single float32 array [batch,seq,dim]: out[b,t] = table[ids[b,t]]; ids are int32 in [0, vocab)",
    Tolerance(rtol=0.0, atol=0.0),
    [(n, lambda shape, kind=n: _embedding_edge(kind, shape)) for n in ("boundary_ids", "single_id")],
    dev=((4, 64, 1024, 128), (8, 128, 4096, 256)), heldout=((2, 200, 3000, 128), (16, 64, 2048, 256)),
    static={"shape_order": ["batch", "seq", "vocab", "dim"], "input_layout": "ids[batch,seq] int32, table[vocab,dim]"})


def _rnn_inputs(seed, shape, dtype):
    steps, batch, inp, hidden = _checked_shape(shape, 4, "rnn_layer")
    _check_input_size(steps * batch * inp + inp * hidden + hidden * hidden + hidden + batch * hidden, "rnn_layer")
    rng = np.random.default_rng(seed)
    return (_normal(rng, steps, batch, inp), _normal(rng, inp, hidden) * np.float32(0.05),
            _normal(rng, hidden, hidden) * np.float32(0.05), _normal(rng, hidden) * np.float32(0.1),
            _normal(rng, batch, hidden))


def _rnn_edge(kind, shape):
    x, w_in, w_rec, bias, h0 = _rnn_inputs(EDGE_SEED, shape, "float32")
    if kind == "no_recurrence":
        w_rec = np.zeros_like(w_rec)
    elif kind == "state_only":             # only the carried state drives the output
        w_in, h0 = np.zeros_like(w_in), h0 * np.float32(5.0)
    return x, w_in, w_rec, bias, h0


RNN_LAYER = _spec(
    "rnn_layer", _rnn_inputs, rnn_layer_f64,
    "single float32 array [steps,batch,hidden]: h_t = tanh(x_t @ w_in + h_(t-1) @ w_rec + bias), h_(-1) = h0",
    Tolerance(rtol=1e-4, atol=1e-5),
    [(n, lambda shape, kind=n: _rnn_edge(kind, shape)) for n in ("no_recurrence", "state_only")],
    dev=((32, 16, 64, 64), (128, 32, 256, 256)), heldout=((96, 16, 128, 192), (64, 64, 128, 128)),
    static={"shape_order": ["steps", "batch", "input", "hidden"],
            "input_layout": "x[steps,batch,input], w_in[input,hidden], w_rec[hidden,hidden], bias[hidden], h0[batch,hidden]"})

OP_TIER = (MATMUL_CHAIN, PAIRWISE_DISTANCES, ATTENTION_LOOP, LAYER_NORM_LOOP, BATCHED_MATMUL_LOOP)
BLOCK_TIER = (GQA_DECODE, TOP_K_ROWS, CROSS_ENTROPY, CONV3X3, EMBEDDING_LOOKUP, RNN_LAYER)
HEADROOM = (*OP_TIER, *BLOCK_TIER)
CONTROLS = (SOFTMAX, LAYER_NORM, MATMUL_GELU)


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

REGISTRY = {s.workload_id: s for s in (*CONTROLS, *HEADROOM, EXAMPLE_ADD)}


def legacy_source_spec(source_path, function_name, args) -> WorkloadSpec:
    """Adapter for `agent.py --source`: a trusted .py file whose TEST_ARGS / make_test_args are the only
    case. Not scientific: one input, no edge cases, no held-out split."""
    arrays = tuple(np.asarray(a) for a in args)
    if not arrays:
        raise ValueError("a --source workload needs at least one input array")
    path = Path(source_path).resolve()
    return WorkloadSpec(
        schema_version=WORKLOAD_SCHEMA, workload_id=f"source:{path.stem}", version=1, reference_path=path,
        function_name=function_name,
        dtypes=tuple(dict.fromkeys(["float32", *(str(a.dtype) for a in arrays)])), static_params={},
        development_shapes=(arrays[0].shape,), heldout_shapes=(), search_seeds=(0,),
        input_factory=lambda seed, shape, dtype: tuple(a.copy() for a in arrays), oracle=None,
        output_contract="as produced by the reference", tolerance=Tolerance(rtol=1e-5, atol=1e-6),
        edge_cases=(), max_input_bytes=max(1 << 20, 2 * sum(a.nbytes for a in arrays)), scientific=False)


def get(workload_id: str) -> WorkloadSpec:
    """Return a workload spec; KeyError lists the available IDs."""
    try:
        return REGISTRY[workload_id]
    except KeyError:
        raise KeyError(f"unknown workload {workload_id!r}; available: {sorted(REGISTRY)}") from None
