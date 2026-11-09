"""Candidate evaluation: correctness checking (AIC-004), the paired-block timing protocol (AIC-005)
and legacy timing.

Correctness is a hard gate before timing. Candidates are compared with the jit-compiled frozen
reference on the same device, dtype and XLA settings; the NumPy oracle only validates the reference
itself (AIC-003). Comparison order: tree structure, leaf count, then per leaf exact shape, exact dtype,
non-finite values, and finally values (integers and bools exactly; floats with atol + rtol * |expected|).
"""
import dataclasses
import functools
import hashlib
import json
import math
import statistics
import time
import timeit
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from workloads.registry import Tolerance

FAILURE_CATEGORIES = ("COMPILE_FAILED", "EXECUTION_FAILED", "STRUCTURE_MISMATCH",
                      "NONFINITE", "NUMERIC_MISMATCH")
# Same tolerance the prototype has always used for evaluate(); integers and bools are compared exactly.
LEGACY_TOLERANCE = Tolerance(rtol=1e-5, atol=1e-6)
LEGACY_NONFINITE_REFERENCE = "reference produced non-finite values; workload contract must declare masks"
MAX_DETAIL_CHARS = 500          # bounds what enters history JSON and the LLM context
MAX_REPORTED_LEAVES = 3


class ReferenceInvalid(RuntimeError):
    """The frozen reference failed or produced non-finite output: a harness/contract bug, not a candidate failure."""


def _clip(text):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= MAX_DETAIL_CHARS else text[:MAX_DETAIL_CHARS - 3] + "..."


def _finite_or_none(value):
    return value if value is None or math.isfinite(value) else None


@dataclass(frozen=True)
class LeafResult:
    path: str                 # jax.tree_util.keystr(path); "" for a single array
    passed: bool
    category: str | None      # one of FAILURE_CATEGORIES when failed
    max_abs_error: float | None
    max_rel_error: float | None
    n_nonfinite: int
    detail: str | None

    def to_json(self) -> dict:
        detail = self.detail
        abs_err, rel_err = _finite_or_none(self.max_abs_error), _finite_or_none(self.max_rel_error)
        if (abs_err, rel_err) != (self.max_abs_error, self.max_rel_error):
            detail = ((detail + "; ") if detail else "") + "non-finite error values reported as null"
        return {"path": self.path, "passed": self.passed, "category": self.category,
                "max_abs_error": abs_err, "max_rel_error": rel_err,
                "n_nonfinite": self.n_nonfinite, "detail": _clip(detail)}


@dataclass(frozen=True)
class CaseResult:
    case: str                 # e.g. "dev(32,128)/seed0", "edge/zeros"
    passed: bool
    category: str | None
    leaves: tuple[LeafResult, ...]
    detail: str | None

    def to_json(self) -> dict:
        failing = [leaf for leaf in self.leaves if not leaf.passed]
        shown = (failing or list(self.leaves))[:MAX_REPORTED_LEAVES]
        return {"case": self.case, "passed": self.passed, "category": self.category,
                "detail": _clip(self.detail), "n_leaves": len(self.leaves),
                "n_failing_leaves": len(failing), "leaves": [leaf.to_json() for leaf in shown]}


@dataclass(frozen=True)
class CorrectnessReport:
    passed: bool
    category: str | None      # category of the first failing case
    policy_hash: str          # sha256 of tolerance, non-finite policy and case list: the freeze evidence
    cases: tuple[CaseResult, ...]

    def to_json(self) -> dict:
        """Strict JSON: non-finite error values become null with a reason in `detail`. Never contains inputs."""
        return {"passed": self.passed, "category": self.category, "policy_hash": self.policy_hash,
                "n_cases": len(self.cases), "n_failed": sum(not c.passed for c in self.cases),
                "cases": [c.to_json() for c in self.cases]}


def policy_hash(tolerance, case_names, workload_id, reference_sha256) -> str:
    """sha256 of everything that defines "correct": tolerance, non-finite policy, cases and reference."""
    policy = {"rtol": tolerance.rtol, "atol": tolerance.atol, "nonfinite": tolerance.nonfinite,
              "cases": list(case_names), "workload": workload_id, "reference_sha256": reference_sha256}
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def _dtype(leaf):
    dtype = getattr(leaf, "dtype", None)
    return np.dtype(dtype) if dtype is not None else np.asarray(leaf).dtype


def _is_float(dtype) -> bool:
    return bool(jnp.issubdtype(dtype, jnp.inexact))


def _relative_errors(diff, e):
    """Relative error with a near-zero floor.

    denominator = max(|expected|, floor), floor = max(1e-3 * max|expected|, float64 tiny), so that tiny
    expected values (e.g. small softmax probabilities) do not blow the relative error up. Reported only;
    the pass/fail verdict uses atol + rtol * |expected|.
    """
    magnitude = np.abs(e)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    floor = max(1e-3 * peak, np.finfo(np.float64).tiny)
    return diff / np.maximum(magnitude, floor)


def _compare_leaf(path, expected_leaf, actual_leaf, tol) -> LeafResult:
    def fail(category, detail, abs_err=None, rel_err=None, n_nonfinite=0):
        return LeafResult(path, False, category, abs_err, rel_err, n_nonfinite, detail)

    e_shape, a_shape = tuple(np.shape(expected_leaf)), tuple(np.shape(actual_leaf))
    if e_shape != a_shape:
        return fail("STRUCTURE_MISMATCH", f"shape: expected {e_shape}, got {a_shape}")
    e_dtype, a_dtype = _dtype(expected_leaf), _dtype(actual_leaf)
    if e_dtype != a_dtype:
        # e.g. bfloat16/float16 instead of float32: a silent precision change, forbidden by AIC-1.1.
        return fail("STRUCTURE_MISMATCH", f"dtype: expected {e_dtype}, got {a_dtype}")

    # Host transfer happens here, in the correctness phase only, never inside timing.
    e = np.asarray(expected_leaf)
    a = np.asarray(actual_leaf)

    if _is_float(e_dtype):
        if tol.nonfinite != "reject":
            raise ValueError(f"unsupported non-finite policy {tol.nonfinite!r}")
        if not np.all(np.isfinite(e)):
            raise ReferenceInvalid(f"reference output {path or '<root>'} contains "
                                   f"{int(np.sum(~np.isfinite(e)))} non-finite values")
        bad = ~np.isfinite(a)
        if bad.any():
            first = tuple(int(i) for i in np.argwhere(bad)[0])
            return fail("NONFINITE", f"{int(bad.sum())} non-finite values; first at index {first}",
                        n_nonfinite=int(bad.sum()))
        e64, a64 = e.astype(np.float64), a.astype(np.float64)   # upcast: no overflow while computing errors
        diff = np.abs(a64 - e64)
        bound = tol.atol + tol.rtol * np.abs(e64)
        exceeded = diff > bound
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float(_relative_errors(diff, e64).max()) if diff.size else 0.0
        if exceeded.any():
            first = tuple(int(i) for i in np.argwhere(exceeded)[0])
            return fail("NUMERIC_MISMATCH",
                        f"{int(exceeded.sum())} of {diff.size} elements exceed atol + rtol*|expected| "
                        f"(atol={tol.atol}, rtol={tol.rtol}); first at index {first}", max_abs, max_rel)
        return LeafResult(path, True, None, max_abs, max_rel, 0, None)

    # Integers and bools: exact.
    if not np.array_equal(a, e):
        mismatched = a != e
        first = tuple(int(i) for i in np.argwhere(mismatched)[0]) if mismatched.ndim else ()
        diff = np.abs(a.astype(np.float64) - e.astype(np.float64))
        return fail("NUMERIC_MISMATCH", f"{int(np.sum(mismatched))} elements differ (exact comparison); "
                    f"first at index {first}", float(diff.max()) if diff.size else 0.0)
    return LeafResult(path, True, None, 0.0, 0.0, 0, None)


def compare_outputs(expected, actual, tol) -> CaseResult:
    """Compare two output PyTrees under `tol`. Returns a CaseResult with an empty case name.

    Raises ReferenceInvalid if `expected` contains non-finite values ("both sides inf" never passes).
    """
    e_tree = jax.tree_util.tree_structure(expected)
    a_tree = jax.tree_util.tree_structure(actual)
    if e_tree != a_tree:
        return CaseResult("", False, "STRUCTURE_MISMATCH", (),
                          f"tree: expected {e_tree}, got {a_tree}")
    e_leaves = jax.tree_util.tree_flatten_with_path(expected)[0]
    a_leaves = jax.tree_util.tree_leaves(actual)
    if len(e_leaves) != len(a_leaves):  # implied by equal tree structures; kept as a defensive check
        return CaseResult("", False, "STRUCTURE_MISMATCH", (),
                          f"leaf count: expected {len(e_leaves)}, got {len(a_leaves)}")
    leaves = tuple(_compare_leaf(jax.tree_util.keystr(path), e_leaf, a_leaf, tol)
                   for (path, e_leaf), a_leaf in zip(e_leaves, a_leaves))
    failing = [leaf for leaf in leaves if not leaf.passed]
    if failing:
        return CaseResult("", False, failing[0].category, leaves,
                          f"leaf {failing[0].path or '<root>'}: {failing[0].detail}")
    return CaseResult("", True, None, leaves, None)


def _shape_name(shape) -> str:
    return "(" + ",".join(str(n) for n in shape) + ")"


def build_cases(spec, *, include_heldout_shapes=False):
    """Return [(name, args)]: development shapes x search seeds, then edge cases on the first development shape.

    include_heldout_shapes=True adds held-out shapes x search seeds as a SMOKE TEST ONLY (it catches
    shape-overfit candidates). This is not AIC-018's held-out evaluation and never evidence of generalisation.
    """
    cases = [(f"dev{_shape_name(shape)}/seed{seed}", spec.make_inputs(seed, shape))
             for shape in spec.development_shapes for seed in spec.search_seeds]
    cases += [(f"edge/{edge.name}", edge.make(spec.development_shapes[0])) for edge in spec.edge_cases]
    if include_heldout_shapes:
        cases += [(f"heldout-smoke{_shape_name(shape)}/seed{seed}", spec.make_inputs(seed, shape))
                  for shape in spec.heldout_shapes for seed in spec.search_seeds]
    return cases


def load_function(source, function_name, filename="<candidate>"):
    """exec a trusted source string and return `function_name` (no isolation yet: AIC-007...009)."""
    namespace = {}
    exec(compile(source, filename, "exec"), namespace)
    function = namespace.get(function_name)
    if not callable(function):
        raise ValueError(f"source does not define callable {function_name!r}")
    return function


@functools.lru_cache(maxsize=16)
def _load_reference(path, sha256, function_name):
    del sha256  # part of the cache key only: a changed reference file is reloaded
    from pathlib import Path
    return load_function(Path(path).read_text(encoding="utf-8"), function_name, filename=str(path))


def _compile(jitted, args):
    """Compile `jitted` for `args`. Module-level so tests can substitute it."""
    return jitted.lower(*args).compile()


def _signature(args):
    return tuple((tuple(np.shape(a)), str(_dtype(a))) for a in args)


def check_correctness(candidate_fn, spec, cases, *, reference_fn=None) -> CorrectnessReport:
    """Hard correctness gate for one candidate over `cases` (from build_cases).

    The candidate is compared with the jit-compiled frozen reference, never with the NumPy oracle.
    A reference that fails or returns non-finite values raises ReferenceInvalid (a harness bug).
    """
    if reference_fn is None:
        reference_fn = _load_reference(str(spec.reference_path), spec.reference_sha256, spec.function_name)
    ref = jax.jit(reference_fn)
    cand = jax.jit(candidate_fn)
    compile_failures = {}       # signature -> name of the case where compilation failed
    executables = {}
    results = []
    for name, raw_args in cases:
        args = tuple(jax.device_put(a) for a in raw_args)
        jax.block_until_ready(args)
        try:
            expected = jax.block_until_ready(ref(*args))
        except Exception as error:
            raise ReferenceInvalid(f"{name}: reference failed: {type(error).__name__}: {error}") from error
        for leaf in jax.tree_util.tree_leaves(expected):
            if _is_float(_dtype(leaf)) and not np.all(np.isfinite(np.asarray(leaf))):
                raise ReferenceInvalid(f"{name}: reference produced non-finite values")

        sig = _signature(args)
        if sig in compile_failures:
            results.append(CaseResult(name, False, "COMPILE_FAILED", (),
                                      f"not retried: same shape/dtype failed to compile in {compile_failures[sig]}"))
            continue
        if sig not in executables:
            try:
                executables[sig] = _compile(cand, args)
            except Exception as error:
                compile_failures[sig] = name
                results.append(CaseResult(name, False, "COMPILE_FAILED", (),
                                          _clip(f"{type(error).__name__}: {error}")))
                continue
        try:
            actual = jax.block_until_ready(executables[sig](*args))
        except Exception as error:
            results.append(CaseResult(name, False, "EXECUTION_FAILED", (),
                                      _clip(f"{type(error).__name__}: {error}")))
            continue
        results.append(replace(compare_outputs(expected, actual, spec.tolerance), case=name))

    first_failure = next((r for r in results if not r.passed), None)
    return CorrectnessReport(
        passed=first_failure is None,
        category=None if first_failure is None else first_failure.category,
        policy_hash=policy_hash(spec.tolerance, [name for name, _ in cases],
                                spec.workload_id, spec.reference_sha256),
        cases=tuple(results),
    )


# ---------------------------------------------------------------------------
# Paired-block timing protocol (AIC-005, AIC-1.1 §5)
#
# The metric is steady-state end-to-end latency of a compiled JAX call, dispatch and synchronisation
# included. Every call is synchronised with jax.block_until_ready, so the clock never measures only the
# asynchronous dispatch. `clock` is injectable (nanoseconds, monotonic) so the protocol logic is
# testable with a deterministic fake clock; real runs use time.perf_counter_ns.
# ---------------------------------------------------------------------------

def warm_up(fn, args, calls):
    """Run `fn` `calls` times, synchronised, outside any timed window."""
    for _ in range(calls):
        jax.block_until_ready(fn(*args))


def _run_block(fn, args, repeats, clock):
    """(start_ns, elapsed_ns) of `repeats` synchronised calls."""
    start = clock()
    for _ in range(repeats):
        jax.block_until_ready(fn(*args))
    return start, clock() - start


def calibrate_repeats(baseline, args, protocol, *, clock=time.perf_counter_ns):
    """Calls per block, chosen from the baseline only and then frozen for both sides (AIC-1.1 §5.3).

    Doubles the repeat count until one block lasts at least `min_block_s` (or the cap is hit), then
    scales it so a block lasts about `target_block_s`. A call slower than `max_block_s` gets one repeat.
    """
    repeats = 1
    while True:
        _, elapsed_ns = _run_block(baseline, args, repeats, clock)
        if elapsed_ns >= protocol.min_block_s * 1e9 or repeats >= protocol.max_repeats:
            break
        repeats = min(protocol.max_repeats, repeats * 2)
    per_call_s = elapsed_ns / 1e9 / repeats
    if per_call_s <= 0:
        return protocol.max_repeats
    return max(1, min(protocol.max_repeats, round(protocol.target_block_s / per_call_s)))


def block_order(n_blocks, seed):
    """Balanced AB/BA order, shuffled with a fixed seed: neither side systematically runs first."""
    import random
    orders = ["AB"] * ((n_blocks + 1) // 2) + ["BA"] * (n_blocks // 2)
    random.Random(seed).shuffle(orders)
    return orders


@dataclass(frozen=True)
class Block:
    """One paired block, raw: both sides ran `repeats` synchronised calls back to back."""
    index: int
    order: str                # "AB" or "BA": which side ran first
    repeats: int
    a_ns: int                 # elapsed nanoseconds for the A calls
    b_ns: int
    a_start_ns: int           # start offsets from the first timed block (monotonic clock)
    b_start_ns: int

    @property
    def a_s(self) -> float:
        """Normalised A latency: seconds per call."""
        return self.a_ns / 1e9 / self.repeats

    @property
    def b_s(self) -> float:
        return self.b_ns / 1e9 / self.repeats


@dataclass(frozen=True)
class PairedSession:
    """Raw result of one measurement session: A (incumbent or baseline) against B (candidate)."""
    protocol_hash: str
    repeats: int
    blocks: tuple[Block, ...]

    def to_json(self) -> dict:
        return {"protocol_hash": self.protocol_hash, "repeats": self.repeats,
                "blocks": [dataclasses.asdict(b) for b in self.blocks]}

    @classmethod
    def from_json(cls, data: dict) -> "PairedSession":
        return cls(data["protocol_hash"], int(data["repeats"]),
                   tuple(Block(**b) for b in data["blocks"]))


def measure_paired(a, b, args, protocol, *, repeats=None, clock=time.perf_counter_ns) -> PairedSession:
    """One paired session of `protocol.n_blocks` blocks in balanced, seeded AB/BA order.

    `a` and `b` should be compiled executables, so no compilation happens inside the timed window.
    Both are warmed up first. `repeats=None` calibrates from `a` (the baseline side); pass the frozen
    value to reuse it across sessions. Inputs must already be on the device.
    """
    from contracts import protocol_hash
    warm_up(a, args, protocol.warmup_calls)
    warm_up(b, args, protocol.warmup_calls)
    if repeats is None:
        repeats = calibrate_repeats(a, args, protocol, clock=clock)
    origin = clock()
    blocks = []
    for index, order in enumerate(block_order(protocol.n_blocks, protocol.order_seed)):
        first, second = (a, b) if order == "AB" else (b, a)
        s1, e1 = _run_block(first, args, repeats, clock)
        s2, e2 = _run_block(second, args, repeats, clock)
        (a_start, a_ns), (b_start, b_ns) = ((s1, e1), (s2, e2)) if order == "AB" else ((s2, e2), (s1, e1))
        blocks.append(Block(index, order, repeats, a_ns, b_ns, a_start - origin, b_start - origin))
    return PairedSession(protocol_hash(protocol), repeats, tuple(blocks))


def timeit_version_1(function, args, num_runs=30, num_tests=1):
    """Return the mean runtime per call across several timing batches."""
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")
    if num_tests <= 0:
        raise ValueError("num_tests must be positive.")

    times = []
    for _ in range(num_tests):
        # värm upp
        jax.block_until_ready(function(*args))   # works for PyTree outputs too

        elapsed = timeit.timeit(
            lambda: jax.block_until_ready(function(*args)),
            number=num_runs,
        )
        times.append(elapsed / num_runs)

    mean_runtime = statistics.mean(times)
    std_runtime = statistics.stdev(times) if len(times) > 1 else 0.0
    print(
        f"Average execution time: {mean_runtime * 1e6:.2f} µs "
        f"(std: {std_runtime * 1e6:.2f} µs)"
    )
    return mean_runtime, std_runtime


def evaluate(code, reference_func, args, num_runs=30, num_tests=5, function_name="kernel"):
    """Legacy single-input evaluation used by controller.py.

    The five keys (syntax_correct, error, output_correct, runtime_mean, runtime_std) keep their meaning.
    Since AIC-004 the comparison uses compare_outputs(): PyTree outputs are supported, and non-finite
    output is rejected even when the reference is non-finite too. A `correctness` key carries the
    size-bounded report (None when the comparison was not reached).
    """
    def failed(syntax_correct, error, correctness=None):
        return {
            "syntax_correct": syntax_correct,
            "error": error,
            "output_correct": False,
            "runtime_mean": None,
            "runtime_std": None,
            "correctness": correctness,
        }

    namespace = {}

    try:
        compiled_code = compile(code, "<generated>", "exec")
    except SyntaxError as error:
        return failed(False, str(error))

    try:
        exec(compiled_code, namespace)
        candidate = jax.jit(namespace[function_name])

        expected = jax.block_until_ready(reference_func(*args))
        actual = jax.block_until_ready(candidate(*args))

    except Exception as error:
        return failed(True, str(error))

    case_name = "legacy/args"
    try:
        case = replace(compare_outputs(expected, actual, LEGACY_TOLERANCE), case=case_name)
    except ReferenceInvalid:
        return failed(True, LEGACY_NONFINITE_REFERENCE)
    report = CorrectnessReport(
        passed=case.passed,
        category=case.category,
        policy_hash=policy_hash(LEGACY_TOLERANCE, [case_name], "legacy", None),
        cases=(case,),
    )

    if not case.passed:
        return failed(True, None, report.to_json())

    runtime_mean, runtime_std = timeit_version_1(
        candidate,
        args,
        num_runs=num_runs,
        num_tests=num_tests,
    )

    return {
        "syntax_correct": True,
        "error": None,
        "output_correct": True,
        "runtime_mean": runtime_mean,
        "runtime_std": runtime_std,
        "correctness": report.to_json(),
    }
