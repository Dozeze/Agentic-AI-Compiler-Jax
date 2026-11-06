"""Dual-backend rules (D-06): the same code and tests must run on CPU (Mac, CI) and on CUDA (NV host).

The rules are in AI_CONTEXT.md §7. This file covers the parts a machine can check:
  rule 1: best-effort AST guard: harness code never picks a backend itself (only environment.select_backend does);
  rule 2: --require-gpu turns the needs_gpu skip into an error (the skip itself is exercised by test_cuda_on_host);
  rule 3: demonstration: a Pallas candidate that picks interpret mode on CPU passes the correctness gate here;
  rule 5: best-effort AST guard: ordinary tests make no speed claims.
Each guard is first run on known-bad and known-good snippets, so it cannot pass by matching nothing.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from backends import cuda_status
from benchmark import build_cases, check_correctness, load_function
from workloads.registry import SOFTMAX

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
SCRATCH = {"runningscripts.py", "algorithm.py", "utils.py"}   # with root test*.py: scratch scripts (AI_CONTEXT §3)


# ---------------------------------------------------------------------------
# Rule 1: only environment.select_backend chooses the backend
# ---------------------------------------------------------------------------

def _dotted(node) -> str:
    """'jax.config.update' for the expression jax.config.update; '' for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


_BACKEND_CONFIG_KEYS = ("jax_platforms", "jax_platform_name", "jax_default_device")


def backend_choices(source: str) -> list[str]:
    """Return one line per place where `source` picks a backend or device by itself.

    Calls are matched by their last name segment, so aliases (`import jax as j`, `from jax import devices`)
    and wrappers (`partial(jax.jit, backend=...)`) are caught too. Best effort, not a proof.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and node.value == "JAX_PLATFORMS":
            found.append(f"line {node.lineno}: touches JAX_PLATFORMS")
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func) or "<call>"
        last = name.rsplit(".", 1)[-1]
        if last in ("devices", "local_devices", "get_backend") and (node.args or node.keywords):
            found.append(f"line {node.lineno}: {name}(...) with a backend argument")
        if last == "default_device":
            found.append(f"line {node.lineno}: {name}(...)")
        if last == "update" and node.args and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value in _BACKEND_CONFIG_KEYS:
            found.append(f"line {node.lineno}: {name}({node.args[0].value!r}, ...)")
        for keyword in node.keywords:
            if keyword.arg in ("backend", "device"):
                found.append(f"line {node.lineno}: {name}(..., {keyword.arg}=...)")
    return found


def _harness_files() -> list[Path]:
    files = [*REPO.glob("*.py"), *(REPO / "workloads").glob("*.py"), *(REPO / "scripts").glob("*.py")]
    return sorted(f for f in files if not f.name.startswith("test") and f.name not in SCRATCH
                  and f.name != "environment.py")


@pytest.mark.parametrize("snippet", [
    'import jax\njax.devices("gpu")',
    'import jax\njax.local_devices(backend="cuda")',
    'import jax\njax.config.update("jax_platforms", "cuda")',
    'import os\nos.environ["JAX_PLATFORMS"] = "cpu"',
    'import jax\nf = jax.jit(lambda x: x, backend="gpu")',
    'import jax\nfrom functools import partial\n@partial(jax.jit, backend="gpu")\ndef f(x):\n    return x',
    'import jax\nf = jax.jit(lambda x: x, device=d)',
    'import jax as j\nj.devices("gpu")',
    'from jax import devices\ndevices("gpu")',
    'import jax\njax.config.update("jax_default_device", d)',
    'import jax\nwith jax.default_device(d):\n    pass',
    'from jax.lib import xla_bridge\nxla_bridge.get_backend("gpu")',
])
def test_backend_guard_flags_hardcoded_choices(snippet):
    assert backend_choices(snippet)


@pytest.mark.parametrize("snippet", [
    "import jax\ndevices = jax.devices()",
    "import jax\nf = jax.jit(lambda x: x)",
    'import jax\njax.config.update("jax_enable_x64", False)',
])
def test_backend_guard_allows_backend_neutral_code(snippet):
    assert backend_choices(snippet) == []


def test_harness_code_does_not_pick_a_backend():
    files = _harness_files()
    assert {"agent.py", "benchmark.py", "controller.py", "probe_target.py"} <= {f.name for f in files}
    problems = [f"{f.relative_to(REPO)} {line}" for f in files
                for line in backend_choices(f.read_text(encoding="utf-8"))]
    assert problems == [], "select the backend with environment.select_backend instead:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# Rule 2: --require-gpu turns the needs_gpu skip into an error
# ---------------------------------------------------------------------------
# The plain skip needs no test of its own: without it, test_cuda_on_host fails on every CPU-only machine.

def test_require_gpu_errors_without_cuda():
    available, reason = cuda_status()
    if available:
        pytest.skip("CUDA is available: test_cuda_on_host runs for real in this session")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--require-gpu",
                           "tests/test_environment.py::test_cuda_on_host"],
                          cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0 and f"--require-gpu: {reason}" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# Rule 3: Pallas candidates choose interpret mode on CPU
# ---------------------------------------------------------------------------

PALLAS_SOFTMAX = '''import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _kernel(x_ref, o_ref):
    x = x_ref[...]
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)


def softmax(x):
    return pl.pallas_call(_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                          interpret={interpret})(x)
'''


@pytest.fixture(scope="module")
def dev_cases():
    # Both Pallas tests assume a CPU process; conftest forces JAX_PLATFORMS=cpu.
    assert jax.default_backend() == "cpu", "the test process must be pinned to CPU (tests/conftest.py)"
    return build_cases(SOFTMAX)


def test_portable_pallas_candidate_passes_on_cpu(dev_cases):
    source = PALLAS_SOFTMAX.format(interpret='jax.default_backend() == "cpu"')
    report = check_correctness(load_function(source, "softmax"), SOFTMAX, dev_cases)
    assert len(report.cases) == len(dev_cases) and report.passed, report.to_json()


def test_native_only_pallas_candidate_fails_to_compile_on_cpu(dev_cases):
    source = PALLAS_SOFTMAX.format(interpret="False")      # native Pallas is not supported on CPU
    report = check_correctness(load_function(source, "softmax"), SOFTMAX, dev_cases)
    assert len(report.cases) == len(dev_cases)
    assert {case.category for case in report.cases} == {"COMPILE_FAILED"}


# ---------------------------------------------------------------------------
# Rule 5: ordinary tests make no speed claims
# ---------------------------------------------------------------------------

_TIMING = re.compile(r"runtime|speedup|elapsed|latency|duration|perf_counter|(?:^|_)times?$|_(?:s|ns|us|ms)$")
_ORDERING = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_TIMING_MARKERS = {"slow", "needs_gpu"}


def _value_names(node):
    """Names that denote a value: variables, attributes and string subscript keys (r["runtime_mean"]).

    Arguments of len() are skipped: `len(durations) > 3` counts samples, it does not compare times.
    """
    if isinstance(node, ast.Call) and _dotted(node.func) == "len":
        return
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str):
        yield node.slice.value
    for child in ast.iter_child_nodes(node):
        yield from _value_names(child)


def _mentions_timing(node) -> bool:
    return any(_TIMING.search(name) and "timeout" not in name for name in _value_names(node))


def _is_presence_check(left, right) -> bool:
    """`value > 0` (a timing exists) on a single value, not on a difference or a call."""
    values = (ast.Name, ast.Attribute, ast.Subscript)
    zero = lambda n: isinstance(n, ast.Constant) and n.value == 0   # noqa: E731
    return (zero(left) and isinstance(right, values)) or (zero(right) and isinstance(left, values))


def _is_approx(node) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func).endswith("approx")


def _mark_names(exprs) -> set[str]:
    """{'slow'} for decorators or pytestmark values such as pytest.mark.slow, mark.slow or [pytest.mark.slow]."""
    names = set()
    for expr in exprs:
        for node in expr.elts if isinstance(expr, (ast.List, ast.Tuple)) else [expr]:
            name = _dotted(node.func if isinstance(node, ast.Call) else node)
            if name.startswith(("pytest.mark.", "mark.")):
                names.add(name.rsplit(".", 1)[-1])
    return names


def _test_functions(tree):
    """Yield (test function, its markers), including markers from its class and the module's pytestmark."""
    module = _mark_names(n.value for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            yield node, module | _mark_names(node.decorator_list)
        elif isinstance(node, ast.ClassDef):
            marks = module | _mark_names(node.decorator_list)
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    yield item, marks | _mark_names(item.decorator_list)


def timing_claims(source: str) -> list[str]:
    """Speed claims in asserts of unmarked tests: ordering, or == pytest.approx, between timing values.

    Each adjacent pair of a chained comparison is checked on its own. Best effort: asserts in helper
    functions and `if ...: raise AssertionError` are not inspected.
    """
    found = []
    for func, marks in _test_functions(ast.parse(source)):
        if marks & _TIMING_MARKERS:
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Assert):
                continue
            for cmp in (n for n in ast.walk(node.test) if isinstance(n, ast.Compare)):
                operands = [cmp.left, *cmp.comparators]
                for left, op, right in zip(operands, cmp.ops, operands[1:]):
                    claim = isinstance(op, _ORDERING) or (isinstance(op, (ast.Eq, ast.NotEq))
                                                          and (_is_approx(left) or _is_approx(right)))
                    if claim and (_mentions_timing(left) or _mentions_timing(right)) \
                            and not _is_presence_check(left, right):
                        found.append(f"{func.name} line {cmp.lineno}")
    return found


@pytest.mark.parametrize("snippet", [
    "def test_x(r):\n    assert r['runtime_mean'] < 5e-6",
    "def test_x(r):\n    assert r['speedup'] >= 1.1",
    "def test_x(new, old):\n    assert new.elapsed_s < old.elapsed_s",
    "def test_x(n, o):\n    assert 0 < n.runtime_mean < o.runtime_mean",
    "def test_x(n, o):\n    assert o.runtime_mean - n.runtime_mean > 0",
    "def test_x(new_time, old_time):\n    assert new_time < old_time",
    "def test_x(a_s, b_s):\n    assert a_s < b_s",
    "import pytest\ndef test_x(r):\n    assert r['speedup'] == pytest.approx(1.5, rel=0.1)",
])
def test_timing_guard_flags_speed_claims(snippet):
    assert timing_claims(snippet)


@pytest.mark.parametrize("snippet", [
    "def test_x(r):\n    assert r['runtime_mean'] > 0",
    "def test_x(r):\n    assert r['runtime_mean'] is None",
    "def test_x(r):\n    assert {'runtime_mean', 'runtime_std'} <= set(r)",
    "def test_x(durations):\n    assert len(durations) > 3",
    "def test_x(cfg):\n    assert cfg.timeout_ms >= 100",
    "import pytest\n@pytest.mark.slow\ndef test_x(r):\n    assert r['speedup'] >= 1.1",
    "import pytest\npytestmark = pytest.mark.slow\ndef test_x(r):\n    assert r['speedup'] >= 1.1",
    "import pytest\n@pytest.mark.needs_gpu\nclass TestX:\n    def test_x(self, r):\n        assert r['speedup'] >= 1.1",
    "from pytest import mark\n@mark.slow\ndef test_x(r):\n    assert r['speedup'] >= 1.1",
])
def test_timing_guard_allows_presence_checks_and_marked_tests(snippet):
    assert timing_claims(snippet) == []


def test_ordinary_tests_make_no_speed_claims():
    problems = [f"{f.name}: {claim}" for f in sorted(TESTS.glob("test_*.py"))
                for claim in timing_claims(f.read_text(encoding="utf-8"))]
    assert problems == [], "mark these tests slow or needs_gpu:\n" + "\n".join(problems)
