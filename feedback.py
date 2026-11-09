"""Compiler feedback (AIC-010, AIC-1.1 §6): what JAX and XLA made of a program, as bounded, provenance-tagged data.

Runs inside the worker, on the selected target, for an already compiled executable. Every level of the
lowering pipeline is summarised so the agent can see what it wrote and what survived compilation:

  jaxpr      jax.make_jaxpr(fn)(*args)                    the program as written (sub-jaxprs walked)
  stablehlo  lowered.as_text()                            what JAX handed the compiler
  hlo        lowered.compiler_ir("hlo").as_hlo_text()     HLO before XLA's optimisation passes
  compiled   compiled.as_text()                           optimised HLO: fusions, surviving ops

The namespaces of AIC-1.1 §6 stay apart: `compiler_cost_estimates` and `compiler_memory_estimates` are XLA's
models, never measurements (zero temporary bytes from XLA is not zero process memory); runtime measurements
live only in benchmark sessions; `runtime_trace` is not collected here (coverage UNAVAILABLE). A value XLA
does not report is null with a reason in `unavailable`, never zero. Counts parsed from text are indicative:
not a semantic analysis and no proof that a fusion pays off.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

PARSER_VERSION = "ir_summary/v1"
MAX_ARTIFACT_BYTES = 1 << 20          # per raw text artifact kept for inspection
TOP_OPS = 10

# `%name = f32[...]{...} opcode(operands)`, with or without ROOT; tuple shapes are a single token too.
_HLO_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?[%\w.\-]+ = (?:\([^=]*?\)|\S+) ([a-z][a-z0-9-]*)\(")
_STABLEHLO_OP = re.compile(r"=\s*\"?stablehlo\.([a-z_0-9]+)")

COST_KEYS = ("flops", "transcendentals", "bytes accessed", "optimal_seconds")
MEMORY_FIELDS = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes",
                 "generated_code_size_in_bytes", "host_temp_size_in_bytes")


def group_of(op: str) -> str | None:
    """The plan's coarse groups, across the vocabularies of the four levels."""
    name = op.replace("-", "_")
    if name in ("dot", "dot_general", "convolution", "conv_general_dilated"):
        return "dots"
    if name.startswith("reduce") or name in ("argmax", "argmin", "cumsum", "cumprod", "cummax", "sort"):
        return "reductions"
    if name == "transpose":
        return "transposes"
    if name.startswith("broadcast"):
        return "broadcasts"
    if name in ("custom_call",):
        return "custom_calls"
    if name == "fusion":
        return "fusions"
    return None


def jaxpr_counts(closed_jaxpr) -> Counter:
    """Primitive counts, descending into the sub-jaxprs of scan, while, cond, pjit and custom rules."""
    counts: Counter = Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts[str(eqn.primitive)] += 1
            for param in eqn.params.values():
                for sub in param if isinstance(param, (list, tuple)) else (param,):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    walk(closed_jaxpr.jaxpr)
    return counts


def stablehlo_counts(text: str) -> Counter:
    return Counter(m.group(1) for m in _STABLEHLO_OP.finditer(text))


def hlo_counts(text: str) -> Counter:
    return Counter(m.group(1) for line in text.splitlines() if (m := _HLO_INSTRUCTION.match(line)))


def level_summary(text: str, counts: Counter) -> dict:
    groups = Counter()
    for op, n in counts.items():
        if (group := group_of(op)) is not None:
            groups[group] += n
    return {"lines": text.count("\n") + 1, "total_ops": sum(counts.values()),
            "groups": {g: groups.get(g, 0) for g in ("dots", "reductions", "transposes", "broadcasts",
                                                     "custom_calls", "fusions")},
            "top_ops": dict(counts.most_common(TOP_OPS))}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def cost_estimates(executable):
    analysis = executable.cost_analysis()
    if isinstance(analysis, (list, tuple)):          # JAX 0.4.38 returns a list with one dict per program
        analysis = analysis[0] if analysis else None
    if not isinstance(analysis, dict):
        return None
    return {key: _finite(analysis.get(key)) for key in COST_KEYS}


def memory_estimates(executable):
    stats = executable.memory_analysis()
    if stats is None:
        return None
    return {name: getattr(stats, name, None) for name in MEMORY_FIELDS}


def analyze(jax, fn, args, compiled, *, artifacts: Path | None = None) -> dict:
    """Feedback for `fn` compiled for `args` (a benchmark.Compiled). Every level is optional."""
    out = {"parser_version": PARSER_VERSION, "levels": {}, "compiler_cost_estimates": None,
           "compiler_memory_estimates": None,
           "runtime_trace": {"coverage": "UNAVAILABLE", "reason": "profiler traces are not collected yet (AIC-011)"},
           "provenance": {"jax": jax.__version__, "backend": jax.default_backend(),
                          "signature": [f"{a.dtype}{list(a.shape)}" for a in args]},
           "unavailable": {}}
    levels = {
        "jaxpr": lambda: (lambda c: (str(c), jaxpr_counts(c)))(jax.make_jaxpr(fn)(*args)),
        "stablehlo": lambda: (lambda t: (t, stablehlo_counts(t)))(compiled.lowered.as_text()),
        "hlo": lambda: (lambda t: (t, hlo_counts(t)))(compiled.lowered.compiler_ir("hlo").as_hlo_text()),
        "compiled": lambda: (lambda t: (t, hlo_counts(t)))(compiled.executable.as_text()),
    }
    for level, extract in levels.items():
        try:
            text, counts = extract()
        except Exception as exc:   # noqa: BLE001 - availability varies by backend and version
            out["unavailable"][level] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        out["levels"][level] = level_summary(text, counts)
        if artifacts is not None:
            artifacts.mkdir(parents=True, exist_ok=True)
            data = text.encode("utf-8")[:MAX_ARTIFACT_BYTES]
            (artifacts / f"{level}.txt").write_bytes(data)
            out["levels"][level]["artifact"] = f"{artifacts.name}/{level}.txt"
            out["levels"][level]["truncated"] = len(text.encode("utf-8")) > MAX_ARTIFACT_BYTES
    for key, extract in (("compiler_cost_estimates", cost_estimates),
                         ("compiler_memory_estimates", memory_estimates)):
        try:
            out[key] = extract(compiled.executable)
        except Exception as exc:   # noqa: BLE001
            out["unavailable"][key] = f"{type(exc).__name__}: {exc}"[:300]
        if out[key] is None and key not in out["unavailable"]:
            out["unavailable"][key] = "not reported by this backend"
    return out
