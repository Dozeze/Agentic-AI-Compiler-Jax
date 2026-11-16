"""Compiler feedback (AIC-010, AIC-1.1 §6): what JAX and XLA made of a program, as bounded, provenance-tagged data.

Runs inside the worker, on the selected target, for an already compiled executable. Every level of the
lowering pipeline is summarised so the agent can see what it wrote and what survived compilation:

  jaxpr      jax.make_jaxpr(fn)(*args)                    the program as written (sub-jaxprs walked)
  stablehlo  lowered.as_text()                            what JAX handed the compiler
  hlo        lowered.compiler_ir("hlo").as_hlo_text()     HLO before XLA's optimisation passes
  compiled   compiled.as_text()                           optimised HLO: fusions, surviving ops

The namespaces of AIC-1.1 §6 stay apart: `compiler_cost_estimates` and `compiler_memory_estimates` are XLA's
models, never measurements (zero temporary bytes from XLA is not zero process memory); runtime measurements
live only in benchmark sessions; `capture_runtime_trace` collects observed events separately. A value XLA
does not report is null with a reason in `unavailable`, never zero. Counts parsed from text are indicative:
not a semantic analysis and no proof that a fusion pays off.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shutil
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from contracts import RuntimeTrace

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
           "runtime_trace": {"coverage": "UNAVAILABLE", "reason": "profiling is collected separately from compiler analysis"},
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


# Runtime profiling (AIC-011). Raw compressed and decoded data both have independent bounds.
MAX_TRACE_BYTES = 8 << 20
MAX_TRACE_DECODED_BYTES = 32 << 20
MAX_TRACE_EVENTS = 100_000
MAX_PROFILE_LANES = 16
PROFILE_CALLS = 5
PROFILE_ANNOTATION = "aic_profile_call"
_DURATION_NOTE = "inclusive interval union on this lane; names and lanes may overlap; no percentages"


def unavailable_trace(reason, provenance=None, **fields):
    return RuntimeTrace("UNAVAILABLE", str(reason)[:500], provenance or {}, **fields).to_json()


def _union(intervals):
    """Non-overlapping intervals covering the input; never sum nested or overlapping durations."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in _union(intervals))


def _lane_id(value):
    return isinstance(value, (int, str)) and not isinstance(value, bool) and len(str(value)) <= 80


def _timestamp(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _matches_device(match, provenance):
    kind, ordinal = match.groups()
    platforms = {"gpu", "cuda", "rocm"} if kind == "GPU" else {"tpu"}
    if provenance.get("backend") not in platforms:
        return False
    devices = provenance.get("devices")
    return devices is None or any(d.get("platform") in platforms and d.get("id") == int(ordinal) for d in devices)


def summarize_trace(data, provenance):
    """Summarize bounded Perfetto complete events within the harness's annotated calls.

    Device coverage requires explicit device process metadata and actual events on that process. CPU
    executable events are recognized only for a CPU backend. Host dispatch never establishes device
    execution. Unsupported phases are counted and omitted rather than guessed into execution intervals.
    """
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list) or len(events) > MAX_TRACE_EVENTS:
        return unavailable_trace("traceEvents must be a list within the event limit", provenance)
    processes, threads, complete = {}, {}, []
    invalid = ignored = 0
    for event in events:
        if not isinstance(event, dict):
            invalid += 1
            continue
        phase, name = event.get("ph"), event.get("name")
        pid, tid = event.get("pid"), event.get("tid")
        if phase == "M" and _lane_id(pid):
            args = event.get("args")
            label = args.get("name") if isinstance(args, dict) else None
            if isinstance(label, str):
                if name == "process_name":
                    processes[pid] = label[:200]
                elif name == "thread_name" and _lane_id(tid):
                    threads[pid, tid] = label[:200]
            continue
        if phase != "X":
            ignored += 1
            continue
        start, duration = event.get("ts"), event.get("dur")
        if not (isinstance(name, str) and _lane_id(pid) and _lane_id(tid)
                and _timestamp(start) and _timestamp(duration) and _timestamp(start + duration)):
            invalid += 1
            continue
        complete.append((pid, tid, name[:200], float(start), float(start + duration)))
    annotations = [e for e in complete if e[2] == PROFILE_ANNOTATION]
    windows = _union((e[3], e[4]) for e in annotations if e[4] > e[3])
    if not windows:
        return unavailable_trace("no valid complete harness call annotations; execution scope is unknown",
                                 provenance, n_events=len(events), invalid_events=invalid, ignored_events=ignored)
    grouped = defaultdict(list)
    names = defaultdict(list)
    counts = Counter()
    for pid, tid, name, start, end in complete:
        if name == PROFILE_ANNOTATION:
            continue
        process = processes.get(pid, "")
        device = re.search(r"/device:(GPU|TPU):(\d+)", process)
        if device:
            kind = "device_execution" if _matches_device(device, provenance) else "unattributed_device"
        elif provenance.get("backend") == "cpu" and "CpuExecutable::Execute" in name:
            kind = "cpu_execution"
        elif any(label in name for label in ("PjitFunction", "PjRt", "Execute", "Dispatch")):
            kind = "host_dispatch"
        else:
            kind = "host_other"
        clipped = [(max(start, lo), min(end, hi)) for lo, hi in windows if max(start, lo) < min(end, hi)]
        if clipped:
            key = (pid, tid, kind)
            grouped[key].extend(clipped)
            names[key + (name,)].extend(clipped)
            counts[key + (name,)] += 1
    kinds = {key[2] for key in grouped}
    if "device_execution" in kinds:
        coverage, reason = "DEVICE_EXECUTION_EVENTS", "execution events observed on explicitly named device lanes"
        hotspot_kind = "device_execution"
    elif "cpu_execution" in kinds:
        coverage, reason = "CPU_EXECUTION_EVENTS", "CPU executable events observed; no per-HLO-op attribution"
        hotspot_kind = "cpu_execution"
    else:
        coverage, reason = "HOST_ONLY", "no attributable CPU or device execution events within the annotated calls"
        hotspot_kind = None
    lanes = []
    for (pid, tid, kind), intervals in grouped.items():
        overlapping = nested = 0
        last_end = -1.0
        for start, end in sorted(intervals, key=lambda v: (v[0], -v[1])):
            overlapping += start < last_end
            nested += end <= last_end
            last_end = max(last_end, end)
        lanes.append({"pid": pid, "tid": tid, "kind": kind, "process": processes.get(pid),
                      "thread": threads.get((pid, tid)), "inclusive_union_us": _duration(intervals),
                      "overlapping_intervals": overlapping, "nested_or_duplicate_intervals": nested,
                      "duration_semantics": _DURATION_NOTE})
    lanes.sort(key=lambda lane: (lane["kind"], str(lane["pid"]), str(lane["tid"])))
    hotspots = [{"pid": pid, "tid": tid, "kind": kind, "name": name, "count": counts[pid, tid, kind, name],
                 "inclusive_union_us": _duration(intervals), "duration_semantics": _DURATION_NOTE}
                for (pid, tid, kind, name), intervals in names.items() if kind == hotspot_kind]
    hotspots.sort(key=lambda event: (-event["inclusive_union_us"], str(event["pid"]), str(event["tid"]), event["name"]))
    return RuntimeTrace(coverage, reason, provenance,
                        annotation={"name": PROFILE_ANNOTATION, "calls": len(annotations),
                                    "inclusive_union_us": _duration(windows),
                                    "intervals_us": [list(w) for w in windows[:PROFILE_CALLS]],
                                    "duration_semantics": "union of harness call intervals; profiled, not benchmark latency"},
                        lanes=tuple(lanes[:MAX_PROFILE_LANES]), hotspots=tuple(hotspots[:TOP_OPS]),
                        n_events=len(events), invalid_events=invalid, ignored_events=ignored,
                        omitted_lanes=max(0, len(lanes) - MAX_PROFILE_LANES)).to_json()


def read_runtime_trace(path, provenance):
    """Read bounded gzip JSON, including a decoded-size bound against compressed expansion."""
    try:
        path = Path(path)
        if path.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("compressed trace exceeds the byte limit")
        with gzip.open(path, "rb") as stream:
            raw = stream.read(MAX_TRACE_DECODED_BYTES + 1)
        if len(raw) > MAX_TRACE_DECODED_BYTES:
            raise ValueError("decoded trace exceeds the byte limit")
        return summarize_trace(json.loads(raw), provenance)
    except (OSError, EOFError, UnicodeError, ValueError, RecursionError, zlib.error) as error:
        return unavailable_trace(f"cannot parse trace: {type(error).__name__}: {error}", provenance)


def capture_runtime_trace(jax, executable, args, *, artifacts, provenance):
    """Profile five synchronized calls of an already compiled executable, after five warmups.

    Called only by a separate profile worker, never from measurement code. Keep one bounded raw
    Perfetto artifact; discard duplicate exporter files. Profiler failure is optional evidence failure.
    """
    from benchmark import warm_up
    provenance = {**provenance, "jax": jax.__version__, "backend": jax.default_backend(),
                  "signature": [f"{a.dtype}{list(a.shape)}" for a in args],
                  "devices": [{"id": d.id, "platform": d.platform, "device_kind": d.device_kind}
                              for d in jax.devices()], "warmup_calls": 5, "profile_calls": PROFILE_CALLS}
    artifact = artifact_hash = None
    try:
        jax.block_until_ready(args)
        warm_up(executable, args, 5)
        with tempfile.TemporaryDirectory(prefix="aic-profile-") as temporary:
            with jax.profiler.trace(temporary, create_perfetto_link=False, create_perfetto_trace=True):
                for _ in range(PROFILE_CALLS):
                    with jax.profiler.TraceAnnotation(PROFILE_ANNOTATION):
                        jax.block_until_ready(executable(*args))
            traces = list(Path(temporary).rglob("perfetto_trace.json.gz"))
            if len(traces) != 1:
                raise ValueError(f"expected one Perfetto trace, found {len(traces)}")
            if traces[0].stat().st_size > MAX_TRACE_BYTES:
                raise ValueError("compressed trace exceeds the byte limit; raw artifact omitted")
            artifacts = Path(artifacts)
            artifacts.mkdir(parents=True, exist_ok=True)
            destination = artifacts / "perfetto_trace.json.gz"
            shutil.copyfile(traces[0], destination)
            artifact = f"{artifacts.name}/{destination.name}"
            artifact_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            summary = read_runtime_trace(destination, provenance)
            return {**summary, "artifact": artifact, "artifact_sha256": artifact_hash}
    except Exception as error:   # noqa: BLE001 - unsupported profilers must not become candidate failures
        return unavailable_trace(f"profile capture failed: {type(error).__name__}: {error}", provenance,
                                 artifact=artifact, artifact_sha256=artifact_hash)
