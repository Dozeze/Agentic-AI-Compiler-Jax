"""Turning a measurement into a prompt.

This module is the part of the project the assignment actually singles out:
"language models do not always know much about a given accelerator or about
compiler behavior, so part of the task is giving the agent the right context and
feedback signals". Everything here is a choice about what earns its place in the
context window - which is why the HLO arrives as a summary and never as its 700
raw lines.
"""

from __future__ import annotations

from halo.agent.protocol import Context
from halo.types import BufferReport, HloSummary, Measurement, ProgramSummary

#: Ordered from least to most information. Each level includes the ones before it.
#: The point of the ordering is the ablation: holding everything else fixed and
#: varying only this is how we find out whether compiler feedback actually helps.
LEVELS = ("code", "timing", "hlo", "full")


SYSTEM = """\
You are a performance engineer optimizing JAX code that is compiled by XLA.

You are given one function and its exact semantics, together with whatever
measurements and compiler feedback are available. You propose one rewrite at a time.

What actually wins on this kind of code, roughly in order:
- Removing Python-level loops that trace into many separate operations, so XLA sees
  one batched operation it can block and vectorise instead of N small ones.
- Letting a single einsum or dot express what several reshapes, transposes and
  matmuls were doing, so the compiler can pick the layout.
- Removing materialised intermediates that force scratch allocations.
- Avoiding recomputation of values that do not depend on the reduction axis.

XLA already fuses elementwise chains and does layout assignment on its own. Rewriting
well-fused elementwise code rarely helps; if the HLO summary shows a small
instruction count and few fusions, say so honestly in your analysis and expect a
smaller win.

Numerical correctness is checked before speed, against a float64 NumPy oracle, on
adversarial inputs you cannot see - including inputs large enough that a softmax
without max-subtraction overflows float32. A faster but less stable rewrite is
rejected outright.
"""


def _format_ops(stats, limit: int = 8) -> str:
    return ", ".join(f"{op}={n}" for op, n in stats.top_ops(limit))


def _format_buffers(report: BufferReport) -> list[str]:
    """What XLA actually materialised, by shape.

    Scalars are dropped: a report is dominated by 4-byte constants that no rewrite
    can remove, and they crowd out the shapes that matter.
    """
    shapes = [
        f"{shape}x{count}"
        for shape, count in report.shape_counts.items()
        if not shape.endswith("[]")
    ][:6]
    lines = [f"     live buffers      {report.total_bytes:,} bytes total"]
    if shapes:
        lines.append(f"     by shape          {', '.join(shapes)}")
    return lines


def _format_program(program: ProgramSummary, level: str) -> str:
    """The same program at every level, so the reader can see what XLA changed."""
    full = level == "full"
    lines: list[str] = []
    if full and program.jaxpr is not None:
        lines.append(f"  1. jaxpr, what you wrote          {program.jaxpr.total_ops:>4} ops")
        lines.append(f"     {_format_ops(program.jaxpr)}")
    if full and program.stablehlo is not None:
        lines.append(f"  2. StableHLO, handed to XLA       {program.stablehlo.total_ops:>4} ops")

    summary = program.hlo
    if summary is not None:
        lines.append(
            f"  3. optimized HLO, what XLA built  {summary.instruction_count:>4} instructions"
            f" in {summary.fusion_count} fusion(s)"
        )
        lines.append(f"     {', '.join(f'{op}={n}' for op, n in summary.top_ops(8))}")
        if summary.flops is not None and summary.bytes_accessed is not None:
            intensity = summary.arithmetic_intensity
            lines.append(
                f"     flops {summary.flops:,.0f} | bytes accessed "
                f"{summary.bytes_accessed:,.0f}"
                + (
                    f" | {intensity:.2f} flop/byte "
                    f"({'compute' if intensity > 10 else 'memory'}-leaning)"
                    if intensity is not None
                    else ""
                )
            )
        if summary.temp_bytes is not None:
            lines.append(f"     scratch memory    {summary.temp_bytes:,} bytes")

    if full and program.buffers is not None:
        lines.extend(_format_buffers(program.buffers))

    return "\n".join(lines) if lines else "  unavailable"


_PREAMBLE = {
    "hlo": "This is what XLA produced for your code.",
    "full": (
        "Your code passes through three levels before it runs. Comparing them shows "
        "what XLA already fixed on its own, and what it could not - the operations "
        "that survive to level 3 are the ones a source rewrite still has leverage over."
    ),
}


def _format_compiler_feedback(measurement: Measurement, level: str) -> str:
    current = measurement.candidate_program
    baseline = measurement.baseline_program
    if current is None:
        return "Compiler feedback unavailable."

    blocks = [
        _PREAMBLE[level],
        "",
        "### Current implementation",
        "",
        _format_program(current, level),
    ]
    if baseline is not None and baseline != current:
        blocks += ["", "### Seed implementation, for comparison", "",
                   _format_program(baseline, level)]
    elif baseline is not None:
        blocks += ["", "(The current implementation is still the seed; there is no second "
                   "program to compare against yet.)"]
    return "\n".join(blocks)


def _format_measurement(measurement: Measurement) -> str:
    device = measurement.device
    lines = []
    if device is not None:
        lines.append(
            f"Device: {device.platform} ({device.device_kind}), "
            f"{device.device_count} device(s), JAX {device.jax_version}"
        )
    if measurement.candidate is not None:
        lines.append(
            f"Current runtime: median {measurement.candidate.median_ms:.3f} ms "
            f"(IQR {measurement.candidate.iqr_ms:.3f} ms, "
            f"min {measurement.candidate.min_ms:.3f} ms)"
        )
    if measurement.baseline is not None:
        lines.append(
            f"Seed runtime:    median {measurement.baseline.median_ms:.3f} ms"
        )
    if measurement.speedup is not None:
        lines.append(f"Current speedup over seed: {measurement.speedup.describe()}")
    if measurement.compile_s is not None:
        lines.append(f"Compile time: {measurement.compile_s:.3f} s")
    return "\n".join(lines)


def _format_history(context: Context) -> str:
    if not context.history:
        return ""
    blocks = ["", "## Previous attempts", ""]
    for item in context.history:
        verdict = "ACCEPTED" if item.accepted else f"REJECTED ({item.rule})"
        blocks.append(f"### Attempt {item.index} - {verdict}")
        blocks.append(f"Reason: {item.reason}")
        blocks.append("Diff against the seed implementation:")
        blocks.append(f"```diff\n{item.diff}```")
        blocks.append("")
    blocks.append(
        "Do not repeat a rewrite that was already rejected. If a previous attempt "
        "failed on correctness, address that specific failure."
    )
    return "\n".join(blocks)


def render(context: Context, *, min_speedup: float, level: str = "full") -> str:
    """Build the prompt, including only the sections ``level`` admits.

    ``code`` is the control: the function and its semantics, nothing measured.
    ``timing`` adds runtimes, ``hlo`` adds what XLA produced, ``full`` adds the
    levels above it and the buffer report.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown context level {level!r}; choose from {LEVELS}")

    sections = [
        f"## Task: {context.task_name}",
        "",
        context.task_description,
        "",
        "## Exact semantics",
        "",
        "This is the oracle the result is graded against. It is read-only reference "
        "material,\nwritten in NumPy float64; it is not the code you are optimizing "
        "and you cannot change it.",
        "",
        f"```python\n{context.reference_source}\n```",
        "",
        "## Current implementation (candidate.py)",
        "",
        f"```python\n{context.current_source}\n```",
    ]

    if level != "code":
        sections += ["", "## Measurement", "", _format_measurement(context.measurement)]
    if level in ("hlo", "full"):
        sections += ["", "## Compiler feedback", "",
                     _format_compiler_feedback(context.measurement, level)]

    history = _format_history(context)
    if history:
        sections.append(history)

    sections += [
        "",
        "## Your task",
        "",
        "Propose ONE rewrite of `candidate.py` that runs faster on this device while",
        "computing the same result.",
        "",
        "Constraints:",
        "- Return the COMPLETE new contents of `candidate.py`, not a diff or a fragment.",
        "- It must define `candidate(...)` with exactly the same signature and semantics.",
        "- Import only from `jax`, `jax.numpy` and the standard library. No new dependencies.",
        "- It must be traceable by `jax.jit`: no Python control flow on array values, no",
        "  `.item()`, no printing, no host callbacks. Shapes are static and given above, so",
        "  you may specialise on them.",
        "- It must remain numerically stable for large-magnitude inputs.",
        "- To be accepted, the 95% lower bound of the measured speedup must exceed",
        f"  {min_speedup:.2f}x. Changes smaller than that are indistinguishable from noise.",
        "- If the code is already optimal, return it unchanged and say so in your analysis.",
        "",
    ]
    return "\n".join(sections)
