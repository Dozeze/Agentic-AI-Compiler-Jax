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
from halo.types import HloSummary, Measurement

SYSTEM = """\
You are a performance engineer optimizing JAX code that is compiled by XLA.

You are given one function, its exact semantics, its current runtime, and a summary
of the HLO that XLA produced for it. You propose one rewrite at a time.

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


def _format_hlo(summary: HloSummary | None, label: str) -> str:
    if summary is None:
        return f"{label}: unavailable"
    lines = [
        f"{label}:",
        f"  instructions      {summary.instruction_count}",
        f"  fusions           {summary.fusion_count}",
        f"  top ops           "
        + ", ".join(f"{op}={n}" for op, n in summary.top_ops(10)),
    ]
    if summary.flops is not None:
        lines.append(f"  flops             {summary.flops:,.0f}")
    if summary.bytes_accessed is not None:
        lines.append(f"  bytes accessed    {summary.bytes_accessed:,.0f}")
    if (intensity := summary.arithmetic_intensity) is not None:
        lines.append(
            f"  arithmetic int.   {intensity:.2f} flop/byte "
            f"({'compute' if intensity > 10 else 'memory'}-leaning)"
        )
    if summary.temp_bytes is not None:
        lines.append(f"  scratch memory    {summary.temp_bytes:,} bytes")
    return "\n".join(lines)


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


def render(context: Context, *, min_speedup: float) -> str:
    return f"""\
## Task: {context.task_name}

{context.task_description}

## Exact semantics

This is the oracle the result is graded against. It is read-only reference material,
written in NumPy float64; it is not the code you are optimizing and you cannot change it.

```python
{context.reference_source}
```

## Current implementation (candidate.py)

```python
{context.current_source}
```

## Measurement

{_format_measurement(context.measurement)}

## Compiler feedback

{_format_hlo(context.measurement.hlo, "Optimized HLO for the current implementation")}

{_format_hlo(context.measurement.baseline_hlo, "Optimized HLO for the seed implementation")}
{_format_history(context)}

## Your task

Propose ONE rewrite of `candidate.py` that runs faster on this device while
computing the same result.

Constraints:
- Return the COMPLETE new contents of `candidate.py`, not a diff or a fragment.
- It must define `candidate(...)` with exactly the same signature and semantics.
- Import only from `jax`, `jax.numpy` and the standard library. No new dependencies.
- It must be traceable by `jax.jit`: no Python control flow on array values, no
  `.item()`, no printing, no host callbacks. Shapes are static and given above, so
  you may specialise on them.
- It must remain numerically stable for large-magnitude inputs.
- To be accepted, the {int(100 * 0.95)}% lower bound of the measured speedup must exceed
  {min_speedup:.2f}x. Changes smaller than that are indistinguishable from noise.
"""
