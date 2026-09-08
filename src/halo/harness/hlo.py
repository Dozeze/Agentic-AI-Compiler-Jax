"""Turn a compiled XLA executable into a summary small enough to put in a prompt.

The raw optimized HLO for even a toy attention kernel is ~700 lines; pasting it
wastes context and buries the signal. What actually distinguishes a good lowering
from a bad one is compact: how many fusions XLA emitted, how many `dot`s survived,
how much scratch memory the schedule needs, and the arithmetic intensity.
"""

from __future__ import annotations

import re
from collections import Counter

from halo.types import HloSummary

#: Matches `%name = f32[...]{...} opcode(operands)`, with or without a ROOT prefix.
_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?[%\w.\-]+ = \S+ ([a-z][a-z0-9-]*)\(")


def parse_op_counts(hlo_text: str) -> Counter[str]:
    return Counter(
        match.group(1)
        for line in hlo_text.splitlines()
        if (match := _INSTRUCTION.match(line))
    )


def _cost(compiled) -> tuple[float | None, float | None]:
    """FLOPs and bytes accessed, if this XLA build reports them."""
    try:
        analysis = compiled.cost_analysis()
    except Exception:  # noqa: BLE001 - availability varies by backend and version
        return None, None
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0] if analysis else {}
    if not isinstance(analysis, dict):
        return None, None
    return analysis.get("flops"), analysis.get("bytes accessed")


def _memory(compiled) -> tuple[int | None, int | None, int | None]:
    try:
        stats = compiled.memory_analysis()
    except Exception:  # noqa: BLE001
        return None, None, None
    if stats is None:
        return None, None, None
    return (
        getattr(stats, "temp_size_in_bytes", None),
        getattr(stats, "output_size_in_bytes", None),
        getattr(stats, "argument_size_in_bytes", None),
    )


def summarize(compiled, hlo_text: str) -> HloSummary:
    counts = parse_op_counts(hlo_text)
    flops, bytes_accessed = _cost(compiled)
    temp, output, argument = _memory(compiled)
    return HloSummary(
        instruction_count=sum(counts.values()),
        fusion_count=counts.get("fusion", 0),
        op_counts=dict(counts),
        flops=flops,
        bytes_accessed=bytes_accessed,
        temp_bytes=temp,
        output_bytes=output,
        argument_bytes=argument,
    )
