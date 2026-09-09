"""Parse XLA's buffer-assignment report into a summary the agent can act on.

``compiled.memory_analysis()`` gives one number for scratch memory. That number
says *how much* was materialised but not *what*, and "what" is the actionable
part: a report naming sixteen live ``f32[4,256,256]`` score matrices points
straight at the per-head loop that created them.

XLA writes this report only as a side effect of ``--xla_dump_to``, which the
runner sets for the measurement subprocess. Format (CPU backend, JAX 0.11):

    Memory Space: default (color=0)
    Total bytes: 19661004 (18.75MiB)
      cumulative_size;       size;       offset; used_by_n_values; shapes_list
      ------------------------------------------------------------
        1.00MiB(  7%);    1.00MiB;      4194304;   3; f32[4,256,64], 2xf32[4,256,256]

Several values may share one offset when XLA reuses a buffer, so ``size`` belongs
to the slot rather than to any single shape.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from halo.types import BufferEntry, BufferReport

_TOTAL = re.compile(r"^Total bytes:\s*(\d+)", re.MULTILINE)
_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
_SIZE = re.compile(r"^([\d.]+)\s*(B|KiB|MiB|GiB)$")
#: A shape token, optionally preceded by a count XLA writes with U+00D7, as in
#: "f32[4,256,64], 2xf32[4,256,256]". Matched whole rather than split on commas,
#: because the shapes contain commas of their own.
_SHAPE = re.compile(r"(?:(\d+)\s*[×x]\s*)?([a-z]\w*\[[^\]]*\])")


def _parse_size(text: str) -> int | None:
    match = _SIZE.match(text.strip())
    if match is None:
        return None
    return int(float(match.group(1)) * _UNITS[match.group(2)])


def _parse_shapes(text: str) -> list[tuple[str, int]]:
    """"f32[4,256,64], 2xf32[4,256,256]" -> [("f32[4,256,64]", 1), (..., 2)]"""
    return [
        (match.group(2), int(match.group(1) or 1))
        for match in _SHAPE.finditer(text)
    ]


def parse(text: str, *, top_n: int = 8) -> BufferReport:
    total = sum(int(m.group(1)) for m in _TOTAL.finditer(text))

    entries: list[BufferEntry] = []
    shape_counts: Counter[str] = Counter()
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(";")]
        if len(fields) != 5:
            continue
        size = _parse_size(fields[1])
        if size is None or not fields[3].isdigit():
            continue
        shapes = _parse_shapes(fields[4])
        for shape, count in shapes:
            shape_counts[shape] += count
        entries.append(
            BufferEntry(
                size_bytes=size,
                n_values=int(fields[3]),
                shapes=tuple(shape for shape, _ in shapes),
            )
        )

    entries.sort(key=lambda e: -e.size_bytes)
    return BufferReport(
        total_bytes=total,
        entries=tuple(entries[:top_n]),
        shape_counts=dict(shape_counts.most_common()),
    )


def read(dump_dir: Path, module_name: str, *, top_n: int = 8) -> BufferReport | None:
    """Find and parse the report XLA wrote for ``jit_<module_name>``.

    Returns None rather than raising: a missing report degrades the agent's
    context but must never fail a measurement.
    """
    pattern = f"*jit_{module_name}.*memory-usage-report.txt"
    matches = sorted(dump_dir.glob(pattern))
    if not matches:
        return None
    return parse(matches[0].read_text(), top_n=top_n)
