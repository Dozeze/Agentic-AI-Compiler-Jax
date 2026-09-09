"""Parsing XLA's buffer-assignment report."""

from __future__ import annotations

from halo.harness import buffers

# Trimmed from a real CPU dump for the seed attention implementation.
REPORT = """\
Memory Space: default (color=0)
Total bytes: 19661004 (18.75MiB)
  cumulative_size;       size;       offset; used_by_n_values; shapes_list
  ------------------------------------------------------------
    1.00MiB(  7%);    1.00MiB;      4194304;                3; f32[4,256,64], 2×f32[4,256,256]
    2.00MiB( 15%);    1.00MiB;      1048576;                3; f32[4,1,256,64], 2×f32[4,256,256]
    8.25MiB( 61%);   256.0KiB;     10747904;                1; f32[4,1,256,64]
   13.50MiB(100%);     4.0KiB;     10489856;                2; 2×f32[4,256]

  cumulative_size;       size;       offset; used_by_n_values; shapes_list
  ------------------------------------------------------------
    2.00MiB(100%);    2.00MiB;            0;                1; f32[4,8,256,64]

  cumulative_size;       size;       offset; used_by_n_values; shapes_list
  ------------------------------------------------------------
         4B(100%);         4B;            0;                1; f32[]
"""


def test_total_bytes():
    assert buffers.parse(REPORT).total_bytes == 19661004


def test_shapes_containing_commas_are_not_split():
    """The regression that made this parser useless: 'f32[4,256,256]' shredded
    into 'f32[4', '256', '256]' by a naive split on commas."""
    counts = buffers.parse(REPORT).shape_counts
    assert counts["f32[4,256,256]"] == 4  # 2 rows x the "2x" multiplier
    assert counts["f32[4,1,256,64]"] == 2
    assert counts["f32[4,8,256,64]"] == 1
    assert not any("[" in shape and "]" not in shape for shape in counts)


def test_multiplier_prefix_counts_repeated_values():
    assert buffers.parse(REPORT).shape_counts["f32[4,256]"] == 2


def test_entries_are_largest_first_with_units_resolved():
    entries = buffers.parse(REPORT).entries
    assert [e.size_bytes for e in entries] == [
        2 * 1024**2, 1024**2, 1024**2, 256 * 1024, 4 * 1024, 4
    ]
    assert entries[0].shapes == ("f32[4,8,256,64]",)


def test_top_n_caps_the_entry_list_but_not_the_shape_histogram():
    report = buffers.parse(REPORT, top_n=2)
    assert len(report.entries) == 2
    assert len(report.shape_counts) > 2


def test_header_and_rule_lines_are_ignored():
    assert all(e.n_values > 0 for e in buffers.parse(REPORT).entries)


def test_missing_report_returns_none_rather_than_raising(tmp_path):
    """A measurement must never fail because the compiler dumped nothing."""
    assert buffers.read(tmp_path, "candidate") is None


def test_read_picks_the_report_for_the_named_module(tmp_path):
    (tmp_path / "module_0002.jit_baseline.cpu_after_optimizations-memory-usage-report.txt").write_text(REPORT)
    (tmp_path / "module_0004.jit_candidate.cpu_after_optimizations-memory-usage-report.txt").write_text(
        "Memory Space: default (color=0)\nTotal bytes: 42 (42B)\n"
    )
    assert buffers.read(tmp_path, "baseline").total_bytes == 19661004
    assert buffers.read(tmp_path, "candidate").total_bytes == 42
