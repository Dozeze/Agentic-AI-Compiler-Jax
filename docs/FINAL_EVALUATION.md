# Final evaluation and split release (AIC-018)

`agent.py --finalize-run` evaluates the verified best of a completed optimization run on the workload's
reserved final split. It never constructs a provider, requests a new proposal or changes the search
run's best pointer. The result is a separate directory with source snapshots, every per-shape
correctness result, raw workers and a committed final release.

The final split is **one use**. Reserving it ends search under that workload/evaluation version, even
when correctness fails, timing is unavailable, the process is interrupted or correctness-only mode is
used. This prevents final feedback from guiding another search presented as held out. Finish candidate
selection first. Use the offline tests for rehearsal; they use private evaluation state and do not
consume the official split. Missing target capability detected during preflight does not reserve it.

## What is frozen before search

Scientific runs now save an opaque `evaluation_split` commitment in `manifest.json` before requesting a
proposal. Its digest binds the workload/version, reference source, full input-recipe file, split
version/seeds/shapes, development population, dtypes, static semantics and tolerance. The prompt still
receives only the positive development allowlist; no final seeds, shapes, arrays or results enter it.
Team decision P16 keeps the seed declarations public in the repository; this is separation of search
and evaluation, not a secrecy guarantee against someone inspecting the repository.

Each scientific family declares twenty final seeds and two unseen shapes: forty final correctness
cases. Commit the new workload declarations before research search. A completed older run without the
pre-search commitment cannot be backfilled and called held out; start a newly frozen run. Changed
reference/input recipe, split, dependency lock, runner/image or observed device/JAX population refuses
final reuse rather than silently comparing different experiments.

## Run the final check

After selected-target entry checks and candidate selection are complete, inside the same WSL2 CUDA
environment and with the current worker images:

```bash
uv run --locked --extra cuda12 python agent.py \
  --finalize-run runs/SEARCH_RUN --require-gpu --output-dir runs/FINAL_RUN
uv run --locked python scripts/heldout.py replay --run runs/FINAL_RUN
```

The output directory must be new. Backend, runner, workload, reference, protocol and budgets come from
the original run. Search options/config overrides are refused. A model-written winner still requires
verified isolation; trusted subprocess execution is permitted only for the exact verified reference
fallback. Python tests may explicitly inject reviewed fixtures, which remain non-scientific.

`--final-correctness-only` omits timing and records performance as unverified. It **still consumes the
split** and is not a way to probe final cases during search. The default final mode checks correctness
on all forty cases before timing. A failure blocks timing for every shape and invalidates the affected
generalization/performance scope.

For each unseen shape, default final timing first calibrates A/A against the original frozen baseline,
then collects at least three fresh paired sessions of baseline versus selected source. Timing uses the
first reserved seed for that shape; the twenty-seed population is the correctness scope. Calibration,
raw blocks, uncertainty, stability and host decisions are retained. There is no retry until a desired
result appears, no threshold chosen from candidate performance and no profile data mixed into latency.
Noise/calibration/time failures remain explicit and prevent a verified performance claim. A correct
baseline fallback is valid, with no claimed improvement. CUDA performance verification additionally
requires a scientific parent run and confirmed improvement in every recorded final shape/session.

## Evidence and replay

- `parent_manifest.json`, `parent_best.json`, `parent_result.json` preserve the validated selection;
  `selection.json`, `baseline.py` and `winner.py` bind exact source identities.
- `split.json` is the released split recipe. The public pre-search commitment must match it exactly.
- `work/` contains actual input/output arrays, requests, results, logs and per-shape calibration/raw
  measurement evidence. Final correctness failures retain every declared shape and case.
- `final_report.json` records status, correctness scope, per-shape results and performance verification.
  `release.json` is published last and hashes the report and all supporting artifacts. A partial
  directory without a valid release is not final evidence.

`storage.verified_final_result(path)` and the replay CLI read JSON/source bytes only. They verify hashes,
selection/commitment, complete shape/case coverage, calibration/session links, raw measurement analysis
and derived verification flags. They never import stored candidate code. This validates the recorded
chain, not arbitrary fabricated files controlled by someone replacing the entire evidence set.

The one-use record lives under `evidence/heldout/publications/`; legacy failed-correctness burns remain
honored. An incomplete reservation stays sealed, including a failure before release publication.
Automatic recovery is not implemented here. A new evaluation after feedback/debugging requires a new
version with fresh seeds and shapes; merely changing the version number is refused.

The old `scripts/heldout.py evaluate --candidate ...` remains a legacy correctness diagnostic. It does
not establish frozen selection, durable final release or performance verification. Use the new mode
for research evidence. General report generation and execution recovery remain AIC-020 and AIC-019.

## Validation scope

Offline tests cover native forty-case reference checks for softmax, layer norm and dense exact GELU;
a reviewed source that passes development but fails both unseen shapes; synthetic timing/confirmation
and noise fixtures; source/report/array tampering; pre-dispatch commitments; sealed-state refusal;
budget failure; failed publication; and provider-free CLI/replay. Synthetic timing records are explicitly
non-scientific and are not speedup evidence. Actual target tests require CUDA/Docker and full final
sessions for all three families. Their acceptance remains pending with the earlier target/live gates.
