# Experiments (batch 04, 2026-11-27)

Sections 1–2: sweep 01, the first end-to-end results (op tier, gemini-2.5-flash, n = 1). Section 3: sweep 02,
the main CPU study (14 workloads, two models, two framings, n = 3). CPU only (Sergio's Mac), so these are development results, not the
CUDA target evidence that M3/M4 require.

## Setup
| Item | Value |
|---|---|
| Host | Apple M5, 10 cores, macOS, battery; Docker Desktop VM with 10 CPUs, no other containers |
| Code | `sergio/batch-04`, JAX 0.4.38, CPU backend, worker image `aic-worker:cpu-940408e17fadd2f2` |
| Runner | `docker` without a CPU quota (P29), the only path for model-written code |
| Protocol | default `MeasurementProtocol` (paired AB/BA blocks, bootstrap CI, accept only if the CI lower bound clears 1 + δ and a fresh session confirms) |
| Calibration | one frozen A/A record per workload, `evidence/calibration/mac-m5-cpu-docker-sweep/` (δ 4.9–6.0 %), shared by the ceiling study and the search |
| Model | `gemini-2.5-flash` on Vertex AI (`gemini-vertex`), thinking off; prices USD 0.30 / 2.50 per 1M input/output tokens, checked 2026-11-27 on the Vertex AI pricing page |
| Budget per cell | 3 candidates, 3 requests, money cap USD 0.30; context condition C2 (code + measured history) |

## Workloads
The three AIC-1.1 workloads (softmax, layer_norm, matmul_gelu) are already compiled well by XLA. On
`claude-experiment` (HALO) these kinds of functions were the controls with no headroom (softmax 1.01×,
rmsnorm 1.01×, matmul 0.99×). A search over them alone can only ever report "no speedup", whether or
not the agent works. So batch 04 adds five **headroom** workloads ported from HALO. Each one is written
the way a user might write it, and a known rewrite that XLA cannot find is faster:

| Workload | Reference (what the user wrote) | Known rewrite (`workloads/ceilings/`) |
|---|---|---|
| `matmul_chain` | `(a @ b) @ c`, a[1024,16], b[16,1024] | `a @ (b @ c)` (reassociation) |
| `pairwise_distances` | builds the (n, m, d) difference tensor | ‖a‖² + ‖b‖² − 2ab (one matmul) |
| `attention_loop` | Python loop over heads | one batched einsum over all heads |
| `layer_norm_loop` | Python loop over rows | vectorized over rows |
| `batched_matmul_loop` | Python loop over the batch | one batched matmul |

The controls stay: softmax, layer_norm, matmul_gelu. Accepting a change on a control is a false positive.
Tolerances were fixed before any search from the measured error of the reference and the rewrite against
float64 oracles (the rewrite needs at most a third of each declared tolerance). The held-out split
(seeds 1000–1019, two unseen shapes) is untouched.

## 1. Ceiling study (no model)
`scripts/measure_ceilings.py`: the known rewrite goes through the same pipeline as an agent candidate.

| Workload | Class | Ceiling, Docker (frozen δ) | Ceiling, native subprocess |
|---|---|---|---|
| matmul_chain | headroom | 13.14× [12.82, 13.56] | 9.10× [8.75, 9.36] |
| layer_norm_loop | headroom | 4.34× [4.28, 4.40] | 5.09× [5.07, 5.12] |
| pairwise_distances | headroom | 2.30× [2.28, 2.31] | A/A unstable |
| attention_loop | headroom | 1.87× [1.85, 1.90] | 1.93× [1.92, 1.94] |
| batched_matmul_loop | headroom | 1.31× [1.30, 1.33] | 1.49× [1.48, 1.49] |
| softmax | control | 1.00× (no change) | 1.00× (no change) |
| layer_norm | control | 0.99× (no change) | A/A unstable |
| matmul_gelu | control | 0.99× (no change) | 1.00× (no change) |

Brackets: 95 % bootstrap CI of the first judged session. Ceilings differ between runners (different
measurement populations); only the Docker column is the reference for the search below.
Results: `evidence/ceilings/*/results.json`.

## 2. Agent search (`scripts/search_sweep.py`, one cell per workload × framing)
Two prompt framings: `baseline` (the original system prompt) and `compiler_complement` (adds the
division of labour between XLA and the agent, after HALO's framing study, but **without** naming any
rewrite or workload; HALO's version named the exact identities of two of its tasks, which would leak the
answers here).

| Workload | Ceiling | baseline: speedup (share) | compiler_complement: speedup (share) |
|---|---|---|---|
| matmul_chain | 13.13× | 13.50× (101 %) | 13.16× (100 %) |
| layer_norm_loop | 4.26× | 4.28× (100 %) | 4.23× (100 %) |
| pairwise_distances | 2.30× | 2.39× (105 %) | 2.30× (100 %) |
| attention_loop | 1.88× | 1.92× (104 %) | 1.88× (100 %) |
| batched_matmul_loop | 1.30× | 1.28× (93 %) | 1.29× (96 %) |
| softmax (control) | 1.00× | no change accepted (0/3) | no change accepted (0/1) |
| layer_norm (control) | 0.99× | no change accepted (0/3) | no change accepted (0/2) |
| matmul_gelu (control) | 0.99× | no change accepted (0/3) | run stopped before a candidate (see below) |

Share of ceiling = ln(speedup) / ln(ceiling); values above 100 % are within measurement noise.
Totals: 10/10 headroom cells reached the ceiling, 0/5 evaluated control cells accepted a change,
46 model requests, USD 0.104 in total. Every cell's requests, attempts and raw sessions are under
`runs/sweep-01/` (not committed; summary in `evidence/sweeps/mac-m5-cpu-docker-sweep-01.json`).

What the agent did: it found the known rewrite (reassociation, the norm expansion, vectorizing the loop,
batching) within three proposals in every headroom cell, and in the first or second in 8 of 10. In
pairwise_distances/compiler_complement the first proposal was already right, but its session was
unstable (MEASUREMENT_UNSTABLE), so it was only accepted on the third. Later proposals were cosmetic
(VALID_NO_CHANGE) or slower and correctly rejected. On the controls it proposed equivalent rewrites
(`jnp.amax`, `rsqrt`, `dot_general`) that the decision rule rejected as no change. With the
compiler_complement framing it more often resubmitted the unchanged code (DUPLICATE) instead of a
cosmetic change, which used fewer evaluations (12 vs 21 in total).

## Limitations (read before quoting any number)
1. **One run per cell.** No variance across runs yet; HALO used n = 3. The next step is n ≥ 3.
2. **These rewrites are textbook patterns** and gemini-2.5-flash finds all of them on the first try. The
   suite is not hard enough to separate the two framings (HALO saw the framing effect with the weaker
   flash-lite, and its prompt also named the answers). Harder workloads are needed to show what the
   loop itself adds, for example HALO's block tier (gqa_decode, top_k, conv) and its tiny GPT loss.
3. **The speedup is over XLA compiling the user's code**, not over the best possible code. That is the
   project's question (agent + XLA vs XLA alone), but the controls show the other side: on code XLA already
   compiles well there was nothing to gain, and the agent claimed nothing.
4. **CPU on a laptop, inside Docker** (δ ≈ 5 %). Not the CUDA target; GPU numbers need the NV host.
5. **No held-out evaluation yet.** The final, one-use split evaluation (`--finalize-run`) was not run.
6. **One cell was lost to a provider guard**: Vertex applied implicit prompt caching (1,622 cached input
   tokens), and `providers.py` stops the run on any cache usage as "unpriced billing". Charging cached
   tokens at the full input price would be a safe upper bound instead (proposed fix, not applied).
7. Model-written code had two avoidable failures: escaped newlines in the source (POLICY_REJECTED syntax
   error) and calls such as `jax.nn.layer_norm` that do not exist in JAX 0.4.38 (COMPILE_FAILED). Both
   cost a candidate slot and were reported back to the model in the next request.

## 3. Sweep 02: 14 workloads × 2 models × 2 framings × 3 repetitions (168 searches)
Same host and protocol as above, new worker image `aic-worker:cpu-d00f283409f6fb94` (block tier added), so every
calibration and ceiling was measured again for this population (`evidence/calibration/mac-m5-cpu-docker-sweep2/`,
δ 4.5–6.1 %; top_k_rows 0.6 %). gqa_decode never passed A/A calibration (four attempts); its cells calibrate
lazily. Models: gemini-2.5-flash and gemini-2.5-flash-lite, thinking off; framings `baseline` and
`compiler_complement`; context C2; 3 candidates and 3 requests per search. 493 requests, USD 0.64 in total,
61 minutes wall time. Summary: `evidence/sweeps/mac-m5-cpu-docker-sweep-02.json`; cells under `runs/sweep-02/`.

### Ceilings (Docker, frozen δ)
| Tier | Workload | Ceiling | | Tier | Workload | Ceiling |
|---|---|---|---|---|---|---|
| op | matmul_chain | 13.5× | | block | top_k_rows | 161× |
| op | layer_norm_loop | 4.3× | | block | embedding_lookup | ~52× (session unstable, see below) |
| op | pairwise_distances | 2.3× | | block | gqa_decode | 2.2× |
| op | attention_loop | 1.9× | | block | conv3x3 | 2.0× |
| op | batched_matmul_loop | 1.3× | | block | rnn_layer | 1.7× |
| control | softmax, layer_norm, matmul_gelu | 1.00× (no change) | | block | cross_entropy | 1.3× |

### Share of the ceiling reached (mean over cells, capped at 100 %; 95 % bootstrap CI, runs resampled within workload)
| Tier | Model | baseline framing | compiler_complement | after 1 candidate → after 3 |
|---|---|---|---|---|
| op (5 workloads) | flash | 83 % [70, 97] | 97 % [95, 98] | 65 → 83 % / 84 → 97 % |
| op | flash-lite | 37 % | 37 % [23, 52] | no gain from iterating |
| block (6 workloads) | flash | 31 % [29, 33] | 42 % [21, 63] | 15 → 31 % / 25 → 42 % |
| block | flash-lite | 22 % [6, 39] | 24 % [11, 39] | 22 → 22 % / 17 → 24 % |

Differences (percentage points, 95 % CI): flash − flash-lite +31 [21, 42] over all headroom cells (+53 on the op
tier, +13 [−4, 30] on the block tier). compiler_complement − baseline: +12 [0, 26] with flash, +1 [−12, 13] with
flash-lite. Iterating (best after 3 candidates − best after 1): +16 [8, 25] with flash (11 of 66 cells improved
after the first candidate), +2 [0, 6] with flash-lite.

### Controls
0 of 48 control cells accepted a change on softmax or layer_norm. On matmul_gelu, 2 of 24 cells accepted a 1.17× and
a 1.20× rewrite (one per model). Both replace `jax.nn.gelu(approximate=False)`, which lowers through `erfc`, by
0.5·x·(1 + erf(x/√2)). It passes every declared case: its largest absolute error against the float64 oracle is
2.5e-6, the same as the reference. But for x < −5.5 it returns 0 where GELU is about 1e-9, a 100 % relative error that
the declared atol (1e-5) allows. The contract permits it; the reference author chose `erfc` for that tail. We report
it as accepted on a control under a contract that does not protect the tail, not as a clean speedup, and the
tolerance is not changed after the fact.

### What the agent found vs what the harness accepted
15 correct candidates were rejected as MEASUREMENT_UNSTABLE, 12 of them with every session's CI lower bound far above
1 + δ: 9 of 12 embedding_lookup cells (48–60×), one matmul_chain (~15×), one cross_entropy (1.33×), one
batched_matmul_loop (~1.1×). Cause: calls per block are calibrated on the reference only (AIC-1.1 §5.3), so a 50×
faster candidate runs blocks of well under a millisecond, whose dispersion exceeds the 10 % IQR limit. The agent then
resubmitted the same correct program 18 times, and the duplicate rule refused to measure it again. Counting those
candidates as found raises the block-tier share from 36 % to 53 % (flash) and from 23 % to 31 % (flash-lite);
embedding_lookup goes from 17 % to 100 % with flash. The official metric stays "accepted"; "found" is a diagnostic of
the protocol, not of the agent.

### Where the agent fails
Most failures are the model's: flash-lite produced 50 COMPILE_FAILED attempts (flash 16), mostly calls to JAX APIs
with invented keyword arguments (`gather(..., axis_size=)`, `expand_dims(..., axis=)`), and 20 CORRECTNESS_FAILED
(flash 13). gqa_decode (grouping query heads) and rnn_layer (hoisting the input projection out of `scan`) are the
hardest: 1 accepted of 6 flash cells each, none with flash-lite. On cross_entropy, flash found
`jax.nn.log_softmax` + `take_along_axis`, 1.31×, slightly faster than our ceiling rewrite (1.27×): the ceiling is
the best rewrite we knew, not an upper bound.

### Lessons for the protocol (proposed, not applied to these results)
1. Calibrate calls per block so that *both* sides reach the minimum block duration (more calls only lengthen both
   sides equally), or accept when every session's CI lower bound clears 1 + δ by a wide margin despite dispersion.
2. Let a candidate rejected only as MEASUREMENT_UNSTABLE be measured again once when it is resubmitted unchanged.
3. Declare an elementwise relative tolerance for functions with small tails (GELU), or test the tail explicitly.
