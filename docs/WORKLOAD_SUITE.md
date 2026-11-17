# Representative workload contracts (AIC-017)

These are development/reference contracts, declared before candidate search. CPU eager/JIT, independent
oracle and native-worker checks pass. Official CUDA correctness, containment/resource acceptance and
per-workload A/A calibration are still pending on the D-02 target. M3/AIC-016 remains blocked; this
independent software work does not accept that gate or complete M4. Commit the new source/input/split
declarations before any search; no search or paid request was performed during this implementation.

| Workload ID | Function and explicit inputs | Fixed semantics | Float32 tolerance |
| --- | --- | --- | --- |
| `softmax` | `softmax(x)` | Last axis, stable max subtraction; existing contract unchanged | rtol `1e-5`, atol `1e-6` |
| `layer_norm` | `layer_norm(x, scale, bias)` | Last-axis two-pass population variance; `(x-mean)/sqrt(var+1e-5)*scale+bias`; vectors broadcast over rows | rtol `1e-4`, atol `1e-4` |
| `matmul_gelu` | `matmul_gelu(x, weight, bias)` | Float32 matmul requesting `Precision.HIGHEST` and float32 accumulation, column bias, exact erfc GELU (`approximate=False`) | rtol `1e-4`, atol `1e-5` |

All three produce a single finite float32 array for the declared finite inputs. The new references reject
non-float32 arguments and invalid ranks/shapes. The input factories reject unsupported dtypes and invalid
dimensions, and check the combined byte bound before allocating a large matrix. Precision, epsilon,
activation variant and tolerances are recorded in development context. Candidate comparisons continue
to use the compiled frozen reference; the independent NumPy float64 oracles validate that reference.

Layer norm's absolute floor covers mean-rounding error amplified by `1/sqrt(epsilon)` in nearly constant
rows near one. Its development edge population includes these rows explicitly. Dense projection's floor
covers float32 dot accumulation and reassociation up to `K=512`, including GELU outputs close to zero.
These policies were declared before the reference checks and were not relaxed to admit a failing
candidate. The CPU native-reference checks observed maximum tolerance fractions 0.412 and 0.043 for
layer norm and dense projection respectively. They do not establish CUDA error bounds. Review the
selected-target results before search; do not loosen a policy after inspecting a faster failing proposal.

## Inputs, shapes and edge cases

| Family | Development shapes | Reserved unseen shape declarations | Largest combined input |
| --- | --- | --- | --- |
| Softmax | `(32,128)`, `(128,1024)` | `(64,257)`, `(256,2048)` | 2 MiB |
| Layer norm `(rows,width)` | `(32,128)`, `(128,1024)` | `(64,257)`, `(256,2048)` | 2,113,536 bytes including scale/bias |
| Dense `(M,K,N)` | `(32,64,64)`, `(128,256,256)` | `(63,127,95)`, `(256,512,384)` | 1,312,256 bytes including weight/bias |

The new families each allow at most 8 MiB of combined inputs. Input-size bounds do not prove that XLA,
profiling and device allocation fit the Docker resource limits; actual selected-target checks are still
required. No plan shape was resized and no baseline was deliberately slowed down.

Each family uses deterministic NumPy PCG64 inputs with search seeds 0–4. Layer norm draws standard-normal
activations, uniform scale in `[0.5,1.5)` and normal bias with standard deviation 0.1. Dense projection
draws standard-normal activations, normal weights scaled by `1/sqrt(K)` and normal bias with standard
deviation 0.1. Parameters are explicit arrays supplied by the harness, not constants embedded in source.
The final split declares seeds 1000–1019 and two disjoint shapes per family. Those fields remain outside
proposal context. Trusted reference tests use seed 0 on the reserved shapes as a shape smoke check;
this is not final-split evaluation or generalization evidence. Split/ledger procedures remain AIC-018.

Layer norm has seven edge cases: zeros, constant rows, nearly constant rows, positive offsets, mixed
magnitudes, a dominant value and signed/nonuniform affine parameters. Dense projection has six:
all zeros, zero activations, zero weights, a bias/activation sweep over `[-6,6]`, dot cancellation and
inversely scaled mixed-magnitude activations/weights. The resulting development populations contain
17 and 16 cases. Negative fixtures reject changed epsilon, omitted affine parameters, unstable variance,
approximate GELU, omitted bias and reduced output precision.

## Validation and target handoff

```bash
uv run --locked pytest tests/test_workload_suite.py -m 'not target' -q
AIC_TARGET_CONFIG=configs/target.json uv run --locked --extra cuda12 \
  pytest tests/test_workload_suite.py -m target -q
```

The two target tests run every declared development/edge case in CUDA Docker workers, compare against
the independent oracles and retain ordinary compiler/profile evidence. Missing capabilities fail with
`AIC_TARGET_CONFIG` set. Rebuild both worker variants after these source additions. The image content
hash now includes the new modules, so an old image cannot be mistaken for the current worker.

After the target checks pass, calibrate each primary (largest development) shape on the same idle
population. Keep the same config, JAX/XLA settings and resource caps. For example, inside WSL2:

```bash
for workload in softmax layer_norm matmul_gelu; do
  uv run --locked --extra cuda12 python scripts/calibrate_aa.py \
    --workload "$workload" --backend cuda --runner docker --config configs/target.json \
    --label nvhost-suite-UNIQUE --work-dir "runs/nvhost-suite-UNIQUE/$workload"
done
```

`--work-dir` must be new. It preserves raw worker requests, source, inputs, results, measurement blocks,
logs and profiles even if noise prevents a usable calibration. `calibration_context.json` records the
configured budgets/protocol, reference/policy hashes, primary shape and runner identity. Evaluation
uses the supplied config's budgets. A successful aggregate calibration remains under
`evidence/calibration/<label>/<workload>-<shape>.json`; when retaining raw work, an existing aggregate
or work directory is refused. Supply that matching file to the optimizer with `--calibration` only
after all pending entry gates are met. CPU calibration is mechanics evidence only.

Native CPU evidence is retained in `runs/aic-017-native-validation-20261116/validation.json`, with two
baseline-only CLI run directories. It verifies actual worker outputs against the oracles, unchanged
sources, exact baseline exports, source/policy hashes and executable-level CPU profile coverage. Zero
provider requests were made. It makes no timing, live-optimization or selected-target acceptance claim.
