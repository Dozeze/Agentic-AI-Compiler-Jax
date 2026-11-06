# Target decision (AIC-031)
Status: **DRAFT**   Decided on: ____   Signed off by: Harry / J. Larsson / J. Tadesse / Sergio

> The implementing AI must not mark this `DECIDED`; the team decides (AIC-1.1). GPU rows are filled after
> phase B on the NV host (`docs/WSL2_CUDA_SETUP.md` §4).

## Selected target (team decision D-02)
NVIDIA <GPU model: RTX 40-series, exact model pending> · Windows <version> · WSL2 <distro, kernel> · driver <x> ·
JAX/jaxlib 0.4.38 (cuda12 wheels)

Evidence: `evidence/targets/nvhost-wsl2-cuda.json` *(pending phase B)* ·
CPU reference: `evidence/targets/mac-cpu.json` (2026-11-06)

## Criteria (AIC-1.1 §11)
| criterion | finding | evidence |
|---|---|---|
| reproducible access (who, when, how long) | Harry's own NV host; access schedule ____ | — |
| supported JAX/jaxlib/runtime versions | Mac: jax/jaxlib 0.4.38, numpy 2.5.2, Python 3.12.16. GPU: ____ | `mac-cpu.json` `environment.versions` |
| profiling coverage (`probe.profiler.coverage`) | Mac CPU: `CPU_EXECUTION_EVENTS` (executable granularity, 208 complete events, no device lanes). GPU: ____ | `mac-cpu.json` `profiler` |
| kernel path (Pallas native? FFI?) | Mac CPU: Pallas native `ValueError: Only interpret mode is supported on CPU backend.`; interpret mode correct; `jax.extend.ffi` available with headers; Apple clang 17. GPU Pallas native: ____ | `mac-cpu.json` `kernel` |
| isolation path (Docker + NVIDIA toolkit? cgroups?) | Mac: no Docker daemon, no cgroup v2, no user namespaces (macOS). WSL2: ____ | `mac-cpu.json` `isolation` |
| memory/compile budget (device memory) | Mac: 16 GiB host RAM, no device memory stats on CPU. GPU: ____ | `mac-cpu.json` `resources` |
| research opportunity | ____ | — |
| cost | No cloud cost (own hardware); electricity only | — |

## Known limitations of this target
- WSL2 GPU support is experimental in JAX.
- No GPU clock locking; limited NVML; one shared consumer GPU that may also drive a display.
- Mac CPU results and WSL2 results are different measurement populations; never compare them.

## Fallbacks (only if the WSL2 path fails a hard requirement)
Dual-boot Ubuntu on the same host (native Linux, best measurement quality), or a cloud NVIDIA GPU.

## What would make us revisit this decision
____
