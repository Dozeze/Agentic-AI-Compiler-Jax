# A/A calibration on CPU, Sergio's Mac (2026-11-27)

Host: Apple M5 (10 cores), macOS, on battery; Docker Desktop VM with 10 CPUs. Code: `sergio/batch-04` @ 4b67828 (rows marked otherwise: 4d33851),
JAX 0.4.38, CPU backend. Command: `scripts/calibrate_aa.py --backend cpu --runner <subprocess|docker> --work-dir ...`,
default protocol (limits: IQR/median <= 0.10, baseline drift <= 0.05, 5 A/A sessions).

## Results
| Setup | softmax 128x1024 | layer_norm 128x1024 | matmul_gelu 128x256x256 |
|---|---|---|---|
| subprocess (native) | stable, δ 1.04 % / 0.32 % (2 runs) | stable, δ 0.34 % | stable, δ 0.55 % |
| docker, default `--cpus 2`, 20 other containers running | UNSTABLE | stable, δ 5.63 % | UNSTABLE (IQR/median ≈ 2) |
| docker, default `--cpus 2`, nothing else running | stable, δ 5.32 % | UNSTABLE (drift 0.068) | UNSTABLE (IQR/median ≈ 2) |
| docker, quota lifted (`--cpus 10`), nothing else running | stable, δ 5.46 % | stable, δ 5.11 % | stable, δ 5.19 % |
| docker, repository default after commit 4d33851 (no quota) | stable, δ 5.16 % | stable, δ 5.60 % | stable, δ 5.19 % |
| docker, `--cpuset-cpus 2,3` instead of a quota | stable, δ 6.53 % | UNSTABLE (drift 0.054) | UNSTABLE (IQR/median 0.13) |

## Finding: the `--cpus` quota makes container timing bimodal
With the default `DockerRunner(cpus=2)`, matmul_gelu blocks alternate between about 375 µs and 1,130 µs
per call (raw blocks in `raw-mac-cpu-docker-quiet-matmul_gelu/0003-session/result.json`). `--cpus 2` is a CFS
quota (2 CPU-periods per period), but XLA's CPU thread pool still sees all 10 cores and uses them. A
multi-threaded call uses up the quota early in a period, then the whole container is throttled until the
next period. Lifting the quota removes the bimodality: all three workloads calibrate (δ ≈ 5 %).

Docker still costs precision on this Mac (δ ≈ 5 % vs 0.3–1 % native): the Docker Desktop Linux VM adds
noise. Pinning two cores (`--cpuset-cpus`) avoids throttling but is borderline here; the core choice may
matter (performance vs efficiency cores are not visible inside the VM).

## Relevance for the GPU host (P26)
The NV host runs the same `DockerRunner` with `--cpus 2` and `--gpus all`. The host-side JAX runtime
(dispatch, synchronization) runs on the throttled CPU quota too, so the GPU A/A instability may have the
same cause. Not verified: it needs the same comparison on the NV host.

## Files
`mac-m5-*` and `exp-*`: calibration records (δ). `raw-*`: per-session raw blocks (JSON), worker logs and
sources. `experiment-scripts/`: the scratch wrappers used for the `--cpus 10` and `--cpuset-cpus` runs (they
patch `DockerRunner` at run time). Since commit 4d33851 the repository default has no quota (P29). The input `.npz` files and compressed
traces are left out of the shared zip.
