# WSL2 + CUDA setup for the target machine (AIC-031)

This is how to rebuild the primary measurement environment: **NVIDIA GPU · Windows host · WSL2 Ubuntu ·
`jax[cuda12]==0.4.38` installed by `uv` inside WSL2**. JAX has no native Windows GPU support; NVIDIA GPUs are only
available through WSL2, which the JAX installation guide lists as **experimental**.

Sources: [JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
[NVIDIA CUDA on WSL user guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Steps marked *(to confirm)* have not yet been executed on the NV host; update this file after phase B.

## 1. Windows side
1. Windows 11 (Windows 10 needs specific builds, see the NVIDIA guide).
2. WSL2 with an Ubuntu distribution; kernel ≥ 5.10.16.3 recommended. Check in PowerShell:
   ```powershell
   wsl --version
   wsl cat /proc/version
   ```
3. Install the **Windows** NVIDIA driver only (supporting CUDA 12; JAX needs driver ≥ 525, CUDA on WSL needs R495+).
   **Do not install any Linux NVIDIA driver inside WSL.** The Windows driver exposes `libcuda.so` to WSL.

## 2. GPU requirements
- Compute capability ≥ 5.2 for JAX CUDA 12 wheels; Pascal or newer under WSL2.
- **Volta or newer for CUPTI profiling** under WSL2 (needed for GPU device events in the profiler probe).
  The NV host has an RTX 40-series (Ada) GPU, so this is expected to hold (AI_CONTEXT D-05).

## 3. Inside WSL2 Ubuntu
1. If a CUDA toolkit is ever needed (JAX's pip CUDA wheels normally need none), use the WSL-Ubuntu package or
   `cuda-toolkit-12-x`. **Never** install `cuda`, `cuda-12-x` or `cuda-drivers`: they overwrite the WSL `libcuda.so`.
2. Install `uv` (see <https://docs.astral.sh/uv/getting-started/installation/>).
3. Clone the repository **into the Linux filesystem**, not under `/mnt/c/...` (cross-filesystem I/O is slow and
   would distort timings; the probe records `repo.on_windows_mount`):
   ```bash
   mkdir -p ~/src && cd ~/src
   git clone https://github.com/Dozeze/Agentic-AI-Compiler-Jax.git
   cd Agentic-AI-Compiler-Jax
   git checkout harry/batch-01
   uv sync --locked --extra cuda12        # the extra is Linux-only; WSL counts as Linux
   ```
4. Verify:
   ```bash
   /usr/lib/wsl/lib/nvidia-smi            # often not on PATH under WSL2
   uv run --locked --extra cuda12 python -c "import jax; print(jax.devices())"
   ```

## 4. Probe and tests (AIC-031 phase B)
```bash
uv run --locked --extra cuda12 pytest -q                     # all pass, including needs_gpu
uv run --locked --extra cuda12 python scripts/probe_target.py --backend cuda --label nvhost-wsl2-cuda --out evidence/targets
```
Then copy the results (`backend.status`, `profiler.coverage`, `profiler.device_lanes`, `kernel.pallas_native`,
`isolation`, `resources`) into `docs/TARGET_DECISION.md`. Record the actual `device.platform` value seen under CUDA
(expected `gpu`) and the device-lane names in `AI_CONTEXT.md` §8.

## 5. Docker (later, for AIC-009)
Docker Desktop with the WSL2 backend, or Docker CE inside WSL, plus NVIDIA Container Toolkit ≥ 2.6.0.
Under WSL2 only `--gpus all` is supported (no per-GPU selection). The probe only tries a GPU container with
`--try-docker-gpu` (it may pull an image).

## 6. Measurement hygiene (record with every official run)
- Close GPU-heavy Windows applications; note the Windows power plan; note whether the GPU also drives a display.
- WSL2 cannot lock GPU clocks, and many NVML queries (utilisation, persistence mode) are unsupported:
  record these as `unavailable`, never as zero.
- Mac CPU results and WSL2 GPU results are different measurement populations — never compare them.
