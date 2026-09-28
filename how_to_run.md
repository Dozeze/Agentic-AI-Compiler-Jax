For Jonathan L:

Install the locked CPU development environment and run a bounded optimization loop:

```bash
uv sync --locked
uv run --locked python agent.py --iterations 1
```

The command reads `ANTHROPIC_API_KEY` from `.env.local`. Increase `--iterations` after
the first smoke run. Add `--plot` only when a graphical runtime plot is wanted.

On Linux with a compatible NVIDIA GPU, use the already locked CUDA 12 extra:

```bash
uv sync --locked --extra cuda12
uv run --locked --extra cuda12 python -c "import jax; print(jax.devices())"
uv run --locked --extra cuda12 python agent.py --iterations 1 --require-gpu
```

`--require-gpu` stops before calling Claude if JAX only finds a CPU. There is
no need to edit `pyproject.toml` manually. `uv run --no-project --with anthropic`
skips this project's declared dependencies and lockfile, so it is not the
recommended way to run this project.
The old `jax[cuda13]==0.4.38` line does not install CUDA 13 support: JAX
0.4.38 does not provide that extra. Check `jax.devices()` before benchmarking.

To optimize a Python workload file, give it a callable function and either a
`TEST_ARGS` tuple or a `make_test_args()` function:

```python
import jax.numpy as jnp

TEST_ARGS = (jnp.arange(32, dtype=jnp.float32), jnp.ones(32, dtype=jnp.float32))

def bad_func(x, y):
    return x + y
```

Run that file with:

```bash
uv run --locked python agent.py --source workloads/example.py \
  --function bad_func --iterations 3 --output-dir runs/example
```

On the GPU machine, add `--extra cuda12` after `--locked` and `--require-gpu`
after `agent.py` to this command.

Each evaluation writes `iteration_0000.json`, `iteration_0001.json`, and so on,
plus a cumulative `history.json`. The JSON records include the complete candidate
source, unified source diff, runtime mean/std, correctness, and accept/reject decision.
