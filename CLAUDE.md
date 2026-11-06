# Instructions for Claude Code

Read `AI_CONTEXT.md` first; it is the team's shared context and overrides anything older in this repo.

- Every commit you prepare must also update `AI_CONTEXT.md` (at least §5 Status, §6 Decisions, §9 Change log) in the same commit.
- All documentation, code comments, docstrings, test names and commit messages are in English.
- Personal work plans live in `docs/plans/` (git-ignored, local only). Never commit them. If one exists for the current branch, follow it and do only the tasks in it.

## Dual-backend code (CPU + CUDA)
Everything must run on CPU (Mac, CI) and on CUDA (NV host). Full rules: `AI_CONTEXT.md` §7, D-06.
1. Never choose a backend in harness code; only `environment.select_backend()` does (from `RunConfig.backend`). Check for a GPU with `environment.has_gpu()`.
2. Tests run on CPU. GPU-only tests get `@pytest.mark.needs_gpu` and run JAX in a subprocess (`tests/backends.gpu_env()`).
3. Pallas kernels use `interpret=jax.default_backend() == "cpu"`, never a hard-coded value (`scripts/probe_target.py` is the one deliberate exception).
4. Tolerances are fixed per workload before any search; a backend-specific tolerance is declared up front, never added after a failure.
5. Ordinary tests never compare runtimes or speedups; such checks are `slow` or `needs_gpu`.
