"""Run the agentic optimization loop on a trusted JAX workload."""
import argparse
import dataclasses
import inspect
import json
import os
import sys
from pathlib import Path

import anthropic
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

from contracts import Budgets, ConfigError, ModelSettings, config_hash, resolve_run_config, to_dict
from controller import Controller, read_python_source
from environment import BACKENDS, BackendUnavailable, collect_environment, has_gpu, select_backend

REPO = Path(__file__).resolve().parent
MANIFEST_SCHEMA = "manifest/v1"

# Read controller.md to see how it works!
#client = anthropic.Anthropic()


# Initialize parameters and PROMPTS

DEFAULT_MODEL = ModelSettings().model_id
# Default number of candidate evaluations. Changed from 10 to 3 to match the AIC-1.1 §7
# development ceiling (team decision P4); the value lives in contracts.Budgets.
DEFAULT_ITERATIONS = Budgets().max_candidates


# Version 1 - only for a .py-file
def _load_source_workload(source_path, function_name):
    """Load a trusted .py workload and return source, function, and test args."""
    source_path = Path(source_path)
    source = read_python_source(source_path)
    namespace = {"__file__": str(source_path)}
    exec(compile(source, str(source_path), "exec"), namespace)

    function = namespace.get(function_name)

    if not callable(function):
        raise ValueError(f"{source_path} does not define callable {function_name!r}")

    make_test_args = namespace.get("make_test_args")
    if callable(make_test_args):
        args = tuple(make_test_args())
    elif "TEST_ARGS" in namespace:
        args = tuple(namespace["TEST_ARGS"])
    else:
        parameter_count = len(inspect.signature(function).parameters)
        if parameter_count == 1:
            args = (jnp.arange(32, dtype=jnp.float32),)
        elif parameter_count == 2:
            args = (
                jnp.arange(32, dtype=jnp.float32),
                jnp.ones(32, dtype=jnp.float32),
            )
        else:
            raise ValueError(
                f"{source_path} must define TEST_ARGS or make_test_args() "
                "for functions with more than two parameters"
            )
    return source, jax.jit(function), args


def build_controller(client, max_iterations=DEFAULT_ITERATIONS, source_path=None, function_name="bad_func",
                     output_dir=None, *, workload_id=None, runner=None, selected_backend=None, protocol=None,
                     calibration=None, context_condition="C2"):
    """Build the controller for a registered workload (`workload_id`) or a trusted --source file.

    Without either, the registered `example_add` workload (the original x + y example) is used. The
    default runner is the isolated DockerRunner; pass another runner explicitly.
    """
    import tempfile
    from contracts import MeasurementProtocol
    from controller import Evaluator
    from runner import DockerRunner
    from workloads import registry

    if source_path is not None:
        _, _, args = _load_source_workload(source_path, function_name)
        spec = registry.legacy_source_spec(source_path, function_name, args)
    else:
        spec = registry.get(workload_id or "example_add")
        function_name = spec.function_name

    prompt = (
        "You optimize pure JAX functions through the existing evaluator. "
        "Preserve the function signature and numerical behavior. Do not use "
        "CUDA-only libraries, custom kernels, approximations, or changes to "
        "precision. Submit each candidate with benchmark_code as a proposal: a "
        "hypothesis, the single transformation, its rationale, the expected effect "
        "and the complete candidate source (raw Python, no Markdown fences or "
        "escaped quotes). The evaluator applies jax.jit itself, so do not add "
        "@jax.jit or call jax.jit in the candidate. Inspect the current code "
        "and history before proposing a candidate."
    )
    workdir = Path(output_dir) / "work" if output_dir is not None else Path(tempfile.mkdtemp(prefix="aic-run-"))
    evaluator = Evaluator(spec, runner or DockerRunner(selected_backend=selected_backend),
                          protocol or MeasurementProtocol(), workdir, selected_backend=selected_backend,
                          calibration=calibration)
    return Controller(
        agent="Claude",
        prompt=prompt,
        evaluator=evaluator,
        client=client,
        function_name=function_name,
        max_iterations=max_iterations,
        output_dir=output_dir,
        source_path=source_path,
        context_condition=context_condition,
    )


def _registry():
    from workloads import registry
    return registry


def output_dir_of(cfg):
    return Path(cfg.output_dir)


def scientific_status(cfg, spec, runner):
    """(scientific, reasons): a run supports a claim only with an explicitly selected backend, a scientific
    workload (seeded inputs, edge cases, held-out split) and an isolated runner. Every missing condition
    is recorded, never hidden behind a single flag."""
    reasons = []
    if cfg.backend is None:
        reasons.append("backend not explicitly selected")
    if not spec.scientific:
        reasons.append(f"workload {spec.workload_id!r} is not a scientific workload")
    if not getattr(runner, "isolated", False):
        reasons.append(f"runner {runner.name} is not isolated")
    return not reasons, reasons


def client_factory():
    """The LLM client for the configured provider. Tests replace anthropic.Anthropic with a fake."""
    return anthropic.Anthropic()


def atomic_write_json(path, obj):
    """Write strict JSON (no NaN/inf) atomically: write a temporary file, then replace."""
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=None,
                        help=f"maximum candidate evaluations (budgets.max_candidates; default {DEFAULT_ITERATIONS})")
    parser.add_argument("--model", default=None,
                        help=f"model id (default: $CLAUDE_MODEL, else the config value, else {DEFAULT_MODEL})")
    parser.add_argument("--workload", default=None, choices=sorted(_registry().REGISTRY),
                        help="registered workload to optimize (default: example_add); --source overrides it")
    parser.add_argument("--source", type=Path, help="trusted Python workload file to optimize (not scientific)")
    parser.add_argument("--function", default=None, help="function name in --source (default: bad_func)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for per-iteration JSON artifacts (default: runs/latest)",
    )
    parser.add_argument("--plot", action="store_true", help="show the runtime history plot")
    parser.add_argument("--require-gpu", action="store_true", help="fail if JAX cannot use a GPU")
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help="JAX backend to select before any JAX computation; never falls back silently. "
                             "Without it, JAX chooses (legacy) and the run is marked non-scientific")
    parser.add_argument("--config", type=Path, default=None,
                        help="JSON RunConfig v1 file; precedence is CLI > file > defaults")
    parser.add_argument("--runner", choices=("docker", "subprocess"), default=None,
                        help="where code runs: docker (default; isolated, required for model-written code) or "
                             "subprocess (trusted code only: the baseline runs, model candidates are refused)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="frozen A/A calibration JSON; without it the run calibrates before its first measurement")
    return parser


def _cli_overrides(args):
    """Map explicit CLI values onto RunConfig keys. None means 'not given'."""
    return {
        "backend": args.backend,
        "budgets.max_candidates": args.iterations,
        # CLAUDE_MODEL keeps its legacy precedence: below --model, above the config file.
        "model.model_id": args.model or os.getenv("CLAUDE_MODEL") or None,
        "workload_id": args.workload,
        "source_path": str(args.source) if args.source is not None else None,
        "function_name": args.function,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "runner": args.runner,
        "calibration_path": str(args.calibration) if args.calibration is not None else None,
    }


def main(argv=None):
    load_dotenv(Path(".env.local"))
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = resolve_run_config(_cli_overrides(args), args.config)
    except ConfigError as error:
        parser.error(str(error))

    if args.require_gpu and cfg.backend in ("cpu", "tpu"):
        parser.error(f"--require-gpu conflicts with backend {cfg.backend!r}")

    # Select the backend BEFORE any JAX computation: building the controller execs the workload file,
    # and JAX 0.4.38 silently ignores a selection made after its backends are initialised.
    if cfg.backend is not None:
        try:
            devices = select_backend(cfg.backend)
        except BackendUnavailable as error:
            print(f"error: {error}", file=sys.stderr)
            sys.exit(2)
    else:
        devices = jax.devices()
    if args.require_gpu and not has_gpu(devices):
        parser.error(f"--require-gpu was requested, but JAX only sees: {devices}")
    print(f"JAX devices: {devices}")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing; put it in .env.local or the environment")

    # Where code runs. Model-written code needs isolation; without it the run stops before any LLM call.
    from runner import DockerRunner, TrustedSubprocessRunner
    runner = DockerRunner(selected_backend=cfg.backend) if cfg.runner == "docker" else TrustedSubprocessRunner()
    reason = runner.preflight()
    if reason is not None:
        print(f"error: ISOLATION_UNAVAILABLE: {reason}", file=sys.stderr)
        sys.exit(2)

    from benchmark import Calibration
    from controller import BaselineFailed
    controller = build_controller(
        client_factory(),
        max_iterations=cfg.budgets.max_candidates,
        source_path=Path(cfg.source_path) if cfg.source_path is not None else None,
        function_name=cfg.function_name,
        output_dir=output_dir_of(cfg),
        workload_id=None if cfg.source_path is not None else cfg.workload_id,
        runner=runner,
        selected_backend=cfg.backend,
        protocol=cfg.measurement,
        calibration=Calibration.load(cfg.calibration_path) if cfg.calibration_path else None,
        context_condition=cfg.context_condition,
    )
    evaluator = controller.evaluator
    # The correctness policy is known once the workload is: record it in the resolved config (AIC-002).
    cfg = dataclasses.replace(cfg, workload_id=evaluator.spec.workload_id,
                              correctness_policy_hash=evaluator.policy_hash)

    # Record the run before any LLM call.
    output_dir = output_dir_of(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    scientific, reasons = scientific_status(cfg, evaluator.spec, runner)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_config_sha256": config_hash(cfg),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "backend_requested": cfg.backend,
        "scientific": scientific,
        "non_scientific_reasons": reasons,
        "workload": {"id": evaluator.spec.workload_id, "reference_sha256": evaluator.spec.reference_sha256,
                     "correctness_policy_hash": evaluator.policy_hash,
                     "heldout_commitment": evaluator.spec.heldout_commitment},
        "runner": runner.identity(),
        "environment": collect_environment(REPO, devices),
    }
    atomic_write_json(output_dir / "run_config.json", to_dict(cfg))
    atomic_write_json(output_dir / "manifest.json", manifest)

    try:
        controller.start()
    except BaselineFailed as error:
        print(f"error: baseline failed, no proposal requested: {error}", file=sys.stderr)
        sys.exit(2)
    if cfg.budgets.max_candidates == 0:
        # A zero candidate budget is valid and means: no provider request at all.
        history = controller.provide_hist()
    else:
        history = controller.agent_loop(model=cfg.model.model_id, max_tokens=cfg.model.max_tokens)
    if controller.stop_reason:
        print(f"Run stopped: {controller.stop_reason}")

    print(f"Completed {len(history) - 1} candidate evaluations.")
    print(f"Best runtime: {controller.best_runtime}")
    print(
        "Accepted candidates: "
        f"{sum(item['accepted'] for item in history[1:])}"
    )
    for item in history[1:]:
        print(f"  iteration {item['iteration']}: {item['state']}")
    if history[-1].get("error"):
        print(f"Last evaluation error: {history[-1]['error']}")

    if args.plot:
        controller.graph_it()

    return controller


if __name__ == "__main__":
    main()


#
# TODO - Create an agent

# Agent should: - Get controller in order to USE function HOWEVER it wants
# Then based on this, it should be able to test strategies to make the code better
# In the controller, we should also implement a max_iter, which is the max number of times
# it can run tests on the hardware.

# Then we should have ANOTHER loop for the agent, and the number of loops, N, is the number of
# times we ask the agent to improve the code.
