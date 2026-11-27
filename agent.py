"""Run the agentic optimization loop on a trusted JAX workload."""
import argparse
import dataclasses
import inspect
import os
import sys
from pathlib import Path

import anthropic
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

from contracts import Budgets, ConfigError, ModelSettings, config_hash, resolve_run_config, to_dict
from context_builder import evaluation_commitment
from controller import Controller, read_python_source
from environment import BACKENDS, BackendUnavailable, collect_environment, has_gpu, select_backend
from storage import RunStore, StorageError, atomic_write_json

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


# Batch 04, after the framing study on `claude-experiment` (HALO). It states the division of labour only;
# it names no identity, rewrite or workload, so it cannot hand the agent a task's answer.
COMPILER_COMPLEMENT = (
    "You work alongside XLA, not in place of it. XLA already does the mechanical work well: it fuses "
    "elementwise operations, assigns layouts, schedules and picks kernels. What it cannot do is change "
    "the algorithm: it compiles faithfully whatever computation it is given. Your job is the part only "
    "you can do: find a different computation that gives the same result, within the tolerance, for less "
    "work or less memory traffic. Reason from the mathematics and the structure of the code first. A clean "
    "compiler summary only says the given algorithm was lowered well, not that a better one does not "
    "exist. If you find no such change, say so and submit the current code unchanged rather than a "
    "cosmetic rewrite."
)


def build_controller(client, max_iterations=DEFAULT_ITERATIONS, source_path=None, function_name="bad_func",
                     output_dir=None, *, workload_id=None, runner=None, selected_backend=None, protocol=None,
                     calibration=None, context_condition="C2", budgets=None, model_settings=None,
                     prompt_framing="baseline"):
    """Build the controller for a registered workload (`workload_id`) or a trusted --source file.

    Without either, the registered `example_add` workload (the original x + y example) is used. The
    default runner is the isolated DockerRunner; pass another runner explicitly.
    """
    from contracts import MeasurementProtocol, validate_budgets
    from controller import Evaluator
    from runner import DockerRunner
    from workloads import registry

    validate_budgets(budgets or Budgets())
    if type(max_iterations) is not int or max_iterations < 0:
        raise ConfigError("max_iterations must be a nonnegative integer")

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
        "precision. Return each candidate with submit_proposal as a proposal: a "
        "hypothesis, the single transformation, its rationale, the expected effect "
        "and the complete candidate source (raw Python, no Markdown fences or "
        "escaped quotes). The evaluator applies jax.jit itself, so do not add "
        "@jax.jit or call jax.jit in the candidate. Inspect the current code "
        "and the measured history provided in context before proposing a candidate."
    )
    if prompt_framing == "compiler_complement":
        prompt += " " + COMPILER_COMPLEMENT
    store = RunStore(output_dir, budgets=budgets)
    workdir = store.path / "work"
    evaluator = Evaluator(spec, runner or DockerRunner(selected_backend=selected_backend),
                          protocol or MeasurementProtocol(), workdir, selected_backend=selected_backend,
                          calibration=calibration, budgets=budgets)
    return Controller(
        agent="Claude",
        prompt=prompt,
        evaluator=evaluator,
        client=client,
        function_name=function_name,
        max_iterations=max_iterations,
        output_dir=store.path,
        store=store,
        source_path=source_path,
        context_condition=context_condition,
        model_settings=model_settings,
    )


def _registry():
    from workloads import registry
    return registry


def output_dir_of(cfg):
    return Path(cfg.output_dir) if cfg.output_dir is not None else None


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


def client_factory(settings=None):
    """The LLM client for the configured provider. Tests replace anthropic.Anthropic with a fake."""
    from contracts import ModelSettings
    from providers import make_client
    return make_client(settings or ModelSettings())


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=None,
                        help=f"maximum candidate evaluations (budgets.max_candidates; default {DEFAULT_ITERATIONS})")
    parser.add_argument("--provider", default=None, choices=("anthropic", "anthropic-vertex", "gemini-vertex"),
                        help="LLM provider: anthropic (API key, default), anthropic-vertex (Claude on Vertex AI) or "
                             "gemini-vertex (Gemini on Vertex AI); both Vertex providers bill the GCP project")
    parser.add_argument("--model", default=None,
                        help="model id (default: $CLAUDE_MODEL for --provider anthropic, else the config value, "
                             f"else the provider default, e.g. {DEFAULT_MODEL})")
    parser.add_argument("--workload", default=None, choices=sorted(_registry().REGISTRY),
                        help="registered workload to optimize (default: example_add); --source overrides it")
    parser.add_argument("--source", type=Path, help="trusted Python workload file to optimize (not scientific)")
    parser.add_argument("--function", default=None, help="function name in --source (default: bad_func)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new artifact directory; must not exist (default: unique runs/<run-id>)",
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
    parser.add_argument("--offline-responses", type=Path, default=None,
                        help="replay an array of local Messages-shaped responses; no credentials or network")
    parser.add_argument("--proposal-only", action="store_true", default=None,
                        help="save one validated proposal without loading or evaluating source code")
    parser.add_argument("--finalize-run", type=Path,
                        help="consume the one-use final split for a completed run's verified best; no provider requests")
    parser.add_argument("--final-correctness-only", action="store_true",
                        help="omit final timing; still consumes the split and leaves performance unverified")
    return parser


def _cli_overrides(args):
    """Map explicit CLI values onto RunConfig keys. None means 'not given'."""
    return {
        "backend": args.backend,
        "budgets.max_candidates": args.iterations,
        "model.provider": args.provider,
        # CLAUDE_MODEL keeps its legacy precedence (below --model, above the config file) for the Anthropic
        # API only: a first-party Claude id means nothing to the Vertex providers.
        "model.model_id": args.model or (os.getenv("CLAUDE_MODEL") if args.provider in (None, "anthropic")
                                         else None) or None,
        "workload_id": args.workload,
        "source_path": str(args.source) if args.source is not None else None,
        "function_name": args.function,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "runner": args.runner,
        "calibration_path": str(args.calibration) if args.calibration is not None else None,
        "model.offline_responses_path": str(args.offline_responses) if args.offline_responses is not None else None,
        "proposal_only": args.proposal_only,
    }


def _proposal_only(cfg, argv, devices, client):
    """Read reference text and save an unverified proposal. No workload import, runner or evaluator."""
    from providers import ProposalFailure, ProposalProvider
    from context_builder import build_context, workload_contract, ContextError
    if cfg.budgets.max_candidates == 0:
        raise ProposalFailure("BUDGET_BLOCKED", "proposal-only mode requires a positive candidate budget")
    if cfg.source_path:
        source = read_python_source(cfg.source_path)
        workload_id, function = f"source:{Path(cfg.source_path).stem}", cfg.function_name
        workload = {"id": workload_id, "split": "development", "output_contract": "not evaluated in proposal-only mode",
                    "tolerance": None}
    else:
        spec = _registry().get(cfg.workload_id)
        source, workload_id, function = spec.reference_source, spec.workload_id, spec.function_name
        workload = workload_contract(spec)
    store = RunStore(output_dir_of(cfg), budgets=cfg.budgets)
    cfg = dataclasses.replace(cfg, output_dir=str(store.path), workload_id=workload_id, function_name=function)
    store.initialize(source, config=to_dict(cfg), manifest={
        "argv": list(argv), "mode": "proposal_only", "scientific": False,
        "non_scientific_reasons": ["proposal only; no correctness or performance evaluation"],
        "workload": {"id": workload_id, "reference_sha256": __import__("hashlib").sha256(source.encode()).hexdigest()},
        "environment": collect_environment(REPO, devices)})
    provider = ProposalProvider(client, cfg.model, cfg.budgets, reserve=store.reserve_request,
                                record=store.record_request)
    try:
        snapshot = store.context_inputs()
        try:
            context = build_context(function_name=function, workload=workload,
                baseline_sha256=__import__("hashlib").sha256(source.encode()).hexdigest(),
                current_source=source, current_parent_id=None, environment=snapshot.pop("environment"),
                evidence=snapshot, selected_backend=cfg.backend, context_condition=cfg.context_condition,
                max_source_bytes=cfg.budgets.max_source_bytes, remaining_budgets=provider.remaining_budget(),
                fits=provider.context_fits, redact=provider._redact)
        except ContextError as error:
            raise ProposalFailure("CONTEXT_BLOCKED", str(error)) from error
        proposal = provider.propose(context)
    except ProposalFailure as error:
        store.record_provider_stop(error.status, str(error), provider.summary())
        raise
    store.save_proposal(proposal, provider.events[-1]["request"])
    print(f"Unverified proposal saved to {store.path / 'proposed.py'}; no code was evaluated.")
    return store


def main(argv=None):
    load_dotenv(Path(".env.local"))
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.finalize_run is not None:
        if any(getattr(args, name) is not None for name in (
                "config", "iterations", "provider", "model", "workload", "source", "function", "backend",
                "runner", "calibration", "offline_responses", "proposal_only")) or args.plot:
            parser.error("final evaluation inherits the frozen run config; only output-dir, require-gpu and final-correctness-only apply")
        from finalization import _load_selection, finalize_run
        from heldout import HeldoutError
        try:
            final_cfg = _load_selection(args.finalize_run)[4]
            devices = select_backend(final_cfg.backend)
            if args.require_gpu and not has_gpu(devices):
                parser.error("--require-gpu: frozen run does not select an available GPU")
            path, report = finalize_run(args.finalize_run, output_dir=args.output_dir,
                                       correctness_only=args.final_correctness_only)
        except (HeldoutError, StorageError, ConfigError, BackendUnavailable, OSError, ValueError, KeyError) as error:
            print(f"error: {error}", file=sys.stderr)
            sys.exit(2)
        print(f"Final release: {path / 'release.json'}; {report['status']}; "
              f"correctness={report['correctness_passed']}; performance_verified={report['performance_verified']}")
        return path, report
    if args.final_correctness_only:
        parser.error("--final-correctness-only requires --finalize-run")

    try:
        cfg = resolve_run_config(_cli_overrides(args), args.config)
    except ConfigError as error:
        parser.error(str(error))

    if cfg.split != "development":
        parser.error("search requires the development split; use --finalize-run for final evaluation")
    if cfg.source_path is None and cfg.budgets.max_candidates > 0:
        from heldout import HeldoutError, check_search_allowed
        try:
            check_search_allowed(_registry().get(cfg.workload_id))
        except HeldoutError as error:
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

    from providers import ProviderError, preflight, require_live_budget
    client = None
    try:
        if cfg.budgets.max_candidates > 0:
            preflight(cfg.model, cfg.budgets)
            client = client_factory(cfg.model)
            require_live_budget(cfg.model, cfg.budgets, client)
    except ProviderError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)

    if cfg.proposal_only:
        try:
            return _proposal_only(cfg, sys.argv[1:] if argv is None else argv, devices, client)
        except (ProviderError, StorageError) as error:
            print(f"error: {error}", file=sys.stderr)
            sys.exit(2)

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
        client,
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
        prompt_framing=cfg.prompt_framing,
        budgets=cfg.budgets,
        model_settings=cfg.model,
    )
    evaluator = controller.evaluator
    # The correctness policy is known once the workload is: record it in the resolved config (AIC-002).
    cfg = dataclasses.replace(cfg, workload_id=evaluator.spec.workload_id,
                              correctness_policy_hash=evaluator.policy_hash, output_dir=str(controller.output_dir))

    # Record the run before any LLM call.
    scientific, reasons = scientific_status(cfg, evaluator.spec, runner)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "run_config_sha256": config_hash(cfg),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "backend_requested": cfg.backend,
        "scientific": scientific,
        "non_scientific_reasons": reasons,
        "evaluation_split": evaluation_commitment(evaluator.spec),
        "workload": {"id": evaluator.spec.workload_id, "reference_sha256": evaluator.baseline_sha256,
                     "correctness_policy_hash": evaluator.policy_hash,
                     "split": "development", "cases": evaluator.case_names,
                     "heldout_version": evaluator.spec.heldout_version,
                     "heldout_cases": len(evaluator.spec.heldout_seeds) * len(evaluator.spec.heldout_shapes)},
        "runner": runner.identity(),
        "environment": collect_environment(REPO, devices),
    }
    try:
        controller.store.initialize(evaluator.baseline_source, config=to_dict(cfg), manifest=manifest)
        controller.start()
    except BaselineFailed as error:
        print(f"error: baseline failed, no proposal requested: {error}", file=sys.stderr)
        sys.exit(2)
    except StorageError as error:
        print(f"error: storage: {error}", file=sys.stderr)
        sys.exit(2)
    try:
        history = controller.agent_loop(model=cfg.model.model_id, max_tokens=cfg.model.max_tokens)
    except StorageError as error:
        print(f"error: storage: {error}", file=sys.stderr)
        sys.exit(2)
    if controller.stop_reason:
        print(f"Run stopped: {controller.stop_reason}")

    totals = controller.usage_totals()
    print(f"Provider {cfg.model.provider} ({cfg.model.model_id}): {totals['requests']} requests, "
          f"{totals['input_tokens']} known input / {totals['output_tokens']} known output tokens; "
          f"{totals['accounted_tokens']} conservatively accounted, "
          f"{totals['unknown_usage_requests']} requests with incomplete usage.")
    print(f"Completed {len(history) - 1} candidate attempts.")
    print(f"Stop: {controller.result['status']}: {controller.result['reason']}")
    print(f"Verified best: {controller.output_dir / 'best.py'}")
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
