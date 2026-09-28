
import argparse
import inspect
import os
from pathlib import Path

import anthropic
import jax
import jax.numpy as jnp
from dotenv import load_dotenv

from controller import Controller, read_python_source

# Read controller.md to see how it works!
#client = anthropic.Anthropic()


# Initialize parameters and PROMPTS

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ITERATIONS = 10 # Only edit this one if we want more iterations


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


def build_controller(client, max_iterations=DEFAULT_ITERATIONS, source_path=None, function_name="bad_func", output_dir=None):
    """Build the small forward-only workload used by the first milestone."""
    if source_path is not None:
        initial_code, reference_func, args = _load_source_workload(source_path, function_name)
    else:
        initial_code = """
import jax.numpy as jnp

def bad_func(x, y):
    return x + y
""".strip()

        namespace = {}
        exec(initial_code, namespace)
        reference_func = jax.jit(namespace[function_name])
        x_test = jnp.arange(32, dtype=jnp.float32)
        y_test = jnp.ones(32, dtype=jnp.float32)
        args = (x_test, y_test)

    prompt = (
        "You optimize pure JAX functions through the existing evaluator. "
        "Preserve the function signature and numerical behavior. Do not use "
        "CUDA-only libraries, custom kernels, approximations, or changes to "
        "precision. Return only raw Python source, without Markdown fences or "
        "escaped quotes. The evaluator applies jax.jit itself, so do not add "
        "@jax.jit or call jax.jit in the candidate. Inspect the current code "
        "and history before proposing a candidate."
    )
    # Initialize agent
    return Controller(
        agent="Claude",
        prompt=prompt,
        reference_func=reference_func,
        args=args,
        initial_code=initial_code,
        client=client,
        function_name=function_name,
        max_iterations=max_iterations,
        output_dir=output_dir,
        source_path=source_path,
    )


def main(argv=None):
    load_dotenv(Path(".env.local"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--model", default=os.getenv("CLAUDE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--source", type=Path, help="trusted Python workload file to optimize")
    parser.add_argument("--function", default="bad_func", help="function name in --source")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/latest"),
        help="directory for per-iteration JSON artifacts",
    )
    parser.add_argument("--plot", action="store_true", help="show the runtime history plot")
    parser.add_argument("--require-gpu", action="store_true", help="fail if JAX cannot use a GPU")
    args = parser.parse_args(argv)

    devices = jax.devices()
    if args.require_gpu and not any(device.platform == "gpu" for device in devices):
        parser.error(f"--require-gpu was requested, but JAX only sees: {devices}")
    print(f"JAX devices: {devices}")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing; put it in .env.local or the environment")

    client = anthropic.Anthropic()
    controller = build_controller(
        client,
        max_iterations=args.iterations,
        source_path=args.source,
        function_name=args.function,
        output_dir=args.output_dir,
    )
    # agent_controller.agent_loop() # JUST A LOOP FOR TESTING! NOT THE ACTUAL LOOP. THE LOOP SHOULD NOT BE INSIDE
    # THE CONTROLLER.
    history = controller.agent_loop(model=args.model)

    print(f"Completed {len(history) - 1} candidate evaluations.")
    print(f"Best runtime: {controller.best_runtime}")
    print(
        "Accepted candidates: "
        f"{sum(item['accepted'] for item in history[1:])}"
    )
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
