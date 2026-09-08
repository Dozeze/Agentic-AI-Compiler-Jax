import os
from getpass import getpass

import jax
import jax.numpy as jnp
from openai import OpenAI

from agent import Agent
from benchmark import evaluate, timeit_version_1
from controller import Controller


def main():
    # Read the key from the environment, or ask for it without showing it.
    api_key = os.environ.get("OPENAI_API_KEY") or getpass("OpenAI API key (hidden): ")
    client = OpenAI(api_key=api_key)

    def generate(prompt):
        print("Asking OpenAI for a faster implementation...")
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions="Return only Python code. Do not use Markdown code fences.",
            input=prompt,
        )
        return response.output_text

    initial_code = """
import jax.numpy as jnp

def bad_func(x, y):
    A = x + y
    B = x + y
    C = x + y
    D = A + B + C + C - C
    E = C - A
    return E
"""
    namespace = {}
    exec(initial_code, namespace)
    reference_func = jax.jit(namespace["bad_func"])
    X_test = jnp.arange(32 * 128, dtype=jnp.float32).reshape(32, 128) / 1000
    Y_test = jnp.cos(X_test)

    def evaluate_code(code):
        return evaluate(
            code, reference_func, X_test, Y_test, function_name="bad_func"
        )

    config = {"evaluate": evaluate_code, "max_iterations": 3}
    controller = Controller(Agent(generate), config)
    baseline_runtime = timeit_version_1(reference_func, X_test, Y_test)
    best_code, best_runtime = controller.run(initial_code, baseline_runtime)

    print("\nAttempts:")
    for attempt in controller.history:
        print(attempt)
    print(f"\nBaseline: {baseline_runtime * 1e6:.2f} µs")
    print(f"Best: {best_runtime * 1e6:.2f} µs")
    print("\nBest code:")
    print(best_code)


if __name__ == "__main__":
    main()
