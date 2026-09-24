import anthropic
import jax
import jax.numpy as jnp

from controller import *

# Read controller.md to see how it works!

#client = anthropic.Anthropic()


# Initialize parameters and PROMPTS

iteration = 0
num_iterations = 10 # Only edit this one if we want more iterations

# Version 1 - only for a .py-file

default_prompt_v1 = (
"You are an agent for our system. Your purpose is to write faster code in a JAX environment. "
"Edit an unoptimized .py file. You have ONE function for testing that reports its performance. "
f"The environment is GPU-based with jax[cuda12], so use parallelization where advantageous. You will have {num_iterations}"
" number of iterations to complete this"
)


# Initialize agent
agent_name = "Version 1"
initial_code = """
import jax.numpy as jnp

def bad_func(x, y):
    return x + y
"""
namespace = {}
exec(initial_code, namespace)
reference_func = jax.jit(namespace["bad_func"])
X_test = jnp.arange(32, dtype=jnp.float32)
Y_test = jnp.ones(32, dtype=jnp.float32)
args = (X_test, Y_test)



#3 lines of code to run a agent
agent_controller = Controller(agent_name, default_prompt_v1, reference_func, 
                              args, initial_code, function_name="bad_func", 
                              max_iterations=num_iterations)
agent_controller.agent_loop()
agent_controller.graph_it()