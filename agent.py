import anthropic
import jax
import jax.numpy as jnp

from controller import Controller

# Read controller.md to see how it works!

client = anthropic.Anthropic()

messages = []


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

agent_controller = Controller(agent_name, default_prompt_v1, reference_func, args, function_name="bad_func", max_iterations=num_iterations)
agent_controller.initHist(initial_code)


# Agent does rest
while iteration < num_iterations:

    # 1. Give agent controller #TODO
    # 1.1 Provide history

    # 2. Give agent tools #TODO
    # 2.1 

    # 3. Edit the python file

    # 3.1 Make the agent run the edit-function s.t. it edits the .py file

    # 3.2 Copy the .py file

    # 4. Save this version to the controller
    # Use the original code until the Claude proposal step is implemented.
    new_code = initial_code
    agent_controller.updateHist(new_code)


    #Prints and counts
    print(f"Iteration: {iteration} completed. ")
    iteration += 1

# 5 #TODO save the history for this agent and its prompt

# 6 Make a graph on how the average execution time VS iterations
