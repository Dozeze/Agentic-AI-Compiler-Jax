import anthropic
import jax
import jax.numpy as jnp

from controller import *

client = anthropic.Anthropic()



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



#
agent_controller = Controller(agent_name, default_prompt_v1, reference_func, 
                              args, initial_code, client, function_name="bad_func", 
                              max_iterations=num_iterations)
agent_controller.agent_loop() # JUST A LOOP FOR TESTING! NOT THE ACTUAL LOOP. THE LOOP SHOULD NOT BE INSIDE
# THE CONTROLLER.
agent_controller.graph_it()




# TODO - Create an agent

# Agent should: - Get controller in order to USE function HOWEVER it wants
# Then based on this, it should be able to test strategies to make the code better
# In the controller, we should also implement a max_iter, which is the max number of times
# it can run tests on the hardware.

# Then we should have ANOTHER loop for the agent, and the number of loops, N, is the number of
# times we ask the agent to improve the code. 