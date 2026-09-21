import anthropic
from controller import Controller

# Read controller.md to see how it works!

client = anthropic.Anthropic()

messages = []

# Prompts

# Version 1 - only for a .py-file
default_prompt_v1 = (
"You are an agent for our system. Your purpose is to write faster code in a JAX environment. "
"Edit an unoptimized .py file. You have ONE function for testing that reports its performance. "
"The environment is GPU-based with jax[cuda12], so use parallelization where advantageous."
)



# Initialize parameters
exit = False
iteration = 0
num_iterations = 10


# Initialize agent
agent_controller = Controller("Agent 47", default_prompt_v1)
agent_controller.initHist("Code goes here")


# Agent does rest
while (not exit):

    # Iteration count
    if (iteration == (num_iterations + 1)): # Exits after num_iterations iterations
        exit = True
        break
    else:
        pass

    # 1. Give agent controller #TODO
    # 1.1 Provide history

    # 2. Give agent tools #TODO
    # 2.1 

    # 3. Edit the python file

    # 3.1 Make the agent run the edit-function s.t. it edits the .py file

    # 3.2 Copy the .py file

    # 4. Save this version to the controller
    agent_controller.updateHist("New code")


    #Prints and counts
    print(f"Iteration: {iteration} completed. ")
    iteration += 1