import anthropic
from controller import Controller

# Read controller.md to see how it works!

client = anthropic.Anthropic()

messages = []


# Initialize parameters and PROMPTS

exit = False
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
agent_controller = Controller(agent_name, default_prompt_v1)
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

# 5 #TODO save the history for this agent and its prompt

# 6 Make a graph on how the average execution time VS iterations