import difflib
from copy import deepcopy
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import benchmark


class Controller:
    def __init__(self, agent: str, prompt, reference_func, args, initial_code, function_name="kernel", max_iterations=10):
        """Store the agent, prompt, benchmark data, and optimization state."""
        self.agent = agent
        self.init_prompt = prompt
        self.reference_func = reference_func #reference funktion
        self.args = args
        self.function_name = function_name
        self.max_iterations = max_iterations
        self.history = []
        self.iteration = 0
        self.best_code = None
        self.best_runtime = None
        self.initial_code = initial_code
        self.initHist(initial_code) #initialize history in the beginning

    def initHist(self, original_code):


        """Evaluate the original code and create the first history entry."""
        self.history = []
        self.iteration = 0

        #kolla om kod är syntaxically korrekt och om den överstämmer med det avsedda outputen
        syntax_correct, error, output_correct, runtime_mean, runtime_std = self.current_runtime(original_code)
        accepted = syntax_correct and output_correct

        if accepted: #om ok
            self.best_code = original_code
            self.best_runtime = runtime_mean
        else:
            self.best_code = None
            self.best_runtime = None

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": 0, "code": original_code, "runtime_mean": runtime_mean, "runtime_std": runtime_std, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": "No changes", "error": error})
        return accepted

    def updateHist(self, new_code):
        """ 
            Updates history AFTER an iteration is completed.    
        """       
        self.iteration += 1
        syntax_correct, error, output_correct, runtime_mean, runtime_std = self.current_runtime(new_code)
        changes = self.changesPython(self.best_code, new_code)
        accepted = syntax_correct and output_correct and self.best_runtime is not None and runtime_mean < self.best_runtime

        if accepted:
            self.best_code = new_code
            self.best_runtime = runtime_mean

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": self.iteration, "code": new_code, "runtime_mean": runtime_mean, "runtime_std": runtime_std, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": changes, "error": error})
        return accepted

    def provideTools(self):
        """Return the tools that the controller exposes to the agent."""
        return [{"name": "get_history", "description": 
                 "Get all optimization attempts and their results.", "input_schema": {"type": "object", "properties": {}}}]



    def current_runtime(self, new_code):
        """Check syntax and output correctness, then return the measured runtime."""
        return benchmark.evaluate(new_code, self.reference_func, self.args, function_name=self.function_name)



    def changesPython(self, old_code, new_code):
        """Return a readable diff between the current best code and new code."""
        if old_code is None:
            return "No previous valid code"

        changes = difflib.unified_diff(old_code.splitlines(), new_code.splitlines(), fromfile="best.py", tofile="candidate.py", lineterm="")
        result = "\n".join(changes)
        return result if result else "No changes"


    def provide_hist(self):
        """copy history"""
        return deepcopy(self.history)


    def provide_prompt(self):
        """ Provides all prompts to agent """

        #TODO Should also return the prompt of each iteration -> work as memory (future)

        return self.init_prompt

    def import_info(self):
        """ Imports the running function """

    def agent_loop(self):
        """ Performs the agent loop with anthropic LLM """
        iteration = 0

        # initialize params etc

        namespace = {}
        #exec(self.reference_funcinitial_code, namespace)


        while iteration <= self.max_iterations:


            #1. Have the agent look at the code with initial_code

            to_agent_1 = self.initial_code

            #2. Give the agent instructions

            to_agent_2 = self.provide_prompt()

            #3. Give agent tools

            to_agent_3 = self.provideTools()

            #4. Make it change the code

            to_agent_4 = 1 #TODO

            #5. Update
            new_code = self.initial_code
            self.updateHist(new_code)

            #Prints and counts
            print(f"Iteration: {iteration} completed. ")
            iteration += 1



    def graph_it(self):
        """ Graphs the benchmarks vs iterations """

        length_of_executed_iterations = len(self.history)
        execution_hist_mean = []
        execution_hist_std = []

        for i in range(length_of_executed_iterations):
            execution_hist_mean.append(self.history[i]["runtime_mean"])
            execution_hist_std.append(self.history[i]["runtime_std"])

        run_time_mean = jnp.array(execution_hist_mean)
        run_time_std = jnp.array(execution_hist_mean)
        run_iterations = jnp.arange(0, len(run_time_mean))



        plt.plot(run_iterations, run_time_mean, label = "Runtime VS agent iteration")
        plt.fill_between(
            run_iterations,
            run_time_mean - 0.5 * run_time_std,
            run_time_mean + 0.5 * run_time_std,
            alpha=0.5,
            label=r"$\pm 1$ 0.5 * standard deviation"
        )
        plt.grid
        plt.legend()
        plt.show()