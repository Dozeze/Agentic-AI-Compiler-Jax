import difflib
from copy import deepcopy

import benchmark


class Controller:
    def __init__(self, agent: str, prompt, reference_func, args, function_name="kernel", max_iterations=10):
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

    def initHist(self, original_code):


        """Evaluate the original code and create the first history entry."""
        self.history = []
        self.iteration = 0

        #kolla om kod är syntaxically korrekt och om den överstämmer med det avsedda outputen
        syntax_correct, error, output_correct, runtime = self.current_runtime(original_code)
        accepted = syntax_correct and output_correct

        if accepted: #om ok
            self.best_code = original_code
            self.best_runtime = runtime
        else:
            self.best_code = None
            self.best_runtime = None

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": 0, "code": original_code, "runtime": runtime, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": "No changes", "error": error})
        return accepted

    def updateHist(self, new_code):
        """ 
            Updates history AFTER an iteration is completed.    
        """       
        self.iteration += 1
        syntax_correct, error, output_correct, runtime = self.current_runtime(new_code)
        changes = self.changesPython(self.best_code, new_code)
        accepted = syntax_correct and output_correct and self.best_runtime is not None and runtime < self.best_runtime

        if accepted:
            self.best_code = new_code
            self.best_runtime = runtime

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": self.iteration, "code": new_code, "runtime": runtime, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": changes, "error": error})
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
        """kopiera history"""
        return deepcopy(self.history)


    def provide_prompt(self):
        return self.init_prompt
