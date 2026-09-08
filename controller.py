class Controller:
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.history = []

    def provide_to_agent(self, current_code, current_runtime):
        return {
            "current_code": current_code,
            "current_runtime": current_runtime,
            "history": self.history.copy(),
        }

    def run(self, initial_code, baseline_runtime):
        
        best_code = initial_code
        best_runtime = baseline_runtime
        self.history = []
        evaluate = self.config["evaluate"]
        max_iterations = self.config.get("max_iterations", 10)

        for iteration in range(max_iterations):

            context = self.provide_to_agent(best_code, best_runtime)
            generated_code = None
            try:
                generated_code = self.agent.propose_code_by_agent(context)
                correct, runtime = evaluate(generated_code)
            except Exception as error:
                # Save the error and move on to the next attempt.
                self.history.append({
                    "iteration": iteration + 1,
                    "code": generated_code,
                    "error": str(error),
                })
                continue

            accepted = False
            if correct and runtime < best_runtime:
                best_code = generated_code
                best_runtime = runtime
                accepted = True

            self.history.append({
                "iteration": iteration + 1,
                "code": generated_code,
                "correct": correct,
                "runtime": runtime,
                "accepted": accepted,
            })

        return best_code, best_runtime
