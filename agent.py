class Agent:
    def __init__(self, generate):
        self.generate = generate

    def getPrompt(self):
        return "You are given JAX code. Please optimize it."

    def build_prompt(self, information):
        return f"""
        {self.getPrompt()}
        You are going to optimize JAX code for execution with XLA.

        CURRENT CODE:
        {information["current_code"]}

        CURRENT RUNTIME:
        {information["current_runtime"]} seconds

        PREVIOUS ATTEMPTS:
        {information["history"]}

        Generate a faster, syntactically correct JAX implementation.
        Keep the same function name, arguments, and results.
        Return only Python code, including any required imports.
        """.strip()

    def propose_code_by_agent(self, info):
        prompt = self.build_prompt(info)
        response = self.generate(prompt)
        return response.strip()
