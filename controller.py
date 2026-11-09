import difflib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import benchmark
import json

from contracts import (ACCEPTED, MEASUREMENT_UNSTABLE, POLICY_REJECTED, PROPOSAL_INVALID, VALID_NO_CHANGE,
                       VALID_SLOWER, Budgets, ContractError, parse_proposal)
from source_policy import check_source


# ---------------------------------------------------------------------------
# Acceptance decision (AIC-006, AIC-1.1 §5.7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    state: str        # a terminal state from contracts
    reason: str


def decide(analysis, delta) -> Decision:
    """Rule on ONE stable-or-not measurement session of a correct candidate against the incumbent.

    Correctness is decided before this is ever called. A session must be stable; then the candidate is
    provisionally better only if the lower bound of its speedup exceeds 1 + delta, where delta is the
    frozen A/A calibration. It is slower only if the upper bound is below 1 / (1 + delta); anything in
    between is indistinguishable from noise.
    """
    if not analysis.stability.stable:
        return Decision(MEASUREMENT_UNSTABLE, "; ".join(analysis.stability.reasons))
    s = analysis.speedup
    interval = f"speedup {s.speedup:.3f}x, {s.confidence:.0%} CI [{s.ci_low:.3f}, {s.ci_high:.3f}]"
    if s.ci_low > 1 + delta:
        return Decision(ACCEPTED, f"{interval}; lower bound clears 1 + delta = {1 + delta:.3f}")
    if s.ci_high < 1 / (1 + delta):
        return Decision(VALID_SLOWER, f"{interval}; upper bound below 1 / (1 + delta)")
    return Decision(VALID_NO_CHANGE, f"{interval}; not distinguishable from noise (delta = {delta:.3f})")


def judge(measure, delta, protocol):
    """(Decision, analyses) for a correct candidate, from fresh sessions produced by `measure()`.

    An unstable session gets `protocol.unstable_retries` fixed retries, never unbounded reruns looking for
    a win. A provisional win must be repeated by `protocol.confirmation_sessions` further fresh sessions
    before promotion; all of them must meet the same rule.
    """
    analyses = []
    for _ in range(1 + protocol.unstable_retries):
        analyses.append(measure())
        decision = decide(analyses[-1], delta)
        if decision.state != MEASUREMENT_UNSTABLE:
            break
    if decision.state != ACCEPTED:
        return decision, analyses
    for i in range(protocol.confirmation_sessions):
        analyses.append(measure())
        confirmation = decide(analyses[-1], delta)
        if confirmation.state != ACCEPTED:
            state = MEASUREMENT_UNSTABLE if confirmation.state == MEASUREMENT_UNSTABLE else VALID_NO_CHANGE
            return Decision(state, f"provisional win not confirmed in fresh session {i + 1}: "
                                   f"{confirmation.reason}"), analyses
    return Decision(ACCEPTED, f"{decision.reason}; confirmed in {protocol.confirmation_sessions} fresh "
                              f"session(s)"), analyses


def verify_winner(measure_against_baseline, shapes, delta_for, protocol):
    """Final check of a frozen winner (AIC-1.1 §5.8): `protocol.final_sessions` fresh sessions against the
    ORIGINAL baseline on every declared shape. Returns (reproduced, {shape: [Decision, ...]}); a single
    session that does not clear 1 + delta on any shape means the effect did not reproduce.
    """
    per_shape = {}
    for shape in shapes:
        per_shape[shape] = [decide(measure_against_baseline(shape), delta_for(shape))
                            for _ in range(protocol.final_sessions)]
    reproduced = all(d.state == ACCEPTED for decisions in per_shape.values() for d in decisions)
    return reproduced, per_shape


def read_python_source(source_path):
    """Read a Python workload file and return its complete source as text."""
    path = Path(source_path)
    if path.suffix != ".py":
        raise ValueError(f"Expected a .py source file, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Python source file does not exist: {path}")
    return path.read_text(encoding="utf-8")


class Controller:
    def __init__(self, agent: str, prompt, reference_func, args, initial_code, client=None, function_name="kernel", max_iterations=10, output_dir=None, source_path=None):
        """Store the agent, prompt, benchmark data, and optimization state."""
        self.agent = agent
        self.init_prompt = prompt
        self.client = client
        self.reference_func = reference_func #reference funktion
        self.args = args
        self.function_name = function_name
        self.max_iterations = max_iterations
        self.history = []
        self.iteration = 0
        self.best_code = None
        self.best_runtime = None
        self.initial_code = initial_code
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.source_path = str(source_path) if source_path is not None else None
        self.initHist(initial_code) #initialize history in the beginning

    def initHist(self, original_code):


        """Evaluate the original code and create the first history entry."""
        self.history = []
        self.iteration = 0

        #kolla om kod är syntaxically korrekt och om den överstämmer med det avsedda outputen
        result = self.current_runtime(original_code)
        syntax_correct = result["syntax_correct"]
        error = result["error"]
        output_correct = result["output_correct"]
        runtime_mean = result["runtime_mean"]
        runtime_std = result["runtime_std"]

        accepted = syntax_correct and output_correct

        if accepted: #om ok
            self.best_code = original_code
            self.best_runtime = runtime_mean
        else:
            self.best_code = None
            self.best_runtime = None

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": 0, "code": original_code, "runtime_mean": runtime_mean, "runtime_std": runtime_std, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": "No changes", "error": error, "source_path": self.source_path, "function_name": self.function_name})
        self._report_runtime(self.history[-1])
        self._write_iteration_json(self.history[-1])
        return accepted

    def updateHist(self, new_code):
        """ 
            Updates history AFTER an iteration is completed.    
        """       
        self.iteration += 1
        result = self.current_runtime(new_code)

        syntax_correct = result["syntax_correct"]
        error = result["error"]
        output_correct = result["output_correct"]
        runtime_mean = result["runtime_mean"]
        runtime_std = result["runtime_std"]

        changes = self.changesPython(self.best_code, new_code)
        accepted = (
            syntax_correct
            and output_correct
            and (
                self.best_runtime is None
                or runtime_mean < self.best_runtime
            )
        )
        if accepted:
            self.best_code = new_code
            self.best_runtime = runtime_mean

        if error is None and not output_correct:
            error = "Output does not match the reference function"

        self.history.append({"iteration": self.iteration, "code": new_code, "runtime_mean": runtime_mean, "runtime_std": runtime_std, "syntax_correct": syntax_correct, "output_correct": output_correct, "accepted": accepted, "changes": changes, "error": error, "source_path": self.source_path, "function_name": self.function_name})
        self._report_runtime(self.history[-1])
        self._write_iteration_json(self.history[-1])
        return accepted
    
    #def provideTools(self):
    #    """Return the tools that the controller exposes to the agent."""
    #    return [{"name": "get_history", "description": 
    #             "Get all optimization attempts and their results.", "input_schema": {"type": "object", "properties": {}}}]
    
    def provideTools(self):
        return [
            {
                "name": "get_history",
                "description": (
                    "Get the complete optimization history. "
                    "Use this to see previous candidate implementations, "
                    "their correctness, runtime, and whether they were accepted."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_current_code",
                "description": (
                    "Get the current best JAX implementation being optimized."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            {
                "name": "benchmark_code",
                "description": (
                    "Submit one proposal: a hypothesis and a complete candidate source. The harness "
                    "checks the source against its policy (pure JAX only), then for correctness, and "
                    "only then measures it. A correct candidate becomes the current best only when it "
                    "is faster than the incumbent."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string",
                                       "description": "Why this change should be faster."},
                        "transformation": {"type": "string",
                                           "description": "The single change made, e.g. 'vectorise the loop'."},
                        "rationale": {"type": "string",
                                      "description": "The evidence (code, history) behind the hypothesis."},
                        "expected_effect": {"type": "string",
                                            "description": "Predicted effect; a hypothesis, not a measurement."},
                        "candidate_source": {
                            "type": "string",
                            "description": (
                                "Complete Python source of the candidate module: imports from jax, "
                                "jax.numpy, jax.lax, jax.nn, jax.scipy, math or functools only, and the "
                                "function itself. Raw source without Markdown fences, escaped quotes or "
                                "@jax.jit decorators; the evaluator applies JIT."
                            ),
                        },
                        "parent_id": {"type": "integer",
                                      "description": "Iteration this builds on (default: current best)."},
                        "constraints": {"type": "array", "items": {"type": "string"},
                                        "description": "Semantics the rewrite preserves."},
                        "notes": {"type": "string"},
                    },
                    "required": ["hypothesis", "transformation", "rationale", "expected_effect",
                                 "candidate_source"],
                    "additionalProperties": False,
                },
            },
        ]

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

    def _report_runtime(self, result):
        """Print one explicit runtime/decision line for every evaluated iteration."""
        runtime = result["runtime_mean"]
        std = result["runtime_std"]
        if runtime is None:
            timing = "runtime unavailable"
        else:
            timing = f"{runtime * 1e6:.2f} µs ± {std * 1e6:.2f} µs"
        decision = "accepted" if result["accepted"] else "rejected"
        error = f"; error: {result['error']}" if result["error"] else ""
        print(f"Iteration {result['iteration']}: {timing}; {decision}{error}")

    def _write_iteration_json(self, result):
        """Persist each source change and the complete history as JSON artifacts."""
        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        iteration_path = self.output_dir / f"iteration_{result['iteration']:04d}.json"
        history_path = self.output_dir / "history.json"
        for path, value in ((iteration_path, result), (history_path, self.history)):
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            temporary_path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
            temporary_path.replace(path)

    def provide_hist(self):
        """Return the history of all completed iterations to the agent"""
        return deepcopy(self.history)


    def provide_prompt(self):
        """ Provides all prompts to agent """

        #TODO Should also return the prompt of each iteration -> work as memory (future)
        #TODO ^- already inside provide_hist

        return self.init_prompt

    def import_info(self, source_path="algorithm.py"):
        """ Imports the running function """

        return read_python_source(source_path)

    def get_current_code(self):
        return self.best_code

    def benchmark_code(self, code):
        return self.current_runtime(code)

    def submit_proposal(self, tool_input):
        """Validate a proposal and its source before anything runs; then evaluate it.

        An invalid proposal or a policy violation is a terminal candidate state recorded in the history,
        and the candidate's code is never executed.
        """
        max_source_bytes = Budgets().max_source_bytes
        try:
            proposal = parse_proposal(tool_input, max_source_bytes=max_source_bytes)
        except ContractError as error:
            return self._record_rejection(PROPOSAL_INVALID, str(error), code=None, proposal=None)
        policy = check_source(proposal.candidate_source, self.function_name, max_source_bytes=max_source_bytes)
        if not policy.ok:
            return self._record_rejection(POLICY_REJECTED, policy.summary(), code=proposal.candidate_source,
                                          proposal=proposal)
        self.updateHist(proposal.candidate_source)
        self.history[-1]["proposal"] = _proposal_json(proposal)
        return self.history[-1]

    def _record_rejection(self, state, reason, *, code, proposal):
        self.iteration += 1
        entry = {"iteration": self.iteration, "state": state, "code": code, "runtime_mean": None,
                 "runtime_std": None, "syntax_correct": state != PROPOSAL_INVALID and "syntax error" not in reason,
                 "output_correct": False, "accepted": False, "changes": None, "error": reason,
                 "proposal": _proposal_json(proposal) if proposal else None,
                 "source_path": self.source_path, "function_name": self.function_name}
        self.history.append(entry)
        self._report_runtime(entry)
        self._write_iteration_json(entry)
        return entry

    def _dispatch_tool(self, name, tool_input):
        """Run one model-requested tool and return JSON-serializable data."""
        if name == "get_history":
            return {"history": self.provide_hist()}

        if name == "get_current_code":
            return {"code": self.get_current_code()}

        if name == "benchmark_code":
            return self.submit_proposal(tool_input)

        return {"error": f"Unknown tool: {name}"}

    def agent_loop(self, model="claude-haiku-4-5-20251001", max_tokens=2048):
        """Ask Claude for bounded candidates and keep only measured improvements."""
        if self.client is None:
            raise RuntimeError("An Anthropic client is required for agent_loop")

        # initialize params etc

        namespace = {}
        #exec(self.reference_funcinitial_code, namespace)

        messages = [{
            "role": "user",
            "content": (
                "Start the optimization run. Inspect the current implementation and "
                "history with the tools. Propose a complete replacement source file "
                f"for `{self.function_name}` and call benchmark_code to evaluate it. "
                "Only measured, correct candidates can become the best implementation. "
                f"You may make at most {self.max_iterations} candidate evaluations."
            ),
        }]
        max_turns = max(1, self.max_iterations * 3)
        candidate_count = 0

        for _ in range(max_turns):
            # 1. get_history()
            # 2. get_current_code()
            # 3. propose code
            # 4. benchmark_code(code)
            # 5. LLM sees result
            # 6.
            # to_agent_4 = 1 #TODO
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self.init_prompt,
                tools=self.provideTools(),
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            #5. Update
            # 5. Update the agent with the benchmark result through tool_result.

            tool_results = []
            evaluated_this_turn = False
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue

                name = block.name
                if name == "benchmark_code" and evaluated_this_turn:
                    result = {"error": "Only one benchmark_code call is allowed per turn"}
                else:
                    result = self._dispatch_tool(name, block.input)
                    if name == "benchmark_code":       # every submission spends a candidate slot
                        evaluated_this_turn = True
                        candidate_count += 1

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            if not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})
            if candidate_count >= self.max_iterations:
                break

            #Prints and counts

        return deepcopy(self.history)



    def graph_it(self):
        """ Graphs the benchmarks vs iterations """

        length_of_executed_iterations = len(self.history)
        execution_hist_mean = []
        execution_hist_std = []

        for i in range(length_of_executed_iterations):
            execution_hist_mean.append(self.history[i]["runtime_mean"])
            execution_hist_std.append(self.history[i]["runtime_std"])

        run_time_mean = jnp.array(execution_hist_mean)
        run_time_std = jnp.array(execution_hist_std)
        run_iterations = jnp.arange(0, len(run_time_mean))

        plt.plot(run_iterations, run_time_mean, label = "Runtime VS agent iteration")
        plt.fill_between(
            run_iterations,
            run_time_mean - run_time_std,
            run_time_mean +  run_time_std,
            alpha=0.5,
            label=r"$\pm 1$ 0.5 * standard deviation"
        )
        plt.grid()
        plt.legend()
        plt.show()


def _proposal_json(proposal):
    """The proposal as recorded in history, without its source (already in `code`)."""
    from dataclasses import asdict
    data = asdict(proposal)
    data.pop("candidate_source")
    return data
