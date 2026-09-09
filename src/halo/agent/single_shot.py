"""The MVP agent: one prompt, one structured response, one candidate."""

from __future__ import annotations

from halo.agent.context import SYSTEM, render
from halo.agent.llm import LLMClient
from halo.agent.protocol import Context
from halo.types import Proposal


class AgentError(RuntimeError):
    """The agent could not produce a usable proposal."""


def _strip_fences(source: str) -> str:
    """Remove a stray ```python fence.

    JSON mode makes this rare rather than impossible, and a fenced string is a
    syntax error the moment it is imported - cheap to defend against here.
    """
    text = source.strip()
    if not text.startswith("```"):
        return source
    lines = text.splitlines()
    if lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines[1:]) + "\n"


class SingleShotAgent:
    """Pure ``Context -> Proposal``. No tools, no benchmarking, no filesystem."""

    def __init__(
        self,
        client: LLMClient,
        min_speedup: float,
        name: str = "llm",
        context_level: str = "full",
    ) -> None:
        self.name = name
        self._client = client
        self._min_speedup = min_speedup
        self._context_level = context_level

    def propose(self, context: Context) -> Proposal:
        prompt = render(
            context, min_speedup=self._min_speedup, level=self._context_level
        )
        response = self._client.complete(SYSTEM, prompt)
        if response.parsed is None:
            raise AgentError(
                "model returned no parseable proposal; raw response begins: "
                f"{response.raw_text[:300]!r}"
            )

        parsed = response.parsed
        source = _strip_fences(parsed.new_candidate_source)
        if "def candidate" not in source:
            raise AgentError("proposed source does not define candidate()")

        return Proposal(
            source=source,
            analysis=parsed.analysis,
            hypothesis=parsed.hypothesis,
            expected_speedup=parsed.expected_speedup,
            risk=parsed.risk,
            usage=response.usage,
            prompt=prompt,
            raw_response=response.raw_text,
        )
