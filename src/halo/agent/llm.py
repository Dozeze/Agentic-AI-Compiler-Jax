"""Provider-agnostic LLM interface and cost accounting.

The project plan commits to comparing providers and reasoning configurations on
both quality *and* cost, so every call records its tokens, its dollars and its
latency from the first commit. Retrofitting cost tracking after the experiments
have run means re-running the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from halo.types import Usage

#: USD per million tokens, (input, output). List prices; verify against the
#: provider's pricing page before quoting these in the report.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class OptimizationProposal(BaseModel):
    """The structured response every provider is asked to produce.

    Full-file replacement rather than a patch: unified diffs from language models
    fail to apply often enough that patch-application errors would dominate the
    failure statistics and tell us nothing about optimization quality.
    """

    analysis: str = Field(
        description="What the profile and HLO summary say about where time goes."
    )
    hypothesis: str = Field(
        description="The single change being made and why it should be faster."
    )
    new_candidate_source: str = Field(
        description="Complete replacement contents of candidate.py."
    )
    expected_speedup: float = Field(
        description="Predicted speedup over the current version, e.g. 1.5."
    )
    risk: str = Field(
        description="What could make this slower or numerically wrong."
    )


@dataclass(frozen=True)
class LLMResponse:
    parsed: OptimizationProposal | None
    raw_text: str
    usage: Usage


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    #: Opaque reasoning state some providers attach to a call and expect back on
    #: the next turn. Carried verbatim; never inspected.
    signature: str | None = None


@dataclass(frozen=True)
class Turn:
    """One model turn in a tool conversation."""

    calls: tuple[ToolCall, ...]
    text: str
    usage: Usage
    #: The provider's own reason the turn ended (STOP, MAX_TOKENS,
    #: MALFORMED_FUNCTION_CALL, ...). Diagnostic; carried into the transcript.
    finish_reason: str = ""


class ToolClient(Protocol):
    """A provider that can hold a tool-calling conversation.

    ``transcript`` is a list of plain dicts so it can be written to disk as-is:
    ``{"role": "user", "text"}``, ``{"role": "model", "text", "calls"}`` and
    ``{"role": "tool", "results": [{"name", "response"}]}``.
    """

    def converse(self, system: str, transcript: list[dict], tools: list[dict]) -> Turn: ...
