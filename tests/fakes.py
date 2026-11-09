"""Deterministic offline stand-in for anthropic.Anthropic. Never touches the network."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


def tool_use(block_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    """A content block shaped like an Anthropic `tool_use` block."""
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def text(value: str) -> SimpleNamespace:
    """A content block shaped like an Anthropic `text` block."""
    return SimpleNamespace(type="text", text=value)


@dataclass
class FakeMessages:
    script: list[list[SimpleNamespace]]          # one turn is popped per create() call
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs):
        # The controller passes the SAME messages list object on every call and appends to it in place,
        # so store a shallow snapshot of what was sent at this moment.
        self.calls.append({**kwargs, "messages_snapshot": list(kwargs.get("messages", []))})
        blocks = self.script.pop(0) if self.script else []
        return SimpleNamespace(content=blocks, stop_reason="tool_use" if blocks else "end_turn",
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))


class FakeAnthropic:
    """Replays a fixed script of assistant turns; an exhausted script returns an empty turn."""

    def __init__(self, script, **_ignored):
        self.messages = FakeMessages(list(script))


def proposal(source: str, **fields) -> dict:
    """A complete benchmark_code input (Proposal v1) around `source`."""
    return {"hypothesis": "test hypothesis", "transformation": "test transformation",
            "rationale": "test rationale", "expected_effect": "none expected",
            "candidate_source": source, **fields}
