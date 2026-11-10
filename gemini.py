"""Gemini on Vertex AI behind the Anthropic Messages surface (P8), billed to the GCP project.

`GeminiVertex(...).messages.create(model=, max_tokens=, system=, tools=, messages=)` accepts exactly what the
controller sends to Anthropic and returns blocks shaped like Anthropic's, so the controller's tool loop,
its history and the offline fakes stay provider-neutral:

  request   system -> system_instruction; tools[].input_schema -> FunctionDeclaration.parameters_json_schema;
            user text -> user parts; assistant text/tool_use blocks -> model text/function_call parts;
            tool_result blocks -> function_response parts (named after the call they answer)
  response  text parts -> {"type": "text"}; function_call parts -> {"type": "tool_use", id, name, input};
            stop_reason "tool_use" when there are calls, else "end_turn"; usage in/out tokens (thinking tokens
            count as output, which is how they are billed)

Gemini 2.5 returns a thought signature with its function calls and needs it back to keep its reasoning; it
rides on our tool_use block and is re-attached when the conversation is sent again. Transient Vertex errors
(429, 500, 503) are retried with backoff, as in the claude-experiment prototype; anything else raises.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

RETRY_STATUS = frozenset({429, 500, 503})
RETRY_DELAYS_S = (5, 15, 30, 60)


class GeminiVertex:
    def __init__(self, *, project: str, location: str, client=None, sleep=time.sleep):
        """`client` is a google.genai.Client; tests pass a stand-in. Credentials: Application Default."""
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=project, location=location)
        self.project, self.location = project, location
        self.messages = _Messages(client, sleep)


class _Messages:
    def __init__(self, client, sleep):
        self._client, self._sleep = client, sleep
        self._calls = 0

    def create(self, *, model, max_tokens, messages, system=None, tools=None, **_ignored):
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t.get("description", ""),
                                          parameters_json_schema=t["input_schema"]) for t in tools])] if tools else None,
            # The controller drives the loop one call at a time; the SDK must not run tools by itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))
            if tools else None,
        )
        contents = to_contents(messages)
        response = self._with_retry(lambda: self._client.models.generate_content(
            model=model, contents=contents, config=config))
        return self._to_message(response, model)

    def _with_retry(self, call):
        from google.genai import errors
        for delay in RETRY_DELAYS_S:
            try:
                return call()
            except errors.APIError as error:
                if error.code not in RETRY_STATUS:
                    raise
                self._sleep(delay)
        return call()

    def _to_message(self, response, model):
        candidate = response.candidates[0] if response.candidates else None
        parts = (candidate.content.parts if candidate and candidate.content else None) or []
        blocks = []
        for part in parts:
            if part.function_call is not None:
                self._calls += 1
                blocks.append(SimpleNamespace(type="tool_use", id=part.function_call.id or f"gemini_call_{self._calls}",
                                              name=part.function_call.name, input=dict(part.function_call.args or {}),
                                              thought_signature=part.thought_signature))
            elif part.text and not part.thought:
                blocks.append(SimpleNamespace(type="text", text=part.text))
        meta = response.usage_metadata
        usage = SimpleNamespace(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
            + (getattr(meta, "thoughts_token_count", 0) or 0))
        return SimpleNamespace(content=blocks, model=model, usage=usage,
                               stop_reason="tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn",
                               finish_reason=str(getattr(candidate, "finish_reason", "") or ""))


def _get(block, key, default=None):
    """Blocks are dicts (sent by the controller) or objects (returned by a provider and appended back)."""
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def to_contents(messages):
    """Anthropic-shaped messages -> google.genai Contents (see the module docstring)."""
    from google.genai import types
    names = {}                                   # tool_use id -> function name, for the responses
    contents = []
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            contents.append(types.Content(role="user" if role == "user" else "model",
                                          parts=[types.Part(text=content)]))
            continue
        parts = []
        for block in content:
            kind = _get(block, "type")
            if kind == "text" and _get(block, "text"):
                parts.append(types.Part(text=_get(block, "text")))
            elif kind == "tool_use":
                names[_get(block, "id")] = _get(block, "name")
                parts.append(types.Part(function_call=types.FunctionCall(name=_get(block, "name"),
                                                                         args=_get(block, "input") or {}),
                                        thought_signature=_get(block, "thought_signature")))
            elif kind == "tool_result":
                raw = _get(block, "content")
                try:
                    payload = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    payload = raw
                response = payload if isinstance(payload, dict) else {"result": payload}
                parts.append(types.Part.from_function_response(name=names.get(_get(block, "tool_use_id"), "unknown"),
                                                               response=response))
        if parts:
            contents.append(types.Content(role="model" if role == "assistant" else "user", parts=parts))
    return contents
