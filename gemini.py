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
rides on our tool_use block and is re-attached when a conversation is sent again. The structured provider
boundary uses a fresh context per call. Transport retries are disabled; only the budget boundary retries.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

class GeminiVertex:
    def __init__(self, *, project: str, location: str, client=None, request_timeout_s=30.0):
        """`client` is a google.genai.Client; tests pass a stand-in. Credentials: Application Default."""
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=project, location=location)
        self.project, self.location = project, location
        from providers import OfflineClient
        self.offline_transport = isinstance(client, OfflineClient)
        self.messages = _Messages(client, request_timeout_s)

    def with_options(self, *, max_retries, timeout):
        if max_retries != 0:
            raise ValueError("Gemini transport retries must be disabled")
        return GeminiVertex(project=self.project, location=self.location, client=self.messages._client,
                            request_timeout_s=timeout)


class _Messages:
    def __init__(self, client, timeout):
        self._client, self._timeout = client, timeout
        self._calls = 0

    def create(self, *, model, max_tokens, messages, system=None, tools=None, temperature=None,
               tool_choice=None, **_ignored):
        from google.genai import types
        config = types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=max(1, int(self._timeout * 1000)),
                                           retry_options=types.HttpRetryOptions(attempts=1)),
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t.get("description", ""),
                                          parameters_json_schema=t["input_schema"]) for t in tools])] if tools else None,
            # The controller drives the loop one call at a time; the SDK must not run tools by itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="ANY" if tool_choice else "AUTO",
                allowed_function_names=[tool_choice["name"]] if tool_choice else None))
            if tools else None,
        )
        contents = to_contents(messages)
        response = self._client.models.generate_content(model=model, contents=contents, config=config)
        return self._to_message(response, model)

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
        out, thoughts = getattr(meta, "candidates_token_count", None), getattr(meta, "thoughts_token_count", None)
        # Vertex implicit caching (P32): cached_content_token_count is a subset of prompt_token_count, billed
        # at a discount. Charging every prompt token at the full input price is an upper bound, so the cached
        # count is kept for diagnostics only and not reported as separate cache billing.
        usage = SimpleNamespace(input_tokens=getattr(meta, "prompt_token_count", None),
                                output_tokens=out + (thoughts or 0) if out is not None else None,
                                implicit_cached_input_tokens=getattr(meta, "cached_content_token_count", None))
        finish = str(getattr(candidate, "finish_reason", "") or "")
        stop = ("max_tokens" if "MAX_TOKENS" in finish else "refusal" if any(
            word in finish for word in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION")) else
            "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
        return SimpleNamespace(content=blocks, model=model, id=getattr(response, "response_id", None), usage=usage,
                               stop_reason=stop,
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
