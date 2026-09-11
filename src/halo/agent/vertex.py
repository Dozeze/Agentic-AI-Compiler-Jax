"""Gemini on Vertex AI, billed to the project's GCP credits.

Vertex rather than an AI Studio API key: the education credits live in a GCP
billing account, and only the Vertex path draws on them. Authentication is
Application Default Credentials (``gcloud auth application-default login``), so
no key material ever enters the repository - which matters, because this repo is
public.
"""

from __future__ import annotations

import base64
import os
import time

from google import genai
from google.genai import errors, types

from halo.agent.llm import LLMResponse, OptimizationProposal, ToolCall, Turn, estimate_cost
from halo.config import LLMConfig
from halo.types import Usage


#: Vertex answers a burst of calls with 429 RESOURCE_EXHAUSTED; a sweep is a
#: burst. Retried with backoff; anything else raises and becomes a failed cell.
RETRY_STATUS = frozenset({429, 500, 503})
RETRY_DELAYS_S = (5, 15, 30, 60, 120, 120)


def _with_retry(call):
    for delay in RETRY_DELAYS_S:
        try:
            return call()
        except errors.APIError as exc:
            if exc.code not in RETRY_STATUS:
                raise
            time.sleep(delay)
    return call()


class VertexGemini:
    def __init__(self, cfg: LLMConfig) -> None:
        project = cfg.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "no GCP project configured: set GOOGLE_CLOUD_PROJECT or llm.project"
            )
        self.cfg = cfg
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", cfg.location),
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        response = _with_retry(lambda: self.client.models.generate_content(
            model=self.cfg.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_output_tokens,
                response_mime_type="application/json",
                response_schema=OptimizationProposal,
                # We use the pydantic model purely as a response schema. Without
                # this the SDK assumes we might want automatic function calling
                # and warns on every single call.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        ))
        usage = self._usage(response, time.perf_counter() - start)

        parsed = response.parsed
        if parsed is not None and not isinstance(parsed, OptimizationProposal):
            parsed = OptimizationProposal.model_validate(parsed)
        return LLMResponse(parsed=parsed, raw_text=response.text or "", usage=usage)

    def converse(self, system: str, transcript: list[dict], tools: list[dict]) -> Turn:
        start = time.perf_counter()
        response = _with_retry(lambda: self.client.models.generate_content(
            model=self.cfg.model,
            contents=[_to_content(entry) for entry in transcript],
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.cfg.temperature,
                max_output_tokens=self.cfg.max_output_tokens,
                tools=[
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=t["name"],
                                description=t["description"],
                                parameters_json_schema=t["parameters"],
                            )
                            for t in tools
                        ]
                    )
                ],
                # The loop is driven here, one call at a time, so every call is
                # logged and budgeted; and the model must act through a tool on
                # every turn - there is nothing else for it to say.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ),
            ),
        ))
        usage = self._usage(response, time.perf_counter() - start)

        calls, texts = [], []
        candidate = response.candidates[0] if response.candidates else None
        finish_reason = str(getattr(candidate, "finish_reason", "") or "")
        for part in (candidate.content.parts if candidate and candidate.content else []) or []:
            if part.function_call is not None:
                signature = getattr(part, "thought_signature", None)
                calls.append(
                    ToolCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args or {}),
                        signature=(
                            base64.b64encode(signature).decode() if signature else None
                        ),
                    )
                )
            elif part.text and not getattr(part, "thought", False):
                texts.append(part.text)
        return Turn(
            calls=tuple(calls), text="\n".join(texts), usage=usage, finish_reason=finish_reason
        )

    def _usage(self, response, latency: float) -> Usage:
        meta = response.usage_metadata
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        # Reasoning models bill thinking tokens at the output rate, so they belong
        # in the cost figure even though they never appear in the response.
        output_tokens = (getattr(meta, "candidates_token_count", 0) or 0) + (
            getattr(meta, "thoughts_token_count", 0) or 0
        )
        return Usage(
            model=self.cfg.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(self.cfg.model, input_tokens, output_tokens),
            latency_s=latency,
        )


def _to_content(entry: dict) -> types.Content:
    """One transcript entry as the SDK wants it. Thought signatures ride along on
    the model's own calls; Gemini 2.5 needs them back to keep its reasoning."""
    role = entry["role"]
    if role == "user":
        return types.Content(role="user", parts=[types.Part(text=entry["text"])])
    if role == "model":
        parts = [types.Part(text=entry["text"])] if entry.get("text") else []
        for call in entry.get("calls", []):
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(name=call["name"], args=call["args"]),
                    thought_signature=(
                        base64.b64decode(call["signature"]) if call.get("signature") else None
                    ),
                )
            )
        return types.Content(role="model", parts=parts)
    if role == "tool":
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(name=r["name"], response=r["response"])
                for r in entry["results"]
            ],
        )
    raise ValueError(f"unknown transcript role {role!r}")
