"""Gemini on Vertex AI, billed to the project's GCP credits.

Vertex rather than an AI Studio API key: the education credits live in a GCP
billing account, and only the Vertex path draws on them. Authentication is
Application Default Credentials (``gcloud auth application-default login``), so
no key material ever enters the repository - which matters, because this repo is
public.
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

from halo.agent.llm import LLMResponse, OptimizationProposal, estimate_cost
from halo.config import LLMConfig
from halo.types import Usage


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
        response = self.client.models.generate_content(
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
        )
        latency = time.perf_counter() - start

        meta = response.usage_metadata
        input_tokens = getattr(meta, "prompt_token_count", 0) or 0
        # Reasoning models bill thinking tokens at the output rate, so they belong
        # in the cost figure even though they never appear in the response.
        output_tokens = (getattr(meta, "candidates_token_count", 0) or 0) + (
            getattr(meta, "thoughts_token_count", 0) or 0
        )

        parsed = response.parsed
        if parsed is not None and not isinstance(parsed, OptimizationProposal):
            parsed = OptimizationProposal.model_validate(parsed)

        return LLMResponse(
            parsed=parsed,
            raw_text=response.text or "",
            usage=Usage(
                model=self.cfg.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=estimate_cost(self.cfg.model, input_tokens, output_tokens),
                latency_s=latency,
            ),
        )
