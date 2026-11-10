"""LLM providers (team decision P8): one client per provider, all with the Anthropic Messages surface.

  anthropic          the Anthropic API with ANTHROPIC_API_KEY (unchanged default)
  anthropic-vertex   Claude on Google Cloud Vertex AI, billed to the GCP project (ADC, no Anthropic key)
  gemini-vertex      Gemini on Vertex AI, billed to the GCP project (ADC), through an adapter

Every client exposes `client.messages.create(model=, max_tokens=, system=, tools=, messages=)` and returns
content blocks shaped like Anthropic's (`text` / `tool_use`), so the controller's tool loop, the history
and the offline fakes are the same for all three. Vertex needs a GCP project (`model.project` or
GOOGLE_CLOUD_PROJECT) and Application Default Credentials (`gcloud auth application-default login`); no
key ever enters the repository.
"""
from __future__ import annotations

import os
import re

PROVIDERS = ("anthropic", "anthropic-vertex", "gemini-vertex")

# Defaults keep today's model on each path. On Vertex, a dated Claude snapshot uses `@` before the date.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "anthropic-vertex": "claude-haiku-4-5@20251001",
    "gemini-vertex": "gemini-2.5-flash",
}
DEFAULT_REGION = "global"


class ProviderError(RuntimeError):
    """The configured provider cannot be used: missing credentials or project."""


def vertex_model_id(model: str) -> str:
    """claude-haiku-4-5-20251001 -> claude-haiku-4-5@20251001; undated ids are unchanged on Vertex."""
    match = re.fullmatch(r"(claude-[a-z0-9-]+?)-(\d{8})", model)
    return f"{match.group(1)}@{match.group(2)}" if match else model


def gcp_project(settings) -> str:
    project = settings.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ProviderError(f"provider {settings.provider!r} needs a GCP project: set GOOGLE_CLOUD_PROJECT "
                            "or model.project, and run `gcloud auth application-default login` once")
    return project


def gcp_region(settings) -> str:
    return settings.region or os.environ.get("CLOUD_ML_REGION") or DEFAULT_REGION


def preflight(settings) -> None:
    """Fail before anything is written when the provider obviously cannot work."""
    if settings.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is missing; put it in .env.local or the environment")
    else:
        gcp_project(settings)


def make_client(settings):
    """The client for `settings.provider` (a contracts.ModelSettings)."""
    import anthropic
    if settings.provider == "anthropic":
        return anthropic.Anthropic()
    if settings.provider == "anthropic-vertex":
        return anthropic.AnthropicVertex(project_id=gcp_project(settings), region=gcp_region(settings))
    if settings.provider == "gemini-vertex":
        from gemini import GeminiVertex
        return GeminiVertex(project=gcp_project(settings), location=gcp_region(settings))
    raise ProviderError(f"unknown provider {settings.provider!r}; choose from {PROVIDERS}")


def provider_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that mean "the provider request failed" (quota, auth, network, server), for every provider.
    They end a run as a request event; they never become a candidate (AIC-1.1 §3)."""
    import anthropic
    errors: list[type[BaseException]] = [anthropic.APIError]
    try:
        from google.genai import errors as genai_errors
        errors.append(genai_errors.APIError)
    except ImportError:
        pass
    try:
        from google.auth import exceptions as auth_errors
        errors.append(auth_errors.GoogleAuthError)
    except ImportError:
        pass
    return tuple(errors)
