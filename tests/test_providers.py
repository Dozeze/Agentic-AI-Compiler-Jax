"""LLM providers (P8): the Anthropic API stays the default; Vertex providers bill the GCP project."""
import pytest

import providers
from contracts import ConfigError, ModelSettings, resolve_run_config


def test_default_stays_the_anthropic_api_with_todays_model():
    cfg = resolve_run_config({}, None)
    assert (cfg.model.provider, cfg.model.model_id) == ("anthropic", "claude-haiku-4-5-20251001")


@pytest.mark.parametrize("provider, model", [("anthropic-vertex", "claude-haiku-4-5@20251001"),
                                             ("gemini-vertex", "gemini-2.5-flash")])
def test_each_provider_gets_its_own_default_model(provider, model):
    assert resolve_run_config({"model.provider": provider}, None).model.model_id == model


def test_an_explicit_model_wins():
    cfg = resolve_run_config({"model.provider": "gemini-vertex", "model.model_id": "gemini-2.5-flash-lite"}, None)
    assert cfg.model.model_id == "gemini-2.5-flash-lite"


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError, match="model.provider"):
        resolve_run_config({"model.provider": "openai"}, None)


@pytest.mark.parametrize("model, vertex", [("claude-haiku-4-5-20251001", "claude-haiku-4-5@20251001"),
                                           ("claude-opus-5-5", "claude-opus-5-5"),
                                           ("claude-haiku-4-5@20251001", "claude-haiku-4-5@20251001")])
def test_vertex_model_id(model, vertex):
    assert providers.vertex_model_id(model) == vertex


def test_vertex_needs_a_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(providers.ProviderError, match="GOOGLE_CLOUD_PROJECT"):
        providers.preflight(ModelSettings(provider="anthropic-vertex"))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "dd2430-test")
    providers.preflight(ModelSettings(provider="anthropic-vertex"))      # no Anthropic key needed


def test_anthropic_needs_its_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="ANTHROPIC_API_KEY"):
        providers.preflight(ModelSettings())


def test_anthropic_vertex_client_targets_the_project_and_region(monkeypatch):
    import anthropic
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "dd2430-test")
    monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
    client = providers.make_client(ModelSettings(provider="anthropic-vertex"))
    assert isinstance(client, anthropic.AnthropicVertex)
    assert (client.project_id, client.region) == ("dd2430-test", "global")
    assert hasattr(client.messages, "create")                            # the loop's only entry point
