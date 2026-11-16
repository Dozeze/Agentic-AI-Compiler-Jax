"""AIC-013: structured proposals, conservative debits and offline protocol/CLI checks."""
import dataclasses
import json
import subprocess
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

import providers
from contracts import Budgets, ConfigError, ModelPrices, ModelSettings, Proposal, resolve_run_config
from fakes import proposal
from storage import RunStore, StorageError

SOURCE = "def kernel(x):\n    return x + 1\n"


def response(payload=None, *, stop="tool_use", usage=None, name="submit_proposal"):
    return {"id": "msg_offline", "model": ModelSettings().model_id, "stop_reason": stop,
            "content": [{"type": "tool_use", "id": "tool_offline", "name": name,
                         "input": proposal(SOURCE) if payload is None else payload}],
            "usage": {"input_tokens": 10, "output_tokens": 7} if usage is None else usage}


class SDKStandIn:
    """Intentionally live-shaped (not OfflineClient): exercise the paid-call gate without a network."""
    def __init__(self, *responses):
        self.script, self.calls, self.options = list(responses), [], []
        self.messages = SimpleNamespace(create=self.create)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def prices(**overrides):
    return ModelPrices(provider="anthropic", model_id=ModelSettings().model_id, currency="USD",
                       input_per_million=1.0, output_per_million=5.0, checked_date="2026-01-01",
                       source_url="https://example.invalid/test-price-fixture", **overrides)


def live(client, *, budgets=None, settings=None, **kwargs):
    return providers.ProposalProvider(client, settings or ModelSettings(prices=prices()),
                                       budgets or Budgets(money_cap=1.0, currency="USD"), **kwargs)


def test_installed_sdk_message_is_locally_validated_and_records_ids():
    from anthropic.types import Message, ToolUseBlock, Usage
    message = Message(id="msg_sdk_fixture", type="message", role="assistant", model=ModelSettings().model_id,
                      stop_reason="tool_use", content=[ToolUseBlock(type="tool_use", id="tool_sdk_fixture",
                      name="submit_proposal", input=proposal(SOURCE))], usage=Usage(input_tokens=12, output_tokens=8))
    client = providers.OfflineClient([message])
    boundary = providers.ProposalProvider(client)
    result = boundary.propose({"current_source": SOURCE})
    assert isinstance(result, Proposal) and result.candidate_source == SOURCE
    assert boundary.events[0]["response_id"] == "msg_sdk_fixture"
    assert boundary.events[0]["status"] == "PROPOSAL_VALID"
    assert [tool["name"] for tool in client.calls[0]["tools"]] == ["submit_proposal"]
    assert client.calls[0]["tool_choice"]["disable_parallel_tool_use"] is True
    assert client.calls[0]["service_tier"] == "standard_only"


@pytest.mark.parametrize("bad", ["unknown_tool", "two_calls", "refusal", "truncated", "text_only",
                                  "empty", "content_object", "content_number", "unknown_block"])
def test_malformed_refused_or_incomplete_responses_never_return_a_proposal(bad):
    item = response()
    expected = "MALFORMED"
    if bad == "unknown_tool":
        item["content"][0]["name"] = "benchmark_code"
    elif bad == "two_calls":
        item["content"] *= 2
    elif bad == "refusal":
        item["stop_reason"], expected = "refusal", "REFUSED"
    elif bad == "truncated":
        item["stop_reason"], expected = "max_tokens", "INCOMPLETE"
    elif bad == "text_only":
        item["content"] = [{"type": "text", "text": "I cannot provide this proposal"}]
        item["stop_reason"] = "end_turn"
    elif bad == "empty":
        item["content"], expected = [], "EMPTY"
    elif bad == "content_object":
        item["content"] = {"type": "tool_use"}
    elif bad == "content_number":
        item["content"] = 42
    else:
        item["content"].append({"type": "server_tool_use"})
    boundary = providers.ProposalProvider(providers.OfflineClient([item]))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == expected
    assert boundary.events[0]["status"] == expected
    assert len(boundary.events) == len(boundary.reservations) == 1


@pytest.mark.parametrize("payload", [{"code": SOURCE}, {**proposal(SOURCE), "hypothesis": ""},
                                     {**proposal(SOURCE), "parent_id": True}, {**proposal(SOURCE), "target": "shell"}])
def test_tool_schema_does_not_replace_local_validation(payload):
    boundary = providers.ProposalProvider(providers.OfflineClient([response(payload)]))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == "PROPOSAL_INVALID"
    assert boundary.events[0]["status"] == "PROPOSAL_INVALID"


@pytest.mark.parametrize("what", ["cap", "prices", "currency", "transport"])
def test_missing_live_authorization_blocks_even_an_injected_sdk(what):
    client = SDKStandIn(response())
    settings, budgets = ModelSettings(prices=prices()), Budgets(money_cap=1, currency="USD")
    if what == "cap":
        budgets = Budgets()
    elif what == "prices":
        settings = ModelSettings()
    elif what == "currency":
        budgets = dataclasses.replace(budgets, currency="SEK")
    else:
        client.with_options = None
    with pytest.raises(providers.ProviderError, match="live mode disabled"):
        providers.ProposalProvider(client, settings, budgets)
    assert client.calls == []


@pytest.mark.parametrize("override", [
    {"model.prices": {"provider": "anthropic"}}, {"model.request_timeout_s": 0},
    {"model.max_retries": -1}, {"model.retry_backoff_s": 6}, {"model.temperature": 2},
    {"model.max_payload_bytes": 2}, {"proposal_only": "true"},
])
def test_new_settings_are_strictly_validated(override):
    with pytest.raises(ConfigError):
        resolve_run_config(override, None)


@pytest.mark.parametrize("field,value", [("model_id", "other"), ("provider", "gemini-vertex"),
                                         ("currency", "usd"), ("checked_date", "2099-01-01"),
                                         ("checked_date", "not-a-date"), ("source_url", "http://example.invalid"),
                                         ("input_per_million", -1), ("output_per_million", float("nan"))])
def test_price_identity_date_and_provenance_are_required(field, value):
    configured = dataclasses.asdict(prices())
    configured[field] = value
    with pytest.raises(ConfigError):
        resolve_run_config({"model.prices": configured}, None)


def test_each_retry_has_a_reservation_and_unknown_failure_usage_is_not_refunded():
    client = SDKStandIn(TimeoutError("timeout"), response())
    boundary = live(client, settings=ModelSettings(prices=prices(), max_retries=1), sleep=lambda _: None)
    boundary.propose({})
    assert len(client.calls) == len(boundary.events) == len(boundary.reservations) == 2
    first = boundary.reservations[0]
    assert boundary.accounted_tokens == first["input_tokens_reserved"] + first["output_tokens_reserved"] + 17
    assert boundary.events[0]["input_tokens"] is None and boundary.events[0]["usage_uncertain"]
    assert boundary.events[1]["retry"] == 1
    assert all(option == {"max_retries": 0, "timeout": 30.0} for option in client.options)
    assert boundary.events[0]["budget"]["unknown_usage_requests"] == 1
    assert boundary.summary()["unknown_usage_requests"] == 1


def test_rate_limit_is_retried_but_authentication_is_not():
    import anthropic
    import httpx2
    request = httpx2.Request("POST", "https://example.invalid/v1/messages")
    busy = anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None)
    client = SDKStandIn(busy, response())
    boundary = live(client, settings=ModelSettings(prices=prices(), max_retries=1), sleep=lambda _: None)
    boundary.propose({})
    assert [event["status"] for event in boundary.events] == ["TRANSPORT_FAILED", "PROPOSAL_VALID"]
    denied = anthropic.AuthenticationError("denied", response=httpx2.Response(401, request=request), body=None)
    client = SDKStandIn(denied, response())
    boundary = live(client, settings=ModelSettings(prices=prices(), max_retries=1), sleep=lambda _: None)
    with pytest.raises(providers.ProposalFailure):
        boundary.propose({})
    assert len(client.calls) == 1


@pytest.mark.parametrize("usage,input_known,output_known", [({"output_tokens": 7}, None, 7),
                                                          ({"input_tokens": 0}, 0, None), ({}, None, None)])
def test_partial_usage_keeps_unknown_components_reserved(usage, input_known, output_known):
    boundary = live(SDKStandIn(response(usage=usage)))
    boundary.propose({})
    reservation = boundary.reservations[0]
    expected_input = input_known if input_known is not None else reservation["input_tokens_reserved"]
    expected_output = output_known if output_known is not None else reservation["output_tokens_reserved"]
    assert boundary.accounted_tokens == expected_input + expected_output
    assert boundary.accounted_money == boundary._cost(expected_input, expected_output)
    assert boundary.events[0]["usage_uncertain"]


@pytest.mark.parametrize("limit", ["requests", "tokens", "money", "wall", "candidates", "output"])
def test_budget_exhaustion_prevents_dispatch(limit):
    client = SDKStandIn(response())
    budgets = Budgets(money_cap=1, currency="USD")
    changes = {"requests": {"max_provider_requests": 0}, "tokens": {"max_total_tokens": 1},
               "money": {"money_cap": 0.000001}, "wall": {"run_wall_s": 0},
               "candidates": {"max_candidates": 0}, "output": {"max_output_tokens_per_request": 0}}
    boundary = live(client, budgets=dataclasses.replace(budgets, **changes[limit]))
    with pytest.raises(providers.ProposalFailure, match="budget|cap"):
        boundary.propose({})
    assert client.calls == [] and boundary.reservations == []


def test_unknown_timeout_debit_can_block_retry_at_the_money_cap():
    settings = ModelSettings(prices=prices(), max_retries=1)
    probe = live(SDKStandIn(), settings=settings)
    reserved = Decimal(probe._reservation(probe._body({}), 0)["money_reserved"])
    client = SDKStandIn(TimeoutError("unknown usage"), response())
    boundary = live(client, settings=settings,
                    budgets=Budgets(money_cap=float(reserved * Decimal("1.5")), currency="USD"), sleep=lambda _: None)
    with pytest.raises(providers.ProposalFailure, match="money cap"):
        boundary.propose({})
    assert len(client.calls) == 1 and boundary.accounted_money == reserved


def test_output_cap_clamps_the_model_request_and_known_usage_releases_reservation():
    client = SDKStandIn(response())
    boundary = live(client, budgets=Budgets(money_cap=1, currency="USD", max_output_tokens_per_request=32))
    boundary.propose({})
    assert client.calls[0]["max_tokens"] == 32
    assert boundary.accounted_tokens == 17
    assert boundary.accounted_money == Decimal("0.000045")


def test_deadline_shrinks_transport_timeout_and_rejects_late_response():
    ticks = [0.0]
    client = SDKStandIn(response())
    original = client.messages.create
    def late(**kwargs):
        ticks[0] += 3
        return original(**kwargs)
    client.messages.create = late
    boundary = live(client, budgets=Budgets(money_cap=1, currency="USD", run_wall_s=2), clock=lambda: ticks[0])
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == "TIME_LIMIT"
    assert client.options[0]["timeout"] == 2
    assert boundary.events[0]["output_tokens"] == 7


@pytest.mark.parametrize("usage", [{"input_tokens": 1_000_000, "output_tokens": 7},
                                   {"input_tokens": 10, "output_tokens": 7, "cache_creation_input_tokens": 3}])
def test_unexpected_billing_or_usage_overrun_stops_without_returning_source(usage):
    boundary = live(SDKStandIn(response(usage=usage)))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == "USAGE_INVALID"
    assert boundary.events[0]["status"] == "USAGE_INVALID"


def test_reservation_is_durable_before_dispatch_and_outcome_links_to_it(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "run")
    store.initialize(SOURCE, config={}, manifest={})
    client = SDKStandIn(response())
    original = client.messages.create
    def inspect(**kwargs):
        assert (store.path / "request_reservations/0001.json").is_file()
        assert not (store.path / "requests/0001.json").exists()
        return original(**kwargs)
    client.messages.create = inspect
    boundary = live(client, reserve=store.reserve_request, record=store.record_request)
    boundary.propose({})
    event = json.loads((store.path / "requests/0001.json").read_bytes())
    assert event["reservation"] == store._reference("request_reservations/0001.json")
    reservation = json.loads((store.path / event["reservation"]["path"]).read_bytes())
    assert reservation["prices"]["source_url"].startswith("https://example.invalid/")
    assert reservation["local_request_id"] == event["local_request_id"]
    def fail(*args):
        raise StorageError("reservation failure")
    other = live(SDKStandIn(response()), reserve=fail)
    with pytest.raises(StorageError):
        other.propose({})
    assert other.client.calls == []


def test_credentials_are_redacted_from_payloads_and_source_is_refused(monkeypatch, tmp_path):
    secret = "sk-provider-fixture-secret-123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    store = RunStore(tmp_path / "run")
    store.initialize(SOURCE, config={}, manifest={})
    client = providers.OfflineClient([response({**proposal(SOURCE), "notes": f"message {secret}"})])
    boundary = providers.ProposalProvider(client, reserve=store.reserve_request, record=store.record_request)
    boundary.propose({"data": secret, "api_key": secret})
    assert secret not in json.dumps(client.calls)
    for path in store.path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    boundary = providers.ProposalProvider(providers.OfflineClient([response(proposal(f"# {secret}\n{SOURCE}"))]))
    with pytest.raises(providers.ProposalFailure, match="credential"):
        boundary.propose({})


def test_oversized_payloads_are_bounded_and_never_returned():
    client = providers.OfflineClient([response()])
    boundary = providers.ProposalProvider(client)
    with pytest.raises(providers.ProposalFailure, match="payload byte"):
        boundary.propose({"data": "x" * (200 << 10)})
    assert client.calls == []
    item = response(proposal(SOURCE, notes="x" * (200 << 10)))
    boundary = providers.ProposalProvider(providers.OfflineClient([item]))
    with pytest.raises(providers.ProposalFailure, match="payload byte"):
        boundary.propose({})
    assert boundary.events[0]["response_payload"]["omitted"]
    assert len(json.dumps(boundary.events[0])) < 10 << 10
    item = response(name="x" * 4000)
    item["content"] *= 100
    boundary = providers.ProposalProvider(providers.OfflineClient([item]))
    with pytest.raises(providers.ProposalFailure):
        boundary.propose({})
    assert boundary.events[0]["omitted_content_blocks"] == 96
    assert len(json.dumps(boundary.events[0])) < 10 << 10


def test_controller_provider_failure_is_persisted_with_baseline_fallback(no_key, repo_root):
    from test_foundation import _build
    _, controller = _build(repo_root, [], no_key / "run")
    controller.client = providers.OfflineClient([TimeoutError("offline timeout")])
    history = controller.agent_loop(model="offline-model")
    assert len(history) == 1 and controller.store.verified_best()["baseline_fallback"]
    event = json.loads((controller.output_dir / "requests/0001.json").read_bytes())
    assert event["status"] == "TRANSPORT_FAILED" and event["usage_uncertain"]
    assert json.loads((controller.output_dir / "provider_stop.json").read_bytes())["status"] == "TRANSPORT_FAILED"


def test_proposal_only_cli_never_imports_reference_or_candidate(repo_root, tmp_path, clean_env):
    original = tmp_path / "original.py"
    original.write_text("raise RuntimeError('reference executed')\n" + SOURCE)
    candidate = "raise RuntimeError('candidate executed')\n" + SOURCE
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps([response(proposal(candidate))]))
    output = tmp_path / "run"
    command = [sys.executable, str(repo_root / "agent.py"), "--backend", "cpu", "--source", str(original),
               "--function", "kernel", "--proposal-only", "--offline-responses", str(replay), "--output-dir", str(output)]
    result = subprocess.run(command, cwd=tmp_path, env=clean_env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (output / "proposed.py").read_text() == candidate
    assert not (output / "work").exists() and not (output / "best.py").exists()
    assert json.loads((output / "proposal_only_result.json").read_bytes())["verified"] is False
    assert original.read_text().startswith("raise RuntimeError('reference executed')")


def test_cli_missing_cap_blocks_before_constructing_a_live_client(repo_root, tmp_path, clean_env):
    code = f"""
import sys
sys.path.insert(0, {str(repo_root)!r})
import agent
def unexpected(*args):
    raise AssertionError('constructed a live client')
agent.client_factory = unexpected
agent.main(['--backend', 'cpu', '--proposal-only', '--runner', 'subprocess'])
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=clean_env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2 and "money_cap" in result.stderr
    assert "Traceback" not in result.stderr and not (tmp_path / "runs").exists()


def test_real_sdk_http_transport_has_no_hidden_retry_and_records_server_request_ids(tmp_path):
    import anthropic
    import httpx2
    calls = []
    key = "sk-literal-offline-http-fixture-123456"
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(429, headers={"request-id": "req_busy"}, json={
                "type": "error", "error": {"type": "rate_limit_error", "message": f"busy {key}"}})
        return httpx2.Response(200, headers={"request-id": "req_ok"}, json={
            **response(), "type": "message", "role": "assistant", "stop_sequence": None})
    store = RunStore(tmp_path / "run")
    store.initialize(SOURCE, config={}, manifest={})
    transport = httpx2.Client(transport=httpx2.MockTransport(handle))
    with anthropic.Anthropic(api_key=key, http_client=transport, max_retries=7) as client:
        boundary = live(client, settings=ModelSettings(prices=prices(), max_retries=1), sleep=lambda _: None,
                        reserve=store.reserve_request, record=store.record_request)
        assert boundary.propose({"current_source": SOURCE}).candidate_source == SOURCE
    assert len(calls) == 2
    assert [event["provider_request_id"] for event in boundary.events] == ["req_busy", "req_ok"]
    assert all(call.headers["x-stainless-retry-count"] == "0" for call in calls)
    assert all(call.extensions["timeout"]["read"] == 30 for call in calls)
    packet = json.loads(calls[0].content)
    assert packet["thinking"] == {"type": "disabled"}
    assert packet["tool_choice"]["name"] == "submit_proposal"
    for path in store.path.rglob("*"):
        if path.is_file():
            assert key.encode() not in path.read_bytes()


def test_outcome_persistence_failure_stops_the_boundary_before_another_call():
    client = providers.OfflineClient([response(), response()])
    def fail(event):
        raise StorageError("outcome failure")
    boundary = providers.ProposalProvider(client, record=fail)
    with pytest.raises(StorageError):
        boundary.propose({})
    with pytest.raises(providers.ProposalFailure, match="cannot resume"):
        boundary.propose({})
    assert len(client.calls) == 1


def test_candidate_limit_and_nonfinite_context_block_without_another_request():
    client = providers.OfflineClient([response(), response()])
    boundary = providers.ProposalProvider(client, budgets=Budgets(max_candidates=1))
    boundary.propose({})
    with pytest.raises(providers.ProposalFailure, match="candidate budget"):
        boundary.propose({})
    assert len(client.calls) == 1
    other = providers.ProposalProvider(providers.OfflineClient([response()]))
    with pytest.raises(providers.ProposalFailure) as caught:
        other.propose({"bad": float("nan")})
    assert caught.value.status == "CONTEXT_INVALID" and not other.client.calls


@pytest.mark.parametrize("extra", [{"input_tokens": True}, {"output_tokens": -1}, {"service_tier": "priority"},
                                   {"server_tool_use": {"web_search_requests": 1}}])
def test_malformed_or_unpriced_usage_retains_the_reservation_and_stops(extra):
    boundary = live(SDKStandIn(response(usage={"input_tokens": 10, "output_tokens": 7, **extra})))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == "USAGE_INVALID"
    reserved = boundary.reservations[0]
    assert boundary.accounted_tokens >= reserved["input_tokens_reserved"] + reserved["output_tokens_reserved"]
    assert boundary.events[0]["usage_uncertain"]


def test_offline_path_or_marker_on_a_live_shaped_client_cannot_bypass_the_cap():
    client = SDKStandIn(response())
    client.offline_transport = True
    with pytest.raises(providers.ProviderError, match="money_cap"):
        providers.ProposalProvider(client, ModelSettings(offline_responses_path="unused.json"))
    assert not client.calls


def test_literal_sdk_key_is_redacted_even_in_malformed_dictionary_keys():
    key = "sk-literal-malformed-key-fixture-123456"
    client = SDKStandIn(response(proposal(SOURCE, notes=f"private {key}")), response({key: "bad proposal"}))
    client.api_key = key
    boundary = live(client)
    assert key not in json.dumps(dataclasses.asdict(boundary.propose({})))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert key not in str(caught.value)
    assert key not in json.dumps(caught.value.submitted)
    assert key not in json.dumps(boundary.events)


def test_unexpected_response_model_is_recorded_and_stops_the_run():
    item = response()
    item["model"] = "unpriced-other-model"
    boundary = live(SDKStandIn(item))
    with pytest.raises(providers.ProposalFailure) as caught:
        boundary.propose({})
    assert caught.value.status == "MODEL_MISMATCH"
    assert boundary.events[0]["response_payload"]["model"] == "unpriced-other-model"


@pytest.mark.parametrize("zero", ["iterations", "budget"])
def test_direct_controller_zero_candidates_never_requests_a_proposal(no_key, repo_root, zero):
    from fakes import tool_use
    from test_foundation import _build
    client, controller = _build(repo_root, [[tool_use("unused", "submit_proposal", proposal(SOURCE))]],
                                no_key / "run", max_iterations=0 if zero == "iterations" else 3)
    if zero == "budget":
        controller.evaluator.budgets = dataclasses.replace(controller.evaluator.budgets, max_candidates=0)
    history = controller.agent_loop(model="offline-fixture")
    assert len(history) == 1 and not client.messages.calls and not controller.requests
    assert controller.store.verified_best()["baseline_fallback"]
