"""Gemini on Vertex behind the Anthropic surface (P8), against a stand-in google.genai client: no network."""
import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

import gemini
from fakes import proposal

TOOLS = [{"name": "benchmark_code", "description": "submit",
          "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}}]


def _response(*parts, prompt=10, out=3, thoughts=2):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)), finish_reason="STOP")],
                           usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=out,
                                                          thoughts_token_count=thoughts))


class StandIn:
    """Records what would be sent to Vertex and replays scripted responses."""

    def __init__(self, *responses):
        self.sent, self._responses = [], list(responses)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.sent.append({"model": model, "contents": contents, "config": config})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(*responses):
    stand_in = StandIn(*responses)
    return gemini.GeminiVertex(project="p", location="global", client=stand_in, sleep=lambda s: None), stand_in


def test_request_translation():
    client, sent = _client(_response(types.Part(text="ok")))
    client.messages.create(model="gemini-2.5-flash", max_tokens=99, system="be brief", tools=TOOLS,
                           messages=[{"role": "user", "content": "start"}])
    request = sent.sent[0]
    assert request["model"] == "gemini-2.5-flash"
    assert request["config"].system_instruction == "be brief" and request["config"].max_output_tokens == 99
    declaration = request["config"].tools[0].function_declarations[0]
    assert declaration.name == "benchmark_code" and declaration.parameters_json_schema == TOOLS[0]["input_schema"]
    assert request["config"].automatic_function_calling.disable is True
    assert [(c.role, c.parts[0].text) for c in request["contents"]] == [("user", "start")]


def test_response_comes_back_in_anthropic_shape_with_usage():
    call = types.Part(function_call=types.FunctionCall(name="benchmark_code", args={"x": "1"}),
                      thought_signature=b"sig")
    client, _ = _client(_response(types.Part(text="thinking aloud", thought=True), types.Part(text="plan"), call))
    message = client.messages.create(model="m", max_tokens=10, messages=[{"role": "user", "content": "go"}])
    assert [b.type for b in message.content] == ["text", "tool_use"]   # thought parts are not text
    tool_use = message.content[1]
    assert (tool_use.name, tool_use.input, tool_use.thought_signature) == ("benchmark_code", {"x": "1"}, b"sig")
    assert message.stop_reason == "tool_use"
    assert (message.usage.input_tokens, message.usage.output_tokens) == (10, 5)


def test_history_round_trip_names_responses_and_keeps_signatures():
    call = SimpleNamespace(type="tool_use", id="gemini_call_1", name="benchmark_code", input={"x": "1"},
                           thought_signature=b"sig")
    messages = [{"role": "user", "content": "go"},
                {"role": "assistant", "content": [SimpleNamespace(type="text", text="plan"), call]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "gemini_call_1",
                                              "content": json.dumps({"state": "ACCEPTED"})}]}]
    contents = gemini.to_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
    sent_call = contents[1].parts[1]
    assert sent_call.function_call.name == "benchmark_code" and sent_call.thought_signature == b"sig"
    reply = contents[2].parts[0].function_response
    assert (reply.name, reply.response) == ("benchmark_code", {"state": "ACCEPTED"})


def test_transient_errors_are_retried_and_others_raise():
    busy = errors.APIError(429, {"error": {"message": "busy", "status": "RESOURCE_EXHAUSTED"}})
    client, sent = _client(busy, _response(types.Part(text="ok")))
    assert client.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}]).content
    assert len(sent.sent) == 2
    bad = errors.APIError(400, {"error": {"message": "bad", "status": "INVALID_ARGUMENT"}})
    client, _ = _client(bad)
    with pytest.raises(errors.APIError):
        client.messages.create(model="m", max_tokens=5, messages=[{"role": "user", "content": "x"}])


def test_the_controller_loop_runs_unchanged_on_gemini(no_key, repo_root):
    """The real controller drives Gemini through the adapter: one proposal, evaluated, result fed back."""
    import agent
    from runner import TrustedSubprocessRunner
    wrong = "import jax.numpy as jnp\n\ndef bad_func(x, y):\n    return x - y\n"
    call = types.Part(function_call=types.FunctionCall(name="benchmark_code", args=proposal(wrong)))
    client, sent = _client(_response(call), _response(types.Part(text="done")))
    controller = agent.build_controller(client, max_iterations=2, source_path=repo_root / "workloads/example.py",
                                        function_name="bad_func", output_dir=no_key / "run", selected_backend="cpu",
                                        runner=TrustedSubprocessRunner(reviewed_fixtures=True))
    history = controller.agent_loop(model="gemini-2.5-flash")
    assert history[-1]["state"] == "CORRECTNESS_FAILED"
    second = sent.sent[1]["contents"]
    assert second[-1].parts[0].function_response.name == "benchmark_code"
    assert second[-1].parts[0].function_response.response["state"] == "CORRECTNESS_FAILED"
