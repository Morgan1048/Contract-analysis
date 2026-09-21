"""Tests for LLMClient.chat (S1/S4 from docs/review/06-requirements-review.md).

Before S1, nothing implemented ``LLMClient.chat`` at all -- these tests would have
failed with an AttributeError. Since no real backend is reachable offline, they
assert on the REQUEST the client builds (tools/tool_choice passed, tool call
arguments parsed) and on how it maps a backend's raw response, rather than on a
live round trip.
"""
from __future__ import annotations

import json

import pytest

from core.agents.chat import ChatResponse, ToolCallRequest
from core.llm.client import LLMClient
from core.providers.config import ProviderConfig

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_contract",
            "description": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]


def _client(monkeypatch, fake_message: dict, capture: dict | None = None):
    cfg = ProviderConfig(backend="ollama", base_url="http://localhost:11434/v1", model="llama3.1")
    client = LLMClient(cfg)

    def fake_post_json(path, payload):
        if capture is not None:
            capture["path"] = path
            capture["payload"] = payload
        return {"choices": [{"message": fake_message}], "usage": {}}

    monkeypatch.setattr(client._client, "_post_json", fake_post_json)
    return client


# 1. (S1) client.chat() exists and passes tools=/tool_choice="auto" to the backend.


def test_chat_passes_tools_and_tool_choice_auto(monkeypatch):
    capture: dict = {}
    client = _client(monkeypatch, {"role": "assistant", "content": None}, capture)
    client.chat([{"role": "user", "content": "hi"}], TOOLS)
    assert capture["payload"]["tools"] == TOOLS
    assert capture["payload"]["tool_choice"] == "auto"
    assert capture["path"] == "/chat/completions"


# 2. (S1) message.tool_calls on the response is mapped into ToolCallRequest objects.


def test_chat_maps_tool_calls_from_response(monkeypatch):
    fake_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search_contract", "arguments": json.dumps({"query": "password rotation"})},
            }
        ],
    }
    client = _client(monkeypatch, fake_message)
    response = client.chat([{"role": "user", "content": "hi"}], TOOLS)
    assert isinstance(response, ChatResponse)
    assert response.tool_calls == [ToolCallRequest("call-1", "search_contract", {"query": "password rotation"})]


# 3. (S1) no tools argument is sent, and no tool_choice, when the caller supplies none.


def test_chat_omits_tools_when_none_supplied(monkeypatch):
    capture: dict = {}
    client = _client(monkeypatch, {"role": "assistant", "content": '{"answer": "ok", "citations": []}'}, capture)
    client.chat([{"role": "user", "content": "hi"}], None)
    assert "tools" not in capture["payload"]
    assert "tool_choice" not in capture["payload"]


# 4. (S4) a well-formed final answer JSON is parsed into content/citations.


def test_chat_parses_final_answer_json(monkeypatch):
    fake_message = {
        "role": "assistant",
        "content": json.dumps({"answer": "Passwords rotate every 90 days.", "citations": [{"chunk_id": "sec:6.6", "quote": "rotate"}]}),
    }
    client = _client(monkeypatch, fake_message)
    response = client.chat([{"role": "user", "content": "hi"}], None)
    assert response.tool_calls is None
    assert response.parse_failed is False
    assert response.content == "Passwords rotate every 90 days."
    assert response.citations == [{"chunk_id": "sec:6.6", "quote": "rotate"}]


# 5. (S4) a final answer that is not the required JSON shape is reported as
# parse_failed rather than as a normal answer with zero citations -- before S4
# this data did not exist at all (ChatResponse.citations was read directly off
# the raw response with no parsing step).


@pytest.mark.parametrize("content", ["just some prose, not JSON", '{"answer": "ok"}', '{"citations": []}', "{}"])
def test_chat_reports_parse_failure_for_malformed_final_answer(monkeypatch, content):
    client = _client(monkeypatch, {"role": "assistant", "content": content})
    response = client.chat([{"role": "user", "content": "hi"}], None)
    assert response.tool_calls is None
    assert response.parse_failed is True
    assert response.citations is None
    assert response.content == content


@pytest.mark.parametrize(
    ("cfg", "expected_cost", "expected_unpriced"),
    [
        (
            ProviderConfig(backend="openai", base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b"),
            0.0,
            False,
        ),
        (
            ProviderConfig(backend="openai", base_url="https://api.siliconflow.cn/v1", model="openai/gpt-oss-120b"),
            pytest.approx(0.000095),
            False,
        ),
        (
            ProviderConfig(backend="openai", base_url="https://api.groq.com/openai/v1", model="unknown-hosted-model"),
            None,
            True,
        ),
    ],
)
def test_complete_json_populates_backend_qualified_response_cost(monkeypatch, cfg, expected_cost, expected_unpriced):
    client = LLMClient(cfg)

    def complete(messages, **kwargs):
        return '{"ok": true}', {"prompt_tokens": 1000, "completion_tokens": 100}

    monkeypatch.setattr(client._client, "complete_with_usage", complete)
    _parsed, response = client.complete_json("system", "user", {"ok": "boolean"})

    assert response.cost_usd == expected_cost
    assert response.unpriced_model is expected_unpriced
