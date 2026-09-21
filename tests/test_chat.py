import json
import logging
from pathlib import Path

import pytest

from core.agents.chat import ChatAgent, ChatResponse, ToolCallRequest
from core.ingest.parse import parse_pdf
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore
from core.store import RunStore
from core.tools.contract_tools import build_tool_registry

FIXTURE = Path(__file__).parents[1] / "Sample Contract.pdf"


class ScriptedClient:
    """A stub chat client scripted with a fixed sequence of ChatResponse steps."""

    def __init__(self, steps: list[ChatResponse]):
        self.steps = list(steps)
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages, tools):
        self.seen_messages.append(messages)
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        return step


@pytest.fixture(scope="module")
def index():
    doc = parse_pdf(FIXTURE)
    return ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=".index/test-chat"))


def _agent(tmp_path, index, client, session_id="s1", run_id="r1", max_steps=6):
    store = RunStore(tmp_path / "runs.db")
    registry = build_tool_registry(index, store)
    return ChatAgent(client, registry, store, session_id, run_id, max_steps=max_steps), store


# 9. a single-step answer with no tool call works


def test_single_step_answer_with_no_tool_call(tmp_path, index):
    client = ScriptedClient([ChatResponse(content="The contract does not appear to address that.")])
    agent, _ = _agent(tmp_path, index, client)
    turn = agent.ask("What color is the cover page?")
    assert turn.role == "assistant"
    assert "does not appear to address" in turn.content
    assert client.calls == 1


# 10. a stub requesting search_contract then answering produces a cited answer


def test_search_then_answer_produces_cited_answer(tmp_path, index):
    real_chunk = next(c for c in index.doc.chunks if c.section_id == "6.6")
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("call-1", "search_contract", {"query": "password rotation"})]),
        ChatResponse(
            content="Passwords must rotate.",
            citations=[{"chunk_id": real_chunk.chunk_id, "quote": real_chunk.text}],
        ),
    ])
    agent, _ = _agent(tmp_path, index, client)
    turn = agent.ask("What does the contract say about password rotation?")
    assert client.calls == 2
    assert turn.citations == [{"chunk_id": real_chunk.chunk_id, "quote": real_chunk.text}]
    assert real_chunk.chunk_id in [c["chunk_id"] for c in turn.citations]


# 11. a multi-tool sequence completes and history() shows user, assistant, tool turns in order


def test_multi_tool_sequence_orders_history_correctly(tmp_path, index):
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "get_requirement", {"requirement_id": "GOV-04"})]),
        ChatResponse(tool_calls=[ToolCallRequest("c2", "list_deviations", {})]),
        ChatResponse(content="Training is required per GOV-04."),
    ])
    agent, _ = _agent(tmp_path, index, client)
    agent.ask("What training is required?")
    roles = [turn.role for turn in agent.history()]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


# 12. reaching max_steps returns a partial answer that states what is missing


def test_max_steps_reached_returns_honest_partial_answer(tmp_path, index):
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "search_contract", {"query": "encryption"})]),
    ])
    agent, _ = _agent(tmp_path, index, client, max_steps=2)
    turn = agent.ask("Tell me everything about encryption.")
    assert client.calls == 2
    assert "step limit" in turn.content or "investigating" in turn.content
    assert "search_contract" in turn.content


# 13. a quote failing canon() verification is stripped from the answer


def test_unverifiable_quote_is_stripped(tmp_path, index):
    real_chunk = next(c for c in index.doc.chunks if c.section_id == "6.6")
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "search_contract", {"query": "password rotation"})]),
        ChatResponse(
            content="Passwords must rotate.",
            citations=[{"chunk_id": real_chunk.chunk_id, "quote": "This sentence was never in the document."}],
        ),
    ])
    agent, _ = _agent(tmp_path, index, client)
    turn = agent.ask("What does the contract say?")
    assert "This sentence was never in the document." not in turn.content
    assert "could not be verified" in turn.content
    assert turn.citations is None


# 14. every turn is persisted to chat_messages and survives reconstructing the agent


def test_history_survives_reconstruction(tmp_path, index):
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "get_requirement", {"requirement_id": "GOV-04"})]),
        ChatResponse(content="GOV-04 covers training."),
    ])
    store = RunStore(tmp_path / "runs.db")
    registry = build_tool_registry(index, store)
    agent1 = ChatAgent(client, registry, store, "session-x", "run-x", max_steps=6)
    agent1.ask("What about training?")
    original_history = agent1.history()

    agent2 = ChatAgent(ScriptedClient([]), registry, store, "session-x", "run-x", max_steps=6)
    reloaded_history = agent2.history()
    assert [(t.role, t.content) for t in reloaded_history] == [(t.role, t.content) for t in original_history]
    assert len(reloaded_history) == 4


# 15. a failing tool returns its error into the loop and the agent still answers rather than crashing


def test_failing_tool_does_not_crash_the_agent(tmp_path, index):
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "get_compliance_result", {"run_id": "no-such-run"})]),
        ChatResponse(content="I could not find a stored run with that id."),
    ])
    agent, _ = _agent(tmp_path, index, client)
    turn = agent.ask("What was the result of run no-such-run?")
    assert turn.role == "assistant"
    assert "could not find" in turn.content
    tool_turns = [t for t in agent.history() if t.role == "tool"]
    assert json.loads(tool_turns[0].content)["found"] is False


# 16. spans named chat:tool:<name> are emitted for each tool call


def test_tool_calls_emit_chat_tool_spans(tmp_path, index, monkeypatch):
    records: list[str] = []

    class FakeLogger:
        def info(self, name, extra=None):
            records.append(name)

        def error(self, name, extra=None):
            records.append(name)

    monkeypatch.setattr("core.obs.trace.get_logger", lambda *a, **k: FakeLogger())
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("c1", "get_requirement", {"requirement_id": "GOV-04"})]),
        ChatResponse(content="Done."),
    ])
    agent, _ = _agent(tmp_path, index, client)
    agent.ask("question")
    assert any(name == "chat:tool:get_requirement" for name in records)


# 17. (S3) the agent sends a system prompt naming the tool list, the citation
# requirement, and the preference for admitting absence -- this would have
# failed before S3, when _messages_for_client sent history only.


def test_system_prompt_states_tools_and_evidence_discipline(tmp_path, index):
    client = ScriptedClient([ChatResponse(content="No comment.")])
    agent, _ = _agent(tmp_path, index, client)
    agent.ask("What does the contract say about encryption?")
    assert client.seen_messages, "client.chat should have been invoked"
    system_messages = [m for m in client.seen_messages[0] if m["role"] == "system"]
    assert len(system_messages) == 1
    prompt = system_messages[0]["content"]
    for tool_name in agent.registry.names():
        assert tool_name in prompt
    assert "does not appear to address" in prompt
    assert "get_compliance_result" in prompt
    assert "source_required: true" in prompt
    assert "re-uploaded" in prompt


# 18. (S2) an assistant turn that requested tool calls is re-sent to the client
# carrying its `tool_calls` array, and the matching tool result carries the
# `tool_call_id` it answers -- this would have failed before S2, when
# _messages_for_client flattened every turn to {role, content}.


def test_messages_preserve_tool_call_linkage(tmp_path, index):
    client = ScriptedClient([
        ChatResponse(tool_calls=[ToolCallRequest("call-42", "get_requirement", {"requirement_id": "GOV-04"})]),
        ChatResponse(content="GOV-04 covers training."),
    ])
    agent, _ = _agent(tmp_path, index, client)
    agent.ask("What training is required?")

    # The second client.chat() call is the one made after the tool ran, so it is
    # the first request that can carry the linkage back.
    messages = client.seen_messages[1]
    assistant_messages = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert assistant_messages, "assistant message must carry its tool_calls array"
    assistant_call = assistant_messages[0]["tool_calls"][0]
    assert assistant_call["id"] == "call-42"
    assert assistant_call["type"] == "function"
    assert assistant_call["function"]["name"] == "get_requirement"

    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert tool_messages, "a tool result message must be present"
    assert tool_messages[0]["tool_call_id"] == "call-42"


# 19. (S4) a client that fails to return the required {"answer", "citations"}
# JSON shape produces an unverified answer with no citations, rather than
# being treated as a normal answer with zero citations. This exercises the
# same `_finalize` contract `LLMClient.chat` relies on.


def test_parse_failed_final_answer_is_reported_as_unverified(tmp_path, index):
    client = ScriptedClient([ChatResponse(content="some free-form prose", parse_failed=True)])
    agent, _ = _agent(tmp_path, index, client)
    turn = agent.ask("What does the contract say?")
    assert turn.citations is None
    assert "could not produce a verifiable answer" in turn.content
    assert "not valid structured output" in turn.content
