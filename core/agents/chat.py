"""The conversational agent: a real agent, not a question-answer box.

It chooses tools from the shared ``ToolRegistry`` (the same registry task 08's MCP
server wraps), may take several steps before answering, and holds state across
turns via the run store's ``chat_messages`` table. It is bound by the same evidence
discipline as the compliance pipeline: every quote it reports back must verify
against text a tool actually returned, using the same ``canon()`` comparison task 04
uses, and it must not reach past the registry into retrieval internals to do it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from core.obs.trace import span
from core.text import canon
from core.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ChatResponse:
    """What a chat-capable client returns for one step of the loop."""
    content: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
    # Citations the model claims for its final answer: [{"chunk_id": ..., "quote": ...}]
    citations: list[dict] | None = None
    # True when a final (no-tool-call) answer failed to parse as the required
    # {"answer": ..., "citations": [...]} JSON shape. Distinguishes "answered with
    # zero citations" from "could not verify anything" so the safety check in
    # _finalize cannot silently become a no-op against a real backend.
    parse_failed: bool = False


@dataclass(frozen=True)
class ChatTurn:
    role: str                 # "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict] | None = None
    # Structured citations attached to a final assistant answer, rendered by the
    # UI rather than concatenated into the message text.
    citations: list[dict] | None = None


SYSTEM_PROMPT = """You are the compliance contract analysis assistant. You answer \
questions about the uploaded contract using only the following tools: {tools}.

Current context:
{context}

Rules you must follow:
- Context is orientation only. It never counts as evidence for numeric or factual \
answers, and numeric questions must be answered with tool calls and grounded in \
tool output.
- Every factual claim must be grounded in a chunk returned by a tool call. Never \
assert something the tools did not return.
- If the contract does not address a question, say so plainly (e.g. "the contract \
does not appear to address this") instead of speculating or inferring an answer \
that is not backed by retrieved text.
- Before disagreeing with, or updating, a previously stored compliance state, call \
`get_compliance_result` and reconcile your answer with what it returns.
- A tool result with `source_required: true` means only stored analysis evidence is \
available. Explain that the original PDF must be re-uploaded for new retrieval or \
analysis; do not imply that a rerun or source search occurred.
- When you are done calling tools and are ready to give your final answer (i.e. you \
are not requesting any more tool calls), respond with *only* a single JSON object of \
the shape {{"answer": "<final answer text>", "citations": [{{"chunk_id": "<id from a \
tool result>", "quote": "<verbatim quote from that chunk>"}}, ...]}}. Every quote must \
be copied verbatim from a chunk's "text" field or a verified "Relevant Quotes" \
entry's "quote" field as returned by a tool; do not \
paraphrase inside a quote. If you have no citations, use an empty list."""


def _harvest_evidence(result: dict) -> dict[str, str]:
    """Pull every (chunk_id, text) pair out of a tool result, however nested.

    This is the only way the agent learns what evidence exists: it never imports
    retrieval or ingest internals directly, only what tools hand back as data.
    """
    harvested: dict[str, str] = {}

    def walk(node) -> None:
        if isinstance(node, dict):
            chunk_id = node.get("chunk_id")
            text = node.get("text")
            if isinstance(node.get("citation"), str):
                chunk_id = node.get("source_chunk_id")
                text = node.get("quote")
            if chunk_id and isinstance(text, str):
                harvested[chunk_id] = "\n".join(filter(None, (harvested.get(chunk_id), text)))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return harvested


def _render_context_block(context: dict | None) -> str:
    if not context:
        return "No additional context."

    lines: list[str] = []
    for key, value in context.items():
        if isinstance(value, dict):
            rendered = ", ".join(f"{k}={v}" for k, v in value.items())
            lines.append(f"- {key}: {rendered}")
        elif isinstance(value, list):
            lines.append(f"- {key}: {', '.join(str(item) for item in value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "No additional context."


class ChatAgent:
    def __init__(self, client, registry: ToolRegistry, store, session_id: str, run_id: str | None, max_steps: int = 6) -> None:
        self.client = client
        self.registry = registry
        self.store = store
        self.session_id = session_id
        self.run_id = run_id
        self.max_steps = max_steps
        self._history: list[ChatTurn] = []
        self._evidence: dict[str, str] = {}
        self._seq = 0
        self._load_history()

    def _load_history(self) -> None:
        for row in self.store.get_chat_history(self.session_id):
            self._history.append(
                ChatTurn(row["role"], row["content"] or "", row["tool_calls"], row.get("citations"))
            )
            self._seq = max(self._seq, row["seq"] + 1)
            if row["role"] == "tool" and row["content"]:
                try:
                    self._evidence.update(_harvest_evidence(json.loads(row["content"])))
                except json.JSONDecodeError:
                    pass

    def history(self) -> list[ChatTurn]:
        return list(self._history)

    def _persist(self, turn: ChatTurn) -> None:
        self._history.append(turn)
        self.store.save_chat_message(
            self.session_id, self.run_id, self._seq, turn.role, turn.content, turn.tool_calls, turn.citations
        )
        self._seq += 1

    def _system_message(self, context: dict | None = None) -> dict:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                tools=", ".join(self.registry.names()),
                context=_render_context_block(context),
            ),
        }

    def _messages_for_client(self, context: dict | None = None) -> list[dict]:
        """Build OpenAI-compatible messages, preserving the tool-call linkage a
        real backend needs: assistant turns carry their ``tool_calls`` array and
        each tool result carries the ``tool_call_id`` it answers.
        """
        messages: list[dict] = [self._system_message(context)]
        for turn in self._history:
            if turn.role == "assistant" and turn.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": turn.content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("arguments", {})),
                            },
                        }
                        for call in turn.tool_calls
                    ],
                })
            elif turn.role == "tool":
                tool_call_id = turn.tool_calls[0]["id"] if turn.tool_calls else None
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": turn.content})
            else:
                messages.append({"role": turn.role, "content": turn.content})
        return messages

    def _finalize(self, response: ChatResponse) -> tuple[str, list[dict] | None]:
        content = (response.content or "").strip()
        if response.parse_failed:
            # The model did not return the required {"answer", "citations"} JSON
            # shape. Treating this as "verified with zero citations" would let the
            # safety check silently pass on an ungrounded answer, so it is
            # reported as unverified instead.
            note = (
                "I could not produce a verifiable answer: my response was not valid "
                "structured output, so no citations could be checked against evidence."
            )
            return (f"{content} ({note})".strip() if content else note), None
        citations = response.citations or []
        verified: list[dict] = []
        stripped = 0
        for citation in citations:
            chunk_id = citation.get("chunk_id")
            quote = citation.get("quote") or ""
            evidence_text = self._evidence.get(chunk_id)
            if quote and evidence_text and canon(quote) in canon(evidence_text):
                verified.append(citation)
            else:
                stripped += 1
        if stripped:
            content = (
                f"{content} (Note: {stripped} citation(s) could not be verified against "
                "retrieved evidence and were removed.)"
            ).strip()
        return (content or "The contract does not appear to address this."), (verified or None)

    def ask(self, question: str, context: dict | None = None) -> ChatTurn:
        with span("chat", run_id=self.run_id, session_id=self.session_id):
            return self._ask(question, context)

    def _ask(self, question, context):
        cfg = getattr(self.client, "cfg", None)
        if cfg is not None and not cfg.is_local:
            run = self.store.get_run(self.run_id) if self.run_id else None
            if run is None or not run.usage.get("allow_remote"):
                raise ValueError("Remote chat requires a non-confidential run explicitly authorized for remote processing. Use a local model for confidential or legacy runs.")
        self._persist(ChatTurn("user", question))
        last_tool_names: list[str] = []
        for _ in range(self.max_steps):
            response = self.client.chat(self._messages_for_client(context), self.registry.specs())
            tool_calls = response.tool_calls or []
            if not tool_calls:
                content, citations = self._finalize(response)
                turn = ChatTurn("assistant", content, citations=citations)
                self._persist(turn)
                return turn

            assistant_calls = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in tool_calls]
            self._persist(ChatTurn("assistant", response.content or "", assistant_calls))
            last_tool_names = [call.name for call in tool_calls]
            for call in tool_calls:
                with span(f"chat:tool:{call.name}", tool_call_id=call.id):
                    result = self.registry.call(call.name, call.arguments)
                self._evidence.update(_harvest_evidence(result))
                self._persist(ChatTurn("tool", json.dumps(result), [{"id": call.id, "name": call.name}]))

        partial = (
            f"I reached my step limit ({self.max_steps}) before finding a confident answer. "
            f"I was still investigating: {', '.join(last_tool_names) or 'no further leads'}."
        )
        turn = ChatTurn("assistant", partial)
        self._persist(turn)
        return turn
