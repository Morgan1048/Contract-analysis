"""Chat tab: the task 06 ``ChatAgent`` over the uploaded contract.

Rendering shows which tools each answer used and any structured citations it
carries, since the assignment's interview walks through agent decision-making
and a chat that hides its tool calls cannot support that.
"""
from __future__ import annotations

from typing import Any

from core.agents.chat import ChatTurn


def tool_calls_before(history: list[ChatTurn], turn_index: int) -> list[str]:
    """The tool names invoked between the previous user turn and the assistant
    turn at ``turn_index`` -- "which tools each answer used".
    """
    names: list[str] = []
    for turn in reversed(history[:turn_index]):
        if turn.role == "user":
            break
        if turn.role == "assistant" and turn.tool_calls:
            names.extend(call["name"] for call in turn.tool_calls)
    return list(reversed(names))


def render(agent: Any, st: Any, context: dict | None = None) -> None:
    """Render the full chat history and an input box for a new question."""
    history = agent.history()

    if hasattr(st, "chat_message"):
        for i, turn in enumerate(history):
            if turn.role == "user":
                with st.chat_message("user"):
                    st.write(turn.content)
            elif turn.role == "assistant" and not turn.tool_calls:
                with st.chat_message("assistant"):
                    used = tool_calls_before(history, i)
                    if used:
                        st.caption("Tools used: " + ", ".join(used))
                    st.write(turn.content)
                    for citation in turn.citations or []:
                        st.markdown(f'> [{citation.get("chunk_id")}] "{citation.get("quote")}"')
    else:
        for i, turn in enumerate(history):
            if turn.role == "user":
                continue
            elif turn.role == "assistant" and not turn.tool_calls:
                used = tool_calls_before(history, i)
                if used:
                    st.caption("Tools used: " + ", ".join(used))
                st.write(turn.content)
                for citation in turn.citations or []:
                    st.markdown(f'> [{citation.get("chunk_id")}] "{citation.get("quote")}"')

    question = st.chat_input("Ask about this contract") if hasattr(st, "chat_input") else None
    if question:
        agent.ask(question, context=context)
        if hasattr(st, "rerun"):
            st.rerun()


__all__ = ["tool_calls_before", "render"]
