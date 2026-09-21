"""Streamlit entrypoint: sidebar intake and Chat, KPI, and Monitoring tabs.

No business logic lives here -- every callback either calls a ``core`` function
directly or delegates to a ``ui.*`` render helper; anything computable is
computed and tested outside Streamlit.
"""
from __future__ import annotations

import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

import streamlit as st

from core.kpi import evaluate_thresholds, snapshot
from core.llm.client import LLMClient
from core.obs import get_logger
from core.pipeline import analyze
from core.providers.config import ProviderConfig
from core.providers.progress import WaitUpdate, backend_wait_updates
from core.retrieval.index import ChunkIndex
from core.store import RunStore
from core.tools.contract_tools import build_tool_registry
from core.agents.chat import ChatAgent
from core.ingest.parse import parse_pdf
from ui import analyze_view, chat_view, kpi_view, monitoring_view

st.set_page_config(page_title="Contract Analyzer", layout="wide")

# CA_RUN_DB lets a reviewer point the app at data/demo.db to look at the seeded
# KPI dashboard shape (python -m eval.seed_demo); the KPI tab shows a banner
# whenever the store it reads is demo-seeded (see ui/kpi_view.py).
DB_PATH = os.getenv("CA_RUN_DB", "data/runs.db")


@st.cache_resource
def _store() -> RunStore:
    return RunStore(DB_PATH)


def _analyze_uploaded_file(uploaded_file, *, allow_remote: bool = False) -> None:
    """The one callback allowed to do I/O: write the upload to a temp path,
    parse/index/analyze it via ``core``, persist the run, and delete the upload
    on success or failure after the analysis worker finishes.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(uploaded_file.getvalue())

        with st.status("Parsing document...", expanded=True) as status:
            updates: Queue[str | WaitUpdate] = Queue()

            def run():
                with backend_wait_updates(updates.put):
                    doc = parse_pdf(tmp_path)
                    updates.put(f"Parsed {len(doc.chunks)} chunks; indexing...")
                    index = ChunkIndex(doc, allow_remote=allow_remote)
                    updates.put("Running compliance analysis...")
                    client = LLMClient(ProviderConfig.from_env("llm"))
                    return index, analyze(tmp_path, client, index=index, allow_remote=allow_remote)

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="contract-analysis") as pool:
                future = pool.submit(run)
                while not future.done() or not updates.empty():
                    try:
                        update = updates.get(timeout=.25)
                    except Empty:
                        continue
                    if isinstance(update, WaitUpdate):
                        if update.phase == "resumed":
                            label = "API wait finished; continuing analysis..."
                        elif update.phase == "blocked":
                            label = "API cooldown exceeds the configured wait budget."
                        else:
                            next_attempt = (f"retry {update.retry}/{update.max_retries}"
                                            if update.retry else "the next request")
                            label = (f"API {update.reason.replace('_', ' ')} (HTTP {update.http_status or '-'}): "
                                     f"waiting {update.remaining_s:.0f}s before {next_attempt}. "
                                     "Analysis is still running.")
                    else:
                        label = update
                    status.update(label=label)
                index, result = future.result()
            if result.incomplete:
                status.update(label="Analysis incomplete.", state="error")
            else:
                status.update(label="Done.", state="complete")

        store = _store()
        store.save_run(result)
        st.session_state["result"] = result
        st.session_state["index"] = index
        st.session_state["registry"] = build_tool_registry(index, store, include_kpi=True)
        st.session_state["loaded_run_id"] = result.run_id
        st.session_state["history_run_id"] = result.run_id
        st.session_state.pop("chat_agent", None)
        st.session_state["backend_status"] = {
            "ok": not result.incomplete,
            "model": result.usage.get("model"),
            "backend": result.usage.get("backend"),
        }
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _load_historical_run(run_id: str) -> None:
    """Load a run entirely from SQLite without reopening its original source."""
    result = _store().get_run(run_id)
    if result is None:
        raise ValueError(f"Run {run_id} no longer exists")
    st.session_state["result"] = result
    st.session_state["index"] = None
    st.session_state["registry"] = build_tool_registry(None, _store(), include_kpi=True)
    st.session_state.pop("chat_agent", None)


def _chat_context() -> dict:
    context: dict[str, Any] = {"active_tab": st.session_state.get("active_tab", "Chat")}
    if "result" in st.session_state and getattr(st.session_state["result"], "run_id", None):
        context["run_id"] = st.session_state["result"].run_id
        context["source_available"] = st.session_state.get("index") is not None

    store = _store()
    snap = snapshot(store)
    alerts = evaluate_thresholds(snap)
    if alerts:
        context["alerting_kpis"] = {
            alert.name: {"value": alert.value, "threshold": alert.threshold, "severity": alert.severity}
            for alert in alerts
        }
    return context


def _ensure_chat_agent() -> ChatAgent:
    store = _store()
    if "registry" not in st.session_state:
        st.session_state["registry"] = build_tool_registry(None, store, include_kpi=True)
    if "chat_agent" not in st.session_state:
        client = LLMClient(ProviderConfig.from_env("llm"))
        run_id = st.session_state.get("result").run_id if "result" in st.session_state else None
        session_id = st.session_state.get("chat_session_id")
        if session_id is None:
            session_id = f"run:{run_id}" if run_id else "global-chat"
        st.session_state["chat_agent"] = ChatAgent(
            client,
            st.session_state["registry"],
            store,
            session_id=session_id,
            run_id=run_id,
        )
    return st.session_state["chat_agent"]


def _render_analyze_tab(st: Any) -> None:
    """Render the evidence portion of Chat before its conversation."""
    if "result" in st.session_state:
        analyze_view.render(st.session_state["result"], st.session_state.get("index"), st)
        st.divider()


def _render_chat_shell(st: Any) -> None:
    if "result" not in st.session_state:
        return
    agent = _ensure_chat_agent()
    chat_view.render(agent, st, context=_chat_context())


def _render_kpi_tab(st: Any) -> None:
    kpi_view.render(_store(), st)


def _render_sidebar(st: Any) -> None:
    sidebar = getattr(st, "sidebar", st)
    getattr(sidebar, "header", lambda *_args, **_kwargs: None)("Contract intake")
    uploaded_file = sidebar.file_uploader("Upload a contract PDF", type=["pdf"])
    allow_remote = False
    if uploaded_file is not None and hasattr(sidebar, "checkbox"):
        import hashlib
        document_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
        allow_remote = sidebar.checkbox(
            "Allow remote processing of this non-confidential document",
            value=False, key=f"allow_remote_{document_key}",
            help="Keep disabled for the confidential assignment and sample contract. Includes inference and embeddings.",
        )
    if uploaded_file is not None and sidebar.button("Analyze"):
        if allow_remote:
            _analyze_uploaded_file(uploaded_file, allow_remote=True)
        else:
            _analyze_uploaded_file(uploaded_file)

    store = _store()
    runs = store.list_runs() if hasattr(store, "list_runs") else []
    if runs and hasattr(sidebar, "selectbox"):
        ids = [run["run_id"] for run in runs]
        started_at = {run["run_id"]: run["started_at"] for run in runs}
        selected = sidebar.selectbox(
            "Run history",
            ids,
            index=None,
            key="history_run_id",
            placeholder="Select a previous run",
            format_func=lambda value: f"{value[:8]} - {started_at.get(value, '')}",
        )
        if selected and st.session_state.get("loaded_run_id") != selected:
            _load_historical_run(selected)
            st.session_state["loaded_run_id"] = selected

    getattr(sidebar, "divider", lambda: None)()
    getattr(sidebar, "header", lambda *_args, **_kwargs: None)("Backend")
    status = st.session_state.get("backend_status")
    cfg = ProviderConfig.from_env("llm")
    getattr(sidebar, "caption", lambda *_args, **_kwargs: None)(f"Configured: {cfg.backend} / {cfg.model}")
    if status:
        if status["ok"]:
            getattr(sidebar, "success", lambda *_args, **_kwargs: None)(
                f"Last call succeeded: {status['backend']} / {status['model']}"
            )
        else:
            getattr(sidebar, "error", lambda *_args, **_kwargs: None)(
                f"Last analysis incomplete: {status['backend']} / {status['model']}"
            )
    else:
        getattr(sidebar, "caption", lambda *_args, **_kwargs: None)("No backend call in this session.")


def _render_tab_safely(tab_name: str, st: Any, render_fn: Callable[[], None]) -> None:
    """Run one tab's body in isolation.

    ``st.tabs`` re-executes the body of *every* tab on *every* rerun, so an
    unhandled exception in one tab (e.g. the KPI view) would otherwise abort
    the whole script run and blank out tabs that were working fine. Contain
    the failure to the tab that raised it: show a short message in that tab
    and send the full traceback to the structured logger instead of the page.
    """
    try:
        render_fn()
    except Exception as exc:  # noqa: BLE001 - isolate every failure per tab
        get_logger().error(
            "ui:tab_error",
            extra={
                "fields": {
                    "tab": tab_name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            },
        )
        st.error(f"The {tab_name} tab failed to render: {type(exc).__name__}: {exc}")


def main() -> None:
    st.title("Contract Analyzer")
    _render_sidebar(st)
    tabs = st.tabs(["Chat", "KPI", "Monitoring"])

    with tabs[0]:
        st.session_state["active_tab"] = "Chat"
        _render_tab_safely("Analyze", st, lambda: _render_analyze_tab(st))
        _render_tab_safely("Chat", st, lambda: _render_chat_shell(st))
    with tabs[1]:
        st.session_state["active_tab"] = "KPI"
        _render_tab_safely("KPI", st, lambda: _render_kpi_tab(st))
    with tabs[2]:
        st.session_state["active_tab"] = "Monitoring"
        _render_tab_safely("Monitoring", st, lambda: monitoring_view.render(_store(), st))


if __name__ == "__main__":
    main()
