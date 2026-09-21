from __future__ import annotations

import builtins
import io
import json
import os

from core.agents.evaluator import Check
from core.pipeline import AnalysisResult
from core.rubric.schema import Finding, QuestionResult
from core.store import RunStore

import app


def _stored_result(source_path: str) -> AnalysisResult:
    finding = Finding("P1", "MET", 0.9, "Stored quote.", "sec:6.6", "Stored rationale.")
    return AnalysisResult(
        "run-history",
        source_path,
        [QuestionResult("Q1", "Question", "Fully Compliant", 0.9, [finding], [], "Rationale")],
        {"P1": [Check("quote_verified", True, "Stored check.")]},
        {"prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.0},
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
    )


def test_load_historical_run_uses_only_sqlite_when_source_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "deleted-upload.pdf"
    store = RunStore(tmp_path / "runs.db")
    store.save_run(_stored_result(str(source)))
    state = {"chat_agent": object()}

    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(
        app,
        "parse_pdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source was parsed")),
    )
    monkeypatch.setattr(
        app,
        "ChunkIndex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("index was built")),
    )
    monkeypatch.setattr(
        app,
        "LLMClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("backend was created")),
    )

    app._load_historical_run("run-history")

    loaded = state["result"]
    assert loaded.questions[0].findings[0].quote == "Stored quote."
    assert loaded.checks["P1"][0].detail == "Stored check."
    assert state["index"] is None
    assert "chat_agent" not in state
    unavailable = state["registry"].call("search_contract", {"query": "new evidence"})
    assert unavailable["source_required"] is True


def test_historical_ui_renders_stored_evidence_and_checks_without_source_access(
    tmp_path, monkeypatch
):
    source = tmp_path / "deleted-upload.pdf"
    store = RunStore(tmp_path / "runs.db")
    store.save_run(_stored_result(str(source)))
    state = {}

    class RenderSt:
        def __init__(self):
            self.session_state = state
            self.infos = []
            self.downloads = []
            self.subheaders = []
            self.markdowns = []
            self.captions = []
            self.writes = []

        def info(self, message):
            self.infos.append(message)

        def download_button(self, _label, **kwargs):
            self.downloads.append(kwargs)

        def subheader(self, message):
            self.subheaders.append(message)

        def markdown(self, message):
            self.markdowns.append(message)

        def caption(self, message):
            self.captions.append(message)

        def write(self, message):
            self.writes.append(message)

        def warning(self, message):
            raise AssertionError(f"unexpected warning: {message}")

        def error(self, message):
            raise AssertionError(f"unexpected error: {message}")

        def divider(self):
            pass

    render_st = RenderSt()
    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(
        app,
        "parse_pdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("source was parsed")),
    )
    monkeypatch.setattr(
        app,
        "ChunkIndex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("index was built")),
    )

    app._load_historical_run("run-history")
    real_stat = os.stat
    real_open = builtins.open
    real_io_open = io.open

    def reject_source(operation, path, *args, **kwargs):
        try:
            resolved = os.fspath(path)
        except TypeError:
            resolved = None
        if resolved == str(source):
            raise AssertionError("historical rendering accessed the original source")
        return operation(path, *args, **kwargs)

    with monkeypatch.context() as source_guard:
        source_guard.setattr(
            os, "stat", lambda path, *args, **kwargs: reject_source(real_stat, path, *args, **kwargs)
        )
        source_guard.setattr(
            builtins,
            "open",
            lambda path, *args, **kwargs: reject_source(real_open, path, *args, **kwargs),
        )
        source_guard.setattr(
            io,
            "open",
            lambda path, *args, **kwargs: reject_source(real_io_open, path, *args, **kwargs),
        )
        app._render_analyze_tab(render_st)

    assert render_st.subheaders == [state["result"].questions[0].question_text]
    assert any('Section 6.6: "Stored quote."' in line for line in render_st.markdowns)
    assert "Rationale" in render_st.writes
    assert any("quote_verified: passed - Stored check." in line for line in render_st.captions)
    assert any("Re-upload the original PDF" in message for message in render_st.infos)
    downloaded = json.loads(render_st.downloads[0]["data"])
    assert downloaded["questions"][0]["Relevant Quotes"][0]["quote"] == "Stored quote."
    assert downloaded["source_path"] == str(source)


def test_sidebar_ignores_empty_history_selection(monkeypatch):
    class Sidebar:
        def file_uploader(self, *_args, **_kwargs):
            return None

        def selectbox(self, *_args, **_kwargs):
            return None

        def button(self, *_args, **_kwargs):
            return False

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class St:
        sidebar = Sidebar()
        session_state = {}

    store = type(
        "Store",
        (),
        {"list_runs": lambda self: [{"run_id": "run-1", "started_at": "2026-01-01"}]},
    )()
    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(
        app,
        "_load_historical_run",
        lambda _run_id: (_ for _ in ()).throw(AssertionError("empty selection was loaded")),
    )
    monkeypatch.setattr(app.ProviderConfig, "from_env", lambda *_args: type("Cfg", (), {"backend": "x", "model": "y"})())

    app._render_sidebar(St())


def test_default_chat_session_is_scoped_to_loaded_run(monkeypatch):
    state = {
        "result": type("Result", (), {"run_id": "run-history"})(),
        "registry": object(),
    }
    captured = {}

    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "_store", lambda: object())
    monkeypatch.setattr(app.ProviderConfig, "from_env", lambda *_args: object())
    monkeypatch.setattr(app, "LLMClient", lambda *_args: object())

    def fake_agent(_client, _registry, _store, *, session_id, run_id):
        captured.update(session_id=session_id, run_id=run_id)
        return object()

    monkeypatch.setattr(app, "ChatAgent", fake_agent)

    app._ensure_chat_agent()

    assert captured == {"session_id": "run:run-history", "run_id": "run-history"}


def test_incomplete_backend_status_is_not_rendered_as_success(tmp_path, monkeypatch):
    class Sidebar:
        def __init__(self):
            self.successes = []
            self.errors = []

        def file_uploader(self, *_args, **_kwargs):
            return None

        def success(self, message):
            self.successes.append(message)

        def error(self, message):
            self.errors.append(message)

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    class St:
        sidebar = Sidebar()
        session_state = {
            "backend_status": {
                "ok": False,
                "backend": "openai",
                "model": "example-model",
            }
        }

    monkeypatch.setattr(app, "_store", lambda: RunStore(tmp_path / "runs.db"))
    monkeypatch.setattr(
        app.ProviderConfig,
        "from_env",
        lambda *_args: type("Cfg", (), {"backend": "openai", "model": "example-model"})(),
    )

    app._render_sidebar(St())

    assert St.sidebar.successes == []
    assert St.sidebar.errors == ["Last analysis incomplete: openai / example-model"]


def test_upload_records_incomplete_backend_status(tmp_path, monkeypatch):
    import threading
    from core.providers.progress import WaitUpdate, observer

    main_thread = threading.get_ident()
    result = _stored_result(str(tmp_path / "upload.pdf"))
    result = AnalysisResult(
        result.run_id,
        result.source_path,
        result.questions,
        result.checks,
        {"backend": "openai", "model": "example-model"},
        result.started_at,
        result.finished_at,
        incomplete=True,
    )

    class Status:
        def __init__(self):
            self.updates = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **kwargs):
            assert threading.get_ident() == main_thread
            self.updates.append(kwargs)

    class St:
        session_state = {}

        def __init__(self):
            self.status_widget = Status()

        def status(self, *_args, **_kwargs):
            return self.status_widget

    fake_st = St()
    store = type("Store", (), {"save_run": lambda self, saved: None})()
    document = type("Document", (), {"chunks": []})()
    index = object()

    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app, "parse_pdf", lambda _path: document)
    monkeypatch.setattr(app, "ChunkIndex", lambda _doc, **_kwargs: index)
    monkeypatch.setattr(app.ProviderConfig, "from_env", lambda *_args: object())
    monkeypatch.setattr(app, "LLMClient", lambda _cfg: object())
    def analyze(*_args, **_kwargs):
        assert threading.get_ident() != main_thread
        observer()(WaitUpdate("groq", "openai/gpt-oss-120b", "rate_limit", 600, 1, 12, 429))
        return result

    monkeypatch.setattr(app, "analyze", analyze)
    monkeypatch.setattr(app, "build_tool_registry", lambda *_args, **_kwargs: object())

    uploaded = type("Upload", (), {"getvalue": lambda self: b"%PDF-test"})()
    app._analyze_uploaded_file(uploaded)

    assert fake_st.session_state["backend_status"] == {
        "ok": False,
        "backend": "openai",
        "model": "example-model",
    }
    assert fake_st.session_state["history_run_id"] == result.run_id
    assert fake_st.status_widget.updates[-1] == {
        "label": "Analysis incomplete.",
        "state": "error",
    }
    labels = [update["label"] for update in fake_st.status_widget.updates]
    assert "API rate limit (HTTP 429): waiting 600s before retry 1/12. Analysis is still running." in labels
    assert all("groq" not in label and "gpt-oss" not in label for label in labels)


def test_real_page_opens_with_legacy_history(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    store = RunStore(tmp_path / "legacy.db")
    result = _stored_result(str(tmp_path / "missing.pdf"))
    store.save_run(result)
    # Pre-snapshot runs reconstruct questions using the current rubric.
    with store.db:
        store.db.execute("DELETE FROM run_snapshots WHERE run_id=?", (result.run_id,))
    monkeypatch.setenv("CA_RUN_DB", str(store.path))
    app.st.cache_resource.clear()
    try:
        page = AppTest.from_file(app.__file__, default_timeout=30).run()
        assert not page.exception
        assert page.sidebar.selectbox[0].value is None
        assert "result" not in page.session_state
        assert not page.tabs[0].info
        assert not page.tabs[0].error
        assert not page.tabs[0].subheader
        assert not page.tabs[0].get("download_button")

        page.sidebar.selectbox[0].select(result.run_id).run()
        assert not page.exception
        assert page.sidebar.selectbox[0].value == result.run_id
        assert page.session_state["result"].questions[0].findings[0].quote == "Stored quote."
        assert page.session_state["loaded_run_id"] == result.run_id
        assert [tab.label for tab in page.tabs] == ["Chat", "KPI", "Monitoring"]
        assert store.get_run(result.run_id) is not None
    finally:
        app.st.cache_resource.clear()
