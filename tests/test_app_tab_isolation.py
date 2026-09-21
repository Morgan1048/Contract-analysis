"""tests/test_app_tab_isolation.py -- a UI robustness regression test.

``st.tabs`` executes the body of *every* tab on *every* rerun. Before this
fix, ``app.main`` rendered the three tab bodies with no error handling, so an
exception raised inside one tab's view (e.g. ``kpi_view.render``) aborted the
whole Streamlit script and the traceback landed on the page regardless of
which tab was selected -- taking down the Analyze and Chat tabs too, even
though they were working fine.

This uses a fake ``st`` object, following the pattern of the other UI tests
(``tests/test_ui_helpers.py``), to run ``app.main()`` end-to-end without a
real Streamlit runtime and assert that a failure in one tab is contained.
"""
from __future__ import annotations

import logging

import app


class FakeTab:
    """A no-op context manager standing in for one of ``st.tabs()``'s tabs."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSt:
    """A minimal fake of the ``streamlit`` module surface ``app.main`` touches."""

    def __init__(self):
        self.session_state: dict = {}
        self.errors: list[str] = []
        self.infos: list[str] = []

    def title(self, *args, **kwargs):
        pass

    def tabs(self, labels):
        return [FakeTab() for _ in labels]

    def file_uploader(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.errors.append(message)


def test_one_tab_raising_does_not_take_down_the_others(monkeypatch):
    fake_st = FakeSt()
    monkeypatch.setattr(app, "st", fake_st)

    analyze_calls: list[bool] = []
    kpi_calls: list[bool] = []

    def _boom_kpi_render(*args, **kwargs):
        kpi_calls.append(True)
        raise ValueError("simulated KPI view failure")

    def _fake_store():
        return object()

    monkeypatch.setattr(app, "_render_analyze_tab", lambda st: analyze_calls.append(True))
    monkeypatch.setattr(app.kpi_view, "render", _boom_kpi_render)
    monkeypatch.setattr(app.monitoring_view, "render", lambda *_args: None)
    monkeypatch.setattr(app, "_store", _fake_store)

    log_records: list[logging.LogRecord] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record):
            log_records.append(record)

    logger = app.get_logger()
    handler = _CapturingHandler()
    logger.addHandler(handler)
    try:
        # Must not raise: the KPI tab's exception is contained, so the whole
        # script run completes and the other tabs still rendered.
        app.main()
    finally:
        logger.removeHandler(handler)

    # The other tabs ran to completion, unaffected by the KPI tab's crash.
    assert analyze_calls == [True]
    # Chat remains blank without an active contract.
    assert fake_st.infos == []

    # The KPI tab's body did run (and blew up) ...
    assert kpi_calls == [True]
    # ... but the failure was contained to that tab: exactly one error shown,
    # naming the exception type and message, not a raw traceback dump.
    assert len(fake_st.errors) == 1
    assert "KPI" in fake_st.errors[0]
    assert "ValueError" in fake_st.errors[0]
    assert "simulated KPI view failure" in fake_st.errors[0]

    # The full traceback went to the structured logger, not to the page.
    assert any(
        getattr(record, "fields", {}).get("tab") == "KPI"
        and getattr(record, "fields", {}).get("error_type") == "ValueError"
        and "Traceback" in getattr(record, "fields", {}).get("traceback", "")
        for record in log_records
    )
    assert not any("Traceback" in message for message in fake_st.errors)
