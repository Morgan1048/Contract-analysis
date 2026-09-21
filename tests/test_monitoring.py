from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

import pytest

from core.kpi import evaluate_thresholds
from core.monitoring import backend_health, monitoring_snapshot, records_by_run, usage_snapshot
from core.obs.logging import get_logger
from core.providers.config import ProviderConfig
from core.providers.errors import BackendError
from core.providers.openai_compat import OpenAICompatClient
from core.store import RunStore
from tests.test_kpi import _insert_finding, _insert_question_result, _insert_run
from ui import monitoring_view


NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)


def seed(store, run_id, *, age=0.5, duration=1000, cost=1, failures=0, count=20, **kwargs):
    _insert_run(store, run_id, started_at=(NOW - timedelta(days=age)).isoformat(),
                duration_ms=duration, cost_usd=cost, **kwargs)
    _insert_question_result(store, run_id, "Q1", .9, ["deviation"])
    for i in range(count):
        _insert_finding(store, run_id, f"P{i}", quote_verified=int(i >= failures))
    payload = {"usage": {"pipeline_version": "langgraph-evidence-v2", "quote_attempts": count,
                         "quote_failures": failures, "budget_exhausted": bool(failures)},
               "incomplete": bool(failures), "questions": [{"findings": [{"verdict": "MET"}] * count}]}
    store.db.execute("INSERT INTO run_snapshots(run_id, schema_version, payload) VALUES (?,2,?)",
                     (run_id, json.dumps(payload)))
    store.db.commit()


def test_empty_monitoring_is_explicitly_missing_not_zero(tmp_path):
    state = monitoring_snapshot(RunStore(tmp_path / "runs.db"), [], now=NOW)
    assert state["health"] == "healthy"
    assert state["alerts"] == state["anomalies"] == []
    for movement in state["movements"].values():
        assert movement == {"value": None, "baseline": None, "baseline_days": 0,
                            "change": None, "relative_change": None}


def test_legacy_quote_flags_do_not_become_attempts_or_budget_exhaustion(tmp_path):
    store = RunStore(tmp_path / "legacy.db")
    seed(store, "legacy", failures=3)
    store.db.execute("DELETE FROM run_snapshots")
    state = monitoring_snapshot(store, [], now=NOW)
    assert state["metrics"]["quote_verification_failure_rate"] is None
    assert state["metrics"]["quote_attempts"] == 0
    assert "Retries exhausted" not in str(state["anomalies"])


@pytest.mark.parametrize(("failures", "health"), [(0, "healthy"), (1, "degraded"), (3, "failing")])
def test_health_and_actions_come_from_current_thresholds(tmp_path, failures, health):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "current", failures=failures)
    seed(store, "ancient", age=100, failures=20, duration=999999)
    state = monitoring_snapshot(store, [], now=NOW)
    assert state["health"] == health
    assert state["metrics"]["runs"] == 1
    assert state["alerts"] == evaluate_thresholds({**state["metrics"], "cost_median_7d": 0})


def test_movements_use_seven_daily_medians_excluding_current_and_empty_days(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for day in range(1, 8):
        seed(store, f"day-{day}", age=day + .5, duration=day * 1000, cost=day, failures=day)
    seed(store, "current", duration=8000, cost=12, failures=10)
    seed(store, "too-old", age=8.01, duration=999999, cost=999, failures=20)
    state = monitoring_snapshot(store, [], now=NOW)
    movement = state["movements"]
    assert movement["p95_latency_ms"]["baseline"] == 4000
    assert movement["p95_latency_ms"]["relative_change"] == 1
    assert movement["cost_per_document"]["baseline"] == 4
    assert movement["cost_per_document"]["change"] == 8
    assert movement["quote_verification_failure_rate"]["baseline"] == .2
    assert movement["quote_verification_failure_rate"]["change"] == pytest.approx(.3)
    assert all(value["baseline_days"] == 7 for value in movement.values())
    cost_alert = next(alert for alert in state["alerts"] if alert.name == "cost_per_document")
    assert (cost_alert.value, cost_alert.threshold) == (12, 8)


def test_cost_is_per_document_and_unpriced_is_missing(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "one", cost=2, count=1)
    seed(store, "many", cost=4, count=100)
    seed(store, "unpriced", cost=None, unpriced_model=1)
    seed(store, "baseline-zero", age=2, cost=0)
    state = monitoring_snapshot(store, [], now=NOW)
    movement = state["movements"]["cost_per_document"]
    assert movement["value"] == 3
    assert movement["baseline"] == 0
    assert movement["relative_change"] is None
    assert monitoring_view._movement_label("cost_per_document", movement) == "+$3.0000 vs zero baseline"


def test_usage_snapshot_exposes_latest_pressure_not_a_raw_ledger(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "baseline", age=2)
    seed(store, "latest", age=.5)
    with store.db:
        store.db.execute(
            "UPDATE runs SET total_prompt_tokens=300,total_completion_tokens=100 "
            "WHERE run_id='baseline'"
        )
        store.db.execute(
            "UPDATE runs SET total_prompt_tokens=900,total_completion_tokens=300 "
            "WHERE run_id='latest'"
        )
        store.db.execute(
            "UPDATE findings SET prompt_tokens=600,completion_tokens=200,cost_usd=.03 "
            "WHERE run_id='latest'"
        )
        store.db.execute(
            "INSERT INTO stage_timings VALUES ('latest','judge',100,1200,.03,0)"
        )
    usage = usage_snapshot(store, now=NOW)
    assert usage["latest_run"]["run_id"] == "latest"
    assert usage["latest_total_tokens"] == 1200
    assert usage["recent_median_tokens"] == 400
    assert usage["departure_from_median"] == 2
    assert usage["stage_share"][0]["stage"] == "judge"
    assert usage["stage_share"][0]["share"] == 1
    assert usage["question_cost"] == [{
        "question_id": "Q1", "prompt_tokens": 12000,
        "completion_tokens": 4000, "cost_usd": pytest.approx(.6),
    }]
    assert usage["trend"][-1]["input_tokens"] == 900


def test_timezone_and_sqlite_timestamp_boundaries(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "current")
    seed(store, "baseline")
    seed(store, "future")
    with store.db:
        store.db.execute("UPDATE runs SET started_at='2026-09-19 12:00:00' WHERE run_id='current'")
        store.db.execute("UPDATE runs SET started_at='2026-09-19T13:59:59+02:00' WHERE run_id='baseline'")
        store.db.execute("UPDATE runs SET started_at='2026-09-21T00:00:00+00:00' WHERE run_id='future'")
    state = monitoring_snapshot(store, [], now=NOW)
    assert state["metrics"]["runs"] == 1
    assert state["movements"]["cost_per_document"]["baseline_days"] == 1
    assert [run["run_id"] for run in state["anomalies"]] == ["current"]


def test_anomalies_cover_failures_breaches_slowest_and_omit_ordinary_runs(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for i in range(6):
        seed(store, f"ordinary-{i}", duration=i + 1)
    seed(store, "incomplete")
    seed(store, "backend")
    seed(store, "exhausted", failures=1)
    seed(store, "slow", duration=130000)
    with store.db:
        store.db.execute("UPDATE runs SET status='incomplete' WHERE run_id='incomplete'")
        store.db.execute("INSERT INTO checks VALUES ('backend','P0','backend',0,'BackendError: unavailable',0)")
    state = monitoring_snapshot(store, [], now=NOW)
    anomalies = {run["run_id"]: run["reasons"] for run in state["anomalies"]}
    assert "Incomplete" in anomalies["incomplete"]
    assert "BackendError" in anomalies["backend"]
    assert "Retries exhausted" in anomalies["exhausted"]
    assert "p95_latency_ms" in anomalies["slow"]
    assert "Slowest run" in anomalies["slow"]
    assert "ordinary-0" not in anomalies
    assert "deviation_applied_rate" not in str(anomalies)


def test_successful_final_retry_is_not_exhaustion_and_list_is_bounded(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for i in range(14):
        seed(store, f"run-{i}", failures=1 if i else 0)
    with store.db:
        store.db.execute("UPDATE findings SET retry_count=2 WHERE run_id='run-0'")
    state = monitoring_snapshot(store, [], now=NOW)
    assert len(state["anomalies"]) == 10
    assert state["anomaly_count"] >= 13
    assert all("Retries exhausted" in run["reasons"] for run in state["anomalies"])


def test_per_run_threshold_catches_outlier_hidden_by_aggregate(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for i in range(20):
        seed(store, f"normal-{i}")
    seed(store, "outlier", duration=120001)
    with store.db:
        store.db.execute("UPDATE findings SET retry_count=2 WHERE run_id='outlier'")
    state = monitoring_snapshot(store, [], now=NOW)
    assert not any(alert.name == "p95_latency_ms" for alert in state["alerts"])
    outlier = next(run for run in state["anomalies"] if run["run_id"] == "outlier")
    assert "p95_latency_ms" in outlier["reasons"]
    assert "Retries exhausted" not in outlier["reasons"]


def test_nested_errors_without_persisted_run_and_reused_trace_ids(tmp_path):
    records = [
        {"run_id": "lost", "span_id": "root", "trace_id": "shared", "ts": NOW.timestamp()},
        {"span_id": "call", "parent_span_id": "root", "message": "BackendError",
         "retry_exhausted": True, "ts": NOW.timestamp()},
        {"run_id": "other", "span_id": "unrelated", "trace_id": "shared", "ts": NOW.timestamp()},
    ]
    assert len(records_by_run(records)["lost"]) == 2
    state = monitoring_snapshot(RunStore(tmp_path / "runs.db"), records, now=NOW)
    assert len(state["anomalies"]) == 1
    assert state["anomalies"][0]["run_id"] == "lost"
    assert state["anomalies"][0]["reasons"] == ["BackendError", "Retries exhausted"]


def test_backend_health_uses_last_terminal_call_not_latest_run_or_start():
    records = [
        {"event": "backend:call", "provider": "old", "model": "old", "message": "end", "ts": 1},
        {"event": "backend:call", "provider": "ollama", "model": "test", "message": "BackendError", "ts": 2},
        {"event": "backend:call", "message": "start", "ts": 3},
    ]
    assert backend_health(records, "configured", "new") == {
        "provider": "ollama", "model": "test", "outcome": "Failed: BackendError", "ts": 2,
    }
    assert "Unknown" in backend_health([], "configured", "new")["outcome"]


@pytest.mark.parametrize(("fail", "retries", "exhausted"), [(False, 0, False), (True, 0, False), (True, 1, True)])
def test_backend_calls_record_real_outcomes(monkeypatch, fail, retries, exhausted):
    import core.providers.openai_compat as provider_module

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(dict(getattr(record, "fields", {}), event=record.msg))

    class Response(io.BytesIO):
        pass

    def request(*args, **kwargs):
        if fail:
            raise HTTPError("http://localhost", 503 if retries else 401, "failed", {}, io.BytesIO())
        return Response(b'{"data": []}')

    monkeypatch.setattr(provider_module, "urlopen", request)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _: None)
    client = OpenAICompatClient(ProviderConfig("ollama", "http://localhost:11434/v1", "test", max_retries=retries))
    handler = Capture()
    logger = get_logger()
    logger.addHandler(handler)
    try:
        if fail:
            with pytest.raises(BackendError):
                client.embed(["synthetic"])
        else:
            assert client.embed(["synthetic"]) == []
    finally:
        logger.removeHandler(handler)
    end = records[-1]
    assert end["event"] == "backend:call"
    assert end["provider"] == "ollama" and end["model"] == "test"
    assert end["message"] == ("BackendError" if fail else "end")
    assert end["retry_exhausted"] is exhausted
    assert end["retry_count"] == retries


class FakeSt:
    def __init__(self, run=None, selected_span=None):
        self.run, self.selected_span = run, selected_span
        self.messages, self.tables, self.span_options = [], [], []

    def subheader(self, text):
        self.messages.append(text)

    caption = info = error = warning = success = write = subheader

    def columns(self, n):
        return [self] * n

    def metric(self, *args, **kwargs):
        pass

    def button(self, *args, **kwargs):
        return False

    def radio(self, label, options, **kwargs):
        return self.run

    def selectbox(self, label, options, **kwargs):
        self.span_options = options
        return self.selected_span

    def dataframe(self, data, **kwargs):
        self.tables.append(data)


@pytest.mark.parametrize(("run", "selected_span"), [(None, None), ("stale", None), ("current", None), ("current", "stale"), ("current", "child-2")])
def test_view_guards_empty_stale_selection_and_duplicate_span_labels(tmp_path, monkeypatch, run, selected_span):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "current")
    records = [
        {"run_id": "current", "span_id": "root", "event": "analyze"},
        {"parent_span_id": "root", "span_id": "child-1", "event": "judge", "message": "start"},
        {"parent_span_id": "root", "span_id": "child-2", "event": "judge", "message": "start"},
    ]
    path = tmp_path / "logs.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    real_snapshot = monitoring_snapshot
    monkeypatch.setattr(monitoring_view, "monitoring_snapshot",
                        lambda store, records: real_snapshot(store, records, now=NOW))
    st = FakeSt(run, selected_span)
    monitoring_view.render(store, st, log_path=path)
    assert "Live log tail" not in st.messages
    if run not in ("current",):
        assert not st.span_options
    elif selected_span == "child-2":
        assert st.tables[-1] == [records[2]]
        assert st.span_options == ["root", "child-1", "child-2"]
    else:
        assert "Select a span to see its log records." in st.messages


def test_view_without_runs_or_logs(tmp_path):
    st = FakeSt()
    monitoring_view.render(RunStore(tmp_path / "runs.db"), st, log_path=tmp_path / "missing.jsonl")
    assert "No recent anomalous runs." in st.messages
    assert any("No stored runs" in message for message in st.messages)


def test_selected_run_without_retained_spans(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "current")
    monkeypatch.setattr(monitoring_view, "monitoring_snapshot",
                        lambda store, records: monitoring_snapshot(store, records, now=NOW))
    st = FakeSt("current")
    monitoring_view.render(store, st, log_path=tmp_path / "missing.jsonl")
    assert any("No retained span records" in message for message in st.messages)
    assert not st.span_options


def test_streamlit_monitoring_empty_selection_and_drilldown(tmp_path):
    from streamlit.testing.v1 import AppTest

    store = RunStore(tmp_path / "runs.db")
    _insert_run(store, "current", started_at=datetime.now(timezone.utc).isoformat())
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(json.dumps({
        "run_id": "current", "span_id": "root", "event": "analyze",
        "message": "end", "duration_ms": 1000,
    }) + "\n")
    page = AppTest.from_string(
        "import streamlit as st\n"
        "from core.store import RunStore\n"
        "from ui.monitoring_view import render\n"
        f"render(RunStore({str(store.path)!r}), st, log_path={str(log_path)!r})\n"
    ).run()
    assert not page.exception
    assert page.radio[0].value is None
    assert not page.selectbox
    page.radio[0].set_value("current").run()
    assert not page.exception
    assert page.selectbox[0].value is None
    assert page.info[-1].value == "Select a span to see its log records."
    page.selectbox[0].set_value("root").run()
    assert not page.exception
    assert page.dataframe[-1].value["span_id"].tolist() == ["root"]
