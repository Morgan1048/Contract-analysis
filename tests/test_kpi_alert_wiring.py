from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from core.kpi import evaluate_thresholds, quality_snapshot, snapshot, summarize_agent_payloads
from core.monitoring import monitoring_snapshot
from core.pipeline import AnalysisResult
from core.rubric.schema import QuestionResult
from core.store import RunStore
from core.tools.contract_tools import build_tool_registry
from tests.test_kpi import _insert_eval_result
from ui.kpi_view import render


NOW = datetime.now(timezone.utc)


def seed(store, run_id, *, age=.1, state="Fully Compliant", warnings=(), demo=False,
         incomplete=False, rubric_hash="synthetic-rubric", version="langgraph-evidence-v2",
         questions=("Q1",), timestamp=None):
    started = timestamp or (NOW - timedelta(days=age)).isoformat()
    store.save_run(AnalysisResult(
        run_id, "independent-fixture.pdf",
        [QuestionResult(qid, "Synthetic requirement", state, .9, [], [], "Test fixture.")
         for qid in questions],
        {}, {"pipeline_version": version, "parse_warnings": list(warnings)},
        started, started, incomplete=incomplete, rubric_hash=rubric_hash,
    ), demo=demo)


def seed_shift(store, *, count=5, changed=None):
    changed = count if changed is None else changed
    for i in range(count):
        seed(store, f"previous-{i}", age=8 + i / 100)
        seed(store, f"current-{i}", age=.1 + i / 100,
             state="Non-Compliant" if i < changed else "Fully Compliant")


@pytest.mark.parametrize("reverse", [False, True])
def test_latest_parse_warning_is_independent_of_order_and_timestamp_format(reverse):
    payloads = [
        {"run_id": "older", "started_at": "2026-09-20T13:00:00+02:00", "incomplete": False,
         "usage": {"pipeline_version": "langgraph-evidence-v2", "parse_warnings": ["old"]},
         "questions": []},
        {"run_id": "newer", "started_at": "2026-09-20 11:30:00", "incomplete": False,
         "usage": {"pipeline_version": "langgraph-evidence-v2", "parse_warnings": ["new", "new2"]},
         "questions": []},
    ]
    if reverse:
        payloads.reverse()
    original = json.dumps(payloads)
    assert summarize_agent_payloads(payloads)["latest_parse_warning_count"] == 2
    assert json.dumps(payloads) == original


@pytest.mark.parametrize("latest_warnings", [[], ["Unreadable page"]])
def test_monitoring_and_dashboard_use_newest_parse_warning(tmp_path, latest_warnings):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "older", age=.4, warnings=[] if latest_warnings else ["Old warning"])
    seed(store, "latest", warnings=latest_warnings)
    for metrics in (snapshot(store, now=NOW), monitoring_snapshot(store, [], now=NOW)["metrics"]):
        assert metrics["latest_parse_warning_count"] == len(latest_warnings)
        assert ("parse_anomaly" in {a.name for a in evaluate_thresholds(metrics)}) == bool(latest_warnings)


class RecordingSt:
    def __init__(self):
        self.messages = []
        self.metrics = {}

    def header(self, text):
        self.messages.append(text)

    subheader = caption = error = warning = header

    def columns(self, count):
        return [self] * count

    def metric(self, name, value):
        self.metrics[name] = value

    def table(self, rows):
        pass

    line_chart = table


def test_quality_alerts_reach_dashboard_chat_monitoring_and_tools(tmp_path, monkeypatch):
    import app

    store = RunStore(tmp_path / "runs.db")
    seed_shift(store)
    _insert_eval_result(store, "gold", "current-0", "Q1", correct=False)
    expected = {"accuracy_vs_gold_set": 0.0, "state_distribution_drift": 1.0}
    snap = snapshot(store, now=NOW)
    assert {key: snap[key] for key in expected} == expected
    assert snap["quality_run_id"] == "current-0"
    assert {a.name for a in evaluate_thresholds(snap)} >= set(expected)

    state = monitoring_snapshot(store, [], now=NOW)
    assert {a.name for a in state["alerts"]} >= set(expected)
    assert {key: state["metrics"][key] for key in expected} == expected
    assert not any("state_distribution_drift" in row["reasons"] for row in state["anomalies"])

    registry = build_tool_registry(None, store)
    tool_snapshot = registry.call("get_kpi_snapshot", {})["snapshot"]
    assert {key: tool_snapshot[key] for key in expected} == expected
    for name, value in expected.items():
        explanation = registry.call("explain_kpi", {"name": name})
        assert explanation["current_value"] == value and explanation["available"]
    assert registry.call("explain_kpi", {"name": "state_distribution_drift"})["baseline"]["current_runs"] == 5

    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state={}))
    alerts = app._chat_context()["alerting_kpis"]
    assert {name: alerts[name]["value"] for name in expected} == expected

    st = RecordingSt()
    render(store, st)
    assert all(any(name in message for message in st.messages) for name in expected)
    assert st.metrics["State-distribution shift"] == "100.0 pp"


def test_unavailable_quality_is_null_not_zero_or_an_old_score(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "older", age=2)
    _insert_eval_result(store, "old-gold", "older", "Q1", correct=False)
    seed(store, "latest")
    with store.db:
        store.db.execute("CREATE TABLE quality_metrics (run_id TEXT, metric TEXT, source TEXT, value REAL)")
        store.db.execute("INSERT INTO quality_metrics VALUES ('latest','answer_correctness','computed',.99)")
    snap = snapshot(store, now=NOW)
    assert snap["accuracy_vs_gold_set"] is None
    assert snap["state_distribution_drift"] is None
    assert not {a.name for a in evaluate_thresholds(snap)} & {"accuracy_vs_gold_set", "state_distribution_drift"}
    registry = build_tool_registry(None, store)
    for name in ("accuracy_vs_gold_set", "state_distribution_drift"):
        result = registry.call("explain_kpi", {"name": name})
        assert result["current_value"] is None and not result["available"]
        json.dumps(result, allow_nan=False)
    st = RecordingSt()
    render(store, st)
    assert st.metrics["State-distribution shift"] == "unavailable"
    assert any("at least 5" in message for message in st.messages)


@pytest.mark.parametrize("changed,expected,alert", [(2, .25, False), (3, .375, True)])
def test_drift_threshold_is_strictly_above_25_percentage_points(tmp_path, changed, expected, alert):
    store = RunStore(tmp_path / "runs.db")
    seed_shift(store, count=8, changed=changed)
    metrics = quality_snapshot(store, now=NOW)
    assert metrics["state_distribution_drift"] == expected
    assert ("state_distribution_drift" in {a.name for a in evaluate_thresholds(metrics)}) == alert


def test_drift_requires_sufficient_matching_questions_and_metadata(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed_shift(store, count=4)
    assert quality_snapshot(store, now=NOW)["state_distribution_drift"] is None
    seed(store, "previous-extra", age=9)
    seed(store, "current-extra", age=.2, questions=("DifferentQuestion",))
    assert quality_snapshot(store, now=NOW)["state_distribution_drift"] is None
    seed(store, "no-metadata", age=.01, rubric_hash="")
    result = quality_snapshot(store, now=NOW)
    assert result["state_distribution_drift"] is None
    assert "versions" in result["state_distribution_baseline"]["reason"]


def test_drift_excludes_demo_incomplete_incompatible_and_out_of_window_runs(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed_shift(store)
    excluded = [
        {"demo": True}, {"incomplete": True}, {"rubric_hash": "another-rubric"},
        {"version": "old-pipeline"}, {"state": "Indeterminate"}, {"age": 15}, {"age": -1},
    ]
    for i, kwargs in enumerate(excluded):
        seed(store, f"excluded-{i}", **({"age": 1} | kwargs))
    _insert_eval_result(store, "demo-label", "excluded-0", "Q1", correct=False)
    result = quality_snapshot(store, now=NOW)
    assert result["quality_run_id"] == "current-0"
    assert result["state_distribution_drift"] == 1.0
    assert result["state_distribution_baseline"]["current_runs"] == 5
    assert result["state_distribution_baseline"]["baseline_runs"] == 5
    assert result["calibration_bins"] == []


def test_drift_windows_have_non_overlapping_timezone_aware_boundaries(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed_shift(store, count=4)
    baseline_edge = (NOW - timedelta(days=14)).astimezone(timezone(timedelta(hours=2))).isoformat()
    current_edge = (NOW - timedelta(days=7)).isoformat()
    seed(store, "baseline-edge", timestamp=baseline_edge)
    seed(store, "current-edge", timestamp=current_edge, state="Non-Compliant")
    seed(store, "outside", timestamp=(NOW - timedelta(days=14, seconds=1)).isoformat())
    result = quality_snapshot(store, now=NOW)
    assert result["state_distribution_drift"] == 1.0
    assert result["state_distribution_baseline"]["current_runs"] == 5
    assert result["state_distribution_baseline"]["baseline_runs"] == 5


def test_state_mix_is_compared_per_question_not_hidden_by_pooled_distribution(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    for window, age in (("previous", 8), ("current", .1)):
        for i in range(5):
            seed(store, f"{window}-{i}", age=age + i / 100, questions=("Q1", "Q2"))
            with store.db:
                question = "Q1" if window == "previous" else "Q2"
                store.db.execute("UPDATE question_results SET state='Non-Compliant' WHERE run_id=? AND question_id=?",
                                 (f"{window}-{i}", question))
    assert quality_snapshot(store, now=NOW)["state_distribution_drift"] == 1.0
