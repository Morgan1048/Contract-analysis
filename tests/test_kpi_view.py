"""The operational dashboard remains usable without the retired RAG evaluator."""
import importlib.util
import sqlite3

import pytest

from core.store import RunStore
from tests.test_kpi_alert_wiring import RecordingSt, seed
from ui.kpi_view import _format_metric, render


def test_dashboard_has_no_rag_quality_panel(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    seed(store, "run")
    st = RecordingSt()
    render(store, st)
    assert "Operational (real time)" in st.messages
    assert "Agent reliability and follow-up" in st.messages
    assert "History" in st.messages
    assert not any("RAG quality" in message or "RAGAS" in message for message in st.messages)
    assert "Runs" in st.metrics


def test_empty_dashboard_still_renders(tmp_path):
    st = RecordingSt()
    render(RunStore(tmp_path / "runs.db"), st)
    assert st.metrics["Runs"] == 0


def test_new_store_does_not_create_retired_evaluation_tables(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    tables = {row["name"] for row in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not tables & {"quality_metrics", "rag_evaluations", "rag_references"}


def test_old_evaluation_data_is_not_deleted_or_resumed(tmp_path):
    path = tmp_path / "runs.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE rag_evaluations (run_id TEXT, status TEXT, score REAL)")
        db.execute("INSERT INTO rag_evaluations VALUES ('old','completed',.8)")
    store = RunStore(path)
    render(store, RecordingSt())
    assert tuple(store.db.execute("SELECT * FROM rag_evaluations").fetchone()) == ("old", "completed", .8)


@pytest.mark.parametrize("module", ["core.quality", "core.rag_metrics", "eval.ragas_eval", "eval.run_ragas"])
def test_retired_backend_modules_are_removed(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("value,percentage,expected", [
    (.9515151515, True, "95.2%"), (1.0, False, "1"), (15.55555, False, "15.6"),
    ("unavailable", True, "unavailable"),
])
def test_operational_metric_formatting(value, percentage, expected):
    assert _format_metric(value, as_percentage=percentage) == expected
