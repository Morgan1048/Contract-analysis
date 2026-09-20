"""tests/test_kpi_view.py -- acceptance list for the KPI tab's "RAG quality:
judged versus computed" table (ui/kpi_view.quality_comparisons).

Before this test file existed, every cell in that table read "unavailable" or
"not applicable" except one raw float (faithfulness), and three rows shared an
identical, non-specific note. Each test below pins one concrete regression.
"""
from __future__ import annotations

import re

import pytest

from core.kpi import snapshot
from core.store import RunStore
from ui.kpi_view import quality_comparisons

SAMPLE_CONTRACT = "Sample Contract.pdf"


def _store(tmp_path):
    return RunStore(tmp_path / "runs.db")


def _insert_run(store, run_id, source_path=SAMPLE_CONTRACT):
    with store.db:
        store.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, source_path, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00",
                "stub", "local", 10, 10, 1.0, 1000.0, "completed", 0, 100, 10, 0,
            ),
        )


def _insert_eval_result(store, eval_id, run_id, question_id, correct):
    with store.db:
        store.db.execute(
            "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?)",
            (eval_id, run_id, question_id, "Fully Compliant", "Fully Compliant", int(correct), "2026-01-01T00:00:00+00:00", 0),
        )


def _rows_by_metric(rows):
    return {row["Metric"]: row for row in rows}


# 1. Offline retrieval diagnostics must not be presented as production KPIs.


def test_context_precision_and_recall_are_omitted_even_when_cached(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    store.save_quality_metrics("r1", "computed", {"context_precision": .75, "context_recall": .8})
    store.save_quality_metrics("r1", "ragas", {"context_precision": .9, "context_recall": .95})
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))

    assert set(rows) == {"Faithfulness / groundedness", "Noise sensitivity", "Answer correctness"}


# 2. An empty store (no runs at all) must still render -- lazily returning
# "unavailable" rather than raising.


def test_context_precision_and_recall_are_omitted_on_empty_store(tmp_path):
    store = _store(tmp_path)
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))

    assert set(rows) == {"Faithfulness / groundedness", "Noise sensitivity", "Answer correctness"}
    assert rows["Answer correctness"]["Exact (computed)"] == "unavailable"


# 3. Answer correctness is wired to eval_results gold-set accuracy.


def test_answer_correctness_is_wired_to_eval_results(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_eval_result(store, "e1", "r1", "Q1", correct=True)
    _insert_eval_result(store, "e2", "r1", "Q2", correct=False)
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))

    cell = rows["Answer correctness"]["Exact (computed)"]
    assert cell != "not applicable"
    assert cell == "50.0%"


def test_latest_quality_does_not_borrow_historical_gold_scores(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "older")
    _insert_eval_result(store, "gold", "older", "Q1", correct=True)
    _insert_run(store, "latest")
    store.db.execute("UPDATE runs SET started_at='2026-09-20T12:00:00+00:00' WHERE run_id='latest'")
    store.db.commit()
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))
    assert rows["Answer correctness"]["Exact (computed)"] == "unavailable"


def test_answer_correctness_is_unavailable_without_any_eval_results(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))
    assert rows["Answer correctness"]["Exact (computed)"] == "unavailable"


# 4. Every numeric cell renders in a short, consistent form -- never a raw
# float repr like 0.9515151515151515.


def test_numeric_cells_are_never_raw_float_repr(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_eval_result(store, "e1", "r1", "Q1", correct=True)
    rows = quality_comparisons(store, snapshot(store))

    raw_float_pattern = re.compile(r"\d\.\d{6,}")
    for row in rows:
        for column in ("RAGAS (LLM-judged)", "Exact (computed)"):
            value = row[column]
            assert not raw_float_pattern.search(str(value)), f"{row['Metric']}/{column} = {value!r}"


# 5. RAGAS "unavailable" must say why (not installed vs. no judge configured),
# never a bare "unavailable".


def test_ragas_unavailable_cell_explains_why(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _insert_run(store, "r1")

    import eval.ragas_eval as ragas_eval
    monkeypatch.setattr(ragas_eval, "ragas_installed", lambda: False)
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))
    cell = rows["Faithfulness / groundedness"]["RAGAS (LLM-judged)"]
    assert cell != "unavailable"
    assert "not installed" in cell
    assert "pip install" in cell

    monkeypatch.setattr(ragas_eval, "ragas_installed", lambda: True)
    monkeypatch.setattr(ragas_eval, "judge_configured", lambda: (False, "no judge"))
    rows = _rows_by_metric(quality_comparisons(store, snapshot(store)))
    cell = rows["Faithfulness / groundedness"]["RAGAS (LLM-judged)"]
    assert cell != "unavailable"
    assert "no judge model configured" in cell


# 6. Notes are per-metric: no two rows share an identical note, and each
# note mentions its own metric. (A test that only checks "non-empty" would
# not catch a repeated generic placeholder -- this is the third time that
# exact failure mode has shown up, after explain_kpi's definition/formula.)


def test_notes_are_unique_and_each_mentions_its_own_metric(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_eval_result(store, "e1", "r1", "Q1", correct=True)
    rows = quality_comparisons(store, snapshot(store))

    notes = [row["Note"] for row in rows]
    assert len(notes) == len(set(notes)), "duplicate note text across metrics"

    for row in rows:
        metric_words = [w for w in re.split(r"[^a-zA-Z]+", row["Metric"].lower()) if len(w) > 3]
        note_lower = row["Note"].lower()
        assert any(word in note_lower for word in metric_words), (
            f"note for {row['Metric']!r} does not mention its own metric: {row['Note']!r}"
        )


# 7. When RAGAS *is* ready but no judged pass has been cached for this run, the
# cell must name the command that produces the number -- "unavailable" sent the
# reader looking for an install problem that did not exist.


def test_ready_ragas_with_no_cached_value_says_how_to_compute_it(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _insert_run(store, "r1")

    import eval.ragas_eval as ragas_eval
    monkeypatch.setattr(ragas_eval, "ragas_installed", lambda: True)
    monkeypatch.setattr(ragas_eval, "ragas_import_error", lambda: None)
    monkeypatch.setenv("CA_LLM_MODEL", "judge-model")

    cell = _rows_by_metric(quality_comparisons(store, snapshot(store)))["Faithfulness / groundedness"]["RAGAS (LLM-judged)"]
    assert "not yet computed for this run" in cell
    assert "python -m eval.run_ragas" in cell
    assert "unavailable" not in cell


# 8. An installed-but-broken ragas must report the conflict, not "not installed".


def test_import_error_cell_carries_the_real_exception(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _insert_run(store, "r1")

    import eval.ragas_eval as ragas_eval
    monkeypatch.setattr(ragas_eval, "ragas_installed", lambda: True)
    monkeypatch.setattr(
        ragas_eval, "ragas_import_error",
        lambda: "ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'",
    )

    cell = _rows_by_metric(quality_comparisons(store, snapshot(store)))["Faithfulness / groundedness"]["RAGAS (LLM-judged)"]
    assert "fails to import" in cell
    assert "langchain_community.chat_models.vertexai" in cell
    assert "not installed" not in cell


# 9. A cached judged value is read from *this run's* stored metrics, not
# averaged across every run in history.


def test_cached_ragas_value_is_shown_for_the_latest_run(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "old")
    _insert_run(store, "new")
    with store.db:
        store.db.execute("UPDATE runs SET started_at=? WHERE run_id=?", ("2026-02-02T00:00:00+00:00", "new"))
    store.save_quality_metrics("old", "ragas", {"faithfulness": 0.10})
    store.save_quality_metrics("new", "ragas", {"faithfulness": 0.90})

    cell = _rows_by_metric(quality_comparisons(store, snapshot(store)))["Faithfulness / groundedness"]["RAGAS (LLM-judged)"]
    assert cell == "90.0%"
