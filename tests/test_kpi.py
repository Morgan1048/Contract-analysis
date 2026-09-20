"""tests/test_kpi.py -- acceptance list from docs/tasks/07-ui-and-kpi-dashboard.md.

Everything here seeds a RunStore's sqlite tables directly (rather than running the
full pipeline) so each KPI computation can be tested in isolation.
"""
from __future__ import annotations

import json

import pytest

from core.kpi import KPI_SPECS, Alert, _percentile, calibration, evaluate_thresholds, history, kpi_metric_names, snapshot
from core.store import RunStore


def _store(tmp_path):
    return RunStore(tmp_path / "runs.db")


def _insert_run(
    store,
    run_id,
    started_at="2026-01-01T00:00:00+00:00",
    cost_usd=1.0,
    duration_ms=1000.0,
    unpriced_model=0,
    chunk_count=100,
    table_row_chunks=10,
):
    with store.db:
        store.db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, "contract.pdf", started_at, started_at, "stub", "local",
                10, 10, cost_usd, duration_ms, "completed", unpriced_model,
                chunk_count, table_row_chunks, 0,
            ),
        )


def _insert_finding(store, run_id, sub_requirement_id, verdict="MET", quote_verified=1, retry_count=0):
    with store.db:
        store.db.execute(
            "INSERT INTO findings "
            "(run_id,question_id,sub_requirement_id,verdict,confidence,quote,"
            "source_chunk_id,quote_verified,retry_count,rationale,demo) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, "Q1", sub_requirement_id, verdict, 0.9, "quote", "sec:1", quote_verified, retry_count, "reason", 0),
        )


def _insert_question_result(store, run_id, question_id, confidence, applied_deviations):
    with store.db:
        store.db.execute(
            "INSERT INTO question_results VALUES (?,?,?,?,?,?,?)",
            (run_id, question_id, "Fully Compliant", confidence, "reason", json.dumps(applied_deviations), 0),
        )


def _insert_eval_result(store, eval_id, run_id, question_id, correct):
    with store.db:
        store.db.execute(
            "INSERT INTO eval_results VALUES (?,?,?,?,?,?,?,?)",
            (eval_id, run_id, question_id, "Fully Compliant", "Fully Compliant", int(correct), "2026-01-01T00:00:00+00:00", 0),
        )


# 1. snapshot over an empty store returns zeros, not an exception


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([1, 2, 3, 4], 0.5, 2.5),
        ([10000.0, 100000.0], 0.5, 55000.0),
        ([4, 1, 3, 2], 0.95, 3.85),
        ([10000.0, 100000.0], 0.95, 95500.0),
        ([3, 1, 2], 0.5, 2.0),
        ([], 0.5, 0.0),
        ([7.0], 0.95, 7.0),
        ([1, 2, 3, 4], 0.0, 1.0),
        ([1, 2, 3, 4], 1.0, 4.0),
        ([1, 2, 3, 4], -0.1, 1.0),
        ([1, 2, 3, 4], 1.1, 4.0),
    ],
)
def test_percentile_interpolates_order_statistics(values, pct, expected):
    original = list(values)
    assert _percentile(values, pct) == pytest.approx(expected)
    assert values == original


def test_snapshot_uses_interpolated_latency_and_cost_percentiles(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import core.kpi as kpi

    store = _store(tmp_path)
    today = datetime.now(timezone.utc).isoformat()
    _insert_run(store, "small", started_at=today, cost_usd=2.0, duration_ms=10000.0)
    _insert_run(store, "large", started_at=today, cost_usd=4.0, duration_ms=100000.0)
    calls = []

    def percentile(values, pct):
        calls.append((values, pct))
        return _percentile(values, pct)

    monkeypatch.setattr(kpi, "_percentile", percentile)
    result = snapshot(store)
    assert ([2.0, 4.0], 0.5) in calls
    assert result["p50_latency_ms"] == 55000.0
    assert result["p95_latency_ms"] == 95500.0
    assert result["cost_median_7d"] == 3.0


def test_snapshot_over_empty_store_returns_zeros(tmp_path):
    store = _store(tmp_path)
    result = snapshot(store)
    assert result["runs"] == 0
    assert result["quote_verification_failure_rate"] is None
    assert result["not_found_rate"] == 0
    assert result["mean_retry_count"] == 0
    assert result["p95_latency_ms"] == 0
    assert result["cost_per_document"] is None
    assert result["deviation_applied_rate"] == 0
    assert result["latest_chunk_count"] == 0
    assert result["latest_table_row_chunks"] == 0


# 2. quote-verification failure rate computed correctly from seeded runs


def test_legacy_findings_do_not_fabricate_attempt_level_quote_metrics(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_finding(store, "r1", "S1", quote_verified=1)
    _insert_finding(store, "r1", "S2", quote_verified=0)
    _insert_finding(store, "r1", "S3", quote_verified=0)
    _insert_finding(store, "r1", "S4", quote_verified=1)
    result = snapshot(store)
    assert result["quote_attempts"] == 0
    assert result["instrumented_runs"] == 0


def test_daily_cost_and_latency_are_not_weighted_by_finding_count(tmp_path):
    from datetime import datetime, timezone
    store = _store(tmp_path)
    today = datetime.now(timezone.utc).isoformat()
    _insert_run(store, "small", started_at=today, cost_usd=2, duration_ms=100)
    _insert_run(store, "large", started_at=today, cost_usd=4, duration_ms=300)
    _insert_finding(store, "small", "one")
    for i in range(25):
        _insert_finding(store, "large", str(i))
    day = history(store)["days"][0]
    assert day["runs"] == 2
    assert day["cost_usd"] == 3
    assert day["duration_ms"] == 200
    assert day["quote_verification_failure_rate"] is None


# 3. NOT_FOUND rate computed per run and aggregated


def test_not_found_rate_computed_and_aggregated(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_run(store, "r2", started_at="2026-01-02T00:00:00+00:00")
    _insert_finding(store, "r1", "S1", verdict="MET")
    _insert_finding(store, "r1", "S2", verdict="NOT_FOUND")
    _insert_finding(store, "r2", "S1", verdict="NOT_FOUND")
    _insert_finding(store, "r2", "S2", verdict="NOT_FOUND")
    result = snapshot(store)
    assert result["not_found_rate"] == pytest.approx(0.75)


# 4. calibration bins map predicted confidence to observed accuracy; empty bins omitted


def test_calibration_bins_map_confidence_to_accuracy(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_question_result(store, "r1", "Q1", confidence=0.95, applied_deviations=[])
    _insert_eval_result(store, "e1", "r1", "Q1", correct=True)
    _insert_run(store, "r2", started_at="2026-01-02T00:00:00+00:00")
    _insert_question_result(store, "r2", "Q1", confidence=0.92, applied_deviations=[])
    _insert_eval_result(store, "e2", "r2", "Q1", correct=False)

    bins = calibration(store, bins=5)
    # Only the top bin (0.8-1.0) has observations; the other four are omitted.
    assert len(bins) == 1
    top = bins[0]
    assert top["bin"] == "0.8-1.0"
    assert top["count"] == 2
    assert top["predicted_confidence"] == pytest.approx((0.95 + 0.92) / 2, abs=1e-3)
    assert top["observed_accuracy"] == pytest.approx(0.5)


# 5. evaluate_thresholds fires warn and page at the documented boundaries


def test_evaluate_thresholds_fires_warn_and_page_at_boundaries():
    warn_snapshot = {
        "runs": 0,
        "quote_verification_failure_rate": 0.05,
        "not_found_rate": 0.0,
        "mean_retry_count": 0.0,
        "p95_latency_ms": 0.0,
        "cost_per_document": 0.0,
        "cost_median_7d": 0.0,
        "deviation_applied_rate": 1.0,
        "latest_chunk_count": 100,
        "latest_table_row_chunks": 10,
    }
    alerts = evaluate_thresholds(warn_snapshot)
    assert alerts == [Alert("quote_verification_failure_rate", 0.05, 0.05, "warn", alerts[0].action)]

    page_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.15)
    alerts = evaluate_thresholds(page_snapshot)
    assert alerts[0].severity == "page"
    assert alerts[0].value == 0.15

    below_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.049)
    assert evaluate_thresholds(below_snapshot) == []

    latency_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.0, p95_latency_ms=120_001)
    alerts = evaluate_thresholds(latency_snapshot)
    assert any(a.name == "p95_latency_ms" and a.severity == "warn" for a in alerts)

    cost_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.0, cost_per_document=10.0, cost_median_7d=4.0)
    alerts = evaluate_thresholds(cost_snapshot)
    assert any(a.name == "cost_per_document" for a in alerts)

    not_found_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.0, not_found_rate=0.41)
    alerts = evaluate_thresholds(not_found_snapshot)
    assert any(a.name == "not_found_rate" and a.severity == "warn" for a in alerts)

    retry_snapshot = dict(warn_snapshot, quote_verification_failure_rate=0.0, mean_retry_count=0.31)
    alerts = evaluate_thresholds(retry_snapshot)
    assert any(a.name == "mean_retry_count" and a.severity == "warn" for a in alerts)


# 6. the deviation-applied-rate alert fires when the rate is zero across seeded runs


def test_zero_modifier_rate_is_not_a_failure(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1")
    _insert_question_result(store, "r1", "Q1", confidence=0.9, applied_deviations=[])
    snap = snapshot(store)
    assert snap["deviation_applied_rate"] == 0
    alerts = evaluate_thresholds(snap)
    assert not any(a.name == "deviation_applied_rate" for a in alerts)

    # a non-zero rate must not fire the alert
    store2 = _store(tmp_path.parent / (tmp_path.name + "-2"))
    _insert_run(store2, "r1")
    _insert_question_result(store2, "r1", "Q1", confidence=0.9, applied_deviations=["RA-2026-001"])
    snap2 = snapshot(store2)
    assert snap2["deviation_applied_rate"] == 1.0
    assert not any(a.name == "deviation_applied_rate" for a in evaluate_thresholds(snap2))


# 7. history returns daily aggregates ordered oldest to newest


def test_history_returns_daily_aggregates_ordered_oldest_to_newest(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1", started_at="2026-01-03T00:00:00+00:00")
    _insert_run(store, "r2", started_at="2026-01-01T00:00:00+00:00")
    _insert_run(store, "r3", started_at="2026-01-02T00:00:00+00:00")
    result = history(store, days=3650)
    days = [row["day"] for row in result["days"]]
    assert days == sorted(days)
    assert days == ["2026-01-01", "2026-01-02", "2026-01-03"]


# 8. parse anomaly triggers below 40 chunks and on zero table_row chunks


def test_short_and_table_free_documents_do_not_trigger_parse_failure(tmp_path):
    store = _store(tmp_path)
    _insert_run(store, "r1", chunk_count=39, table_row_chunks=5)
    snap = snapshot(store)
    alerts = evaluate_thresholds(snap)
    assert not any(a.name == "parse_anomaly" for a in alerts)

    store2 = _store(tmp_path.parent / (tmp_path.name + "-zero-rows"))
    _insert_run(store2, "r1", chunk_count=200, table_row_chunks=0)
    snap2 = snapshot(store2)
    alerts2 = evaluate_thresholds(snap2)
    assert not any(a.name == "parse_anomaly" for a in alerts2)
    assert any(a.name == "parse_anomaly" for a in evaluate_thresholds(dict(snap2, latest_parse_warning_count=1)))

    store3 = _store(tmp_path.parent / (tmp_path.name + "-healthy"))
    _insert_run(store3, "r1", chunk_count=200, table_row_chunks=10)
    _insert_question_result(store3, "r1", "Q1", confidence=0.9, applied_deviations=["RA-2026-001"])
    snap3 = snapshot(store3)
    alerts3 = evaluate_thresholds(snap3)
    assert not any(a.name == "parse_anomaly" for a in alerts3)


def test_kpi_metadata_has_real_definitions_and_formulas():
    placeholder_definition = "Operational KPI for the contract analyzer."
    placeholder_formula = "Derived from the current operational snapshot and evaluated using the same threshold logic as the dashboard."

    for name in kpi_metric_names():
        spec = KPI_SPECS[name]
        assert spec.definition.strip()
        assert spec.formula.strip()
        assert spec.definition != placeholder_definition
        assert spec.formula != placeholder_formula
