"""Problem-oriented monitoring over durable run data and retained span records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from statistics import median
from typing import Any

from core.kpi import _percentile, _time, evaluate_thresholds, quality_snapshot, summarize_agent_payloads


def records_by_run(records: list[dict]) -> dict[str, list[dict]]:
    """Resolve nested spans by parent ID, never by a potentially reused trace ID."""
    owners = {
        record["span_id"]: record["run_id"] for record in records
        if record.get("span_id") and record.get("run_id")
    }
    changed = True
    while changed:
        changed = False
        for record in records:
            span_id = record.get("span_id")
            parent = record.get("parent_span_id")
            if span_id and span_id not in owners and parent in owners:
                owners[span_id] = owners[parent]
                changed = True
    result: dict[str, list[dict]] = {}
    for record in records:
        run_id = record.get("run_id") or owners.get(record.get("span_id"))
        if run_id:
            result.setdefault(run_id, []).append(record)
    return result


def _metrics(runs: list[dict], findings: dict[str, list[dict]], deviations: set[str], snapshots: dict) -> dict:
    judgments = [finding for run in runs for finding in findings.get(run["run_id"], [])]
    durations = [run["duration_ms"] for run in runs if run["duration_ms"] is not None]
    costs = [run["cost_usd"] for run in runs if not run["unpriced_model"] and run["cost_usd"] is not None]
    latest = max(runs, key=lambda run: _time(run["started_at"])) if runs else {}
    agent = summarize_agent_payloads([
        {**snapshots[r["run_id"]], "started_at": r["started_at"], "run_id": r["run_id"]}
        for r in runs if r["run_id"] in snapshots
    ])
    return {
        **agent,
        "runs": len(runs),
        "p95_latency_ms": _percentile(durations, .95) if durations else None,
        "cost_per_document": sum(costs) / len(costs) if costs else None,
        "quote_verification_failure_rate": (
            agent["quote_failures"] / agent["quote_attempts"] if agent["quote_attempts"] else None
        ),
        "not_found_rate": sum(f["verdict"] == "NOT_FOUND" for f in judgments) / len(judgments) if judgments else 0,
        "mean_retry_count": sum(f["retry_count"] for f in judgments) / len(judgments) if judgments else 0,
        "latest_chunk_count": latest.get("chunk_count", 0),
        "latest_table_row_chunks": latest.get("table_row_chunks", 0),
        "deviation_applied_rate": sum(run["run_id"] in deviations for run in runs) / len(runs) if runs else 0,
    }


def _threshold_input(metrics: dict, cost_baseline: float | None) -> dict:
    return {**{key: value for key, value in metrics.items() if value is not None},
            "cost_median_7d": cost_baseline or 0}


def backend_health(records: list[dict], configured_provider: str, configured_model: str) -> dict:
    calls = [record for record in records if record.get("event") == "backend:call"
             and record.get("message") != "start"]
    if not calls:
        return {"provider": configured_provider, "model": configured_model,
                "outcome": "Unknown - no recorded backend call", "ts": None}
    last = max(calls, key=lambda record: record.get("ts", 0))
    return {
        "provider": last.get("provider") or last.get("backend") or "Unknown",
        "model": last.get("model") or "Unknown",
        "outcome": "Succeeded" if last.get("message") == "end" else f"Failed: {last.get('message', 'unknown error')}",
        "ts": last.get("ts"),
    }


def usage_snapshot(store: Any, *, now: datetime | None = None) -> dict:
    """Return compact, problem-oriented token and cost signals for Monitoring."""
    now = now or datetime.now(timezone.utc)
    trend_start = now - timedelta(days=30)
    runs = [dict(row) for row in store.db.execute(
        "SELECT run_id, started_at, total_prompt_tokens, total_completion_tokens, "
        "unpriced_model FROM runs WHERE julianday(started_at) >= julianday(?) "
        "AND julianday(started_at) <= julianday(?) ORDER BY julianday(started_at)",
        (trend_start.isoformat(), now.isoformat()),
    )]
    latest = runs[-1] if runs else None
    recent_start = now - timedelta(days=7)
    baseline_totals = [
        run["total_prompt_tokens"] + run["total_completion_tokens"]
        for run in runs
        if latest is not None
        and recent_start <= _time(run["started_at"]) < _time(latest["started_at"])
    ]
    recent_median = median(baseline_totals) if baseline_totals else None
    latest_total = (
        latest["total_prompt_tokens"] + latest["total_completion_tokens"]
        if latest else None
    )
    departure = (
        (latest_total - recent_median) / recent_median
        if latest_total is not None and recent_median else None
    )

    stage_share = []
    question_cost = []
    if latest:
        stage_rows = store.db.execute(
            "SELECT stage, SUM(tokens) AS tokens, "
            "CASE WHEN COUNT(cost_usd)=COUNT(*) THEN SUM(cost_usd) ELSE NULL END AS cost_usd "
            "FROM stage_timings WHERE run_id=? GROUP BY stage HAVING SUM(tokens)>0 "
            "ORDER BY tokens DESC",
            (latest["run_id"],),
        ).fetchall()
        stage_total = sum(row["tokens"] for row in stage_rows)
        stage_share = [
            {
                "stage": row["stage"],
                "tokens": row["tokens"],
                "share": row["tokens"] / stage_total if stage_total else 0,
                "cost_usd": row["cost_usd"],
            }
            for row in stage_rows
        ]
        question_cost = [dict(row) for row in store.db.execute(
            "SELECT question_id, SUM(prompt_tokens) AS prompt_tokens, "
            "SUM(completion_tokens) AS completion_tokens, "
            "CASE WHEN COUNT(cost_usd)=COUNT(*) THEN SUM(cost_usd) ELSE NULL END AS cost_usd "
            "FROM findings WHERE run_id=? GROUP BY question_id ORDER BY cost_usd DESC",
            (latest["run_id"],),
        )]
        audit = store.audit_trail(latest["run_id"])
        if audit:
            grouped = {}
            for event in audit:
                row = grouped.setdefault(event["question_id"], {
                    "question_id": event["question_id"], "prompt_tokens": 0,
                    "completion_tokens": 0, "cost_usd": 0.0,
                })
                row["prompt_tokens"] += event["prompt_tokens"]
                row["completion_tokens"] += event["completion_tokens"]
                row["cost_usd"] = (row["cost_usd"] + event["cost_usd"]
                                   if row["cost_usd"] is not None and event["cost_usd"] is not None else None)
            question_cost = list(grouped.values())

    trend_by_day: dict[str, dict[str, float | int | str]] = {}
    for run in runs:
        day = _time(run["started_at"]).date().isoformat()
        bucket = trend_by_day.setdefault(
            day, {"day": day, "input_tokens": 0, "output_tokens": 0, "runs": 0}
        )
        bucket["input_tokens"] += run["total_prompt_tokens"]
        bucket["output_tokens"] += run["total_completion_tokens"]
        bucket["runs"] += 1
    trend = [
        {
            "day": bucket["day"],
            "input_tokens": round(bucket["input_tokens"] / bucket["runs"]),
            "output_tokens": round(bucket["output_tokens"] / bucket["runs"]),
        }
        for bucket in trend_by_day.values()
    ]
    run_tokens = [
        {
            "run": run["run_id"][:8],
            "input_tokens": run["total_prompt_tokens"],
            "output_tokens": run["total_completion_tokens"],
        }
        for run in runs[-12:]
    ]
    return {
        "latest_run": latest,
        "latest_total_tokens": latest_total,
        "recent_median_tokens": recent_median,
        "departure_from_median": departure,
        "run_tokens": run_tokens,
        "stage_share": stage_share,
        "question_cost": question_cost,
        "trend": trend,
    }


def monitoring_snapshot(store: Any, records: list[dict], *, now: datetime | None = None) -> dict:
    """Last 24h versus the median of seven preceding, non-overlapping 24h buckets.

    Missing observations stay missing. Cost is averaged per document, not per
    finding; quotation failures are weighted by evidence-bearing verification attempts.
    """
    now = now or datetime.now(timezone.utc)
    current_start = now - timedelta(days=1)
    baseline_start = current_start - timedelta(days=7)
    rows = store.db.execute(
        "SELECT * FROM runs WHERE julianday(started_at) >= julianday(?) "
        "AND julianday(started_at) <= julianday(?) ORDER BY julianday(started_at) DESC",
        (baseline_start.isoformat(), now.isoformat()),
    ).fetchall()
    runs = [dict(row) for row in rows]
    snapshots = {row["run_id"]: json.loads(row["payload"]) for row in store.db.execute(
        "SELECT s.run_id, s.payload FROM run_snapshots s JOIN runs r ON r.run_id=s.run_id "
        "WHERE r.demo=0 AND julianday(r.started_at)>=julianday(?) "
        "AND julianday(r.started_at)<=julianday(?)",
        (baseline_start.isoformat(), now.isoformat()),
    )}
    findings: dict[str, list[dict]] = {}
    for row in store.db.execute(
        "SELECT f.* FROM findings f JOIN runs r ON r.run_id=f.run_id "
        "WHERE julianday(r.started_at) >= julianday(?) AND julianday(r.started_at) <= julianday(?)",
        (baseline_start.isoformat(), now.isoformat()),
    ):
        findings.setdefault(row["run_id"], []).append(dict(row))
    deviations = {row["run_id"] for row in store.db.execute(
        "SELECT DISTINCT q.run_id FROM question_results q JOIN runs r ON r.run_id=q.run_id "
        "WHERE q.applied_deviations IS NOT NULL AND q.applied_deviations != '[]' "
        "AND julianday(r.started_at) >= julianday(?) AND julianday(r.started_at) <= julianday(?)",
        (baseline_start.isoformat(), now.isoformat()),
    )}
    backend_failures = {row["run_id"] for row in store.db.execute(
        "SELECT DISTINCT c.run_id FROM checks c JOIN runs r ON r.run_id=c.run_id "
        "WHERE c.name='backend' AND c.passed=0 "
        "AND julianday(r.started_at) >= julianday(?) AND julianday(r.started_at) <= julianday(?)",
        (current_start.isoformat(), now.isoformat()),
    )}
    current = [run for run in runs if _time(run["started_at"]) >= current_start]
    metrics = _metrics(current, findings, deviations, snapshots)
    metrics.update(quality_snapshot(store, now=now))
    daily = [
        _metrics([run for run in runs if
                  current_start - timedelta(days=day + 1) <= _time(run["started_at"])
                  < current_start - timedelta(days=day)], findings, deviations, snapshots)
        for day in range(7)
    ]
    movements = {}
    for name in ("p95_latency_ms", "cost_per_document", "quote_verification_failure_rate"):
        observed = [bucket[name] for bucket in daily if bucket[name] is not None]
        baseline = median(observed) if observed else None
        value = metrics[name]
        movements[name] = {
            "value": value, "baseline": baseline, "baseline_days": len(observed),
            "change": value - baseline if value is not None and baseline is not None else None,
            "relative_change": (value - baseline) / baseline if value is not None and baseline else None,
        }
    cost_baseline = movements["cost_per_document"]["baseline"]
    weekly = [run for run in runs if _time(run["started_at"]) >= now - timedelta(days=7)]
    alert_input = _threshold_input(metrics, cost_baseline)
    alert_input["deviation_applied_rate"] = _metrics(weekly, findings, deviations, snapshots)["deviation_applied_rate"]
    alerts = evaluate_thresholds(alert_input)
    health = "failing" if any(alert.severity == "page" for alert in alerts) else "degraded" if alerts else "healthy"

    by_run = records_by_run(records)
    slowest = {run["run_id"] for run in sorted(
        (run for run in current if run["duration_ms"] is not None),
        key=lambda run: run["duration_ms"], reverse=True,
    )[:3]}
    anomalies = []
    for run in current:
        run_id = run["run_id"]
        events = by_run.get(run_id, [])
        reasons = []
        if run["status"] == "incomplete":
            reasons.append("Incomplete")
        if run_id in backend_failures or any(
            f["verdict"] == "ERROR" for f in findings.get(run_id, [])
        ) or any(record.get("message") == "BackendError" for record in events):
            reasons.append("BackendError")
        if any(record.get("retry_exhausted") for record in events) or snapshots.get(
            run_id, {}
        ).get("usage", {}).get("budget_exhausted"):
            reasons.append("Retries exhausted")
        per_run = evaluate_thresholds(_threshold_input(_metrics([run], findings, deviations, snapshots), cost_baseline))
        # Deviation absence is a weekly signal, not a per-document requirement.
        breaches = [alert.name for alert in per_run if alert.name != "deviation_applied_rate"]
        reasons.extend(breaches)
        if run_id in slowest:
            reasons.append("Slowest run")
        prior_totals = [
            other["total_prompt_tokens"] + other["total_completion_tokens"]
            for other in runs
            if baseline_start <= _time(other["started_at"]) < _time(run["started_at"])
        ]
        token_median = median(prior_totals) if prior_totals else None
        run_tokens = run["total_prompt_tokens"] + run["total_completion_tokens"]
        if token_median and abs(run_tokens - token_median) / token_median >= .5:
            reasons.append("Token usage >=50% from recent median")
        if reasons:
            anomalies.append({**run, "reasons": reasons})
    # Failed analyses may never reach SQLite. Their spans still identify a run.
    known = {run["run_id"] for run in runs}
    for run_id, events in by_run.items():
        recent = [event for event in events if current_start.timestamp() <= event.get("ts", 0) <= now.timestamp()]
        if run_id in known or not recent:
            continue
        reasons = []
        if any(event.get("message") == "BackendError" for event in recent):
            reasons.append("BackendError")
        if any(event.get("retry_exhausted") for event in recent):
            reasons.append("Retries exhausted")
        if reasons:
            anomalies.append({"run_id": run_id, "started_at": datetime.fromtimestamp(
                min(event["ts"] for event in recent), timezone.utc).isoformat(),
                "duration_ms": None, "reasons": reasons})
    anomalies.sort(key=lambda run: (
        not any(reason in run["reasons"] for reason in ("Incomplete", "BackendError", "Retries exhausted")),
        -_time(run["started_at"]).timestamp(),
    ))
    return {"health": health, "alerts": alerts, "metrics": metrics, "movements": movements,
            "usage": usage_snapshot(store, now=now),
            "anomalies": anomalies[:10], "anomaly_count": len(anomalies)}
