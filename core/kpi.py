"""KPI computation for the Streamlit dashboard (task 07).

Every function here is a pure computation over ``RunStore`` data; nothing renders.
That is deliberate: the UI layer (``ui/kpi_view.py``) only calls these functions and
draws widgets, so every number on screen traces back to something testable without
a browser.

Thresholds are provisional operational guardrails, not measured quality guarantees.
``evaluate_thresholds`` applies the definitions below consistently across UI and MCP.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json


@dataclass(frozen=True)
class Alert:
    name: str
    value: float
    threshold: float
    severity: str   # "warn" | "page"
    action: str


@dataclass(frozen=True)
class KpiTrigger:
    threshold: float
    severity: str
    action: str


@dataclass(frozen=True)
class KpiSpec:
    name: str
    definition: str
    formula: str
    triggers: tuple[KpiTrigger, ...]

    @property
    def threshold(self) -> float:
        return self.triggers[0].threshold

    @property
    def severity(self) -> str:
        return self.triggers[0].severity

    @property
    def action(self) -> str:
        return self.triggers[0].action


def _percentile(values: list[float], pct: float) -> float:
    """Linearly interpolate at pct * (n - 1), clamped to the observed range."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, pct * (len(ordered) - 1)))
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def snapshot(store, *, now: datetime | None = None) -> dict:
    """Current operational KPI state, aggregated across every stored run.

    Missing quotation attempts and price observations remain unavailable.
    """
    db = store.db

    runs = db.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]

    findings_row = db.execute(
        "SELECT "
        "COALESCE(AVG(CASE WHEN quote_verified = 0 THEN 1.0 ELSE 0 END), 0) AS failure_rate, "
        "COALESCE(AVG(CASE WHEN verdict = 'NOT_FOUND' THEN 1.0 ELSE 0 END), 0) AS not_found_rate, "
        "COALESCE(AVG(retry_count), 0) AS mean_retry_count "
        "FROM findings"
    ).fetchone()

    durations = [
        row["duration_ms"] for row in db.execute("SELECT duration_ms FROM runs").fetchall()
        if row["duration_ms"] is not None
    ]

    priced_costs = [
        row["cost_usd"] for row in db.execute(
            "SELECT cost_usd FROM runs WHERE unpriced_model = 0"
        ).fetchall()
        if row["cost_usd"] is not None
    ]
    cost_7d = [
        row["cost_usd"] for row in db.execute(
            "SELECT cost_usd FROM runs WHERE unpriced_model = 0 AND started_at >= datetime('now', '-7 days')"
        ).fetchall()
        if row["cost_usd"] is not None
    ]

    deviation_row = db.execute(
        "SELECT COALESCE(AVG(CASE WHEN applied_deviations IS NOT NULL AND applied_deviations != '[]' "
        "THEN 1.0 ELSE 0 END), 0) AS rate FROM question_results"
    ).fetchone()

    latest = db.execute(
        "SELECT chunk_count, table_row_chunks FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()

    agent = agent_metrics(store)
    return {
        "runs": runs,
        "quote_verification_failure_rate": agent["quote_failures"] / agent["quote_attempts"] if agent["quote_attempts"] else None,
        "not_found_rate": findings_row["not_found_rate"],
        "mean_retry_count": findings_row["mean_retry_count"],
        "p95_latency_ms": _percentile(durations, 0.95),
        "p50_latency_ms": _percentile(durations, 0.50),
        "cost_per_document": (sum(priced_costs) / len(priced_costs)) if priced_costs else None,
        "priced_runs": len(priced_costs),
        "cost_median_7d": _percentile(cost_7d, 0.5),
        "deviation_applied_rate": deviation_row["rate"],
        "latest_chunk_count": latest["chunk_count"] if latest else 0,
        "latest_table_row_chunks": latest["table_row_chunks"] if latest else 0,
        **agent,
        **quality_snapshot(store, now=now),
    }


def agent_metrics(store) -> dict:
    payloads = [{**json.loads(row["payload"]), "started_at": row["started_at"], "run_id": row["run_id"]}
                for row in store.db.execute(
        "SELECT s.payload, r.started_at, r.run_id FROM run_snapshots s JOIN runs r ON r.run_id=s.run_id "
        "WHERE r.demo=0 ORDER BY r.started_at"
    )]
    return summarize_agent_payloads(payloads)


def summarize_agent_payloads(payloads: list[dict]) -> dict:
    """Aggregate only versioned, instrumented results; legacy attempts are unknown."""
    payloads = [p for p in payloads if p["usage"].get("pipeline_version") == "langgraph-evidence-v2"]
    usage = [p["usage"] for p in payloads]
    total = lambda key: sum(u.get(key, 0) for u in usage)
    n = len(usage)
    checks = [f for p in payloads for q in p["questions"] for f in q["findings"]]
    followups = total("followup_checks")
    latest = max(payloads, key=lambda p: (_time(p["started_at"]), p["run_id"])) if payloads else None
    return {
        "instrumented_runs": n,
        "completion_rate": sum(not p["incomplete"] for p in payloads) / n if n else None,
        "review_rate": sum(f["verdict"] in {"UNRESOLVED", "ERROR"} for f in checks) / len(checks) if checks else None,
        "budget_exhaustion_rate": sum(bool(u.get("budget_exhausted")) for u in usage) / n if n else None,
        "followup_resolution_rate": total("resolved_followup_checks") / followups if followups else None,
        "followup_checks": followups,
        "quote_attempts": total("quote_attempts"), "quote_failures": total("quote_failures"),
        "mean_llm_calls": total("llm_calls") / n if n else None,
        "mean_tool_calls": total("tool_calls") / n if n else None,
        "mean_tokens": total("total_tokens") / n if n else None,
        "latest_parse_warning_count": len(latest["usage"].get("parse_warnings", [])) if latest else 0,
    }


def history(store, days: int = 30) -> dict:
    """Daily aggregates over the last ``days`` days, oldest first."""
    db = store.db
    rows = db.execute(
        "SELECT substr(r.started_at, 1, 10) AS day, "
        "COUNT(r.run_id) AS runs, "
        "AVG(CASE WHEN r.unpriced_model=0 THEN r.cost_usd END) AS cost_usd, "
        "AVG(r.duration_ms) AS duration_ms, "
        "NULL AS quote_verification_failure_rate, "
        "COALESCE(1.0 * SUM(f.missing) / NULLIF(SUM(f.total), 0), 0) AS not_found_rate "
        "FROM runs r LEFT JOIN (SELECT run_id, COUNT(*) total, "
        "SUM(CASE WHEN verdict='NOT_FOUND' THEN 1 ELSE 0 END) missing "
        "FROM findings GROUP BY run_id) f ON f.run_id = r.run_id "
        "WHERE r.started_at >= datetime('now', ?) "
        "GROUP BY day ORDER BY day ASC",
        (f"-{days} days",),
    ).fetchall()
    days = [dict(row) for row in rows]
    for day in days:
        audit = [json.loads(r["payload"]) for r in db.execute(
            "SELECT s.payload FROM run_snapshots s JOIN runs r ON r.run_id=s.run_id WHERE substr(r.started_at,1,10)=? AND r.demo=0",
            (day["day"],),
        )]
        usage = [p["usage"] for p in audit if p["usage"].get("pipeline_version") == "langgraph-evidence-v2"]
        attempts = sum(u.get("quote_attempts", 0) for u in usage)
        day["quote_verification_failure_rate"] = sum(u.get("quote_failures", 0) for u in usage) / attempts if attempts else None
        day["llm_calls"] = sum(u.get("llm_calls", 0) for u in usage) if usage else None
        day["tokens"] = sum(u.get("total_tokens", 0) for u in usage) if usage else None
        day["completed_runs"] = sum(not p["incomplete"] for p in audit) if audit else None
    return {"days": days}


def calibration(store, bins: int = 5) -> list[dict]:
    """Bin predicted confidence (from ``question_results``) against observed
    accuracy (from ``eval_results``, i.e. a hand-built gold set). Empty bins are
    omitted rather than reported as zero, since a bin with no observations says
    nothing about calibration.
    """
    db = store.db
    rows = db.execute(
        "SELECT q.confidence AS confidence, e.correct AS correct "
        "FROM eval_results e JOIN question_results q "
        "ON q.run_id = e.run_id AND q.question_id = e.question_id "
        "JOIN runs r ON r.run_id=e.run_id WHERE e.demo=0 AND q.demo=0 AND r.demo=0"
    ).fetchall()

    width = 1.0 / bins
    buckets: dict[int, list[tuple[float, int]]] = {}
    for row in rows:
        confidence = row["confidence"] if row["confidence"] is not None else 0.0
        idx = min(bins - 1, max(0, int(confidence / width)))
        buckets.setdefault(idx, []).append((confidence, row["correct"]))

    result: list[dict] = []
    for idx in sorted(buckets):
        items = buckets[idx]
        low, high = idx * width, (idx + 1) * width
        predicted = sum(c for c, _ in items) / len(items)
        observed = sum(correct for _, correct in items) / len(items)
        result.append({
            "bin": f"{low:.1f}-{high:.1f}",
            "predicted_confidence": round(predicted, 3),
            "observed_accuracy": round(observed, 3),
            "count": len(items),
        })
    return result


def gold_set_accuracy(store, run_id: str | None = None) -> float | None:
    """Overall exact compliance-state match rate against the hand-built gold
    set (``eval_results``), i.e. the "Answer correctness" exact value.

    Returns ``None`` -- not 0.0 -- when no run has ever been scored against
    the gold set, so callers can tell "no data" from "0% accurate".
    """
    row = store.db.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(e.correct), 0) AS correct "
        "FROM eval_results e JOIN runs r ON r.run_id=e.run_id "
        "WHERE e.demo=0 AND r.demo=0 AND (? IS NULL OR e.run_id=?)", (run_id, run_id),
    ).fetchone()
    if not row["n"]:
        return None
    return row["correct"] / row["n"]


def quality_snapshot(store, *, now: datetime | None = None) -> dict:
    """Shared quality-alert inputs; no inference or fabricated missing scores."""
    now = _time((now or datetime.now(timezone.utc)).isoformat())
    latest = store.db.execute(
        "SELECT r.run_id, json_extract(s.payload, '$.rubric_hash') AS rubric_hash, "
        "json_extract(s.payload, '$.usage.pipeline_version') AS pipeline_version "
        "FROM runs r LEFT JOIN run_snapshots s ON s.run_id=r.run_id "
        "WHERE r.demo=0 AND julianday(r.started_at)<=julianday(?) "
        "ORDER BY julianday(r.started_at) DESC, r.run_id DESC LIMIT 1",
        (now.isoformat(),),
    ).fetchone()
    current_start, baseline_start = now - timedelta(days=7), now - timedelta(days=14)
    baseline = {
        "current_start": current_start.isoformat(), "baseline_start": baseline_start.isoformat(),
        "end": now.isoformat(), "current_runs": 0, "baseline_runs": 0,
        "minimum_runs_per_question": 5,
        "reason": "The latest non-demo run needs recorded rubric and pipeline versions.",
    }
    result = {
        "quality_run_id": latest["run_id"] if latest else None,
        "accuracy_vs_gold_set": gold_set_accuracy(store, latest["run_id"]) if latest else None,
        "calibration_bins": calibration(store),
        "state_distribution_drift": None,
        "state_distribution_baseline": baseline,
    }
    if not latest or not latest["rubric_hash"] or not latest["pipeline_version"]:
        return result
    rows = store.db.execute(
        "SELECT r.run_id, r.started_at, q.question_id, q.state "
        "FROM runs r JOIN question_results q ON q.run_id=r.run_id "
        "JOIN run_snapshots s ON s.run_id=r.run_id "
        "WHERE r.demo=0 AND q.demo=0 AND r.status='completed' "
        "AND q.state IN ('Fully Compliant','Partially Compliant','Non-Compliant') "
        "AND json_extract(s.payload, '$.rubric_hash')=? "
        "AND json_extract(s.payload, '$.usage.pipeline_version')=? "
        "AND json_extract(s.payload, '$.incomplete')=0 "
        "AND julianday(r.started_at)>=julianday(?) AND julianday(r.started_at)<=julianday(?)",
        (latest["rubric_hash"], latest["pipeline_version"], baseline_start.isoformat(), now.isoformat()),
    ).fetchall()
    counts: dict[str, dict[str, dict[str, int]]] = {"current": {}, "baseline": {}}
    runs: dict[str, set[str]] = {"current": set(), "baseline": set()}
    for row in rows:
        window = "current" if _time(row["started_at"]) >= current_start else "baseline"
        runs[window].add(row["run_id"])
        states = counts[window].setdefault(row["question_id"], {})
        states[row["state"]] = states.get(row["state"], 0) + 1
    baseline.update(current_runs=len(runs["current"]), baseline_runs=len(runs["baseline"]),
                    rubric_hash=latest["rubric_hash"], pipeline_version=latest["pipeline_version"])
    if (not counts["current"] or counts["current"].keys() != counts["baseline"].keys()
        or any(sum(states.values()) < baseline["minimum_runs_per_question"]
               for window in counts.values() for states in window.values())):
        baseline["reason"] = (
            "Need the same questions and at least 5 completed, non-demo runs per question in each "
            "7-day window, using the latest run's rubric and pipeline versions."
        )
        return result
    shifts = []
    for question, current in counts["current"].items():
        previous = counts["baseline"][question]
        shifts.extend(abs(current.get(state, 0) / sum(current.values())
                          - previous.get(state, 0) / sum(previous.values()))
                      for state in current.keys() | previous.keys())
    result["state_distribution_drift"] = max(shifts)
    baseline["reason"] = "Comparable observations available; shifts may reflect contract mix, not model regression."
    return result


def latest_analysis_diagnostics(store, run_id: str | None = None) -> dict | None:
    from core.obs.otel import _redact
    run_id = run_id or store.latest_run_id()
    result = store.get_run(run_id) if run_id else None
    if result is None:
        return None
    findings = [f for q in result.questions for f in q.findings]
    completed = sum(f.verdict not in {"ERROR", "UNRESOLVED"} and f.quote_verified for f in findings)
    return {
        "run_id": run_id, "incomplete": result.incomplete,
        "instrumented": result.usage.get("pipeline_version") == "langgraph-evidence-v2",
        "completed_checks": completed, "total_checks": len(findings),
        "stops": [
            {"Question": e["question_id"], "Stage": e["node"], "Stop": e["stop_reason"],
             "Detail": _redact(e.get("error") or e["stop_reason"])}
            for e in result.audit_events if e.get("stop_reason") not in {"", "complete", None}
        ],
    }


# KPI definitions, formulas, thresholds, severities, and actions are kept
# together so dashboard alerting and explain_kpi cannot drift.
KPI_SPECS: dict[str, KpiSpec] = {
    "quote_verification_failure_rate": KpiSpec(
        "quote_verification_failure_rate",
        "The share of evidence-bearing verification attempts with an invalid source quotation. This detects fabricated quotations, not semantic entailment. Legacy runs without attempt telemetry are excluded.",
        "failed evidence-bearing verification attempts / all evidence-bearing verification attempts",
        (
            KpiTrigger(
                0.05,
                "warn",
                "Inspect recent prompts; a spike usually means a model or prompt change, not a document change.",
            ),
            KpiTrigger(
                0.15,
                "page",
                "Inspect recent prompts; a spike usually means a model or prompt change, not a document change.",
            ),
        ),
    ),
    "not_found_rate": KpiSpec(
        "not_found_rate",
        "The share of checks with no obligation found after retrieval. This is a diagnostic signal, not measured retrieval failure or model accuracy.",
        "NOT_FOUND sub-requirement judgments / total sub-requirement judgments",
        (KpiTrigger(
            0.40,
            "warn",
            "Inspect document mix, parsing quality and search traces before attributing missing obligations to retrieval.",
        ),),
    ),
    "mean_retry_count": KpiSpec(
        "mean_retry_count",
        "The average evaluator retries per judgment; it surfaces prompt or model degradation before accuracy moves.",
        "sum of evaluator retry counts / total sub-requirement judgments",
        (KpiTrigger(0.3, "warn", "Check for malformed output; verify constrained decoding is active."),),
    ),
    "p95_latency_ms": KpiSpec(
        "p95_latency_ms",
        "The 95th percentile end-to-end analysis duration; it measures user-facing responsiveness and the local-inference cost signal.",
        "95th percentile of run duration_ms values",
        (KpiTrigger(120_000, "warn", "Break down by stage span; judge calls dominate and are parallelisable."),),
    ),
    "cost_per_document": KpiSpec(
        "cost_per_document",
        "The average priced model cost per analyzed document; it is required by the assignment and is the metric finance asks about.",
        "sum priced run cost_usd / count priced runs; alert threshold is 2 x the 7-day median cost",
        (KpiTrigger(2.0, "warn", "Check token growth per judgment -- usually pack size creeping up."),),
    ),
    "parse_anomaly": KpiSpec(
        "parse_anomaly",
        "A latest-run signal for malformed or scanned PDFs; a malformed PDF silently yielding few chunks is the worst failure because downstream results look merely non-compliant.",
        "latest recorded parse warning count",
        (KpiTrigger(1, "warn", "Review flagged pages and extraction warnings; a short or table-free contract is valid."),),
    ),
    "deviation_applied_rate": KpiSpec(
        "deviation_applied_rate",
        "The share of question results with verified applicable modifiers. Zero is legitimate; document mix determines the rate.",
        "question results with non-empty applied modifiers / total question results; descriptive only",
        (KpiTrigger(
            0.0,
            "page",
            "Use labeled modifier fixtures to test missed exceptions; do not infer failure from zero observed modifiers.",
        ),),
    ),
    "accuracy_vs_gold_set": KpiSpec(
        "accuracy_vs_gold_set",
        "The latest non-demo run's exact compliance-state match rate against recorded hand-built gold labels; unavailable when that run has no labels.",
        "correct exact-state eval results / total eval results for the latest non-demo run",
        (KpiTrigger(
            0.80,
            "warn",
            "Inspect per-sub-requirement verdicts; state errors usually trace to one sub-requirement.",
        ),),
    ),
    "confidence_calibration": KpiSpec(
        "confidence_calibration",
        "The largest gap between predicted confidence and observed accuracy in any calibration bin; confidence that does not track correctness is worse than no confidence because it is trusted.",
        "max(abs(predicted_confidence - observed_accuracy)) across non-empty calibration bins",
        (KpiTrigger(
            0.20,
            "warn",
            "Recalibrate the aggregation rule, not the model's self-reported number.",
        ),),
    ),
    "state_distribution_drift": KpiSpec(
        "state_distribution_drift",
        "A diagnostic week-over-week shift in compliance states, not measured accuracy or proof of model regression. Compare completed non-demo runs with matching rubric and pipeline versions; at least five observations per question per window.",
        "max over questions and states of abs(current 7-day state share - preceding 7-day state share)",
        (KpiTrigger(0.25, "warn", "Compare against corpus mix before assuming a model problem."),),
    ),
}


KPI_SPECS.update({
    "review_rate": KpiSpec("review_rate", "Checks requiring review in instrumented real runs.",
                         "unresolved or errored checks / assessed checks",
                         (KpiTrigger(0.20, "warn", "Inspect parse warnings, unmet evidence needs and backend errors."),)),
    "budget_exhaustion_rate": KpiSpec("budget_exhaustion_rate", "Instrumented runs that exhausted an agent budget.",
                                    "budget-exhausted runs / instrumented runs",
                                    (KpiTrigger(0.10, "warn", "Inspect repeat searches and evidence size before raising budgets."),)),
    "followup_resolution_rate": KpiSpec("followup_resolution_rate", "Previously pending checks accepted after targeted follow-up; not independently measured accuracy.",
                                      "resolved follow-up checks / checks sent to follow-up",
                                      (KpiTrigger(0.25, "warn", "Inspect critic objectives and novel evidence retrieval on low-yield loops."),)),
})


def kpi_metric_names() -> list[str]:
    return list(KPI_SPECS)


def kpi_spec(name: str) -> KpiSpec:
    return KPI_SPECS[name]


def _alert(spec: KpiSpec, value: float, threshold: float | None = None, trigger: KpiTrigger | None = None) -> Alert:
    active = trigger or spec.triggers[0]
    return Alert(spec.name, value, active.threshold if threshold is None else threshold, active.severity, active.action)


def kpi_current_value(name: str, snapshot: dict, *, calibration_bins: list[dict] | None = None, accuracy: float | None = None, state_distribution_shift: float | None = None) -> float | None:
    if name == "parse_anomaly":
        return snapshot.get("latest_parse_warning_count", 0)
    if name == "accuracy_vs_gold_set":
        return snapshot.get(name) if accuracy is None else accuracy
    if name == "confidence_calibration":
        if calibration_bins is None:
            calibration_bins = snapshot.get("calibration_bins")
        if not calibration_bins:
            return None
        return max(abs(b["predicted_confidence"] - b["observed_accuracy"]) for b in calibration_bins)
    if name == "state_distribution_drift":
        return snapshot.get(name) if state_distribution_shift is None else state_distribution_shift
    return snapshot.get(name, 0.0)


def kpi_threshold(name: str, snapshot: dict | None = None) -> float:
    spec = kpi_spec(name)
    if name == "cost_per_document" and snapshot is not None:
        return spec.threshold * snapshot.get("cost_median_7d", 0.0)
    return spec.threshold


def kpi_explanation(name: str, snapshot: dict, *, calibration_bins: list[dict] | None = None, accuracy: float | None = None, state_distribution_shift: float | None = None) -> dict:
    spec = kpi_spec(name)
    value = kpi_current_value(name, snapshot, calibration_bins=calibration_bins, accuracy=accuracy,
                              state_distribution_shift=state_distribution_shift)
    result = {
        "found": True,
        "name": spec.name,
        "definition": spec.definition,
        "formula": spec.formula,
        "threshold": kpi_threshold(name, snapshot),
        "severity": spec.severity,
        "current_value": value,
        "available": value is not None,
        "action": spec.action,
    }
    if name == "accuracy_vs_gold_set":
        result["run_id"] = snapshot.get("quality_run_id")
    if name == "state_distribution_drift":
        result["baseline"] = snapshot.get("state_distribution_baseline")
    return result


def evaluate_thresholds(
    snapshot: dict,
    calibration_bins: list[dict] | None = None,
    accuracy: float | None = None,
    state_distribution_shift: float | None = None,
) -> list[Alert]:
    """Turn a KPI snapshot (plus optional historical/quality inputs, when a gold
    set or prior-week comparison is available) into structured alerts: name,
    value, threshold, severity, and the action to take, per the documented table.
    """
    alerts: list[Alert] = []
    runs = snapshot.get("runs", 0)

    failure_rate = snapshot.get("quote_verification_failure_rate") or 0.0
    quote_spec = kpi_spec("quote_verification_failure_rate")
    quote_warn, quote_page = quote_spec.triggers
    if failure_rate >= quote_page.threshold:
        alerts.append(_alert(quote_spec, failure_rate, trigger=quote_page))
    elif failure_rate >= quote_warn.threshold:
        alerts.append(_alert(quote_spec, failure_rate, trigger=quote_warn))

    not_found_rate = snapshot.get("not_found_rate", 0.0)
    if not_found_rate > kpi_spec("not_found_rate").threshold:
        alerts.append(_alert(kpi_spec("not_found_rate"), not_found_rate))

    retry_rate = snapshot.get("mean_retry_count", 0.0)
    if retry_rate > kpi_spec("mean_retry_count").threshold:
        alerts.append(_alert(kpi_spec("mean_retry_count"), retry_rate))

    p95_latency = snapshot.get("p95_latency_ms", 0.0)
    if p95_latency > kpi_spec("p95_latency_ms").threshold:
        alerts.append(_alert(kpi_spec("p95_latency_ms"), p95_latency))

    cost_per_document = snapshot.get("cost_per_document") or 0.0
    cost_median_7d = snapshot.get("cost_median_7d", 0.0)
    cost_threshold = kpi_spec("cost_per_document").threshold * cost_median_7d
    if cost_median_7d > 0 and cost_per_document > cost_threshold:
        alerts.append(_alert(kpi_spec("cost_per_document"), cost_per_document, cost_threshold))

    if runs > 0:
        warnings = snapshot.get("latest_parse_warning_count", 0)
        if warnings >= kpi_spec("parse_anomaly").threshold:
            alerts.append(_alert(kpi_spec("parse_anomaly"), warnings))

    accuracy = kpi_current_value("accuracy_vs_gold_set", snapshot, accuracy=accuracy)
    if accuracy is not None and accuracy < kpi_spec("accuracy_vs_gold_set").threshold:
        alerts.append(_alert(kpi_spec("accuracy_vs_gold_set"), accuracy))

    max_deviation = kpi_current_value("confidence_calibration", snapshot, calibration_bins=calibration_bins)
    if max_deviation is not None and max_deviation > kpi_spec("confidence_calibration").threshold:
        alerts.append(_alert(kpi_spec("confidence_calibration"), max_deviation))

    state_distribution_shift = kpi_current_value("state_distribution_drift", snapshot,
                                                  state_distribution_shift=state_distribution_shift)
    if state_distribution_shift is not None and state_distribution_shift > kpi_spec("state_distribution_drift").threshold:
        alerts.append(_alert(kpi_spec("state_distribution_drift"), state_distribution_shift))

    if snapshot.get("instrumented_runs", 0) >= 5:
        for name in ("review_rate", "budget_exhaustion_rate", "followup_resolution_rate"):
            value = snapshot.get(name)
            if value is None:
                continue
            spec = kpi_spec(name)
            breached = value < spec.threshold if name == "followup_resolution_rate" else value > spec.threshold
            if breached:
                alerts.append(_alert(spec, value))

    return alerts
