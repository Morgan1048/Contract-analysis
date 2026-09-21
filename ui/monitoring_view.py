"""An exception-focused health view, with one selected run/span drill-down."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.monitoring import backend_health, monitoring_snapshot
from core.providers.config import ProviderConfig
from core.obs.langfuse import config as langfuse_config


def read_records(path: str | Path = "logs/contract-analyzer.jsonl") -> list[dict[str, Any]]:
    """Read current and retained JSONL files, ignoring an interrupted final line."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    for candidate in sorted(path.parent.glob(f"{path.name}*")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return sorted(records, key=lambda record: record.get("ts", 0))


def filter_records(
    records: Iterable[dict[str, Any]], *, level: str | None = None,
    event: str | None = None, trace_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        record for record in records
        if (not level or record.get("level") == level)
        and (not event or record.get("event") == event)
        and (not trace_id or record.get("trace_id") == trace_id)
    ]


def span_tree(records: Iterable[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """Rebuild one run's span tree from its durable structured log records."""
    all_records = list(records)
    direct = {
        record["span_id"] for record in all_records
        if record.get("run_id") == run_id and record.get("span_id")
    }
    span_ids = set(direct)
    changed = True
    while changed:
        changed = False
        for record in all_records:
            span_id = record.get("span_id")
            if span_id and record.get("parent_span_id") in span_ids and span_id not in span_ids:
                span_ids.add(span_id)
                changed = True

    spans: dict[str, dict[str, Any]] = {}
    for record in all_records:
        span_id = record.get("span_id")
        if span_id not in span_ids:
            continue
        node = spans.setdefault(span_id, {
            "span_id": span_id, "name": record.get("event", "unknown"),
            "parent_span_id": record.get("parent_span_id"), "duration_ms": None,
            "children": [],
        })
        node["name"] = record.get("event", node["name"])
        if record.get("duration_ms") is not None:
            node["duration_ms"] = record["duration_ms"]

    roots: list[dict[str, Any]] = []
    for node in spans.values():
        parent = spans.get(node["parent_span_id"])
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)

    def sort(nodes: list[dict[str, Any]]) -> None:
        nodes.sort(key=lambda node: (node["name"], node["span_id"]))
        for node in nodes:
            sort(node["children"])
    sort(roots)
    return roots


def records_for_span(records: Iterable[dict[str, Any]], span_id: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("span_id") == span_id]


def _flatten(nodes: Iterable[dict[str, Any]], depth: int = 0) -> list[tuple[str, dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any]]] = []
    for node in nodes:
        duration = "running" if node["duration_ms"] is None else f"{node['duration_ms']:.1f} ms"
        output.append((f"{'  ' * depth}{node['name']} ({duration})", node))
        output.extend(_flatten(node["children"], depth + 1))
    return output


def render(store: Any, st: Any, *, log_path: str | Path = "logs/contract-analyzer.jsonl") -> None:
    _render_langfuse(store, st)
    records = read_records(log_path)
    state = monitoring_snapshot(store, records)
    st.subheader("Is anything wrong right now?")
    if store.is_demo_store():
        st.error("DEMO DATA - synthetic runs, not measured system health.")
    message = f"{state['health'].title()} - {len(state['alerts'])} active threshold alerts"
    if state["health"] == "failing":
        st.error(message)
    elif state["health"] == "degraded":
        st.warning(message)
    else:
        st.success(message)
    st.caption("Operations: last 24 hours. Quality alerts use the latest non-demo run's gold labels, "
               "historical calibration and comparable 7-day state distributions. "
               "Health reflects observed threshold breaches, not a backend availability or quality guarantee.")
    if state["metrics"]["accuracy_vs_gold_set"] is None:
        st.caption("Gold-set accuracy unavailable: no recorded labels for the latest non-demo run.")
    if state["metrics"]["state_distribution_drift"] is None:
        st.caption("State-distribution drift unavailable: " + state["metrics"]["state_distribution_baseline"]["reason"])
    if not state["metrics"]["runs"]:
        st.info("No stored runs in the last 24 hours. No current run metrics to assess.")
    st.button("Refresh health", key="monitoring_refresh")

    st.subheader("Active alerts")
    if state["alerts"]:
        st.dataframe([asdict(alert) for alert in state["alerts"]], hide_index=True, use_container_width=True)
    else:
        st.caption("No thresholds currently breached.")

    st.subheader("Movement that matters")
    st.caption("Last 24h against the median of daily values in the preceding 7 days. Empty days are excluded.")
    names = {
        "p95_latency_ms": "p95 latency",
        "cost_per_document": "Cost per document",
        "quote_verification_failure_rate": "Quote-verification failure rate",
    }
    for column, (name, label) in zip(st.columns(3), names.items()):
        movement = state["movements"][name]
        column.metric(label, _metric_value(name, movement["value"]),
                      delta=_movement_label(name, movement), delta_color="inverse")
        column.caption(f"7-day median: {_metric_value(name, movement['baseline'])} "
                       f"({movement['baseline_days']}/7 observed days)")

    _render_usage(state["usage"], st)

    st.subheader("Backend health")
    config = ProviderConfig.from_env("llm")
    backend = backend_health(records, config.provider, config.model)
    st.write(f"Provider: {backend['provider']} | Model: {backend['model']} | Last call: {backend['outcome']}")
    st.caption(f"Configured now: {config.provider} / {config.model}")
    if backend["ts"] is not None:
        st.caption(f"Last recorded call: {datetime.fromtimestamp(backend['ts'], timezone.utc).isoformat()}")

    st.subheader("Recent anomalies")
    anomalies = {run["run_id"]: run for run in state["anomalies"]}
    if not anomalies:
        st.caption("No recent anomalous runs.")
        return
    st.caption(f"Showing {len(anomalies)} of {state['anomaly_count']} candidates from the last 24h, "
               "including the three slowest. Failures are listed first.")
    run_id = st.radio(
        "Select one run to investigate", list(anomalies), index=None, key="monitoring_run",
        format_func=lambda value: (
            f"{value[:8]} - {anomalies[value]['started_at']} - {', '.join(anomalies[value]['reasons'])}"
        ),
    )
    if run_id not in anomalies:
        st.info("Select an anomalous run to see its span tree.")
        return
    flattened = _flatten(span_tree(records, run_id))
    if not flattened:
        st.info("No retained span records for this run. Stored results remain available in run history.")
        return
    st.subheader("Selected run span tree")
    st.dataframe([{"Span": label, "Span ID": node["span_id"]} for label, node in flattened],
                 hide_index=True, use_container_width=True)
    by_span = {node["span_id"]: label for label, node in flattened}
    selected = st.selectbox("Span", list(by_span), index=None,
                            format_func=by_span.get, key=f"monitoring_span_{run_id}")
    if selected not in by_span:
        st.info("Select a span to see its log records.")
        return
    st.subheader("Selected span log records")
    st.dataframe(records_for_span(records, selected), hide_index=True, use_container_width=True)


def _render_langfuse(store, st):
    st.subheader("Langfuse observability")
    try:
        config = langfuse_config()
    except ValueError as exc:
        st.warning(str(exc))
        return
    if config is None:
        st.caption("Langfuse is disabled. Start the local dashboard with ./setup.sh --langfuse, then restart the app.")
        return
    st.markdown(f"[Open local Langfuse dashboard]({config.dashboard_url})")
    st.caption("Configured for local OTLP export; this is not a connectivity probe. "
               "JSON logs and OTel spans remain on disk even if Langfuse is unavailable.")
    runs = store.list_runs(limit=5)
    for run in runs:
        result = store.get_run(run["run_id"])
        trace_id = result.usage.get("trace_id") if result else None
        if trace_id and len(trace_id) == 32:
            st.markdown(f"[Trace for run {run['run_id'][:8]}]({config.trace_url(trace_id)})")


def _render_usage(usage: dict, st: Any) -> None:
    st.subheader("LLM token and cost pressure")
    latest = usage["latest_run"]
    if latest is None:
        st.caption("No token usage has been recorded in the last 30 days.")
        return

    departure = usage["departure_from_median"]
    columns = st.columns(3)
    columns[0].metric("Latest input tokens", f"{latest['total_prompt_tokens']:,}")
    columns[1].metric("Latest output tokens", f"{latest['total_completion_tokens']:,}")
    columns[2].metric(
        "Latest total tokens",
        f"{usage['latest_total_tokens']:,}",
        delta=(
            f"{departure:+.1%} vs prior 7-day median"
            if departure is not None else "No prior 7-day baseline"
        ),
        delta_color="inverse",
    )
    if departure is not None and abs(departure) >= .5:
        st.warning(
            f"Latest run token usage is {abs(departure):.0%} "
            f"{'above' if departure > 0 else 'below'} its recent median."
        )

    st.caption("Input/output tokens for the 12 most recent runs; a sudden change exposes context growth or truncated work.")
    _chart(st, "bar_chart", usage["run_tokens"], x="run",
           y=["input_tokens", "output_tokens"], use_container_width=True)

    left, right = st.columns(2)
    left.caption("Latest run token share by stage")
    if usage["stage_share"]:
        _chart(left, "bar_chart", usage["stage_share"], x="stage", y="share",
               use_container_width=True)
        dominant = usage["stage_share"][0]
        left.caption(f"Dominant stage: {dominant['stage']} ({dominant['share']:.1%} of tokens)")
    else:
        left.caption("No stage-level token usage was recorded for the latest run.")

    right.caption("Latest LLM cost by compliance question (including Router and Critic for audited runs)")
    priced = [row for row in usage["question_cost"] if row["cost_usd"] is not None]
    if priced:
        _chart(right, "bar_chart", priced, x="question_id", y="cost_usd",
               use_container_width=True)
        highest = max(priced, key=lambda row: row["cost_usd"])
        right.caption(
            f"Highest-cost question: {highest['question_id']} (${highest['cost_usd']:.4f}; "
            f"{highest['prompt_tokens'] + highest['completion_tokens']:,} tokens)"
        )
    elif usage["question_cost"]:
        right.caption("Question usage is recorded, but the latest model has no configured price.")
    else:
        right.caption("No question-level usage was recorded for the latest run.")

    st.caption("30-day daily average token trend per run")
    _chart(st, "line_chart", usage["trend"], x="day",
           y=["input_tokens", "output_tokens"], use_container_width=True)


def _chart(target: Any, method: str, data: list[dict], **kwargs: Any) -> None:
    chart = getattr(target, method, None)
    if chart is not None:
        chart(data, **kwargs)
    else:
        target.dataframe(data, hide_index=True, use_container_width=True)


def _metric_value(name: str, value: float | None) -> str:
    if value is None:
        return "No data"
    if name == "p95_latency_ms":
        return f"{value / 1000:.2f} s"
    if name == "cost_per_document":
        return f"${value:.4f}"
    return f"{value:.2%}"


def _movement_label(name: str, movement: dict) -> str:
    change = movement["change"]
    if change is None:
        return "No comparison available"
    if change == 0:
        return "Unchanged"
    if name == "quote_verification_failure_rate":
        return f"{change * 100:+.2f} pp vs 7-day median"
    relative = movement["relative_change"]
    if relative is not None:
        return f"{relative:+.1%} vs 7-day median"
    return f"+{_metric_value(name, change)} vs zero baseline"


__all__ = ["filter_records", "read_records", "records_for_span", "span_tree", "render"]
