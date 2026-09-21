"""KPI tab: every number traces to ``core/kpi.py``; this module only draws it.
"""
from __future__ import annotations

from typing import Any

from core.kpi import evaluate_thresholds, history, snapshot


def _format_metric(value: Any, *, as_percentage: bool | None = None) -> Any:
    """Render a numeric cell in a short, consistent form instead of a raw
    float repr such as ``0.9515151515151515``.

    ``as_percentage`` picks the format explicitly (a rate/ratio vs. a count);
    when omitted it falls back to inferring from magnitude, which is only
    safe for metrics that are always ratios (never a count that can land in
    ``[0, 1]``, e.g. a mean of 1.0 calls/document).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if as_percentage if as_percentage is not None else (0.0 <= value <= 1.0):
        return f"{value:.1%}"
    return f"{value:.3g}"


def render(store: Any, st: Any) -> None:
    if hasattr(store, "is_demo_store") and store.is_demo_store():
        st.error(
            "\u26a0\ufe0f DEMO DATA -- every number below was fabricated by "
            "`python -m eval.seed_demo` for dashboard development. None of it is a "
            "real measurement. Do not use it to make claims about system quality."
        )

    snap = snapshot(store)
    bins = snap["calibration_bins"]
    alerts = evaluate_thresholds(snap)

    st.header("Operational (real time)")
    st.caption(f"All-history operations: {snap['runs']} runs. Agent reliability below uses only "
               f"{snap['instrumented_runs']} instrumented, non-demo runs; quality scores refer to the latest non-demo run.")
    cols = st.columns(4)
    cols[0].metric("Runs", snap["runs"])
    cols[1].metric("Quote verification failure rate",
                   f"{snap['quote_verification_failure_rate']:.1%}" if snap["quote_attempts"] else "unavailable")
    cols[2].metric("NOT_FOUND rate", f"{snap['not_found_rate']:.1%}")
    cols[3].metric("p95 latency (ms)", f"{snap['p95_latency_ms']:.0f}")
    cols = st.columns(3)
    cols[0].metric("Cost per document", f"${snap['cost_per_document']:.4f}"
                   if snap["cost_per_document"] is not None else "unavailable")
    cols[1].metric("Evaluator retry rate", f"{snap['mean_retry_count']:.2f}")
    cols[2].metric("Deviation-applied rate", f"{snap['deviation_applied_rate']:.1%}")

    st.subheader("Agent reliability and follow-up")
    st.caption(
        f"Based on {snap['instrumented_runs']} instrumented, non-demo runs. "
        "Missing denominators display unavailable; legacy runs are not retroactively re-evaluated."
    )
    from core.kpi import latest_analysis_diagnostics
    diagnostic = latest_analysis_diagnostics(store)
    if diagnostic and diagnostic["incomplete"]:
        st.warning(
            f"Latest run {diagnostic['run_id'][:8]} is incomplete: "
            f"{diagnostic['completed_checks']}/{diagnostic['total_checks']} checks completed. "
            "This is not a completed compliance assessment; successful regression fixtures are stored separately."
        )
        if diagnostic["stops"]:
            st.table(diagnostic["stops"])
    st.table([
        {
            "Metric": label,
            "Value": _format_metric(snap[key], as_percentage=pct) if snap[key] is not None else "unavailable",
        }
        for label, key, pct in [
            ("Completion rate", "completion_rate", True), ("Checks needing review", "review_rate", True),
            ("Follow-up resolution rate", "followup_resolution_rate", True),
            ("Budget exhaustion rate", "budget_exhaustion_rate", True),
            ("Mean LLM calls/document", "mean_llm_calls", False), ("Mean tool calls/document", "mean_tool_calls", False),
            ("Mean tokens/document", "mean_tokens", False),
        ]
    ])
    st.caption(
        f"Quote validation: {snap['quote_failures']} failed / {snap['quote_attempts']} evidence-bearing attempts. "
        "This is not semantic faithfulness. Compliance distribution is a business result, not model accuracy. "
        "Local $0 means no API charge; hardware and energy costs are not measured."
        f" Cost average covers {snap['priced_runs']} priced runs out of {snap['runs']}."
    )

    if alerts:
        st.subheader("Alerts")
        for alert in alerts:
            render_fn = st.error if alert.severity == "page" else st.warning
            render_fn(
                f"[{alert.severity.upper()}] {alert.name} = {alert.value:.4g} "
                f"(threshold {alert.threshold:.4g}) -- {alert.action}"
            )

    st.header("Quality (historical)")
    shift = snap["state_distribution_drift"]
    baseline = snap["state_distribution_baseline"]
    st.metric("State-distribution shift", f"{shift * 100:.1f} pp" if shift is not None else "unavailable")
    st.caption(
        f"Last 7 days vs preceding 7 days: {baseline['current_runs']} vs {baseline['baseline_runs']} "
        f"eligible runs. {baseline['reason']}"
    )
    if bins:
        st.subheader("Confidence calibration")
        st.table(bins)
    else:
        st.caption("No evaluated runs yet -- calibration needs a hand-built gold set (eval_results).")

    st.header("History")
    days = history(store)["days"]
    if days:
        st.line_chart({row["day"]: row["runs"] for row in days})
        st.table(days)
    else:
        st.caption("No runs recorded yet.")


__all__ = ["render"]
