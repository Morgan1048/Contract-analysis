"""KPI tab: every number traces to ``core/kpi.py``; this module only draws it.
"""
from __future__ import annotations

from typing import Any

from core.kpi import calibration, evaluate_thresholds, gold_set_accuracy, history, snapshot
from eval.ragas_eval import COMPUTE_HINT

# Each note is specific to its own metric: what it measures, and the concrete
# reason a judged value and a computed value can disagree for *that* metric.
# (A single shared sentence across rows previously hid this.)
_NOTES: dict[str, str] = {
    "faithfulness": (
        "Faithfulness / groundedness of claims is assessed by an optional LLM judge. There is no exact "
        "programmatic equivalent: quotation matching proves provenance, not entailment. "
        "These scores are judge estimates, not gold-set accuracy."
    ),
    "noise_sensitivity": (
        "Noise sensitivity is how much the answer changes when irrelevant retrieved context is "
        "added. It has no set-arithmetic equivalent -- there is no gold label for 'this chunk "
        "is noise' -- so it is judged-only and the exact column is not applicable."
    ),
    "answer_correctness": (
        "Answer correctness is the exact compliance-state match rate against the hand-built "
        "gold set in eval_results. The exact value is a strict equality count; a RAGAS judge "
        "would instead score semantic similarity to a reference answer, which can rate a "
        "differently-worded but equivalent state as correct where the exact match would not."
    ),
}


def _format_metric(value: Any) -> Any:
    """Render a numeric cell in a short, consistent form (percentage for a
    0..1 ratio, otherwise a few significant figures) instead of a raw float
    repr such as ``0.9515151515151515``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if 0.0 <= value <= 1.0:
        return f"{value:.1%}"
    return f"{value:.3g}"


def _ragas_cell(judged: Any, status: Any) -> Any:
    """What the RAGAS column should say, given the three availability states.

    Each state needs a *different* action from the reader, so each gets its own
    text: install the extra, fix a version conflict, configure a judge, or run
    the (deliberately manual, because it costs model calls) scoring pass.
    """
    if judged is not None:
        return _format_metric(judged)
    if status.ready:
        return f"not yet computed for this run -- run {COMPUTE_HINT}"
    return f"unavailable ({status.reason} -- {status.action})" if status.action else f"unavailable ({status.reason})"


def _ragas_unavailable_reason() -> str:
    """Why the RAGAS column has no number: not installed, installed but failing
    to import (with the real exception), or importable with no judge configured.
    """
    import eval.ragas_eval as ragas_eval

    status = ragas_eval.ragas_status()
    if status.ready:
        return f"not yet computed for this run -- run {COMPUTE_HINT}"
    return status.message


def quality_comparisons(store: Any, snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows for adjacent judged/computed metrics; unavailable data stays explicit."""
    import eval.ragas_eval as ragas_eval

    run_id = store.latest_run_id() if hasattr(store, "latest_run_id") else None
    # Run-scoped: a cached RAGAS pass belongs to the run it scored, and
    # averaging it across history would misattribute the number.
    if run_id is not None and hasattr(store, "quality_metrics_for_run"):
        persisted = store.quality_metrics_for_run(run_id)
    else:
        persisted = store.quality_metrics() if hasattr(store, "quality_metrics") else {}
    accuracy = gold_set_accuracy(store, run_id=run_id)
    status = ragas_eval.ragas_status()
    from core.kpi import latest_analysis_diagnostics
    diagnostic = latest_analysis_diagnostics(store) if hasattr(store, "get_run") else None

    rows = [
        ("Faithfulness / groundedness", "faithfulness", False, None),
        ("Noise sensitivity", "noise_sensitivity", False, None),
        ("Answer correctness", "answer_correctness", True, accuracy),
    ]
    output = []
    for label, key, has_pair, fallback in rows:
        values = persisted.get(key, {})
        computed = values.get("computed", fallback)
        judged = _ragas_cell(values.get("ragas"), status)
        note = _NOTES[key]
        if diagnostic and diagnostic["instrumented"]:
            if values.get("ragas") is None:
                if diagnostic["completed_checks"] == 0:
                    judged = "blocked: this run has no completed findings to score"
                elif key != "faithfulness":
                    judged = "requires reviewed reference labels for this document/rubric version"
                elif status.ready:
                    judged = f"not computed; run {COMPUTE_HINT} --run-id {run_id} --metrics faithfulness"
            if has_pair and computed is None:
                note = "No matching recorded gold evaluation for this run. " + note
        output.append({
            "Metric": label,
            "RAGAS (LLM-judged)": judged,
            "Exact (computed)": _format_metric(computed) if has_pair and computed is not None else (
                "not applicable" if not has_pair else "unavailable"
            ),
            "Note": note,
        })
    return output


def render(store: Any, st: Any) -> None:
    if hasattr(store, "is_demo_store") and store.is_demo_store():
        st.error(
            "\u26a0\ufe0f DEMO DATA -- every number below was fabricated by "
            "`python -m eval.seed_demo` for dashboard development. None of it is a "
            "real measurement. Do not use it to make claims about system quality."
        )

    snap = snapshot(store)
    bins = calibration(store)
    alerts = evaluate_thresholds(snap, calibration_bins=bins or None)

    st.header("Operational (real time)")
    st.caption(f"All-history operations: {snap['runs']} runs. Agent reliability below uses only "
               f"{snap['instrumented_runs']} instrumented, non-demo runs; quality scores refer to the latest run.")
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
        {"Metric": label, "Value": _format_metric(snap[key]) if snap[key] is not None else "unavailable"}
        for label, key in [
            ("Completion rate", "completion_rate"), ("Checks needing review", "review_rate"),
            ("Follow-up resolution rate", "followup_resolution_rate"),
            ("Budget exhaustion rate", "budget_exhaustion_rate"),
            ("Mean LLM calls/document", "mean_llm_calls"), ("Mean tool calls/document", "mean_tool_calls"),
            ("Mean tokens/document", "mean_tokens"),
        ]
    ])
    st.caption(
        f"Quote validation: {snap['quote_failures']} failed / {snap['quote_attempts']} evidence-bearing attempts. "
        "This is not semantic faithfulness. Compliance distribution is a business result, not model accuracy. "
        "Local $0 means no API charge; hardware and energy costs are not measured."
        f" Cost average covers {snap['priced_runs']} priced runs out of {snap['runs']}."
    )

    st.header("RAG quality: judged versus computed")
    st.caption("No score is fabricated for failed runs. Reference-based metrics need reviewed labels for this run; "
               "faithfulness can be judged only when completed findings and source evidence exist. "
               "Use a local judge for confidential documents.")
    st.caption(
        "RAGAS is LLM-judged, not ground truth. Exact quality scores require saved evaluations "
        "against independently labeled evidence or answers. "
        f"Judged values are computed on demand by `{COMPUTE_HINT}` and cached per run -- a judge "
        "pass costs more model calls than the analysis itself, so it never runs on a dashboard rerun."
    )
    st.table(quality_comparisons(store, snap))

    if alerts:
        st.subheader("Alerts")
        for alert in alerts:
            render_fn = st.error if alert.severity == "page" else st.warning
            render_fn(
                f"[{alert.severity.upper()}] {alert.name} = {alert.value:.4g} "
                f"(threshold {alert.threshold:.4g}) -- {alert.action}"
            )

    st.header("Quality (historical)")
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


__all__ = ["quality_comparisons", "render"]
