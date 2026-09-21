"""Analyze tab: upload a PDF, run the pipeline, render the five compliance
answers. All the logic that decides *what* to show is a plain function so it
can be tested without a browser (``tests/test_ui_helpers.py``); ``render`` only
calls those functions and draws widgets.
"""
from __future__ import annotations

import json
from typing import Any

from core.output import reference_label, to_output_document

STATE_COLORS = {
    "Fully Compliant": "green",
    "Partially Compliant": "orange",
    "Non-Compliant": "red",
    "Indeterminate": "gray",
}


def state_color(state: str) -> str:
    """Colour name (a Streamlit-recognized colour) for a compliance state."""
    return STATE_COLORS.get(state, "gray")


def format_citation(citation: dict) -> str:
    """One display line for a Relevant-Quotes citation dict, as produced by
    ``core.output.to_compliance_json``: ``'Section 6.7: "quoted text"'``.
    """
    return f'{citation["citation"]}: "{citation["quote"]}"'


def download_payload(result: Any, index: Any | None) -> str:
    """The exact JSON document offered for download: ``to_output_document``,
    unmodified, so what a user downloads is provably what was computed.
    """
    return json.dumps(to_output_document(result, index), indent=2)


def render(result: Any, index: Any | None, st: Any) -> None:
    """Render the Analyze tab for one completed ``AnalysisResult``."""
    document = to_output_document(result, index)
    if index is None:
        st.info(
            "Showing stored results and evidence from this historical run. "
            "Re-upload the original PDF to run new retrieval or analysis."
        )
    if document["status"] == "incomplete":
        st.error(
            "This run is INCOMPLETE: evidence, document readability, validation, or processing limits "
            "prevented a reliable judgment. Review items are not compliance classifications."
        )
    st.download_button(
        "Download JSON",
        data=json.dumps(document, indent=2),
        file_name=f"{result.run_id}.json",
        mime="application/json",
    )

    for item in document["review_required"]:
        st.subheader(item["question"])
        st.warning("Needs review: " + item["reason"])
    for question in result.questions:
        if question.state == "Indeterminate":
            continue
        from core.output import to_compliance_json
        rendered = to_compliance_json(question, index, evidence_snapshot=getattr(result, "evidence_snapshot", {}))
        st.subheader(rendered["Compliance Question"])
        st.markdown(f":{state_color(rendered['Compliance State'])}[{rendered['Compliance State']}]")
        st.caption(f"Confidence: {rendered['Confidence']:.2f}")
        if question.applied_deviations:
            st.warning(
                "Applicable source-backed modifiers: " + ", ".join(question.applied_deviations)
            )
        for citation in rendered["Relevant Quotes"]:
            st.markdown(f"> {format_citation(citation)}")
        st.write(rendered["Rationale"])
        question_sub_requirements = {
            finding.sub_requirement_id for finding in question.findings
        }
        checks = [
            (sub_requirement_id, check)
            for sub_requirement_id in question_sub_requirements
            for check in result.checks.get(sub_requirement_id, [])
        ]
        if checks:
            st.caption("Evaluator checks")
            for sub_requirement_id, check in checks:
                outcome = "passed" if check.passed else "failed"
                st.caption(
                    f"{sub_requirement_id} - {check.name}: {outcome} - {check.detail}"
                )


__all__ = [
    "STATE_COLORS",
    "state_color",
    "format_citation",
    "download_payload",
    "render",
    "reference_label",
]
