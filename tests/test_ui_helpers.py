"""tests/test_ui_helpers.py -- acceptance tests 9-12 from
docs/tasks/07-ui-and-kpi-dashboard.md.

These exercise the pure helper functions in ``ui/analyze_view.py`` -- the
functions that decide *what* to show -- without touching Streamlit itself, so
the UI's own contract (colour mapping, citation formatting, download payload
shape, and never rendering an unverified quote) is testable without a browser.
"""
from __future__ import annotations

import json

import pytest

from core.ingest.chunk import chunk_blocks
from core.models import Chunk, SourceBlock
from core.output import reference_label, reference_label_from_chunk_id, to_compliance_json, to_output_document
from core.pipeline import AnalysisResult
from core.rubric.schema import Finding, QuestionResult
from ui.analyze_view import download_payload, format_citation, state_color


# 9. state colour mapping covers all three states


def test_state_color_covers_all_three_states():
    assert state_color("Fully Compliant") == "green"
    assert state_color("Partially Compliant") == "orange"
    assert state_color("Non-Compliant") == "red"


# 10. citation formatting renders "Section 6.7" and "Exhibit G13 (NET-01)" from
# chunk metadata


def test_citation_formatting_renders_section_and_exhibit_labels():
    section_chunk = Chunk("sec:6.7", "clause", "text", 1, "6.7", None, None, None)
    assert reference_label(section_chunk) == "Section 6.7"

    exhibit_chunk = Chunk(
        "exh:G:NET-01", "table_row", "TLS is required.", 4, None, "G", "NET-01",
        "G13. Network Authentication and Authorization (Added)",
    )
    assert reference_label(exhibit_chunk) == "Exhibit G13 (NET-01)"

    citation = {"quote": "TLS is required.", "citation": reference_label(exhibit_chunk), "source_chunk_id": exhibit_chunk.chunk_id}
    assert format_citation(citation) == 'Exhibit G13 (NET-01): "TLS is required."'


# 11. the download payload is exactly the assignment output schema, five fields
# per question


def test_download_payload_is_exactly_the_assignment_output_schema():
    chunk = Chunk("sec:6.6", "clause", "Passwords must rotate.", 1, "6.6", None, "PASS-01", None)
    doc = type("Doc", (), {"chunks": [chunk]})()
    index = type("Index", (), {"doc": doc})()
    finding = Finding("P1", "MET", 0.9, chunk.text, chunk.chunk_id, "reason", True)
    question = QuestionResult("Q1", "Question one", "Fully Compliant", 0.9, [finding], [], "reason")
    result = AnalysisResult(
        "run-ui", "contract.pdf", [question], {}, {},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )

    payload = json.loads(download_payload(result, index))
    assert payload == to_output_document(result, index)
    for rendered_question in payload["questions"]:
        assert set(rendered_question.keys()) == {
            "Compliance Question", "Compliance State", "Confidence", "Relevant Quotes", "Rationale",
        }


# 12. an unverified quote never appears in rendered output


def test_unverified_quote_never_appears_in_rendered_output():
    chunk = Chunk("sec:6.6", "clause", "Passwords must rotate.", 1, "6.6", None, "PASS-01", None)
    doc = type("Doc", (), {"chunks": [chunk]})()
    index = type("Index", (), {"doc": doc})()
    verified = Finding("P1", "MET", 0.9, chunk.text, chunk.chunk_id, "reason", True)
    unverified = Finding("P2", "MET", 0.5, "This was never in the document.", chunk.chunk_id, "reason", False)
    question = QuestionResult("Q1", "Question one", "Partially Compliant", 0.7, [verified, unverified], [], "reason")
    result = AnalysisResult(
        "run-ui-2", "contract.pdf", [question], {}, {},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )

    payload = json.loads(download_payload(result, index))
    quotes = [q["quote"] for q in payload["questions"][0]["Relevant Quotes"]]
    assert "This was never in the document." not in quotes
    assert chunk.text in quotes


def test_historical_output_uses_stored_quotes_without_an_index():
    findings = [
        Finding("P1", "MET", 0.9, "Stored clause quote.", "sec:6.6", "reason", True),
        Finding("P2", "MET", 0.9, "Stored exhibit quote.", "exh:G:PASS-01", "reason", True),
        Finding("P3", "PARTIAL", 0.8, "Stored risk quote.", "risk:H:RA-2026-001", "reason", True),
        Finding("P4", "MET", 0.8, "Unverified.", "sec:7.2", "reason", False),
    ]
    question = QuestionResult(
        "Q1", "Question one", "Partially Compliant", 0.85, findings, ["RA-2026-001"], "reason"
    )
    result = AnalysisResult(
        "historical-run", "/missing/original.pdf", [question], {}, {},
        "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00",
    )

    payload = json.loads(download_payload(result, None))
    citations = payload["questions"][0]["Relevant Quotes"]
    assert [(item["citation"], item["quote"]) for item in citations] == [
        ("Section 6.6", "Stored clause quote."),
        ("Exhibit G (PASS-01)", "Stored exhibit quote."),
        ("Exhibit H (RA-2026-001)", "Stored risk quote."),
    ]


@pytest.mark.parametrize("exhibit", ["C", "Z12"])
def test_risk_row_citations_use_document_exhibit(exhibit):
    blocks = [
        SourceBlock(f"Exhibit {exhibit} Risk register", 7, "heading"),
        SourceBlock("RISK-42 Temporary exception.", 7, "table_row",
                    cells=("RISK-42", "Temporary exception."), headers=("Risk", "Acceptance")),
    ]
    chunks = chunk_blocks(blocks)
    risk = chunks[-1]
    assert risk.kind == "risk_entry"
    assert risk.chunk_id == f"risk:{exhibit}:RISK-42"
    assert chunk_blocks(blocks) == chunks
    label = f"Exhibit {exhibit} (RISK-42)"
    assert reference_label(risk) == reference_label_from_chunk_id(risk.chunk_id) == label
    finding = Finding("P1", "PARTIAL", .8, risk.text, risk.chunk_id, "Exception applies.", True)
    question = QuestionResult("Q1", "Question", "Partially Compliant", .8, [finding], [], "reason")
    assert to_compliance_json(question, None)["Relevant Quotes"] == [
        {"quote": risk.text, "citation": label, "source_chunk_id": risk.chunk_id}
    ]


@pytest.mark.parametrize("requirement_id", ["RISK-42", ""])
def test_risk_row_without_exhibit_cites_page(requirement_id):
    block = SourceBlock(f"{requirement_id} Temporary exception.", 7, "table_row",
                        cells=(requirement_id, "Temporary exception."), headers=("Risk", "Acceptance"))
    risk, = chunk_blocks([block])
    assert risk.kind == "risk_entry"
    assert reference_label(risk) == reference_label_from_chunk_id(risk.chunk_id) == "Page 7"
    assert chunk_blocks([block]) == [risk]


def test_split_risk_rows_keep_unique_ids_and_exhibit_labels():
    chunks = chunk_blocks([
        SourceBlock("Exhibit C Risk register", 7, "heading"),
        SourceBlock("RISK-42 " + "Temporary exception. " * 10, 7, "table_row",
                    cells=("RISK-42",), headers=("Risk", "Acceptance")),
    ], max_chars=50)
    risks = [chunk for chunk in chunks if chunk.kind == "risk_entry"]
    assert len(risks) > 1
    assert [chunk.chunk_id for chunk in risks] == [
        "risk:C:RISK-42", *[f"risk:C:RISK-42:part{i}" for i in range(2, len(risks) + 1)]
    ]
    assert all(reference_label_from_chunk_id(chunk.chunk_id) == "Exhibit C (RISK-42)" for chunk in risks)


def test_legacy_risk_id_does_not_invent_an_exhibit():
    assert reference_label_from_chunk_id("risk:OLD-01") == "risk:OLD-01"
