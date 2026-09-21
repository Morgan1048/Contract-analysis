from dataclasses import replace

import pytest

from core.ingest.parse import parse_pdf
from core.models import Chunk, ParsedDocument
from core.retrieval.deviations import extract_deviations


def test_extract_deviations():
    doc = parse_pdf("Sample Contract.pdf")
    deviations = extract_deviations(doc)
    by_risk = {d.risk_id: d for d in deviations}
    assert "RA-2026-001" in by_risk
    assert by_risk["RA-2026-001"].status == "Open"
    assert "PASS-02" in by_risk["RA-2026-001"].requirement_ids
    assert "IAM-01" in by_risk["RA-2026-001"].requirement_ids
    assert "IAM-02" in by_risk["RA-2026-001"].requirement_ids

    ra_2026_002 = next(d for d in deviations if d.risk_id == "RA-2026-002")
    assert ra_2026_002.status == "In Progress"

    synthetic = next(d for d in deviations if d.risk_id == "RA-2026-003")
    assert synthetic.status == "Open"
    chunks = {chunk.chunk_id: chunk for chunk in doc.chunks}
    assert not any(chunks[d.source_chunk_id].cells == chunks[d.source_chunk_id].headers
                   for d in deviations if chunks[d.source_chunk_id].headers)
    assert all("expiration" not in (d.expiration or "").lower() for d in deviations)


@pytest.mark.parametrize("cells", [
    ("Risk", "Status", "Expiration"),
    ("R isk", " STA\nTUS ", "Expi ration"),
])
def test_header_candidates_are_excluded_but_kept_in_document(cells):
    header = Chunk("header", "risk_entry", "Risk Status Expiration", 1, None, None, None, None,
                   cells=cells, headers=("Risk", "Status", "Expiration"))
    real = replace(header, chunk_id="actual", text="A temporary waiver applies.", cells=("Waiver", "", "2030-01-01"))
    prose = replace(header, chunk_id="prose", kind="clause", text="An amendment supersedes the delivery terms.",
                    cells=(), headers=())
    doc = ParsedDocument("synthetic", 1, "\n".join(c.text for c in (header, real, prose)), [header, real, prose])
    results = extract_deviations(doc)
    assert [d.source_chunk_id for d in results] == ["actual", "prose"]
    assert results[0].status == "Unknown" and results[0].expiration == "2030-01-01"
    assert doc.chunks[0] == header


def test_partial_header_match_is_not_enough_to_discard_a_real_row():
    chunk = Chunk("real", "risk_entry", "Risk Open 2030-01-01", 1, None, None, None, None,
                  cells=("Risk", "Open", "2030-01-01"), headers=("Risk", "Status", "Expiration"))
    result = extract_deviations(ParsedDocument("synthetic", 1, chunk.text, [chunk]))
    assert len(result) == 1 and result[0].status == "Open"


def test_modifier_ids_exclude_hyphenated_prose_without_removing_the_candidate():
    text = ("A time-bound waiver applies to Non-productionenvironments and Break-glass access. "
            "It affects IAM-01, IAM-01, AUTH-V2 and on-call response; MFA-capable devices are required.")
    chunk = Chunk("modifier", "clause", text, 1, None, None, None, None)
    result = extract_deviations(ParsedDocument("synthetic", 1, text, [chunk]))
    assert len(result) == 1
    assert result[0].requirement_ids == ["IAM-01", "AUTH-V2"]
    assert result[0].text == text


def test_explicit_document_table_ids_are_preserved_even_without_uppercase_or_digits():
    table_ids = ("ctrl-7", "ACCESS-CORE")
    rows = [
        Chunk(f"row-{i}", "table_row", f"{identifier} Required", 1, None, None, identifier, None,
              cells=(identifier, "Required"), headers=("Control ID", "Obligation"))
        for i, identifier in enumerate(table_ids)
    ]
    text = "A waiver applies to ctrl-7 and ACCESS-CORE, not the unobserved time-bound wording."
    modifier = Chunk("modifier", "clause", text, 1, None, None, None, None)
    doc = ParsedDocument("synthetic", 1, "\n".join(c.text for c in [*rows, modifier]), [*rows, modifier])
    result = extract_deviations(doc)
    assert result[0].requirement_ids == list(table_ids)


def test_uppercase_prefix_requires_a_digit_without_a_document_id():
    text = "Exception for ABC-DEF, mixed-123, Mixed-123 and CTRL-123."
    chunk = Chunk("modifier", "clause", text, 1, None, None, None, None)
    result = extract_deviations(ParsedDocument("synthetic", 1, text, [chunk]))
    assert result[0].requirement_ids == ["CTRL-123"]


def test_a_table_header_cannot_whitelist_a_non_numeric_identifier():
    header = Chunk("header", "table_row", "CONTROL-ID Requirement", 1, None, None, "CONTROL-ID", None,
                   cells=("CONTROL-ID", "Requirement"), headers=("CONTROL-ID", "Requirement"))
    text = "Exception entries use the CONTROL-ID column."
    modifier = Chunk("modifier", "clause", text, 1, None, None, None, None)
    doc = ParsedDocument("synthetic", 1, header.text + "\n" + text, [header, modifier])
    result = extract_deviations(doc)
    assert len(result) == 1 and result[0].requirement_ids == []
