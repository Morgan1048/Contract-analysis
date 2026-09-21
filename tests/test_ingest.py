"""Acceptance tests for Task 01 — structure-aware parsing and chunking.

Run against the real "Sample Contract.pdf" fixture in the repo root (no
mocking): these tests are the acceptance criteria from
docs/tasks/01-parse-and-chunk.md and must not be weakened.
"""

from pathlib import Path

from core.ingest.parse import parse_pdf
from core.models import ParsedDocument

SAMPLE_CONTRACT = Path(__file__).resolve().parents[1] / "Sample Contract.pdf"


def by_section(doc: ParsedDocument, section_id: str):
    for c in doc.chunks:
        if c.section_id == section_id:
            return c
    return None


def by_requirement(doc: ParsedDocument, requirement_id: str):
    for c in doc.chunks:
        if c.requirement_id == requirement_id:
            return c
    return None


def test_page_count_and_chunk_volume():
    doc = parse_pdf(SAMPLE_CONTRACT)
    assert doc.page_count == 21
    assert len(doc.chunks) > 40


def test_substring_invariant():
    doc = parse_pdf(SAMPLE_CONTRACT)
    for c in doc.chunks:
        assert c.text in doc.text, f"{c.chunk_id} is not a substring of the document text"


def test_body_clauses_resolve():
    doc = parse_pdf(SAMPLE_CONTRACT)
    pw = by_section(doc, "6.6")
    assert "14 characters" in pw.text and "break-glass" in pw.text
    assert "TLS 1.2" in by_section(doc, "7.2").text
    for sid in ("9.1", "9.2", "9.3"):
        assert by_section(doc, sid) is not None


def test_exhibit_rows_are_individual_chunks():
    doc = parse_pdf(SAMPLE_CONTRACT)
    gov04 = by_requirement(doc, "GOV-04")
    gov05 = by_requirement(doc, "GOV-05")
    assert gov04.kind == "table_row" and gov04.exhibit_id == "G"
    assert "training" in gov04.text.lower() and "annual" in gov04.text.lower()
    assert "background" in gov05.text.lower()


def test_split_ids_reassembled():
    doc = parse_pdf(SAMPLE_CONTRACT)
    assert by_requirement(doc, "NET-01") is not None
    assert by_requirement(doc, "ASSET-01") is not None


def test_risk_register_reachable():
    doc = parse_pdf(SAMPLE_CONTRACT)
    ra = by_requirement(doc, "RA-2026-001")
    assert ra.kind == "risk_entry" and ra.exhibit_id == "H"
    assert "MFA" in ra.text
    assert "PASS-02" in ra.text and "IAM-01" in ra.text


def test_chunk_ids_unique():
    doc = parse_pdf(SAMPLE_CONTRACT)
    assert len({c.chunk_id for c in doc.chunks}) == len(doc.chunks)
