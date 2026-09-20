from dataclasses import asdict
from pathlib import Path

import pytest

from core.ingest.chunk import chunk_blocks
from core.ingest.parse import parse_pdf
from core.models import Chunk, SourceBlock
from core.output import reference_label, reference_label_from_chunk_id, to_compliance_json
from core.rubric.schema import Finding, QuestionResult
from eval.variants import _write_text_pdf


@pytest.mark.parametrize("text", [
    "Exhibit A sets forth the approved subprocessors...",
    "Appendix B lists the approved controls",
    "Annex C describes the applicable terms.",
    "Schedule D shall govern delivery.",
    "Exhibit A: " + "Approved subprocessors and related contractual obligations " * 4,
])
def test_inline_appendix_reference_does_not_start_exhibit(text):
    chunks = chunk_blocks([
        SourceBlock(text, 1),
        SourceBlock("7.2 Data in Transit. Vendor shall encrypt...", 2),
    ])
    assert [chunk.exhibit_id for chunk in chunks] == [None, None]
    assert chunks[1].chunk_id == "sec:7.2"
    assert reference_label(chunks[1]) == "Section 7.2"


@pytest.mark.parametrize("heading", [
    "Exhibit A", "Appendix B: Approved controls", "Annex C - Delivery terms",
    "Schedule Z12 Risk register",
])
def test_short_appendix_heading_scopes_following_blocks(heading):
    chunks = chunk_blocks([
        SourceBlock(heading, 1),
        SourceBlock("The following controls apply.", 2),
    ])
    label = heading.split()[1].rstrip(":")
    assert [chunk.exhibit_id for chunk in chunks] == [label, label]
    assert all(chunk.section_id is None for chunk in chunks)
    assert reference_label(chunks[1]) == f"Exhibit {label}"


def test_pdf_appendix_heading_stays_separate_from_following_prose(tmp_path):
    heading = "Appendix Z - Control register"
    prose = [
        "Vendor shall document controls and retain evidence of implementation.",
        "Vendor must review the register annually and record changes.",
        "Each review must include delivery, receipt, and storage controls.",
    ]
    path = tmp_path / "appendix-heading.pdf"
    _write_text_pdf([heading, *prose, "7.2 Delivery. Shipments must be sealed."], path)

    doc = parse_pdf(path)

    assert doc.chunks[0].text == heading
    assert doc.chunks[0].kind == "preamble"
    assert doc.chunks[1].text == " ".join(prose)
    assert doc.chunks[1].exhibit_id == "Z"
    assert reference_label(doc.chunks[1]) == "Exhibit Z"
    assert doc.chunks[-1].exhibit_id is None
    assert reference_label(doc.chunks[-1]) == "Section 7.2"
    assert all(chunk.text in doc.text for chunk in doc.chunks)


@pytest.mark.parametrize(("body", "section"), [
    ("7.2 Data in Transit. Vendor shall encrypt...", "7.2"),
    ("Section 42 Delivery terms. Shipments must be sealed.", "42"),
    ("Article IX Returns. Returned goods must be logged.", "IX"),
])
def test_body_section_clears_prior_exhibit_scope(body, section):
    chunks = chunk_blocks([
        SourceBlock("Exhibit G Control register", 1, "heading"),
        SourceBlock(body, 2),
        SourceBlock("The obligation continues on the next page.", 3),
    ])
    assert chunks[0].exhibit_id == "G"
    assert [reference_label(chunk) for chunk in chunks[1:]] == [f"Section {section}"] * 2
    assert all(chunk.section_id == section and chunk.exhibit_id is None for chunk in chunks[1:])


@pytest.fixture(scope="module")
def sample_chunks():
    doc = parse_pdf(Path(__file__).resolve().parents[1] / "Sample Contract.pdf")
    return {
        chunk.chunk_id: (chunk.section_id, chunk.exhibit_id, reference_label(chunk))
        for chunk in doc.chunks
    }


def test_sample_chunks_never_have_both_section_and_exhibit(sample_chunks):
    mixed_ids = [chunk_id for chunk_id, (section, exhibit, _) in sample_chunks.items() if section and exhibit]
    assert mixed_ids == []


def test_sample_password_clause_cites_section_6_6(sample_chunks):
    assert sample_chunks["sec:6.6"][2] == "Section 6.6"


def test_section_label_wins_over_stale_exhibit_in_snapshot():
    chunk = Chunk("sec:6.6", "clause", "Passwords must rotate.", 2, "6.6", "G", None, None)
    assert reference_label(chunk) == "Section 6.6"
    finding = Finding("P1", "MET", .9, chunk.text, chunk.chunk_id, "reason", True)
    question = QuestionResult("Q1", "Question", "Fully Compliant", .9, [finding], [], "reason")
    rendered = to_compliance_json(question, None, evidence_snapshot={chunk.chunk_id: asdict(chunk)})
    assert rendered["Relevant Quotes"][0]["citation"] == "Section 6.6"


@pytest.mark.parametrize(("chunk_id", "label"), [
    ("sec:6.6:part2", "Section 6.6"),
    ("exh:G:GOV-04:part2", "Exhibit G (GOV-04)"),
    ("risk:C:RISK-42:part12", "Exhibit C (RISK-42)"),
    ("p7:block:3:part2", "Page 7"),
    ("risk:OLD-01:part2", "risk:OLD-01"),
])
def test_historical_split_chunk_citations_omit_part_suffix(chunk_id, label):
    assert reference_label_from_chunk_id(chunk_id) == label
    finding = Finding("P1", "MET", .9, "Stored quote.", chunk_id, "reason", True)
    question = QuestionResult("Q1", "Question", "Fully Compliant", .9, [finding], [], "reason")
    assert to_compliance_json(question, None)["Relevant Quotes"] == [
        {"quote": "Stored quote.", "citation": label, "source_chunk_id": chunk_id},
    ]


@pytest.mark.parametrize("max_chars", [2400, 25])
def test_requirement_row_without_exhibit_cites_page(max_chars):
    blocks = [
        SourceBlock("GOV-04 Annual security training is required.", 7, "table_row",
                    cells=("GOV-04", "Annual security training is required."),
                    headers=("Control", "Requirement")),
    ]
    chunks = chunk_blocks(blocks, max_chars=max_chars)
    assert all(chunk.requirement_id == "GOV-04" and chunk.exhibit_id is None for chunk in chunks)
    for chunk in chunks:
        assert reference_label_from_chunk_id(chunk.chunk_id) == "Page 7"
        assert reference_label(chunk) == "Page 7"
        assert chunk.chunk_id.startswith("p7:")
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunk_blocks(blocks, max_chars=max_chars) == chunks
