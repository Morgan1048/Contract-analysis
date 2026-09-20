"""Acceptance tests for eval/gold_contexts.yaml (task 10, Part A).

Test 2 is the one that matters most here: a typo in gold_contexts.yaml would
otherwise show up as a permanently depressed recall score that looks like a
retrieval regression, and nobody would find it.
"""
from __future__ import annotations

from core.ingest.parse import parse_pdf
from core.rag_metrics import load_gold
from core.rubric.aggregate import load_rubric

GOLD_PATH = "eval/gold_contexts.yaml"


def _parsed_doc():
    return parse_pdf("Sample Contract.pdf")


def test_all_25_sub_requirements_are_labelled():
    gold = load_gold(GOLD_PATH)
    rubric = load_rubric()
    all_sub_ids = {sub.id for question in rubric.values() for sub in question.sub_requirements}
    assert len(all_sub_ids) == 25
    assert set(gold.keys()) == all_sub_ids


def test_every_labelled_chunk_id_resolves_in_parsed_contract():
    """The typo guard: a bad label here would silently depress recall forever."""
    gold = load_gold(GOLD_PATH)
    doc = _parsed_doc()
    known_chunk_ids = {chunk.chunk_id for chunk in doc.chunks}
    for sub_id, chunk_ids in gold.items():
        for chunk_id in chunk_ids:
            assert chunk_id in known_chunk_ids, f"{sub_id}: {chunk_id!r} does not resolve in the parsed sample contract"


def test_p6_p7_n2_include_the_open_risk_acceptance():
    gold = load_gold(GOLD_PATH)
    for sub_id in ("P6", "P7", "N2"):
        assert "risk:H:RA-2026-001" in gold[sub_id]


def test_t1_through_t4_reference_only_exhibit_rows():
    """No main-body clause exists for Q3 (Security Training & Background Checks)."""
    gold = load_gold(GOLD_PATH)
    for sub_id in ("T1", "T2", "T3", "T4"):
        for chunk_id in gold[sub_id]:
            assert not chunk_id.startswith("sec:"), f"{sub_id} unexpectedly references a body clause: {chunk_id!r}"
