"""Acceptance tests for core/rag_metrics.py (task 10, Part A)."""
from __future__ import annotations

import pytest

from core.ingest.parse import parse_pdf
from core.rag_metrics import (
    context_precision,
    context_recall,
    load_gold,
    rag_report,
    recall_by_strategy,
)
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.router import route
from core.retrieval.vectorstore import VectorStore
from core.rubric.aggregate import load_rubric

GOLD_PATH = "eval/gold_contexts.yaml"


def _index():
    doc = parse_pdf("Sample Contract.pdf")
    return ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=".index/test-rag-metrics"))


def _all_packs(index):
    rubric = load_rubric()
    packs = []
    for question in rubric.values():
        packs.extend(route(question, index))
    return packs


def test_recall_is_1_when_pack_contains_every_required_chunk():
    required = {"sec:6.6", "exh:G:PASS-01"}
    pack = required
    assert context_recall(pack, required) == 1.0


def test_recall_is_half_when_pack_contains_half():
    required = {"sec:6.6", "exh:G:PASS-01"}
    pack = {"sec:6.6"}
    assert context_recall(pack, required) == 0.5


def test_precision_falls_as_unrelated_chunks_are_added_while_recall_is_unchanged():
    required = {"sec:6.6", "exh:G:PASS-01"}
    tight_pack = {"sec:6.6", "exh:G:PASS-01"}
    padded_pack = {"sec:6.6", "exh:G:PASS-01", "exh:G:ASSET-01", "exh:G:ASSET-02", "risk:H:RA-2026-001"}

    assert context_recall(tight_pack, required) == context_recall(padded_pack, required) == 1.0
    assert context_precision(padded_pack, required) < context_precision(tight_pack, required)


def test_empty_required_set_returns_defined_values_not_zero_division():
    pack = {"sec:6.6"}
    assert context_recall(pack, set()) == 1.0
    assert context_precision(set(), set()) == 0.0
    assert context_precision(pack, set()) == 0.0


def test_recall_by_strategy_attributes_each_chunk_to_the_first_strategy_only_once():
    index = _index()
    rubric = load_rubric()
    gold = load_gold(GOLD_PATH)
    packs = route(rubric["Q1"], index)
    result = recall_by_strategy(packs, gold, index)

    # Every strategy attribution must come from a disjoint slice of the required set:
    # summing per-strategy hit counts can never exceed the total required chunks for Q1.
    total_required = sum(len(gold.get(pack.sub_requirement_id, set())) for pack in packs)
    counted = sum(round(share * total_required) for share in result.values())
    assert counted <= total_required


def test_modifier_scan_is_reported_with_actual_strategy_names():
    index = _index()
    rubric = load_rubric()
    gold = load_gold(GOLD_PATH)

    packs = route(rubric["Q1"], index) + route(rubric["Q5"], index)
    result = recall_by_strategy(packs, gold, index)
    assert "modifier_scan" in result
    assert "requirement_id_lookup" not in result


def test_rag_report_covers_all_five_questions_with_per_question_and_overall_figures():
    index = _index()
    rubric = load_rubric()
    gold = load_gold(GOLD_PATH)

    question_packs = {question_id: route(question, index) for question_id, question in rubric.items()}
    report = rag_report(question_packs, gold)

    assert set(report["questions"].keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    for question_id, figures in report["questions"].items():
        assert 0.0 <= figures["recall"] <= 1.0
        assert 0.0 <= figures["precision"] <= 1.0
    assert 0.0 <= report["overall"]["recall"] <= 1.0
    assert 0.0 <= report["overall"]["precision"] <= 1.0
    assert set(report["sub_requirements"].keys()) == {
        sub.id for question in rubric.values() for sub in question.sub_requirements
    }


def test_gold_entry_naming_a_nonexistent_chunk_id_fails_the_typo_guard_not_recall():
    """Mirrors test_gold_contexts.test_every_labelled_chunk_id_resolves_in_parsed_contract:

    a bogus chunk id must be caught by the resolution guard rather than silently
    lowering a recall figure that would look like a retrieval regression.
    """
    doc = parse_pdf("Sample Contract.pdf")
    known_chunk_ids = {chunk.chunk_id for chunk in doc.chunks}
    bad_gold = {"P1": {"sec:6.6", "exh:G:TYPO-DOES-NOT-EXIST"}}

    with pytest.raises(AssertionError):
        for sub_id, chunk_ids in bad_gold.items():
            for chunk_id in chunk_ids:
                assert chunk_id in known_chunk_ids, f"{sub_id}: {chunk_id!r} does not resolve"

    # And confirm recall itself would NOT raise or flag it -- it silently scores low,
    # which is exactly why a dedicated resolution guard (test 2) is required.
    pack = {"sec:6.6"}
    assert context_recall(pack, bad_gold["P1"]) == 0.5


def test_semantic_baseline_reports_measured_recall_without_hardcoded_gold_lookups():
    """Historical exact-ID labels are diagnostics, not forced routing instructions."""
    index = _index()
    rubric = load_rubric()
    gold = load_gold(GOLD_PATH)

    packs = {qid: route(question, index) for qid, question in rubric.items()}
    report = rag_report(packs, gold)
    assert len(report["sub_requirements"]) == 25
    for group in packs.values():
        for pack in group:
            actual = context_recall({c.chunk_id for c in pack.chunks}, gold[pack.sub_requirement_id])
            assert report["sub_requirements"][pack.sub_requirement_id]["recall"] == actual
            assert "section_lookup" not in pack.strategies_used
