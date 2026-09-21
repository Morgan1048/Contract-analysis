from core.ingest.parse import parse_pdf
from core.rubric.aggregate import load_rubric
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.router import route
from core.retrieval.vectorstore import VectorStore


def _index():
    doc = parse_pdf("Sample Contract.pdf")
    return ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=".index/test-router"))


def test_router_covers_every_sub_requirement():
    rubric = load_rubric()
    index = _index()
    for question in rubric.values():
        packs = route(question, index)
        assert len(packs) == len(question.sub_requirements)
        for pack in packs:
            assert pack.chunks
            assert pack.sub_requirement_id in {sr.id for sr in question.sub_requirements}
            assert len(pack.chunks) == len({chunk.chunk_id for chunk in pack.chunks})


def test_q3_and_q1_router_targets():
    rubric = load_rubric()
    index = _index()
    q1 = rubric["Q1"]
    q3 = rubric["Q3"]
    q4 = rubric["Q4"]
    q5 = rubric["Q5"]

    q3_packs = {pack.sub_requirement_id: pack for pack in route(q3, index)}
    assert {chunk.requirement_id for chunk in q3_packs["T1"].chunks} >= {"GOV-04"}
    assert {chunk.requirement_id for chunk in q3_packs["T2"].chunks} >= {"GOV-04"}
    assert {chunk.requirement_id for chunk in q3_packs["T3"].chunks} >= {"GOV-05"}
    assert {chunk.requirement_id for chunk in q3_packs["T4"].chunks} >= {"GOV-05"}

    q1_packs = {pack.sub_requirement_id: pack for pack in route(q1, index)}
    assert "RA-2026-001" in {chunk.requirement_id for chunk in q1_packs["P6"].chunks}
    assert any("vault" in chunk.text.lower() for chunk in q1_packs["P6"].chunks)
    assert any("rotat" in chunk.text.lower() for chunk in q1_packs["P7"].chunks)

    q5_packs = {pack.sub_requirement_id: pack for pack in route(q5, index)}
    assert "RA-2026-001" in {chunk.requirement_id for chunk in q5_packs["N2"].chunks}

    q4_packs = {pack.sub_requirement_id: pack for pack in route(q4, index)}
    assert any(chunk.section_id == "7.2" for chunk in q4_packs["E1"].chunks)


def test_strategy_usage_and_determinism():
    rubric = load_rubric()
    index = _index()
    q1 = rubric["Q1"]
    q4 = rubric["Q4"]
    packs = route(q1, index)
    p6 = next(pack for pack in packs if pack.sub_requirement_id == "P6")
    assert set(p6.strategies_used) == {"parent_context", "cross_reference", "hybrid_search", "modifier_scan"}
    p3 = next(pack for pack in packs if pack.sub_requirement_id == "P3")
    assert "hybrid_search" in p3.strategies_used
    assert "requirement_id_lookup" not in p3.strategies_used

    pack1 = route(q4, index)
    pack2 = route(q4, index)
    assert [p.sub_requirement_id for p in pack1] == [p.sub_requirement_id for p in pack2]
    assert [chunk.chunk_id for p in pack1 for chunk in p.chunks] == [chunk.chunk_id for p in pack2 for chunk in p.chunks]
