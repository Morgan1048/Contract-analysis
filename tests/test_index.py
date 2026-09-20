import hashlib
import re

import pytest

from core.ingest.parse import parse_pdf
from core.models import Chunk, ParsedDocument
from core.retrieval.embeddings import FakeEmbedder
from core.retrieval.index import ChunkIndex
from core.retrieval.vectorstore import VectorStore


def test_references_ignore_lowercase_words_after_keywords(tmp_path):
    source = Chunk(
        "source", "clause", "comply with this Section and with Exhibit G thereto",
        1, None, None, None, None,
    )
    target = Chunk("target", "table_row", "Required controls.", 2, None, "G", None, None)
    doc = ParsedDocument("synthetic", 2, source.text + "\n" + target.text, [source, target])
    index = ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=tmp_path))

    resolved, unresolved = index.references(source.chunk_id)

    assert [chunk.chunk_id for chunk in resolved] == ["target"]
    assert unresolved == []


@pytest.mark.parametrize("keyword", ["section", "CLAUSE", "exhibit", "APPENDIX", "annex", "SCHEDULE"])
def test_reference_keywords_remain_case_insensitive(tmp_path, keyword):
    is_section = keyword.lower() in {"section", "clause"}
    label = "12.3" if is_section else "Z"
    source = Chunk("source", "clause", f"See {keyword} {label}.", 1, None, None, None, None)
    target = Chunk(
        "target", "clause", "Required controls.", 2,
        label if is_section else None, None if is_section else label, None, None,
    )
    doc = ParsedDocument("synthetic", 2, source.text + "\n" + target.text, [source, target])
    index = ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=tmp_path))

    resolved, unresolved = index.references(source.chunk_id)

    assert [chunk.chunk_id for chunk in resolved] == ["target"]
    assert unresolved == []


def test_references_preserve_genuinely_missing_labels(tmp_path):
    source = Chunk("source", "clause", "See Section 99.1 and Exhibit Z.", 1, None, None, None, None)
    doc = ParsedDocument("synthetic", 1, source.text, [source])
    index = ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=tmp_path))

    resolved, unresolved = index.references(source.chunk_id)

    assert resolved == []
    assert unresolved == ["Section 99.1", "Exhibit Z"]


def test_sample_contract_real_cross_references_resolve(tmp_path):
    doc = parse_pdf("Sample Contract.pdf")
    index = ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=tmp_path))
    checked_kinds = set()
    for source in doc.chunks:
        resolved, unresolved = index.references(source.chunk_id)
        resolved_ids = {chunk.chunk_id for chunk in resolved}
        for kind, label in re.findall(
            r"\b(Section|Clause|Exhibit|Appendix|Annex|Schedule)\s+([A-Z0-9]+(?:\.\d+)*)",
            source.text, re.I,
        ):
            if label != label.upper():
                continue
            is_section = kind.lower() in {"section", "clause"}
            expected_ids = {
                chunk.chunk_id for chunk in doc.chunks
                if (chunk.section_id if is_section else chunk.exhibit_id) == label
            }
            if expected_ids:
                assert expected_ids <= resolved_ids
                assert f"{kind} {label}" not in unresolved
                checked_kinds.add("section" if is_section else "exhibit")
    assert checked_kinds == {"section", "exhibit"}


def test_basic_exact_lookups_and_search():
    doc = parse_pdf("Sample Contract.pdf")
    store = VectorStore(FakeEmbedder(), persist_dir=".index/test-index")
    index = ChunkIndex(doc, store=store)
    assert len(index.by_requirement_id("GOV-04")) == 1
    section = index.by_section_id("6.6")
    assert section
    assert any("14 characters" in chunk.text for chunk in section)
    results = index.search("password rotation break-glass", k=8)
    assert results
    top_ids = [chunk.chunk_id for chunk, _ in results[:3]]
    assert any(chunk_id.startswith("sec:6.6") for chunk_id in top_ids) or any("PASS-02" in chunk_id for chunk_id in top_ids)
    table_results = index.search("password rotation break-glass", k=8, kinds=("table_row",))
    assert all(chunk.kind == "table_row" for chunk, _ in table_results)


def test_persistence_and_no_reembed(monkeypatch):
    doc = parse_pdf("Sample Contract.pdf")
    store = VectorStore(FakeEmbedder(), persist_dir=".index/test-index-persist")
    index = ChunkIndex(doc, store=store)
    doc_hash = index.doc_hash
    assert doc_hash != hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
    assert store.has(doc_hash) is True
    called = {"count": 0}
    original = store.upsert
    def wrapped(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "upsert", wrapped)
    second = ChunkIndex(doc, store=store)
    assert second.store is store
    assert called["count"] == 0


def test_rrf_uses_single_arm_hit():
    doc = parse_pdf("Sample Contract.pdf")
    index = ChunkIndex(doc, store=VectorStore(FakeEmbedder(), persist_dir=".index/test-index-rrf"))
    lexical = index._lexical_ranking("password rotation break-glass", k=8)
    vector = index.store.query("password rotation break-glass", k=8)
    combined = index.search("password rotation break-glass", k=8)
    assert combined
    assert any(chunk.chunk_id in {c.chunk_id for c, _ in lexical} or chunk.chunk_id in {c.chunk_id for c, _ in vector} for chunk, _ in combined)
