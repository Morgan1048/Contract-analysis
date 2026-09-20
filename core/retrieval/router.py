"""Evidence packs and a reproducible baseline; adaptive routing lives in the graph."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Chunk
from core.rubric.schema import Question
from core.tools.evidence import build_evidence_registry


@dataclass(frozen=True)
class EvidencePack:
    sub_requirement_id: str
    chunks: list[Chunk]
    strategies_used: list[str]
    strategy_chunks: dict[str, list[Chunk]] | None = None
    unresolved_references: list[str] = field(default_factory=list)
    truncated: bool = False
    tool_errors: list[str] = field(default_factory=list)


def collect_evidence(sub_id, index, queries, previous=None, max_chars=48000):
    registry = build_evidence_registry(index)
    selected = {c.chunk_id: c for c in previous.chunks} if previous else {}
    strategies = dict(previous.strategy_chunks or {}) if previous else {}
    unresolved = set(previous.unresolved_references if previous else ())

    def add(name, response):
        if response.get("error"):
            raise ValueError(response["error"])
        chunks = [index._by_id[item["chunk_id"]] for item in response.get("results", [])]
        strategies.setdefault(name, []).extend(chunks)
        selected.update((c.chunk_id, c) for c in chunks)
        unresolved.update(response.get("unresolved", []))
        return chunks

    hits = []
    for query in queries:
        hits.extend(add("hybrid_search", registry.call("search_evidence", {"query": query, "k": 6})))
    for hit in dict((c.chunk_id, c) for c in hits).values():
        add("parent_context", registry.call("read_context", {"chunk_id": hit.chunk_id}))
        add("cross_reference", registry.call("resolve_reference", {"chunk_id": hit.chunk_id}))
    add("modifier_scan", registry.call("find_modifiers", {}))
    kept, size = [], 0
    for chunk in selected.values():
        if size + len(chunk.text) <= max_chars:
            kept.append(chunk)
            size += len(chunk.text)
    return EvidencePack(sub_id, kept, list(strategies), strategies, sorted(unresolved),
                        len(kept) != len(selected))


def route(question: Question, index, k: int = 3) -> list[EvidencePack]:
    del k
    return [collect_evidence(sub.id, index, [sub.text]) for sub in question.sub_requirements]
