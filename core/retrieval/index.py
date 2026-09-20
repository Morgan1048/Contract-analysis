from __future__ import annotations

import hashlib
import math
import re
import logging
from collections import Counter

from core.models import Chunk, ParsedDocument
from core.providers.config import ProviderConfig
from core.providers.errors import BackendError
from core.retrieval.embeddings import Embedder, FakeEmbedder
from core.retrieval.vectorstore import VectorStore


class ChunkIndex:
    def __init__(self, doc: ParsedDocument, store: VectorStore | None = None, *, allow_remote: bool = False) -> None:
        self.doc = doc
        self.doc_hash = hashlib.sha256((doc.format_version + "\n" + repr(doc.chunks)).encode("utf-8")).hexdigest()
        self._by_id = {chunk.chunk_id: chunk for chunk in doc.chunks}
        self.degraded_reason: str | None = None
        self._search_cache: dict[tuple, list[tuple[Chunk, float]]] = {}
        self._reported_embedding_usage = {"calls": 0, "prompt_tokens": 0, "cost_usd": 0.0}
        self._chunks_by_requirement: dict[str, list[Chunk]] = {}
        self._chunks_by_section: dict[str, list[Chunk]] = {}
        for chunk in doc.chunks:
            if chunk.requirement_id:
                self._chunks_by_requirement.setdefault(chunk.requirement_id, []).append(chunk)
            if chunk.section_id:
                self._chunks_by_section.setdefault(chunk.section_id, []).append(chunk)

        if store is None:
            try:
                config = ProviderConfig.from_env("embedding")
                if not config.is_local and not allow_remote:
                    raise ValueError("Remote embeddings require explicit authorization for non-confidential input")
                embedder = Embedder(config)
            except (BackendError, RuntimeError, OSError, ValueError) as exc:
                self.degraded_reason = f"Embedding initialization failed; lexical-only retrieval: {exc}"
                embedder = None
            store = VectorStore(embedder, persist_dir=".index/chroma") if embedder is not None else None
        self.store = store
        try:
            # Z3: computing the collection key now calls the embedder's
            # identity, which encodes a probe vector to learn the real
            # dimension -- that can hit the network, so it must fall back
            # the same way `upsert` below does when the backend is down.
            if self.store is not None:
                needs_upsert = not self.store.has(self.doc_hash)
                if needs_upsert:
                    self.store.upsert(self.doc_hash, doc.chunks)
                self.store._active_doc_hash = self.doc_hash
        except (BackendError, RuntimeError, OSError, ValueError) as exc:
            self.degraded_reason = f"Embedding indexing failed; lexical-only retrieval: {exc}"
            self.store = None
        if self.degraded_reason:
            logging.getLogger(__name__).warning(self.degraded_reason)
        self._lexical_stats = self._build_lexical_stats()

    def _build_lexical_stats(self) -> tuple[dict[str, int], dict[str, int], int]:
        doc_terms: dict[str, int] = {}
        df: dict[str, int] = {}
        for chunk in self.doc.chunks:
            text = chunk.text.lower()
            terms = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text)
            seen: set[str] = set()
            for term in terms:
                doc_terms[term] = doc_terms.get(term, 0) + 1
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)
        return doc_terms, df, len(self.doc.chunks)

    def _lexical_ranking(self, query: str, k: int = 8, kinds: tuple[str, ...] | None = None,
                         source_ids: set[str] | None = None) -> list[tuple[Chunk, float]]:
        kinds = tuple(kinds or ())
        if not query:
            return []
        query_terms = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower())
        if not query_terms:
            return []
        candidate_chunks = [chunk for chunk in self.doc.chunks
                            if (not kinds or chunk.kind in kinds)
                            and (source_ids is None or chunk.chunk_id in source_ids)]
        doc_terms, df, n_docs = self._lexical_stats
        scores: list[tuple[Chunk, float]] = []
        for chunk in candidate_chunks:
            terms = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", chunk.text.lower())
            term_counts = Counter(terms)
            if not term_counts:
                continue
            score = 0.0
            doc_len = max(len(terms), 1)
            average_len = sum(len(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", c.text.lower())) for c in candidate_chunks) / max(len(candidate_chunks), 1)
            for term in query_terms:
                if term not in term_counts:
                    continue
                term_freq = term_counts[term]
                idf = math.log((n_docs + 1) / (df.get(term, 1) + 1)) + 1.0
                score += term_freq * idf
                if average_len:
                    score += (term_freq * (1.5 + 1.0)) / (term_freq + 1.5 * (1.0 + 0.75 * (doc_len / average_len)))
            if score > 0:
                scores.append((chunk, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    def by_requirement_id(self, requirement_id: str) -> list[Chunk]:
        return list(self._chunks_by_requirement.get(requirement_id, []))

    def by_section_id(self, section_id: str) -> list[Chunk]:
        return list(self._chunks_by_section.get(section_id, []))

    def search(self, query: str, k: int = 8, kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]:
        k = max(1, min(k, 30))
        cache_key = (query, k, kinds)
        if cache_key in self._search_cache:
            return list(self._search_cache[cache_key])
        lexical_hits = self._lexical_ranking(query, k=k, kinds=kinds)
        vector_hits = []
        if self.store is not None:
            try:
                vector_hits = self.store.query(query, k=k, kinds=kinds)
            except (BackendError, RuntimeError, OSError, ValueError) as exc:
                self.degraded_reason = f"Embedding query failed; lexical-only retrieval: {exc}"
                logging.getLogger(__name__).warning(self.degraded_reason)
        vector_hits = [(self._by_id[c.chunk_id], score) for c, score in vector_hits if c.chunk_id in self._by_id]

        lexical_rank = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(lexical_hits)}
        vector_rank = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(vector_hits)}
        combined: dict[str, tuple[Chunk, float]] = {}

        for rank, (chunk, _) in enumerate(lexical_hits, start=1):
            combined[chunk.chunk_id] = (chunk, 1.0 / (60 + rank))
        for rank, (chunk, _) in enumerate(vector_hits, start=1):
            key = chunk.chunk_id
            prior = combined.get(key, (chunk, 0.0))[1]
            combined[key] = (chunk, prior + 1.0 / (60 + rank))

        ordered = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        self._search_cache[cache_key] = ordered[:k]
        return list(ordered[:k])

    def context(self, chunk_id: str) -> list[Chunk]:
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            return []
        if chunk.table_id:
            return [c for c in self.doc.chunks if c.table_id == chunk.table_id]
        if chunk.parent_id:
            return [c for c in self.doc.chunks if c.parent_id == chunk.parent_id]
        return [chunk]

    def context_window(self, chunk_id: str, radius: int = 2) -> list[Chunk]:
        if not 0 <= radius <= 12:
            raise ValueError("context radius must be between 0 and 12")
        chunks = self.context(chunk_id)
        position = next((i for i, c in enumerate(chunks) if c.chunk_id == chunk_id), None)
        if position is None:
            return []
        selected = chunks[max(0, position - radius):position + radius + 1]
        if chunks[0].table_id and chunks[0] not in selected:
            selected.insert(0, chunks[0])
        return selected

    def reference_candidates(self, chunk_id: str, query: str, k: int = 3):
        chunks, unresolved = self.references(chunk_id)
        if len(chunks) <= k:
            return chunks, unresolved, len(chunks)
        matches = self._lexical_ranking(query, k=k, source_ids={c.chunk_id for c in chunks})
        selected = {c.chunk_id: c for c, _ in matches}
        for chunk in chunks:
            if chunk.kind == "preamble":
                selected[chunk.chunk_id] = chunk
        return list(selected.values()), unresolved, len(chunks)

    def references(self, chunk_id: str) -> tuple[list[Chunk], list[str]]:
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            return [], [f"Unknown source: {chunk_id}"]
        selected, unresolved = [], []
        patterns = re.findall(r"\b((?i:Section|Clause|Exhibit|Appendix|Annex|Schedule))\s+([A-Z0-9]+(?:\.\d+)*)", chunk.text)
        for kind, label in patterns:
            matches = self.by_section_id(label) if kind.lower() in {"section", "clause"} else [c for c in self.doc.chunks if c.exhibit_id and c.exhibit_id.casefold() == label.casefold()]
            if matches:
                selected.extend(matches)
            else:
                unresolved.append(f"{kind} {label}")
        from core.ingest.chunk import IDENTIFIER
        for label in IDENTIFIER.findall(chunk.text):
            selected.extend(self.by_requirement_id(label))
        return list({c.chunk_id: c for c in selected}.values()), unresolved

    def profile(self) -> dict:
        return {
            "pages": self.doc.page_count,
            "chunks": len(self.doc.chunks),
            "tables": len({c.table_id for c in self.doc.chunks if c.table_id}),
            "headings": list(dict.fromkeys(c.heading for c in self.doc.chunks if c.heading)),
            "warnings": self.doc.warnings,
            "retrieval_warning": self.degraded_reason,
        }

    def consume_embedding_usage(self) -> dict:
        embedder = self.store.embedder if self.store else None
        totals = {key: getattr(embedder, key, 0) for key in self._reported_embedding_usage}
        usage = {key: None if value is None or self._reported_embedding_usage[key] is None
                 else value - self._reported_embedding_usage[key] for key, value in totals.items()}
        usage["estimated"] = getattr(embedder, "usage_estimated", False)
        self._reported_embedding_usage = totals
        return usage
