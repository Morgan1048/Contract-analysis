from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.models import Chunk
from core.retrieval.embeddings import cosine_similarity


class VectorStore:
    def __init__(self, embedder, persist_dir: str | Path = ".index/chroma") -> None:
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._active_doc_hash: str | None = None
        self._use_chroma = False
        self._client: Any = None
        try:
            import chromadb  # type: ignore
        except Exception:
            chromadb = None
        if chromadb is not None:
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
                self._use_chroma = True
            except Exception:
                self._use_chroma = False

    def _collection_name(self, doc_hash: str) -> str:
        identity = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedder.identity)
        return f"{doc_hash}-{identity}"

    def _persist_path(self, doc_hash: str) -> Path:
        return self.persist_dir / f"{self._collection_name(doc_hash)}.json"

    def _normalize_kind_filter(self, kinds: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if kinds is None:
            return None
        return tuple(sorted(set(kinds)))

    def upsert(self, doc_hash: str, chunks: list[Chunk]) -> None:
        self._active_doc_hash = doc_hash
        texts = [chunk.retrieval_text or chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        if self._use_chroma and self._client is not None:
            collection = self._client.get_or_create_collection(name=self._collection_name(doc_hash))
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {key: value for key, value in {
                        "chunk_id": chunk.chunk_id,
                        "kind": chunk.kind,
                        "section_id": chunk.section_id,
                        "exhibit_id": chunk.exhibit_id,
                        "requirement_id": chunk.requirement_id,
                        "page": chunk.page,
                    }.items() if value is not None}
                    for chunk in chunks
                ],
            )
            return

        payload = {
            "doc_hash": doc_hash,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "kind": chunk.kind,
                    "text": chunk.text,
                    "page": chunk.page,
                    "section_id": chunk.section_id,
                    "exhibit_id": chunk.exhibit_id,
                    "requirement_id": chunk.requirement_id,
                    "heading": chunk.heading,
                }
                for chunk in chunks
            ],
            "embeddings": embeddings,
        }
        self._persist_path(doc_hash).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def query(self, text: str, k: int = 8, kinds: tuple[str, ...] | None = None) -> list[tuple[Chunk, float]]:
        kinds = self._normalize_kind_filter(kinds)
        doc_hash = self._active_doc_hash or self._latest_doc_hash()
        if doc_hash is None:
            return []

        if self._use_chroma and self._client is not None:
            collection = self._client.get_collection(name=self._collection_name(doc_hash))
            query_embedding = self.embedder.encode([text])[0]
            where = None
            if kinds:
                where = {"kind": kinds[0]} if len(kinds) == 1 else {"$or": [{"kind": kind} for kind in kinds]}
            response = collection.query(query_embeddings=[query_embedding], n_results=k, where=where)
            items: list[tuple[Chunk, float]] = []
            for i, chunk_id in enumerate(response.get("ids", [[]])[0]):
                metadata = response.get("metadatas", [[]])[0][i]
                candidate = Chunk(
                    chunk_id=str(chunk_id),
                    kind=str(metadata.get("kind", "preamble")),
                    text=str(response.get("documents", [[]])[0][i]),
                    page=int(metadata.get("page", 1)),
                    section_id=metadata.get("section_id"),
                    exhibit_id=metadata.get("exhibit_id"),
                    requirement_id=metadata.get("requirement_id"),
                    heading=None,
                )
                similarity = float(response.get("distances", [[]])[0][i]) if response.get("distances") else 0.0
                items.append((candidate, similarity))
            return items

        payload = json.loads(self._persist_path(doc_hash).read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                kind=item["kind"],
                text=item["text"],
                page=item["page"],
                section_id=item.get("section_id"),
                exhibit_id=item.get("exhibit_id"),
                requirement_id=item.get("requirement_id"),
                heading=item.get("heading"),
            )
            for item in payload.get("chunks", [])
        ]
        filtered = [chunk for chunk in chunks if kinds is None or chunk.kind in kinds]
        if not filtered:
            return []
        embedding = self.embedder.encode([text])[0]
        scored = []
        for chunk in filtered:
            index = next((i for i, item in enumerate(payload["chunks"]) if item["chunk_id"] == chunk.chunk_id), None)
            if index is None:
                continue
            candidate_vector = payload["embeddings"][index]
            score = cosine_similarity(embedding, candidate_vector)
            scored.append((chunk, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def has(self, doc_hash: str) -> bool:
        if self._use_chroma and self._client is not None:
            collection_name = self._collection_name(doc_hash)
            try:
                self._client.get_collection(name=collection_name)
                return True
            except Exception:
                return False
        return self._persist_path(doc_hash).exists()

    def _latest_doc_hash(self) -> str | None:
        candidates = sorted(self.persist_dir.glob("*.json"), key=lambda p: p.name)
        if not candidates:
            return None
        name = candidates[-1].name
        return name.split("-", 1)[0] if "-" in name else None
