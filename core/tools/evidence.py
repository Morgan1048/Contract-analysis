"""Evidence operations shared by LangGraph, chat, and MCP."""
from __future__ import annotations

from dataclasses import asdict

from core.retrieval.deviations import extract_deviations
from core.tools.registry import Tool, ToolRegistry


def build_evidence_registry(index) -> ToolRegistry:
    registry = ToolRegistry()

    def result(chunks, **extra):
        return {"found": bool(chunks), "results": [asdict(c) for c in chunks], **extra}

    def available(handler):
        def call(**arguments):
            if index is None:
                return {"found": False, "results": [], "error": "Original document is unavailable; re-upload it."}
            return handler(**arguments)
        return call

    def context(chunk_id, radius=2):
        chunks = index.context_window(chunk_id, radius)
        return result(chunks, available_sources=len(index.context(chunk_id)),
                      selection="Bounded neighboring sources plus table header; use read_table for a full table.")

    def references(chunk_id, query=None):
        if query:
            chunks, unresolved, total = index.reference_candidates(chunk_id, query)
            return result(chunks, unresolved=unresolved, available_sources=total,
                          selection="Query-ranked candidates within actual reference targets; not exhaustive.")
        chunks, unresolved = index.references(chunk_id)
        return result(chunks, unresolved=unresolved)

    operations = [
        ("search_evidence", "Hybrid search over every kind of document content.",
         {"query": {"type": "string"}, "k": {"type": "integer", "default": 6}}, ["query"],
         lambda query, k=6: result([c for c, _ in index.search(query, k=k)])),
        ("read_context", "Read neighboring parent-clause/table sources plus table header; radius can expand the window.",
         {"chunk_id": {"type": "string"}, "radius": {"type": "integer", "default": 2, "minimum": 0, "maximum": 12}}, ["chunk_id"],
         context),
        ("read_table", "Read table rows with their original cells, headers and page positions.",
         {"chunk_id": {"type": "string"}}, ["chunk_id"],
         lambda chunk_id: result([c for c in index.context(chunk_id) if c.table_id])),
        ("resolve_reference", "Follow references actually present in a known source; report missing targets.",
         {"chunk_id": {"type": "string"}, "query": {"type": "string", "description": "Select relevant candidates within broad reference targets."}}, ["chunk_id"],
         references),
        ("find_modifiers", "Scan the whole document for potential exceptions/amendments; candidates are NOT automatic overrides.",
         {}, [],
         lambda: result([index._by_id[d.source_chunk_id] for d in extract_deviations(index.doc)])),
    ]
    for name, description, properties, required, handler in operations:
        registry.register(Tool(name, description, {"type": "object", "properties": properties,
                                                 "required": required, "additionalProperties": False}, available(handler)))
    return registry
