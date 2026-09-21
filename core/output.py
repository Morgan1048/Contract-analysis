from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from core.rubric.schema import QuestionResult
from core.models import Chunk


@dataclass(frozen=True)
class Citation:
    quote: str
    citation: str
    source_chunk_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def reference_label(chunk: Any) -> str:
    """Human-readable citation label for a chunk, e.g. "Section 6.7" or
    "Exhibit G13 (NET-01)". Public so the UI can render the same label for
    citations sourced outside ``to_compliance_json`` (e.g. chat tool results).
    """
    if chunk.section_id:
        return f"Section {chunk.section_id}"
    if chunk.exhibit_id:
        label_id = chunk.exhibit_id
        if chunk.heading:
            # The chunk's own heading names its sub-section, e.g. "G13. Network
            # Authentication and Authorization (Added)" -> "G13"; use it when the
            # assignment's own citation style ("Exhibit G13 (...)") is available.
            match = re.match(rf"({re.escape(chunk.exhibit_id)}\d+)\.", chunk.heading)
            if match:
                label_id = match.group(1)
        label = f"Exhibit {label_id}"
        if chunk.requirement_id:
            label += f" ({chunk.requirement_id})"
        return label
    return f"Page {chunk.page}"


def reference_label_from_chunk_id(chunk_id: str) -> str:
    """Best available citation label when only persisted finding data exists."""
    chunk_id = re.sub(r":part\d+$", "", chunk_id)
    if chunk_id.startswith("sec:"):
        return f"Section {chunk_id.removeprefix('sec:')}"
    if chunk_id.startswith("exh:"):
        parts = chunk_id.split(":", 2)
        label = f"Exhibit {parts[1]}" if len(parts) > 1 else chunk_id
        if len(parts) > 2 and parts[2]:
            label += f" ({parts[2]})"
        return label
    risk = re.fullmatch(r"risk:([A-Za-z0-9]+):([^:]+)", chunk_id)
    if risk:
        return f"Exhibit {risk[1]} ({risk[2]})"
    page = re.match(r"p(\d+):", chunk_id)
    if page:
        return f"Page {page[1]}"
    return chunk_id


def to_compliance_json(result: QuestionResult, index: Any | None, *,
                       evidence_snapshot: dict | None = None) -> dict[str, Any]:
    """Render the assignment's five-field, reader-facing question object."""
    if result.state == "Indeterminate":
        raise ValueError("Incomplete questions must be rendered as review items, not compliance classifications.")
    chunks = {chunk.chunk_id: chunk for chunk in index.doc.chunks} if index is not None else None
    citations = []
    seen = set()
    for finding in result.findings:
        if not finding.quote_verified:
            continue
        evidence = finding.evidence or [{"quote": finding.quote, "source_chunk_id": finding.source_chunk_id}]
        for item in evidence:
            quote, source = item["quote"], item["source_chunk_id"]
            if not quote or not source or (quote, source) in seen:
                continue
            chunk = chunks.get(source) if chunks is not None else None
            # The run's own saved evidence is source-backed, so it still grounds
            # a label when the live index no longer holds that chunk (a result
            # rendered against a re-parsed or different document). Only a source
            # neither the index nor the snapshot knows is dropped.
            if chunk is None and evidence_snapshot and source in evidence_snapshot:
                chunk = Chunk(**evidence_snapshot[source])
            if chunks is not None and chunk is None:
                continue
            label = reference_label(chunk) if chunk else reference_label_from_chunk_id(source)
            citations.append(Citation(quote, label, source).to_dict())
            seen.add((quote, source))
    return {
        "Compliance Question": result.question_text,
        "Compliance State": result.state,
        "Confidence": result.confidence,
        "Relevant Quotes": citations,
        "Rationale": result.rationale,
    }


def to_output_document(result: Any, index: Any | None) -> dict[str, Any]:
    """Assemble every question in a run's result into the single document the UI
    downloads and the MCP server returns."""
    return {
        "run_id": result.run_id,
        "source_path": result.source_path,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        # X1: a run containing an Indeterminate question (a backend failure
        # prevented judging) must be visibly marked incomplete, never silently
        # reported as a normal completed run.
        "status": "incomplete" if getattr(result, "incomplete", False) else "completed",
        "questions": [to_compliance_json(question, index, evidence_snapshot=getattr(result, "evidence_snapshot", {}))
                      for question in result.questions if question.state != "Indeterminate"],
        "review_required": [
            {"question_id": q.question_id, "question": q.question_text, "reason": q.rationale}
            for q in result.questions if q.state == "Indeterminate"
        ],
        "check_results": [asdict(f) for q in result.questions for f in q.findings],
        "audit": {
            "assessment_date": getattr(result, "assessment_date", ""),
            "document_hash": getattr(result, "document_hash", ""),
            "rubric_hash": getattr(result, "rubric_hash", ""),
            "pipeline_version": result.usage.get("pipeline_version", "legacy"),
            "model": result.usage.get("model"),
            "provider": result.usage.get("provider"),
            "langgraph_version": result.usage.get("langgraph_version"),
            "confidence_method": result.usage.get("confidence_method", "legacy"),
        },
    }
