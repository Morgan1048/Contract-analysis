"""Document-wide modifier candidates, never automatic compliance overrides."""
from __future__ import annotations

import re

from core.ingest.chunk import IDENTIFIER
from core.models import Chunk, ParsedDocument
from core.rubric.schema import Deviation

MODIFIER = re.compile(
    r"\b(?:except(?:ion|ions)?|waiver|deviation|risk acceptance|notwithstanding|"
    r"supersed\w*|amendment|order of precedence|does not support|exempt\w*)\b", re.I
)


def _infer_status(text: str) -> str:
    compact = re.sub(r"\s+", "", text).lower()
    return {"open": "Open", "closed": "Closed", "inprogress": "In Progress"}.get(compact, "Unknown")


def _is_header_row(chunk: Chunk) -> bool:
    return bool(
        chunk.cells and len(chunk.cells) == len(chunk.headers) and any(h.strip() for h in chunk.headers)
        and all(re.sub(r"\s+", "", cell).casefold() == re.sub(r"\s+", "", header).casefold()
                for cell, header in zip(chunk.cells, chunk.headers))
    )


def extract_deviations(doc: ParsedDocument) -> list[Deviation]:
    items = []
    observed_ids = {chunk.requirement_id for chunk in doc.chunks
                    if chunk.cells and chunk.requirement_id and not _is_header_row(chunk)}
    for chunk in doc.chunks:
        if _is_header_row(chunk):
            continue
        if chunk.kind != "risk_entry" and not MODIFIER.search(chunk.text):
            continue
        status, expiry = "Unknown", None
        for header, cell in zip(chunk.headers, chunk.cells):
            label = re.sub(r"\s+", "", header).lower()
            if label == "status":
                status = _infer_status(cell)
            if "expir" in label:
                expiry = cell
        items.append(Deviation(
            chunk.requirement_id or chunk.chunk_id, status,
            [identifier for identifier in dict.fromkeys(IDENTIFIER.findall(chunk.text))
             if identifier in observed_ids or (
                 identifier.split("-", 1)[0].isupper() and any(c.isdigit() for c in identifier)
             )],
            chunk.text, chunk.chunk_id, expiry,
        ))
    return items
