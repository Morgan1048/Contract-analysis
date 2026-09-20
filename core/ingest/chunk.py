"""Question-independent structural chunks with source-backed table cells."""
from __future__ import annotations

import re
from core.models import Chunk, SourceBlock

SECTION = re.compile(
    r"^(?:(?:Section|Clause|Article)\s+(?P<explicit>[A-Z0-9]+(?:\.\d+)*)[.:)]?"
    r"|(?P<numbered>\d+(?:\.\d+)+|\d+[.)]|[IVX]+[.)]))\s+(?P<title>.+)", re.I
)
APPENDIX = re.compile(r"^(?:Exhibit|Appendix|Annex|Schedule)\s+([A-Z0-9]+)\b", re.I)
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)).strip()


def appendix_heading(text: str) -> re.Match[str] | None:
    match = APPENDIX.match(text)
    if match and (
        len(text) < 150
        and not re.search(r"[.!?]$|\b(?:shall|must|will)\b", text, re.I)
        and not re.match(r"\s+[a-z]", text[match.end():])
    ):
        return match
    return None


def chunk_blocks(blocks: list[SourceBlock], max_chars: int = 2400) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = exhibit = heading = None
    counters: dict[str, int] = {}
    for block in blocks:
        text = normalize(block.text)
        if not text:
            continue
        appendix = appendix_heading(text) if block.kind != "table_row" else None
        numbered = SECTION.match(text) if block.kind != "table_row" else None
        if appendix:
            exhibit, section, heading = appendix[1], None, text
        elif numbered:
            exhibit = None
            section = (numbered["explicit"] or numbered["numbered"]).rstrip(".)")
            heading = numbered["title"].split(". ", 1)[0][:180]
        elif block.kind == "heading":
            heading, section = text, None

        requirement_id = None
        if block.cells:
            candidate = re.sub(r"\s+", "", block.cells[0])
            if IDENTIFIER.fullmatch(candidate) and (any(c.isdigit() for c in candidate) or candidate == candidate.upper()):
                requirement_id = candidate
        labels = " ".join(block.headers).lower()
        risk = block.kind == "table_row" and bool(
            re.search(r"exception|deviation|waiver|acceptance", labels)
            or ("risk" in labels and re.search(r"status|remediation|expiration", labels))
        )
        kind = "risk_entry" if risk else "table_row" if block.kind == "table_row" else "clause"
        if block.kind == "heading" or appendix:
            kind = "preamble"
        parent = f"exhibit:{exhibit}:section:{section or heading}" if exhibit else f"section:{section or heading or 'body'}"
        prefix = (
            f"risk:{exhibit}:{requirement_id}" if risk and requirement_id and exhibit else
            f"p{block.page}:risk:{requirement_id}" if risk and requirement_id else
            f"exh:{exhibit}:{requirement_id}" if requirement_id and exhibit else
            f"p{block.page}:requirement:{requirement_id}" if requirement_id else
            f"sec:{section}" if section and kind == "clause" else
            f"p{block.page}:block:{len(chunks) + 1}"
        )
        # Short rows remain intact; oversized prose/rows use contiguous source slices.
        pieces = []
        start = 0
        while start < len(text):
            stop = min(start + max_chars, len(text))
            if stop < len(text):
                boundary = text.rfind(" ", start, stop)
                if boundary > start:
                    stop = boundary
            pieces.append(text[start:stop].strip())
            start = stop
        for piece in pieces:
            count = counters.get(prefix, 0)
            counters[prefix] = count + 1
            chunk_id = prefix if count == 0 else f"{prefix}:part{count + 1}"
            chunks.append(Chunk(
                chunk_id, kind, piece, block.page, section, exhibit,
                requirement_id, heading, parent, block.table_id, block.cells,
                block.headers, block.bbox, block.page,
                normalize(" ".join([heading or "", *block.headers, piece])),
            ))
    return chunks


def build_chunks(pages: list[str], doc_text: str) -> list[Chunk]:
    """Text-only fixture adapter; PDF ingestion supplies layout blocks directly."""
    del doc_text
    blocks = []
    for page, text in enumerate(pages, 1):
        paragraph: list[str] = []
        for line in text.splitlines():
            if not line.strip() or SECTION.match(line) or APPENDIX.match(line):
                if paragraph:
                    blocks.append(SourceBlock(normalize(" ".join(paragraph)), page))
                    paragraph = []
            if line.strip():
                paragraph.append(line)
        if paragraph:
            blocks.append(SourceBlock(normalize(" ".join(paragraph)), page))
    return chunk_blocks(blocks)
