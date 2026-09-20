from dataclasses import dataclass, field
from typing import Literal

ChunkKind = Literal["clause", "table_row", "risk_entry", "preamble"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str               # stable and human-readable: "sec:6.6", "exh:G:GOV-04", "risk:C:RA-2026-001"
    kind: ChunkKind
    text: str                   # VERBATIM source text; must be a substring of ParsedDocument.text
    page: int                   # 1-indexed
    section_id: str | None      # "6.6", "9.1"  — body clauses only
    exhibit_id: str | None      # document-derived exhibit/appendix label, e.g. "C", "Z12"
    requirement_id: str | None  # "GOV-04", "NET-01", "RA-2026-001"
    heading: str | None         # nearest enclosing heading, e.g. "6. Identity, Access, ..."
    parent_id: str | None = None
    table_id: str | None = None
    cells: tuple[str, ...] = ()
    headers: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    end_page: int | None = None
    retrieval_text: str = ""


@dataclass(frozen=True)
class SourceBlock:
    text: str
    page: int
    kind: str = "paragraph"
    bbox: tuple[float, float, float, float] | None = None
    table_id: str | None = None
    cells: tuple[str, ...] = ()
    headers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedDocument:
    source_path: str
    page_count: int
    text: str                   # whole document, normalized; the authority for quote verification
    chunks: list[Chunk]
    embedded_image_count: int = 0
    warnings: list[str] = field(default_factory=list)
    unreadable_pages: list[int] = field(default_factory=list)
    format_version: str = "structure-v2"
