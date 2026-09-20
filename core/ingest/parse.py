"""Layout-first PDF extraction; source text and retrieval enrichment stay separate."""
from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from core.ingest.chunk import APPENDIX, IDENTIFIER, SECTION, appendix_heading, chunk_blocks, normalize
from core.models import ParsedDocument, SourceBlock

TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}


def _clean_cell(cell: str | None) -> str:
    return normalize(cell or "")


def _reassemble_row(row: list[str | None]) -> str:
    cells = [_clean_cell(cell) for cell in row]
    if cells and IDENTIFIER.fullmatch(cells[0].replace(" ", "")):
        cells[0] = cells[0].replace(" ", "")
    return " ".join(cell for cell in cells if cell)


def _inside(word: dict, box) -> bool:
    return box[0] <= (word["x0"] + word["x1"]) / 2 <= box[2] and box[1] <= (word["top"] + word["bottom"]) / 2 <= box[3]


def _page_blocks(page, page_number: int) -> tuple[list[SourceBlock], list[str]]:
    warnings: list[str] = []
    tables = list(page.find_tables(TABLE_SETTINGS))
    # Borderless candidates require a header-like first row and aligned data rows.
    for table in page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"}):
        if any(not (table.bbox[2] <= t.bbox[0] or table.bbox[0] >= t.bbox[2] or table.bbox[3] <= t.bbox[1] or table.bbox[1] >= t.bbox[3]) for t in tables):
            continue
        rows = [r for r in table.extract() if any(c and c.strip() for c in r)]
        if len(rows) >= 3 and len(rows[0]) >= 2 and sum(bool(c and c.strip()) for c in rows[0]) >= 2:
            header = " ".join(c or "" for c in rows[0]).lower()
            if len(header) < 180 and len(re.findall(r"\b(?:id|requirement|control|frequency|status|description|standard|field|value|scope|owner)\b", header)) >= 2:
                tables.append(table)

    blocks: list[SourceBlock] = []
    for number, table in enumerate(sorted(tables, key=lambda t: t.bbox[1]), 1):
        rows = table.extract()
        headers = tuple(_clean_cell(c) for c in rows[0]) if rows else ()
        table_id = f"table:p{page_number}:{number}"
        for row_index, row in enumerate(rows):
            cells = tuple(_clean_cell(c) for c in row)
            text = _reassemble_row(row)
            if text:
                box = tuple(table.rows[row_index].bbox)
                blocks.append(SourceBlock(text, page_number, "table_row", box, table_id, cells, headers))
        if rows and not any(headers):
            warnings.append(f"Page {page_number}: table header could not be recovered ({table_id}).")

    words = [w for w in page.extract_words() if not any(_inside(w, t.bbox) for t in tables)]
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if not lines or abs(word["top"] - lines[-1][0]["top"]) > 3:
            lines.append([word])
        else:
            lines[-1].append(word)
    paragraphs: list[list[dict]] = []
    after_appendix_heading = False
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line)
        boundary = SECTION.match(text) or APPENDIX.match(text) or re.match(r"^[A-Z]+\d+\.\s", text)
        if not paragraphs or boundary or after_appendix_heading or line[0]["top"] - max(w["bottom"] for w in paragraphs[-1]) > 9:
            paragraphs.append(list(line))
        else:
            paragraphs[-1].extend(line)
        after_appendix_heading = appendix_heading(text) is not None
    gutter_lines = 0
    for line in lines:
        for left, right in zip(line, line[1:]):
            if right["x0"] - left["x1"] > 100:
                gutter_lines += 1
                break
    if gutter_lines >= 3:
        warnings.append(f"Page {page_number}: uncertain multi-column reading order; manual review required.")
    for paragraph in paragraphs:
        text = normalize(" ".join(w["text"] for w in paragraph))
        box = (min(w["x0"] for w in paragraph), min(w["top"] for w in paragraph),
               max(w["x1"] for w in paragraph), max(w["bottom"] for w in paragraph))
        kind = "heading" if (len(text) < 150 and not re.search(r"\b(?:shall|must|will)\b", text, re.I) and not text.endswith(".")) else "paragraph"
        blocks.append(SourceBlock(text, page_number, kind, box))
    blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))
    return blocks, warnings


def _build_page_text(page) -> str:
    return "\n".join(block.text for block in _page_blocks(page, page.page_number)[0])


def parse_pdf(path: str | Path) -> ParsedDocument:
    path = Path(path)
    blocks: list[SourceBlock] = []
    warnings: list[str] = []
    unreadable: list[int] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        image_count = sum(len(page.images) for page in pdf.pages)
        for page_number, page in enumerate(pdf.pages, 1):
            page_blocks, page_warnings = _page_blocks(page, page_number)
            blocks.extend(page_blocks)
            warnings.extend(page_warnings)
            if not page_blocks or (len(page.chars) < 40 and page.images):
                unreadable.append(page_number)
                warnings.append(f"Page {page_number}: insufficient readable text; local OCR/manual review required.")
            elif page.images:
                warnings.append(f"Page {page_number}: embedded images are not interpreted; review required.")
    chunks = chunk_blocks(blocks)
    # Keep physical table boundaries but link repeated headers across adjacent pages.
    from dataclasses import replace
    previous: dict[tuple, tuple[int, str]] = {}
    physical_tables: dict[str, str] = {}
    updated = []
    for chunk in chunks:
        if chunk.table_id:
            key = (chunk.headers, chunk.exhibit_id, chunk.section_id, chunk.heading)
            prior = previous.get(key)
            table_id = physical_tables.get(chunk.table_id)
            if table_id is None:
                table_id = prior[1] if prior and chunk.page == prior[0] + 1 else chunk.table_id
                physical_tables[chunk.table_id] = table_id
            previous[key] = (chunk.page, table_id)
            chunk = replace(chunk, table_id=table_id)
        updated.append(chunk)
    return ParsedDocument(str(path), page_count, normalize(" ".join(b.text for b in blocks)),
                          updated, image_count, warnings, sorted(set(unreadable)))
