from __future__ import annotations

import re
from pathlib import Path

from core.ingest.parse import parse_pdf

DEFAULT_FIXTURE_DIR = Path(__file__).with_name("fixtures")


def _pdf_escape(text: str) -> str:
    encoded = text.encode("latin-1", "replace").decode("latin-1")
    return encoded.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_text_pdf(lines: list[str], path: Path, *, drawing_commands: list[str] | None = None) -> None:
    """Write extractable text into a minimal PDF.

    The generated fixtures deliberately avoid image-only rendering: pdfplumber
    must be able to extract the edited lines and feed the normal parser.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines_per_page = 60
    pages = [lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]
    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    catalog_id = add_object(b"")
    pages_id = add_object(b"")
    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []

    for page_lines in pages:
        content_lines = ["BT", "/F1 7 Tf", "9 TL", "36 792 Td"]
        for line in page_lines:
            content_lines.append(f"({_pdf_escape(line)}) Tj")
            content_lines.append("T*")
        content_lines.append("ET")
        if drawing_commands:
            content_lines.extend(drawing_commands)
        stream = "\n".join(content_lines).encode("latin-1", "replace")
        content_id = add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode("ascii")
        )
        page_ids.append(page_id)

    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(output))


def _source_lines(source_pdf: Path) -> list[str]:
    doc = parse_pdf(source_pdf)
    return [chunk.text for chunk in doc.chunks]


def _no_tls_version(lines: list[str]) -> list[str]:
    edited = []
    for line in lines:
        if line.startswith("CRYP-01 "):
            continue
        if line.startswith("7.2 "):
            line = line.replace(
                "Vendor will use TLS 1.2 or higher (TLS 1.3 where feasible), will not knowingly enable "
                "insecure cipher suites, and will manage certificates using documented procedures.",
                "Vendor will not knowingly enable insecure cipher suites and will manage certificates "
                "using documented procedures.",
            )
        line = line.replace(
            "TLS In transit encryption TLS 1.2+ (prefer 1.3)",
            "TLS In transit encryption encrypted channels",
        )
        edited.append(line)
    return edited


def _no_training_row(lines: list[str]) -> list[str]:
    return [line for line in lines if not line.startswith(("GOV-04 ", "GOV-05 "))]


def _deviation_closed(lines: list[str]) -> list[str]:
    edited = []
    for line in lines:
        if line.startswith("RA-2026-001 "):
            line = re.sub(r"\bOpen$", "Closed", line)
        edited.append(line)
    return edited


def generate_variants(source_pdf: str | Path = "Sample Contract.pdf", out_dir: str | Path = DEFAULT_FIXTURE_DIR) -> dict[str, Path]:
    source_pdf = Path(source_pdf)
    out_dir = Path(out_dir)
    base_lines = _source_lines(source_pdf)
    variants = {
        "no_tls_version": _no_tls_version(base_lines),
        "no_training_row": _no_training_row(base_lines),
        "deviation_closed": _deviation_closed(base_lines),
    }
    paths: dict[str, Path] = {}
    for name, lines in variants.items():
        path = out_dir / f"{name}.pdf"
        _write_text_pdf(lines, path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    for variant_name, variant_path in generate_variants().items():
        print(f"{variant_name}: {variant_path}")
