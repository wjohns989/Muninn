"""
Safe parser adapters for multi-source ingestion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple

from muninn.ingestion.models import IngestionChunk

SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def infer_source_type(path: Path) -> str:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "unsupported")


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_text_file(path: Path) -> str:
    # utf-8 first; replace errors to preserve ingestion flow.
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return path.read_text(encoding="latin-1", errors="replace")


def _parse_json(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _parse_csv(path: Path, delimiter: str) -> str:
    rows: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            rows.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(rows)


def _parse_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.get_text()


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("PDF parsing requires optional dependency 'pypdf'.") from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages.append(extracted.strip())
    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("DOCX parsing requires optional dependency 'python-docx'.") from exc

    document = Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def parse_source(path: Path, source_type: str) -> str:
    if source_type in {"text", "markdown"}:
        return _read_text_file(path)
    if source_type == "json":
        return _parse_json(path)
    if source_type == "csv":
        return _parse_csv(path, delimiter=",")
    if source_type == "tsv":
        return _parse_csv(path, delimiter="\t")
    if source_type == "html":
        return _parse_html(path)
    if source_type == "pdf":
        return _parse_pdf(path)
    if source_type == "docx":
        return _parse_docx(path)
    raise RuntimeError(f"Unsupported source type: {source_type}")


def chunk_text(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> List[Tuple[int, int, str]]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be >= 0")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be < chunk_size_chars")

    chunks: List[Tuple[int, int, str]] = []
    step = chunk_size_chars - chunk_overlap_chars
    start = 0
    n = len(normalized)

    while start < n:
        end = min(n, start + chunk_size_chars)
        snippet = normalized[start:end].strip()
        if len(snippet) >= min_chunk_chars or (start == 0 and snippet):
            chunks.append((start, end, snippet))
        if end >= n:
            break
        start += step

    return chunks


def build_chunks(
    *,
    path: Path,
    source_type: str,
    source_sha256: str,
    text: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> List[IngestionChunk]:
    raw_chunks = chunk_text(
        text,
        chunk_size_chars=chunk_size_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        min_chunk_chars=min_chunk_chars,
    )
    chunk_count = len(raw_chunks)
    chunks: List[IngestionChunk] = []
    source_bytes = os.path.getsize(path)
    for idx, (start, end, snippet) in enumerate(raw_chunks):
        chunks.append(
            IngestionChunk(
                source_path=str(path),
                source_type=source_type,
                content=snippet,
                chunk_index=idx,
                chunk_count=chunk_count,
                source_sha256=source_sha256,
                metadata={
                    "source_path": str(path),
                    "source_name": path.name,
                    "source_type": source_type,
                    "source_sha256": source_sha256,
                    "source_size_bytes": source_bytes,
                    "chunk_index": idx,
                    "chunk_count": chunk_count,
                    "char_start": start,
                    "char_end": end,
                },
            )
        )
    return chunks
