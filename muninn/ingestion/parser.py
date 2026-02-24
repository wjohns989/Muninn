"""
Safe parser adapters for multi-source ingestion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

from muninn.ingestion.models import IngestionChunk

SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    ".vscdb": "sqlite",
    # Zed AI conversation files (plain-text role/separator format)
    ".zed": "text",
    # Images (Phase 20)
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".bmp": "image",
    # Audio (Phase 20)
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".flac": "audio",
}

CHAT_ROLE_KEYS = ("role", "speaker", "author", "sender")
CHAT_CONTENT_KEYS = (
    "content",
    "text",
    "message",
    "prompt",
    "response",
    "value",
    "body",
)
MAX_PARSED_OUTPUT_CHARS = 2_000_000


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
    # Prefer UTF-8, but fail open to latin-1 when decoding errors are encountered.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")
    except OSError as exc:
        raise RuntimeError(f"Unable to read text source: {path}") from exc


def _stringify_chat_content(value, *, depth: int = 0, max_depth: int = 8) -> str:
    if value is None:
        return ""
    if depth > max_depth:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"].strip()
        parts: List[str] = []
        for key in CHAT_CONTENT_KEYS:
            if key not in value:
                continue
            piece = _stringify_chat_content(value[key], depth=depth + 1, max_depth=max_depth)
            if piece:
                parts.append(piece)
        if parts:
            return "\n".join(parts)
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            piece = _stringify_chat_content(item, depth=depth + 1, max_depth=max_depth)
            if piece:
                parts.append(piece)
        return "\n".join(parts)
    return ""


def _extract_chat_lines(value, lines: List[str], *, depth: int = 0, max_depth: int = 10, max_lines: int = 4000) -> None:
    if depth > max_depth or len(lines) >= max_lines:
        return
    if isinstance(value, list):
        for item in value:
            if len(lines) >= max_lines:
                break
            _extract_chat_lines(item, lines, depth=depth + 1, max_depth=max_depth, max_lines=max_lines)
        return
    if not isinstance(value, dict):
        return

    role = ""
    for key in CHAT_ROLE_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            role = candidate.strip().lower()
            break

    content = ""
    for key in CHAT_CONTENT_KEYS:
        if key not in value:
            continue
        content = _stringify_chat_content(value[key], depth=depth + 1)
        if content:
            break

    if role and content:
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if content:
            lines.append(f"[{role}] {content}")
            return

    for nested in value.values():
        if len(lines) >= max_lines:
            break
        _extract_chat_lines(nested, lines, depth=depth + 1, max_depth=max_depth, max_lines=max_lines)


def _truncate_output(text: str) -> str:
    if len(text) <= MAX_PARSED_OUTPUT_CHARS:
        return text
    return text[:MAX_PARSED_OUTPUT_CHARS] + "\n\n[TRUNCATED]"


def _parse_json(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    chat_lines: List[str] = []
    _extract_chat_lines(payload, chat_lines)
    if chat_lines:
        return _truncate_output("\n".join(chat_lines))
    return _truncate_output(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _parse_jsonl(path: Path) -> str:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw_line = raw.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                lines.append(raw_line)
                continue

            extracted: List[str] = []
            _extract_chat_lines(payload, extracted)
            if extracted:
                lines.extend(extracted)
            else:
                lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return _truncate_output("\n".join(lines))


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
    """Parse a PDF file via the subprocess sandbox for process isolation (Phase 17)."""
    from muninn.ingestion.sandbox import sandboxed_parse_binary
    return sandboxed_parse_binary(path, "pdf", timeout=30.0)


def _parse_docx(path: Path) -> str:
    """Parse a DOCX file via the subprocess sandbox for process isolation (Phase 17)."""
    from muninn.ingestion.sandbox import sandboxed_parse_binary
    return sandboxed_parse_binary(path, "docx", timeout=30.0)


def _parse_sqlite(path: Path) -> str:
    uri = f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        if not table_names:
            return ""

        preferred = [
            name
            for name in table_names
            if any(token in name.lower() for token in ("chat", "conversation", "message", "session", "copilot", "ai", "prompt"))
        ]
        ordered_tables = preferred + [name for name in table_names if name not in preferred]
        ordered_tables = ordered_tables[:12]

        lines: List[str] = []
        for table_name in ordered_tables:
            safe_name = table_name.replace('"', '""')
            query = f'SELECT * FROM "{safe_name}" LIMIT 250'
            try:
                rows = conn.execute(query).fetchall()
            except sqlite3.DatabaseError:
                continue
            if not rows:
                continue

            lines.append(f"# table: {table_name}")
            row_count = 0
            for row in rows:
                row_count += 1
                row_map = dict(row)

                extracted: List[str] = []
                _extract_chat_lines(row_map, extracted)
                if extracted:
                    lines.extend(extracted)
                    continue

                fallback_texts = []
                for key, raw_value in row_map.items():
                    if raw_value is None:
                        continue
                    if isinstance(raw_value, bytes):
                        continue
                    value_text = str(raw_value).strip()
                    if not value_text:
                        continue
                    if len(value_text) > 2000:
                        value_text = value_text[:2000] + "..."
                    key_lower = str(key).lower()
                    if any(token in key_lower for token in ("content", "text", "prompt", "response", "message", "body", "value")):
                        fallback_texts.append(f"[{key}] {value_text}")
                if fallback_texts:
                    lines.extend(fallback_texts)

                if row_count >= 250:
                    break

            if sum(len(line) for line in lines) >= MAX_PARSED_OUTPUT_CHARS:
                break

        return _truncate_output("\n".join(lines))
    finally:
        conn.close()


def parse_source(path: Path, source_type: str) -> str:
    if source_type in {"text", "markdown"}:
        return _read_text_file(path)
    if source_type == "json":
        return _parse_json(path)
    if source_type == "jsonl":
        return _parse_jsonl(path)
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
    if source_type == "sqlite":
        return _parse_sqlite(path)
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