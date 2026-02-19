"""
Subprocess worker for sandboxed binary file parsing (Phase 17).

This module is the subprocess entry point for isolated PDF and DOCX parsing.
It is never imported directly — it is launched by sandbox.py as a child
process to contain any parser library exploits, memory bombs, or side-effects
within a separate process boundary.

Protocol:
  stdin:   (unused — path and type are argv)
  argv[1]: source_type  ("pdf" | "docx")
  argv[2]: file_path    (absolute path to the file to parse)
  stdout:  single JSON line: {"text": "<extracted text>"}
           or on error: {"error": "<message>"}
  exit 0:  success
  exit 1:  parse error (bad file, missing library, etc.)
  exit 2:  usage error (wrong arguments)

Security properties:
  - Runs in a separate OS process — parser exploits cannot affect the parent
  - Output is bounded to MAX_OUTPUT_CHARS (2 MB) before JSON serialization
  - Only stdout is trusted; stderr is consumed by parent for logging only
  - No network access, no env var secrets passed, no side-channel paths
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MAX_OUTPUT_CHARS = 2_000_000  # 2 MB cap on extracted text


def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "PDF parsing requires optional dependency 'pypdf'. "
            "Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        stripped = extracted.strip()
        if stripped:
            pages.append(stripped)
    return "\n\n".join(pages)


def _parse_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "DOCX parsing requires optional dependency 'python-docx'. "
            "Install with: pip install python-docx"
        ) from exc

    document = Document(str(path))
    paragraphs = [
        p.text.strip()
        for p in document.paragraphs
        if p.text and p.text.strip()
    ]
    return "\n".join(paragraphs)


def main() -> int:
    if len(sys.argv) != 3:
        out = {"error": f"usage: _parser_subprocess.py <source_type> <file_path>"}
        print(json.dumps(out), flush=True)
        return 2

    source_type = sys.argv[1]
    file_path = Path(sys.argv[2])

    if source_type not in {"pdf", "docx"}:
        out = {"error": f"unsupported source_type: {source_type!r}; expected 'pdf' or 'docx'"}
        print(json.dumps(out), flush=True)
        return 1

    if not file_path.exists():
        out = {"error": f"file not found: {file_path}"}
        print(json.dumps(out), flush=True)
        return 1

    try:
        if source_type == "pdf":
            text = _parse_pdf(file_path)
        else:  # docx
            text = _parse_docx(file_path)

        # Apply output size cap before serialization
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n\n[TRUNCATED]"

        out = {"text": text}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return 0

    except RuntimeError as exc:
        # Library not installed or file-level error from our wrappers
        out = {"error": str(exc)}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        # Catch-all: parser library exceptions (malformed file, decompression bomb, etc.)
        out = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(out, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
