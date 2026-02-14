"""
Fail-open multi-source ingestion pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

from muninn.ingestion.models import IngestionReport, IngestionSourceResult
from muninn.ingestion.parser import (
    SUPPORTED_EXTENSIONS,
    build_chunks,
    compute_file_sha256,
    infer_source_type,
    parse_source,
)


class IngestionPipeline:
    def __init__(
        self,
        *,
        max_file_size_bytes: int = 5 * 1024 * 1024,
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 150,
        min_chunk_chars: int = 120,
    ):
        self.max_file_size_bytes = max_file_size_bytes
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def _expand_sources(self, sources: Iterable[str], recursive: bool) -> List[Path]:
        resolved: List[Path] = []
        seen: Set[str] = set()
        for source in sources:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                if str(path) not in seen:
                    resolved.append(path)
                    seen.add(str(path))
                continue

            if path.is_file():
                key = str(path)
                if key not in seen:
                    resolved.append(path)
                    seen.add(key)
                continue

            if path.is_dir():
                iterator = path.rglob("*") if recursive else path.glob("*")
                for child in iterator:
                    if not child.is_file():
                        continue
                    if child.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    key = str(child.resolve())
                    if key in seen:
                        continue
                    resolved.append(child.resolve())
                    seen.add(key)

        resolved.sort(key=lambda p: str(p))
        return resolved

    def ingest(
        self,
        sources: Iterable[str],
        *,
        recursive: bool = False,
        max_file_size_bytes: int | None = None,
        chunk_size_chars: int | None = None,
        chunk_overlap_chars: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> IngestionReport:
        max_bytes = max_file_size_bytes if max_file_size_bytes is not None else self.max_file_size_bytes
        chunk_size = chunk_size_chars if chunk_size_chars is not None else self.chunk_size_chars
        chunk_overlap = (
            chunk_overlap_chars if chunk_overlap_chars is not None else self.chunk_overlap_chars
        )
        min_chunk = min_chunk_chars if min_chunk_chars is not None else self.min_chunk_chars

        expanded = self._expand_sources(sources, recursive)
        source_results: List[IngestionSourceResult] = []
        processed_sources = 0
        skipped_sources = 0
        total_chunks = 0

        for path in expanded:
            source_type = infer_source_type(path)
            result = IngestionSourceResult(
                source_path=str(path),
                source_type=source_type,
                status="failed",
            )

            if not path.exists():
                result.status = "skipped"
                result.skipped_reason = "source_not_found"
                result.errors.append("Source path does not exist")
                skipped_sources += 1
                source_results.append(result)
                continue

            if source_type == "unsupported":
                result.status = "skipped"
                result.skipped_reason = "unsupported_extension"
                skipped_sources += 1
                source_results.append(result)
                continue

            file_size = path.stat().st_size
            if max_bytes > 0 and file_size > max_bytes:
                result.status = "skipped"
                result.skipped_reason = "file_too_large"
                result.errors.append(
                    f"File size {file_size} exceeds configured limit {max_bytes}"
                )
                skipped_sources += 1
                source_results.append(result)
                continue

            try:
                text = parse_source(path, source_type)
                source_sha256 = compute_file_sha256(path)
                chunks = build_chunks(
                    path=path,
                    source_type=source_type,
                    source_sha256=source_sha256,
                    text=text,
                    chunk_size_chars=chunk_size,
                    chunk_overlap_chars=chunk_overlap,
                    min_chunk_chars=min_chunk,
                )
                if not chunks:
                    result.status = "skipped"
                    result.skipped_reason = "empty_content"
                    skipped_sources += 1
                else:
                    result.status = "processed"
                    result.chunks = chunks
                    processed_sources += 1
                    total_chunks += len(chunks)
            except Exception as exc:
                result.status = "failed"
                result.errors.append(str(exc))
                skipped_sources += 1

            source_results.append(result)

        return IngestionReport(
            total_sources=len(expanded),
            processed_sources=processed_sources,
            skipped_sources=skipped_sources,
            total_chunks=total_chunks,
            source_results=source_results,
        )
