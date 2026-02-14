"""
Fail-open multi-source ingestion pipeline.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from muninn.ingestion.models import IngestionReport, IngestionSourceResult
from muninn.ingestion.parser import (
    SUPPORTED_EXTENSIONS,
    build_chunks,
    compute_file_sha256,
    infer_source_type,
    parse_source,
)

MAX_INGEST_FILE_SIZE_BYTES = 100 * 1024 * 1024
MAX_CHUNK_SIZE_CHARS = 20_000
MAX_CHUNK_OVERLAP_CHARS = 5_000


def _default_allowed_roots() -> List[Path]:
    return [
        Path.home().expanduser().resolve(),
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class IngestionPipeline:
    def __init__(
        self,
        *,
        max_file_size_bytes: int = 5 * 1024 * 1024,
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 150,
        min_chunk_chars: int = 120,
        allowed_roots: Sequence[str] | None = None,
    ):
        self.max_file_size_bytes = max_file_size_bytes
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars
        roots = (
            [Path(root).expanduser().resolve() for root in allowed_roots]
            if allowed_roots
            else _default_allowed_roots()
        )
        self.allowed_roots = sorted({str(root): root for root in roots}.values(), key=str)

    def resolve_source_path(self, source: str) -> Path:
        return Path(source).expanduser().resolve()

    def is_path_allowed(self, path: Path) -> bool:
        return any(_is_relative_to(path, root) for root in self.allowed_roots)

    def ensure_allowed_path(self, source: str) -> Path:
        resolved = self.resolve_source_path(source)
        if not self.is_path_allowed(resolved):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise ValueError(
                f"Source path is outside configured ingestion roots: {resolved} (allowed_roots={roots})"
            )
        return resolved

    def _validate_runtime_limits(
        self,
        *,
        max_bytes: int,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk: int,
    ) -> None:
        if max_bytes <= 0 or max_bytes > MAX_INGEST_FILE_SIZE_BYTES:
            raise ValueError(
                f"max_file_size_bytes must be between 1 and {MAX_INGEST_FILE_SIZE_BYTES}"
            )
        if chunk_size <= 0 or chunk_size > MAX_CHUNK_SIZE_CHARS:
            raise ValueError(
                f"chunk_size_chars must be between 1 and {MAX_CHUNK_SIZE_CHARS}"
            )
        if chunk_overlap < 0 or chunk_overlap > MAX_CHUNK_OVERLAP_CHARS:
            raise ValueError(
                f"chunk_overlap_chars must be between 0 and {MAX_CHUNK_OVERLAP_CHARS}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")
        if min_chunk <= 0 or min_chunk > chunk_size:
            raise ValueError("min_chunk_chars must be between 1 and chunk_size_chars")

    def _sort_paths(self, paths: List[Path], chronological_order: str) -> List[Path]:
        if chronological_order not in {"none", "oldest_first", "newest_first"}:
            raise ValueError(
                "chronological_order must be one of: none, oldest_first, newest_first"
            )

        if chronological_order == "none":
            return sorted(paths, key=lambda p: str(p))

        existing: List[Path] = []
        missing: List[Path] = []
        for path in paths:
            if path.exists():
                existing.append(path)
            else:
                missing.append(path)

        reverse = chronological_order == "newest_first"
        existing.sort(
            key=lambda p: (p.stat().st_mtime, str(p)),
            reverse=reverse,
        )
        missing.sort(key=lambda p: str(p))
        return [*existing, *missing]

    def _expand_sources(
        self,
        sources: Iterable[str],
        recursive: bool,
        chronological_order: str,
    ) -> List[Path]:
        resolved: List[Path] = []
        seen: Set[str] = set()
        for source in sources:
            path = self.resolve_source_path(source)
            if not self.is_path_allowed(path):
                if str(path) not in seen:
                    resolved.append(path)
                    seen.add(str(path))
                continue
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
                    child_resolved = child.resolve()
                    if not self.is_path_allowed(child_resolved):
                        continue
                    key = str(child_resolved)
                    if key in seen:
                        continue
                    resolved.append(child_resolved)
                    seen.add(key)

        return self._sort_paths(resolved, chronological_order)

    def ingest(
        self,
        sources: Iterable[str],
        *,
        recursive: bool = False,
        chronological_order: str = "none",
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
        self._validate_runtime_limits(
            max_bytes=max_bytes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk=min_chunk,
        )

        expanded = self._expand_sources(
            sources,
            recursive,
            chronological_order,
        )
        source_results: List[IngestionSourceResult] = []
        processed_sources = 0
        skipped_sources = 0
        total_chunks = 0

        for source_order, path in enumerate(expanded):
            source_type = infer_source_type(path)
            result = IngestionSourceResult(
                source_path=str(path),
                source_type=source_type,
                status="failed",
            )

            if not self.is_path_allowed(path):
                result.status = "skipped"
                result.skipped_reason = "outside_allowed_roots"
                result.errors.append("Source path is outside configured ingestion roots")
                skipped_sources += 1
                source_results.append(result)
                continue

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
                source_mtime = path.stat().st_mtime
                source_mtime_iso = datetime.fromtimestamp(
                    source_mtime, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z")
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
                    for chunk in chunks:
                        chunk.metadata["source_mtime_epoch"] = source_mtime
                        chunk.metadata["source_mtime_iso"] = source_mtime_iso
                        chunk.metadata["source_ingest_order"] = source_order
                        chunk.metadata["chronological_order"] = chronological_order
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
