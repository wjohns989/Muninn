"""
Fail-open multi-source ingestion pipeline.
"""

from __future__ import annotations

import tempfile
import concurrent.futures
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Any, Dict

from muninn.ingestion.models import IngestionReport, IngestionSourceResult
from muninn.ingestion.parser import (
    SUPPORTED_EXTENSIONS,
    build_chunks,
    compute_file_sha256,
    infer_source_type,
    parse_source,
)
from muninn.extraction.vision_adapter import VisionAdapter
from muninn.extraction.audio_adapter import AudioAdapter

MAX_INGEST_FILE_SIZE_BYTES = 100 * 1024 * 1024
MAX_CHUNK_SIZE_CHARS = 20_000
MAX_CHUNK_OVERLAP_CHARS = 5_000
MAX_WORKERS = 4  # Cap concurrency to avoid OOM

logger = logging.getLogger("Muninn.Ingest")


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


def _ingest_worker(
    source_order: int,
    path: Path,
    max_bytes: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk: int,
    chronological_order: str,
    allowed_roots_str: List[str],
    vision_config: Dict[str, Any] | None = None,
    audio_config: Dict[str, Any] | None = None,
) -> IngestionSourceResult:
    """Worker function for parallel ingestion."""
    # Reconstruct allowed roots from strings to ensure clean pickle state
    allowed_roots = [Path(p) for p in allowed_roots_str]
    
    source_type = infer_source_type(path)
    result = IngestionSourceResult(
        source_path=str(path),
        source_type=source_type,
        status="failed",
    )

    if not any(_is_relative_to(path, root) for root in allowed_roots):
        result.status = "skipped"
        result.skipped_reason = "outside_allowed_roots"
        result.errors.append("Source path is outside configured ingestion roots")
        return result

    if not path.exists():
        result.status = "skipped"
        result.skipped_reason = "source_not_found"
        result.errors.append("Source path does not exist")
        return result

    if source_type == "unsupported":
        result.status = "skipped"
        result.skipped_reason = "unsupported_extension"
        return result

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        result.status = "failed"
        result.errors.append(f"Stat failed: {exc}")
        return result

    if max_bytes > 0 and file_size > max_bytes:
        result.status = "skipped"
        result.skipped_reason = "file_too_large"
        result.errors.append(
            f"File size {file_size} exceeds configured limit {max_bytes}"
        )
        return result

    try:
        if source_type == "image":
            # Phase 20: Vision support
            if not vision_config or not vision_config.get("enabled"):
                result.status = "skipped"
                result.skipped_reason = "vision_disabled"
                return result
            
            vision = VisionAdapter(
                enabled=True,
                provider=vision_config.get("provider", "ollama"),
                base_url=vision_config.get("ollama_url", "http://localhost:11434"),
                model=vision_config.get("model", "llava"),
                timeout_seconds=vision_config.get("timeout_seconds", 30.0),
            )
            text = vision.describe_image_sync(str(path))
            if not text:
                result.status = "failed"
                result.errors.append("Vision generation failed or returned empty")
                return result
        elif source_type == "audio":
            # Phase 20: Audio support
            if not audio_config or not audio_config.get("enabled"):
                result.status = "skipped"
                result.skipped_reason = "audio_disabled"
                return result
            
            audio = AudioAdapter(
                enabled=True,
                provider=audio_config.get("provider", "openai_compatible"),
                base_url=audio_config.get("base_url", "http://localhost:8000/v1"),
                model=audio_config.get("model", "whisper-1"),
                api_key=audio_config.get("api_key", "not-needed"),
                timeout_seconds=audio_config.get("timeout_seconds", 60.0),
            )
            text = audio.transcribe_audio_sync(str(path))
            if not text:
                result.status = "failed"
                result.errors.append("Audio transcription failed or returned empty")
                return result
        else:
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
        else:
            for chunk in chunks:
                chunk.metadata["source_mtime_epoch"] = source_mtime
                chunk.metadata["source_mtime_iso"] = source_mtime_iso
                chunk.metadata["source_ingest_order"] = source_order
                chunk.metadata["chronological_order"] = chronological_order
            result.status = "processed"
            result.chunks = chunks
    except Exception as exc:
        result.status = "failed"
        result.errors.append(str(exc))

    return result


class IngestionPipeline:
    def __init__(
        self,
        *,
        max_file_size_bytes: int = 5 * 1024 * 1024,
        chunk_size_chars: int = 1200,
        chunk_overlap_chars: int = 150,
        min_chunk_chars: int = 120,
        allowed_roots: Sequence[str] | None = None,
        vision_config: Dict[str, Any] | None = None,
        audio_config: Dict[str, Any] | None = None,
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
        self.vision_config = vision_config
        self.audio_config = audio_config

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

        # Pre-serialize allowed roots for worker
        allowed_roots_str = [str(r) for r in self.allowed_roots]

        # Use ProcessPoolExecutor for true parallelism (CPU bound parsing) and isolation
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    _ingest_worker,
                    idx,
                    path,
                    max_bytes,
                    chunk_size,
                    chunk_overlap,
                    min_chunk,
                    chronological_order,
                    allowed_roots_str,
                    self.vision_config,
                    self.audio_config,
                ): idx
                for idx, path in enumerate(expanded)
            }

            for future in concurrent.futures.as_completed(future_map):
                try:
                    result = future.result(timeout=60)  # 60s timeout per file
                    if result.status == "processed":
                        processed_sources += 1
                        total_chunks += len(result.chunks)
                    else:
                        skipped_sources += 1
                    source_results.append(result)
                except concurrent.futures.TimeoutError:
                    # Handle timeout
                    idx = future_map[future]
                    path = expanded[idx]
                    logger.error(f"Ingestion timed out for {path}")
                    source_results.append(IngestionSourceResult(
                        source_path=str(path),
                        source_type=infer_source_type(path),
                        status="failed",
                        errors=["Parsing timed out"],
                    ))
                    skipped_sources += 1
                except Exception as exc:
                    # Handle pickling error or other worker launch failures
                    idx = future_map[future]
                    path = expanded[idx]
                    logger.error(f"Ingestion worker failed for {path}: {exc}")
                    source_results.append(IngestionSourceResult(
                        source_path=str(path),
                        source_type=infer_source_type(path),
                        status="failed",
                        errors=[f"Worker error: {exc}"],
                    ))
                    skipped_sources += 1

        # Sort results back to original order? 
        # Not strictly required by schema but nice for deterministic reports.
        # But source_results contains IngestionSourceResult which doesn't have an order field easily accessible 
        # unless we parse it or rely on source_ingest_order in chunks.
        # Actually IngestionSourceResult doesn't carry the index explicitly unless we added it.
        # But future_map has the index. We can reconstruct order if we collected tuples.
        # Current implementation appends as they complete.
        
        return IngestionReport(
            total_sources=len(expanded),
            processed_sources=processed_sources,
            skipped_sources=skipped_sources,
            total_chunks=total_chunks,
            source_results=source_results,
        )