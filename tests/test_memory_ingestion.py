"""Tests for MuninnMemory.ingest_sources feature-gated behavior."""

from pathlib import Path

import pytest

from muninn.core.memory import MuninnMemory
from muninn.ingestion.models import IngestionChunk, IngestionReport, IngestionSourceResult


class _Flags:
    def require(self, flag_name: str):
        if flag_name != "multi_source_ingestion":
            raise RuntimeError("wrong flag")


class _Pipeline:
    def ensure_allowed_path(self, source: str) -> Path:
        return Path(source).expanduser().resolve()

    def is_path_allowed(self, path: Path) -> bool:
        return True

    def ingest(self, sources, **kwargs):
        chunk = IngestionChunk(
            source_path="/tmp/a.txt",
            source_type="text",
            content="hello",
            chunk_index=0,
            chunk_count=1,
            source_sha256="a" * 64,
            metadata={"source_path": "/tmp/a.txt", "chunk_index": 0, "chunk_count": 1},
        )
        return IngestionReport(
            total_sources=1,
            processed_sources=1,
            skipped_sources=0,
            total_chunks=1,
            source_results=[
                IngestionSourceResult(
                    source_path="/tmp/a.txt",
                    source_type="text",
                    status="processed",
                    chunks=[chunk],
                )
            ],
        )


class _CapturePipeline:
    def __init__(self):
        self.kwargs = {}

    def ensure_allowed_path(self, source: str) -> Path:
        return Path(source).expanduser().resolve()

    def is_path_allowed(self, path: Path) -> bool:
        return True

    def ingest(self, sources, **kwargs):
        self.kwargs = dict(kwargs)
        return IngestionReport(
            total_sources=0,
            processed_sources=0,
            skipped_sources=0,
            total_chunks=0,
            source_results=[],
        )


class _LegacyPipeline:
    def __init__(self):
        self.last_sources = []

    def ensure_allowed_path(self, source: str) -> Path:
        return Path(source).expanduser().resolve()

    def is_path_allowed(self, path: Path) -> bool:
        return True

    def ingest(self, sources, **kwargs):
        self.last_sources = list(sources)
        chunks = []
        source_results = []
        for idx, source in enumerate(self.last_sources):
            chunk = IngestionChunk(
                source_path=source,
                source_type="markdown",
                content=f"content-{idx}",
                chunk_index=0,
                chunk_count=1,
                source_sha256="b" * 64,
                metadata={"source_path": source, "chunk_index": 0, "chunk_count": 1},
            )
            chunks.append(chunk)
            source_results.append(
                IngestionSourceResult(
                    source_path=source,
                    source_type="markdown",
                    status="processed",
                    chunks=[chunk],
                )
            )

        return IngestionReport(
            total_sources=len(self.last_sources),
            processed_sources=len(self.last_sources),
            skipped_sources=0,
            total_chunks=len(chunks),
            source_results=source_results,
        )


class _RestrictedPipeline:
    def __init__(self, allowed_root: Path):
        self.allowed_root = allowed_root.expanduser().resolve()

    def _resolve(self, source: str) -> Path:
        return Path(source).expanduser().resolve()

    def ensure_allowed_path(self, source: str) -> Path:
        resolved = self._resolve(source)
        if not self.is_path_allowed(resolved):
            raise ValueError(f"outside allow-list: {resolved}")
        return resolved

    def is_path_allowed(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.allowed_root)
            return True
        except ValueError:
            return False

    def ingest(self, sources, **kwargs):
        return IngestionReport(
            total_sources=0,
            processed_sources=0,
            skipped_sources=0,
            total_chunks=0,
            source_results=[],
        )


@pytest.mark.asyncio
async def test_memory_ingest_sources_happy_path(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    memory._ingestion = _Pipeline()

    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())

    async def fake_add(**kwargs):
        return {"id": "m1", "event": "ADD"}

    monkeypatch.setattr(memory, "add", fake_add)

    result = await memory.ingest_sources(sources=["/tmp/a.txt"], project="muninn")

    assert result["event"] == "INGEST_COMPLETED"
    assert result["added_memories"] == 1
    assert result["failed_chunks"] == 0


@pytest.mark.asyncio
async def test_memory_ingest_sources_counts_skips(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    memory._ingestion = _Pipeline()

    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())

    async def fake_add(**kwargs):
        return {"event": "DEDUP_SKIP"}

    monkeypatch.setattr(memory, "add", fake_add)

    result = await memory.ingest_sources(sources=["/tmp/a.txt"], project="muninn")

    assert result["added_memories"] == 0
    assert result["skipped_chunks"] == 1


@pytest.mark.asyncio
async def test_memory_ingest_sources_passes_chronological_order(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    pipeline = _CapturePipeline()
    memory._ingestion = pipeline

    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())

    result = await memory.ingest_sources(
        sources=["/tmp/a.txt"],
        project="muninn",
        chronological_order="oldest_first",
    )

    assert result["event"] == "INGEST_COMPLETED"
    assert pipeline.kwargs["chronological_order"] == "oldest_first"


@pytest.mark.asyncio
async def test_memory_discover_legacy_sources_filters_and_counts(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    memory._ingestion = _Pipeline()

    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())
    monkeypatch.setattr(
        "muninn.core.memory.discover_legacy_sources_catalog",
        lambda **kwargs: [
            {
                "source_id": "src1",
                "provider": "codex_cli",
                "category": "assistant_chat",
                "path": "/tmp/codex.jsonl",
                "source_type": "jsonl",
                "parser_supported": True,
                "confidence": "high",
                "size_bytes": 100,
                "notes": "",
            },
            {
                "source_id": "src2",
                "provider": "chatgpt_desktop",
                "category": "assistant_chat",
                "path": "/tmp/chatgpt.json",
                "source_type": "json",
                "parser_supported": True,
                "confidence": "low",
                "size_bytes": 200,
                "notes": "",
            },
        ],
    )

    result = await memory.discover_legacy_sources(providers=["codex_cli"])

    assert result["event"] == "LEGACY_DISCOVERY_COMPLETED"
    assert result["total_discovered"] == 1
    assert result["provider_counts"] == {"codex_cli": 1}
    assert result["sources"][0]["source_id"] == "src1"


@pytest.mark.asyncio
async def test_memory_ingest_legacy_sources_injects_context_metadata(monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    pipeline = _LegacyPipeline()
    memory._ingestion = pipeline

    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())
    monkeypatch.setattr(
        "muninn.core.memory.discover_legacy_sources_catalog",
        lambda **kwargs: [
            {
                "source_id": "src_serena",
                "provider": "serena_memory",
                "category": "mcp_memory",
                "path": "/tmp/serena.md",
                "source_type": "markdown",
                "parser_supported": True,
                "confidence": "high",
                "size_bytes": 321,
                "notes": "Serena memory files",
            },
            {
                "source_id": "src_unknown",
                "provider": "chatgpt_desktop",
                "category": "assistant_chat",
                "path": "/tmp/unsupported.bin",
                "source_type": "unsupported",
                "parser_supported": False,
                "confidence": "low",
                "size_bytes": 654,
                "notes": "Unsupported",
            },
        ],
    )

    captured = []

    async def fake_add(**kwargs):
        captured.append(kwargs)
        return {"id": "m1", "event": "ADD"}

    monkeypatch.setattr(memory, "add", fake_add)

    result = await memory.ingest_legacy_sources(
        selected_source_ids=["src_serena", "src_unknown"],
        include_unsupported=True,
        project="muninn",
    )

    assert result["event"] == "LEGACY_INGEST_COMPLETED"
    assert result["selected_supported_sources"] == 1
    assert result["selected_unsupported_sources"] == 1
    assert pipeline.last_sources == ["/tmp/serena.md"]
    assert captured
    metadata = captured[0]["metadata"]
    assert metadata["legacy_source_provider"] == "serena_memory"
    assert metadata["legacy_source_category"] == "mcp_memory"
    assert metadata["legacy_import"] is True


@pytest.mark.asyncio
async def test_memory_discover_legacy_sources_rejects_root_outside_allow_list(tmp_path, monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    memory._ingestion = _RestrictedPipeline(allowed)
    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())

    with pytest.raises(ValueError, match="outside allow-list"):
        await memory.discover_legacy_sources(roots=[str(outside)])


@pytest.mark.asyncio
async def test_memory_ingest_legacy_sources_rejects_selected_path_outside_allow_list(tmp_path, monkeypatch):
    memory = MuninnMemory()
    memory._initialized = True
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    blocked_file = outside / "blocked.txt"
    blocked_file.write_text("secret", encoding="utf-8")
    memory._ingestion = _RestrictedPipeline(allowed)
    monkeypatch.setattr("muninn.core.memory.get_flags", lambda: _Flags())

    monkeypatch.setattr(
        "muninn.core.memory.discover_legacy_sources_catalog",
        lambda **kwargs: [],
    )

    with pytest.raises(ValueError, match="outside allow-list"):
        await memory.ingest_legacy_sources(selected_paths=[str(blocked_file)])
