"""Tests for MuninnMemory.ingest_sources feature-gated behavior."""

import pytest

from muninn.core.memory import MuninnMemory
from muninn.ingestion.models import IngestionChunk, IngestionReport, IngestionSourceResult


class _Flags:
    def require(self, flag_name: str):
        if flag_name != "multi_source_ingestion":
            raise RuntimeError("wrong flag")


class _Pipeline:
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
