"""Tests for reindex, legacy import and legacy-store detection."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from muninn.core.maintenance import content_hash, import_memories, normalize_legacy_record, reindex
from muninn.core.memory import MuninnMemory
from muninn.core.types import MemoryRecord
from muninn.retrieval.bm25 import BM25Index
from muninn.store.sqlite_metadata import SQLiteMetadataStore
from muninn.store.vector_store import VectorStore

MEM0_RECORD = {
    "id": "8c1f-legacy",
    "memory": "Prefers blue-green deploys on Fridays",
    "hash": "abc",
    "metadata": {"category": "ops", "user_id": "ignored-here"},
    "created_at": "2024-05-01T10:00:00.123-07:00",
    "user_id": "wjohn",
}


def test_normalizes_mem0_export():
    item = normalize_legacy_record(MEM0_RECORD, source="mem0")
    assert item["content"] == "Prefers blue-green deploys on Fridays"
    assert item["user_id"] == "global_user" and item["scope"] == "global"
    assert item["created_at"] == pytest.approx(1714582800.123)
    assert item["metadata"] == {"category": "ops", "legacy_user_id": "wjohn",
                                "legacy_id": "8c1f-legacy", "import_source": "mem0"}


def test_normalize_handles_other_shapes():
    assert normalize_legacy_record({"memory": "   "}) is None
    ms = normalize_legacy_record({"text": "x", "timestamp": 1714582800123})
    assert ms["created_at"] == pytest.approx(1714582800.123)
    scoped = normalize_legacy_record({"content": "y", "project": "muninn"})
    assert scoped["scope"] == "project" and scoped["metadata"]["project"] == "muninn"


def _engine(tmp_path, with_vectors=False):
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    engine = SimpleNamespace(_metadata=store, _check_initialized=lambda: None, _bm25=BM25Index())
    if with_vectors:
        engine._vectors = VectorStore(tmp_path / "qdrant", embedding_dims=4)

        async def embed(text):
            return [0.1, 0.2, 0.3, 0.4]

        engine._embed = embed

        async def rebuild():
            await MuninnMemory._rebuild_bm25(engine)

        engine._rebuild_bm25 = rebuild
    return engine


@pytest.mark.asyncio
async def test_import_dry_run_then_apply_keeps_timestamps_and_skips_duplicates(tmp_path):
    engine = _engine(tmp_path)
    engine._metadata.add(MemoryRecord(id="existing", content="Already known fact"))

    async def fake_add(content, **kwargs):
        record_id = f"new-{content_hash(content)[:8]}"
        engine._metadata.add(MemoryRecord(id=record_id, content=content, metadata=kwargs["metadata"]))
        return {"id": record_id, "event": "ADD"}

    engine.add = AsyncMock(side_effect=fake_add)
    records = [MEM0_RECORD, {"memory": "already  known FACT"}, dict(MEM0_RECORD), {"memory": ""}]

    dry = await import_memories(engine, records, source="mem0")
    assert dry == {"dry_run": True, "read": 4, "invalid": 1, "duplicates": 2,
                   "imported": 1, "merged": 0, "skipped": 0}
    engine.add.assert_not_awaited()

    applied = await import_memories(engine, records, source="mem0", dry_run=False)
    assert applied["imported"] == 1 and applied["duplicates"] == 2
    kwargs = engine.add.await_args.kwargs
    assert kwargs["scope"] == "global" and kwargs["user_id"] == "global_user"
    stored = engine._metadata.get(f"new-{content_hash(MEM0_RECORD['memory'])[:8]}")
    assert stored.created_at == pytest.approx(1714582800.123)

    again = await import_memories(engine, records, source="mem0", dry_run=False)
    assert again["imported"] == 0 and again["duplicates"] == 3


@pytest.mark.asyncio
async def test_reindex_rebuilds_vectors_and_bm25_from_metadata(tmp_path):
    engine = _engine(tmp_path, with_vectors=True)
    try:
        for i in range(5):
            engine._metadata.add(MemoryRecord(id=f"m{i}", content=f"rollback note {i}",
                                              metadata={"user_id": "global_user"}))
        engine._metadata.update("m4", archived=True)

        dry = await reindex(engine)
        assert dry["dry_run"] is True and dry["live_memories"] == 4 and dry["vector_points_before"] == 0
        assert engine._vectors.count() == 0

        report = await reindex(engine, dry_run=False, recreate_vectors=True)
        assert report["vectors_embedded"] == 4 and report["vector_points_after"] == 4
        assert report["bm25_documents"] == 4
        hits = engine._vectors.search([0.1, 0.2, 0.3, 0.4], 10,
                                      filters={"user_id": "global_user", "archived": False})
        assert {mid for mid, _ in hits} == {"m0", "m1", "m2", "m3"}
        assert len(engine._bm25.search("rollback", user_id="global_user")) == 4
    finally:
        engine._vectors.close()


def test_cli_reads_jsonl_array_and_mem0_response(tmp_path):
    from muninn.cli import _read_export

    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text('{"memory": "a"}\n\n{"memory": "b"}\n', encoding="utf-8")
    array = tmp_path / "b.json"
    array.write_text(json.dumps([{"memory": "c"}]), encoding="utf-8")
    mem0 = tmp_path / "c.json"
    mem0.write_text(json.dumps({"results": [{"memory": "d"}, {"memory": "e"}]}), encoding="utf-8")

    assert [r["memory"] for r in _read_export(jsonl)] == ["a", "b"]
    assert [r["memory"] for r in _read_export(array)] == ["c"]
    assert [r["memory"] for r in _read_export(mem0)] == ["d", "e"]


def test_detect_legacy_stores_reports_booleans_only(tmp_path, monkeypatch):
    from muninn import platform

    monkeypatch.setattr(platform.Path, "home", classmethod(lambda cls: tmp_path))
    assert platform.detect_legacy_stores() == {"muninn_legacy_data_dir": False, "mem0_store": False}
    (tmp_path / ".muninn" / "data").mkdir(parents=True)
    (tmp_path / ".muninn" / "data" / "metadata.db").write_bytes(b"")
    (tmp_path / ".mem0").mkdir()
    assert platform.detect_legacy_stores() == {"muninn_legacy_data_dir": True, "mem0_store": True}
