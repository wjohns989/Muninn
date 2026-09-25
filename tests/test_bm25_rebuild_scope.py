"""Regression: BM25 rebuilt at startup must keep user/namespace scope and cover every live memory."""

from types import SimpleNamespace

import pytest

from muninn.core.memory import MuninnMemory
from muninn.core.types import MemoryRecord
from muninn.retrieval.bm25 import BM25Index
from muninn.store.sqlite_metadata import SQLiteMetadataStore


def test_rebuild_keeps_scopes_for_user_filtered_search():
    index = BM25Index()
    index.rebuild({"m1": "blue green rollback"}, {"m1": ("global_user", "global")})
    assert [doc for doc, _ in index.search("rollback", user_id="global_user")] == ["m1"]
    assert index.search("rollback", user_id="someone_else") == []


@pytest.mark.asyncio
async def test_startup_rebuild_indexes_all_live_memories_with_scope(tmp_path):
    store = SQLiteMetadataStore(tmp_path / "meta.db")
    for i in range(2500):
        store.add(MemoryRecord(id=f"m{i:05d}", content=f"note {i} about rollback",
                               namespace="global", metadata={"user_id": "global_user"}))
    store.update("m00007", archived=True)
    fake = SimpleNamespace(_metadata=store, _bm25=BM25Index())

    await MuninnMemory._rebuild_bm25(fake)

    assert fake._bm25.size == 2499
    hits = {doc for doc, _ in fake._bm25.search("rollback", limit=5000, user_id="global_user")}
    assert len(hits) == 2499 and "m00007" not in hits
