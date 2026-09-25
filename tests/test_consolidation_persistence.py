"""Consolidation phases must persist their results to the real metadata store."""

import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from muninn.consolidation.daemon import ConsolidationDaemon
from muninn.core.types import MemoryRecord, MemoryType
from muninn.store.sqlite_metadata import SQLiteMetadataStore


def _daemon(tmp_path):
    config = MagicMock()
    config.decay_threshold = -1.0
    config.working_memory_ttl_hours = 100000
    graph = MagicMock()
    graph.get_memory_node_degrees_batch.side_effect = lambda ids: {i: 0.0 for i in ids}
    return ConsolidationDaemon(
        config=config,
        metadata=SQLiteMetadataStore(tmp_path / "meta.db"),
        vectors=MagicMock(),
        graph=graph,
        bm25=MagicMock(),
    )


def _record(record_id, content, **kwargs):
    kwargs.setdefault("metadata", {"user_id": "u1"})
    return MemoryRecord(id=record_id, content=content, vector_id=record_id, **kwargs)


@pytest.mark.asyncio
async def test_decay_persists_recalculated_importance(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.metadata.add(_record("m1", "stale fact", importance=0.99, created_at=time.time() - 90 * 86400))

    await daemon._phase_decay()

    assert daemon.metadata.get("m1").importance < 0.99


@pytest.mark.asyncio
async def test_promote_persists_new_memory_type(tmp_path):
    daemon = _daemon(tmp_path)
    daemon.metadata.add(_record("m1", "frequently used fact", access_count=50))

    result = await daemon._phase_promote()

    assert result["promoted"] == 1
    stored = daemon.metadata.get("m1")
    assert stored.memory_type == MemoryType.SEMANTIC
    assert stored.consolidated is True


@pytest.mark.parametrize(
    "primary_importance, secondary_importance",
    [(0.9, 0.2), (0.2, 0.9)],
    ids=["primary-survives", "secondary-survives"],
)
@pytest.mark.asyncio
async def test_merge_persists_survivor_and_archives_absorbed(
    tmp_path, monkeypatch, primary_importance, secondary_importance
):
    daemon = _daemon(tmp_path)
    daemon._embed_fn = lambda text: [0.1, 0.2, 0.3, 0.4]
    daemon.metadata.add(_record("a", "deploys run on Fridays", importance=primary_importance))
    daemon.metadata.add(_record("b", "rollbacks use blue-green", importance=secondary_importance))
    monkeypatch.setattr(
        "muninn.consolidation.daemon.find_merge_candidates",
        AsyncMock(return_value=[("a", "b", 0.95)]),
    )

    result = await daemon._phase_merge()

    assert result["merged"] == 1
    survivor_id, absorbed_id = ("a", "b") if primary_importance > secondary_importance else ("b", "a")
    survivor = daemon.metadata.get(survivor_id)
    assert survivor is not None
    assert "deploys run on Fridays" in survivor.content
    assert "rollbacks use blue-green" in survivor.content
    assert survivor.consolidated is True
    assert survivor.archived is False
    absorbed = daemon.metadata.get(absorbed_id)
    assert absorbed.archived is True
    assert absorbed.parent_id == survivor_id
    assert absorbed.metadata["archived_reason"] == "merged"
    daemon.vectors.delete.assert_called_once_with([absorbed_id])
    daemon.bm25.remove.assert_called_once_with(absorbed_id)
    # The survivor's rewritten content is searchable by keyword and vector.
    bm25_id, bm25_text = daemon.bm25.add.call_args.args[:2]
    assert bm25_id == survivor_id and "rollbacks use blue-green" in bm25_text and "Fridays" in bm25_text
    daemon.vectors.update_vector.assert_called_once_with(survivor_id, [0.1, 0.2, 0.3, 0.4])
    daemon.vectors.set_payload.assert_called_once_with(survivor_id, {"content": survivor.content[:500]})


@pytest.mark.asyncio
async def test_retrieval_feedback_slows_decay_after_consolidation(tmp_path):
    from muninn.scoring.elo import INITIAL_ELO, calculate_elo_update

    daemon = _daemon(tmp_path)
    created = time.time() - 21 * 86400
    for memory_id in ("helpful", "ignored"):
        daemon.metadata.add(_record(memory_id, f"{memory_id} fact", importance=0.5, created_at=created))

    elo = INITIAL_ELO
    for _ in range(5):
        daemon.metadata.add_retrieval_feedback(
            user_id="u1", namespace="global", project="global", query_text="q",
            memory_id="helpful", outcome=1.0, rank=1, sampling_prob=None,
            signals={}, source="test",
        )
        elo = calculate_elo_update(elo, 1.0)
    daemon.metadata.update_elo_rating("helpful", elo)

    await daemon._phase_decay()

    assert daemon.metadata.get("helpful").importance > daemon.metadata.get("ignored").importance


def _config(**overrides):
    from muninn.core.config import ConsolidationConfig

    return ConsolidationConfig(**overrides)


def _real_daemon(tmp_path, **config):
    graph = MagicMock()
    graph.get_memory_node_degrees_batch.side_effect = lambda ids: {i: 0.0 for i in ids}
    return ConsolidationDaemon(
        config=_config(**config),
        metadata=SQLiteMetadataStore(tmp_path / "meta.db"),
        vectors=MagicMock(),
        graph=graph,
        bm25=MagicMock(),
        images_dir=tmp_path / "images",
    )


@pytest.mark.asyncio
async def test_dry_run_proposes_changes_without_writing(tmp_path):
    daemon = _real_daemon(tmp_path, dry_run=True, decay_threshold=0.99)
    daemon.metadata.add(_record("m1", "any fact", importance=0.5))

    result = await daemon._phase_decay()

    stored = daemon.metadata.get("m1")
    assert result["decayed"] == 1
    assert stored.importance == 0.5 and stored.archived is False
    daemon.vectors.delete.assert_not_called()
    actions = daemon.status["proposed_actions"]
    assert {"phase": "decay", "action": "archive", "memory_id": "m1",
            "reason": "below_decay_threshold", "parent_id": None} in actions
    assert daemon.metadata.get_meta("consolidation_cursor:decay") is None


@pytest.mark.asyncio
async def test_decay_pages_through_every_memory(tmp_path):
    daemon = _real_daemon(tmp_path, batch_size=100)
    for i in range(250):
        daemon.metadata.add(_record(f"m{i:03d}", f"fact {i}", importance=0.999))

    await daemon._phase_decay()
    await daemon._phase_decay()
    untouched = [r for r in daemon.metadata.get_all(limit=1000) if r.importance == 0.999]
    assert len(untouched) == 50

    await daemon._phase_decay()
    assert all(r.importance != 0.999 for r in daemon.metadata.get_all(limit=1000))
    assert daemon.metadata.get_meta("consolidation_cursor:decay") == ""


@pytest.mark.asyncio
async def test_decay_archives_redundant_stale_memory_using_stored_novelty(tmp_path):
    from muninn.core.types import Provenance

    daemon = _real_daemon(tmp_path)
    old = time.time() - 3650 * 86400
    daemon.metadata.add(_record("dup", "restated fact", provenance=Provenance.INGESTED,
                                novelty_score=0.02, created_at=old))
    daemon.metadata.add(_record("unique", "distinct fact", provenance=Provenance.INGESTED,
                                novelty_score=1.0, created_at=old))

    result = await daemon._phase_decay()

    assert result["decayed"] == 1
    assert daemon.metadata.get("dup").archived is True
    assert daemon.metadata.get("unique").archived is False


@pytest.mark.asyncio
async def test_expired_working_memory_is_removed_from_every_store(tmp_path):
    daemon = _real_daemon(tmp_path, working_memory_ttl_hours=1)
    images = tmp_path / "images"
    images.mkdir()
    (images / "abc.png").write_bytes(b"png")
    daemon.metadata.add(_record(
        "w1", "scratch", memory_type=MemoryType.WORKING, created_at=time.time() - 7200,
        metadata={"user_id": "u1", "image_stored_name": "abc.png"},
    ))

    result = await daemon._phase_decay()

    assert result["expired"] == 1
    assert daemon.metadata.get("w1") is None
    daemon.bm25.remove.assert_called_once_with("w1")
    daemon.graph.delete_memory_references.assert_called_once_with("w1")
    assert not (images / "abc.png").exists()


@pytest.mark.asyncio
async def test_restore_reindexes_archived_memory(tmp_path):
    from types import SimpleNamespace

    from muninn.core.memory import MuninnMemory

    store = SQLiteMetadataStore(tmp_path / "meta.db")
    store.add(_record("a1", "archived fact"))
    store.update("a1", archived=True,
                 metadata={"user_id": "u1", "archived_reason": "merged", "archived_at": 1.0})
    fake = SimpleNamespace(_metadata=store, _check_initialized=lambda: None, update=AsyncMock())

    result = await MuninnMemory.restore(fake, "a1")

    assert result == {"id": "a1", "restored": True, "event": "RESTORE"}
    kwargs = fake.update.await_args.kwargs
    assert kwargs["data"] == "archived fact" and kwargs["archived"] is False
    assert "archived_reason" not in kwargs["metadata"] and "restored_at" in kwargs["metadata"]
    assert await MuninnMemory.restore(fake, "missing") == {"error": "Memory missing not found"}


@pytest.mark.asyncio
async def test_replay_reembeds_live_memories_with_real_signature(tmp_path):
    from unittest.mock import create_autospec

    from muninn.store.vector_store import VectorStore

    daemon = _real_daemon(tmp_path)
    daemon.vectors = create_autospec(VectorStore, instance=True)
    daemon._embed_fn = lambda text: [0.1, 0.2, 0.3, 0.4]
    daemon.metadata.add(_record("hot", "important fact", importance=0.9))
    daemon.metadata.add(_record("gone", "archived fact", importance=0.95))
    daemon.metadata.update("gone", archived=True)

    result = await daemon._phase_replay()

    assert result["re_embedded"] == 1
    daemon.vectors.update_vector.assert_called_once_with("hot", [0.1, 0.2, 0.3, 0.4])


def test_update_vector_preserves_scope_payload(tmp_path):
    from muninn.store.vector_store import VectorStore

    vs = VectorStore(tmp_path / "vectors", embedding_dims=4)
    try:
        vs.upsert("m1", [1.0, 0.0, 0.0, 0.0], {"user_id": "u1", "project": "p1", "scope": "project"})
        vs.update_vector("m1", [0.0, 1.0, 0.0, 0.0])

        hits = vs.search([0.0, 1.0, 0.0, 0.0], 5, filters={"user_id": "u1", "project": "p1"})
        assert hits and hits[0][0] == "m1" and hits[0][1] > 0.99
    finally:
        vs._get_client().close()


@pytest.mark.asyncio
async def test_replay_reembeds_live_memories_with_real_signature(tmp_path):
    from unittest.mock import create_autospec

    from muninn.store.vector_store import VectorStore

    daemon = _real_daemon(tmp_path)
    daemon.vectors = create_autospec(VectorStore, instance=True)
    daemon._embed_fn = lambda text: [0.1, 0.2, 0.3, 0.4]
    daemon.metadata.add(_record("hot", "important fact", importance=0.9))
    daemon.metadata.add(_record("gone", "archived fact", importance=0.95))
    daemon.metadata.update("gone", archived=True)

    result = await daemon._phase_replay()

    assert result["re_embedded"] == 1
    daemon.vectors.update_vector.assert_called_once_with("hot", [0.1, 0.2, 0.3, 0.4])


def test_update_vector_preserves_scope_payload(tmp_path):
    from muninn.store.vector_store import VectorStore

    vs = VectorStore(tmp_path / "vectors", embedding_dims=4)
    try:
        vs.upsert("m1", [1.0, 0.0, 0.0, 0.0], {"user_id": "u1", "project": "p1", "scope": "project"})
        vs.update_vector("m1", [0.0, 1.0, 0.0, 0.0])

        hits = vs.search([0.0, 1.0, 0.0, 0.0], 5, filters={"user_id": "u1", "project": "p1"})
        assert hits and hits[0][0] == "m1" and hits[0][1] > 0.99
    finally:
        vs._get_client().close()


def test_set_payload_merges_fields(tmp_path):
    from muninn.store.vector_store import VectorStore

    vs = VectorStore(tmp_path / "vectors", embedding_dims=4)
    try:
        vs.upsert("m1", [1.0, 0.0, 0.0, 0.0], {"user_id": "u1", "content": "old"})
        vs.set_payload("m1", {"content": "new"})
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "m1"))
        point = vs._get_client().retrieve(vs.collection_name, ids=[point_id])[0]
        assert point.payload["content"] == "new" and point.payload["user_id"] == "u1"
    finally:
        vs._get_client().close()
