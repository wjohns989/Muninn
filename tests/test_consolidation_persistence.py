"""Consolidation phases must persist their results to the real metadata store."""

import time
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
async def test_merge_persists_survivor_and_deletes_only_absorbed(
    tmp_path, monkeypatch, primary_importance, secondary_importance
):
    daemon = _daemon(tmp_path)
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
    assert daemon.metadata.get(absorbed_id) is None


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
