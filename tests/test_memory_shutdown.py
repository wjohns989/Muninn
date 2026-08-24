from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from muninn.core.memory import MuninnMemory
from muninn.store.graph_store import GraphStore


def _uninitialized_memory() -> MuninnMemory:
    instance = object.__new__(MuninnMemory)
    instance._initialized = True
    instance._user_scope_migration_complete = True
    instance._feedback_multiplier_cache = OrderedDict({("u", "n", "p"): (0.0, {})})
    instance._consolidation = SimpleNamespace(stop=AsyncMock())
    instance._bm25 = MagicMock()
    for name in (
        "_metadata",
        "_vectors",
        "_graph",
        "_embed_model",
        "_reranker",
        "_conflict_detector",
    ):
        setattr(instance, name, MagicMock())
    for name in (
        "_conflict_resolver",
        "_colbert_indexer",
        "_extraction",
        "_retriever",
        "_scout",
        "_goal_compass",
        "_ingestion",
        "_chain_detector",
        "_dedup",
        "_ingestion_manager",
        "_temporal_kg",
        "_federation",
    ):
        setattr(instance, name, object())
    return instance


@pytest.mark.asyncio
async def test_shutdown_closes_stores_models_and_clears_references():
    memory = _uninitialized_memory()
    consolidation = memory._consolidation
    closable_names = (
        "_metadata",
        "_vectors",
        "_graph",
        "_embed_model",
        "_reranker",
        "_conflict_detector",
    )
    closables = {name: getattr(memory, name) for name in closable_names}
    bm25 = memory._bm25

    await memory.shutdown()

    consolidation.stop.assert_awaited_once()
    bm25.clear.assert_called_once()
    for resource in closables.values():
        resource.close.assert_called_once()
    for name in closable_names + (
        "_consolidation",
        "_bm25",
        "_retriever",
        "_scout",
        "_ingestion_manager",
        "_temporal_kg",
        "_federation",
    ):
        assert getattr(memory, name) is None
    assert memory._initialized is False
    assert memory._user_scope_migration_complete is False
    assert memory._feedback_multiplier_cache == {}

    await memory.shutdown()


@pytest.mark.asyncio
async def test_shutdown_continues_after_one_resource_close_fails(caplog):
    memory = _uninitialized_memory()
    memory._vectors.close.side_effect = RuntimeError("synthetic close failure")
    graph = memory._graph
    metadata = memory._metadata

    await memory.shutdown()

    graph.close.assert_called_once()
    metadata.close.assert_called_once()
    assert "shutdown cleanup failed" in caplog.text.lower()


def test_graph_close_explicitly_closes_all_owned_connections_before_database():
    graph = object.__new__(GraphStore)
    connection = MagicMock()
    worker_connection = MagicMock()
    database = MagicMock()
    graph._thread_local = SimpleNamespace(conn=connection)
    graph._connections_lock = __import__("threading").Lock()
    graph._connections = {id(connection): connection, id(worker_connection): worker_connection}
    graph._closed = False
    graph._db = database

    graph.close()

    connection.close.assert_called_once()
    worker_connection.close.assert_called_once()
    database.close.assert_called_once()
    assert graph._db is None
    assert graph._connections == {}
    assert graph._closed is True
    assert not hasattr(graph._thread_local, "conn")


@pytest.mark.asyncio
async def test_health_reports_content_free_resource_and_cache_bounds():
    memory = _uninitialized_memory()
    memory._metadata.count.return_value = 3
    memory._vectors.count.return_value = 3
    memory._graph.get_all_entities.return_value = ["entity"]
    memory._bm25.size = 3
    memory._reranker.is_available = True
    memory._conflict_detector.is_available = True
    memory._feedback_multiplier_cache[("u2", "n", "p")] = (0.0, {})
    memory.config = SimpleNamespace(
        embedding=SimpleNamespace(provider="fastembed"),
        retrieval_feedback=SimpleNamespace(cache_max_entries=64, cache_ttl_seconds=30),
        ingestion=SimpleNamespace(max_workers=2),
    )
    memory._consolidation.status = {"running": False, "integrity_resources_loaded": False}

    health = await memory.health()

    resources = health["runtime_resources"]
    assert resources["embedding"] == {"provider": "fastembed", "loaded": True}
    assert resources["reranker_loaded"] is True
    assert resources["conflict_detector_loaded"] is True
    assert resources["feedback_cache"] == {"entries": 2, "max_entries": 64, "ttl_seconds": 30}
    assert resources["ingestion_max_workers"] == 2
