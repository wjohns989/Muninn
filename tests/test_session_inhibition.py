"""Tests for CoALA-style session inhibition in hybrid retrieval."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muninn.core.types import MemoryRecord
from muninn.retrieval.hybrid import HybridRetriever
from muninn.retrieval.session_inhibition import SessionInhibitor


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_seen_items_are_demoted_by_rank_penalty():
    inhibitor = SessionInhibitor(rank_penalty=2)
    inhibitor.record("s1", ["a"])
    assert inhibitor.rerank("s1", ["a", "b", "c", "d"], key=str) == ["b", "c", "a", "d"]


def test_unseen_session_and_zero_penalty_leave_order_unchanged():
    inhibitor = SessionInhibitor(rank_penalty=0)
    inhibitor.record("s1", ["a"])
    assert inhibitor.rerank("s1", ["a", "b"], key=str) == ["a", "b"]
    assert SessionInhibitor().rerank("other", ["a", "b"], key=str) == ["a", "b"]


def test_sessions_are_isolated():
    inhibitor = SessionInhibitor(rank_penalty=5)
    inhibitor.record("s1", ["a"])
    assert inhibitor.rerank("s2", ["a", "b"], key=str) == ["a", "b"]


def test_entries_expire_after_ttl():
    clock = _Clock()
    inhibitor = SessionInhibitor(ttl_seconds=60, clock=clock)
    inhibitor.record("s1", ["a"])
    clock.now += 30
    inhibitor.record("s1", ["b"])
    clock.now += 45
    assert inhibitor.seen("s1") == {"b"}


def test_state_is_bounded():
    inhibitor = SessionInhibitor(max_sessions=2, max_ids_per_session=3)
    inhibitor.record("s1", ["a", "b", "c", "d"])
    assert inhibitor.seen("s1") == {"b", "c", "d"}
    inhibitor.record("s2", ["x"])
    inhibitor.record("s3", ["y"])
    assert len(inhibitor) == 2
    assert inhibitor.seen("s1") == set()


def test_from_env_can_disable(monkeypatch):
    monkeypatch.setenv("MUNINN_SESSION_INHIBITION", "0")
    assert SessionInhibitor.from_env() is None
    monkeypatch.setenv("MUNINN_SESSION_INHIBITION", "1")
    monkeypatch.setenv("MUNINN_SESSION_INHIBITION_RANK_PENALTY", "7")
    assert SessionInhibitor.from_env().rank_penalty == 7


def _retriever(records, inhibitor):
    metadata = MagicMock()
    metadata.get_by_ids.side_effect = lambda ids: [r for r in records if r.id in ids]
    metadata.get_all.return_value = []
    vector = MagicMock()
    # Descending relevance: m0 is the best match.
    vector.search.return_value = [(r.id, 1.0 - i * 0.05) for i, r in enumerate(records)]
    graph = MagicMock()
    graph.find_related_memories.return_value = []
    bm25 = MagicMock()
    bm25.search.return_value = []
    return HybridRetriever(
        metadata_store=metadata,
        vector_store=vector,
        graph_store=graph,
        bm25_index=bm25,
        embed_fn=AsyncMock(return_value=[0.1] * 8),
        session_inhibitor=inhibitor,
    )


@pytest.mark.asyncio
async def test_repeat_search_keeps_strongest_repeat_and_surfaces_fresh_memories():
    records = [MemoryRecord(id=f"m{i}", content=f"memory {i}", importance=0.5) for i in range(6)]
    retriever = _retriever(records, SessionInhibitor(rank_penalty=3))

    first = await retriever.search("q", limit=3, rerank=False, session_id="s1")
    second = await retriever.search("q", limit=3, rerank=False, session_id="s1")

    assert [r.memory.id for r in first] == ["m0", "m1", "m2"]
    # m0 moves from position 0 to 3 and ties behind unseen m3; weaker repeats drop out.
    assert [r.memory.id for r in second] == ["m3", "m0", "m4"]


@pytest.mark.asyncio
async def test_large_penalty_fully_refreshes_results():
    records = [MemoryRecord(id=f"m{i}", content=f"memory {i}", importance=0.5) for i in range(6)]
    retriever = _retriever(records, SessionInhibitor(rank_penalty=10))

    await retriever.search("q", limit=3, rerank=False, session_id="s1")
    second = await retriever.search("q", limit=3, rerank=False, session_id="s1")

    assert [r.memory.id for r in second] == ["m3", "m4", "m5"]


@pytest.mark.asyncio
async def test_search_without_session_is_unchanged():
    records = [MemoryRecord(id=f"m{i}", content=f"memory {i}", importance=0.5) for i in range(6)]
    retriever = _retriever(records, SessionInhibitor(rank_penalty=3))

    first = await retriever.search("q", limit=3, rerank=False)
    second = await retriever.search("q", limit=3, rerank=False)

    assert [r.memory.id for r in first] == [r.memory.id for r in second] == ["m0", "m1", "m2"]


def test_mcp_search_forwards_a_per_process_session_for_stdio(monkeypatch):
    from muninn.mcp import handlers

    captured = {}
    response = MagicMock()
    response.json.return_value = {"success": True, "data": [{"id": "m1"}]}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs["json"])
        return response

    monkeypatch.setattr(handlers, "make_request_with_retry", fake_request)
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "p", "branch": "b"})

    handlers._do_search_memory({"query": "q"}, None)

    assert captured["session_id"] == handlers._STDIO_SEARCH_SESSION_ID
    assert captured["session_id"].startswith("stdio-")


def test_mcp_search_forwards_http_session_id(monkeypatch):
    from muninn.mcp import handlers
    from muninn.mcp.state import _thread_local

    captured = {}
    response = MagicMock()
    response.json.return_value = {"success": True, "data": [{"id": "m1"}]}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs["json"])
        return response

    monkeypatch.setattr(handlers, "make_request_with_retry", fake_request)
    monkeypatch.setattr(handlers, "get_git_info", lambda: {"project": "p", "branch": "b"})
    _thread_local.mcp_session_id = "http-session-1"
    try:
        handlers._do_search_memory({"query": "q"}, None)
    finally:
        del _thread_local.mcp_session_id

    assert captured["session_id"] == "http-session-1"


def test_health_reports_content_free_inhibition_utilization():
    from types import SimpleNamespace

    from muninn.core.memory import MuninnMemory

    inhibitor = SessionInhibitor(max_sessions=4)
    inhibitor.record("s1", ["secret-memory-id"])
    fake = SimpleNamespace(_retriever=SimpleNamespace(_session_inhibitor=inhibitor))

    status = MuninnMemory._session_inhibition_status(fake)

    assert status == {
        "enabled": True,
        "sessions": 1,
        "max_sessions": 4,
        "max_ids_per_session": 200,
        "ttl_seconds": 1800.0,
        "rank_penalty": 3,
    }
    assert MuninnMemory._session_inhibition_status(SimpleNamespace(_retriever=None)) == {"enabled": False}
