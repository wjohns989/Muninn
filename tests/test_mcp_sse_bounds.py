import asyncio

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from muninn.mcp import sse
from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK


@pytest_asyncio.fixture(autouse=True)
async def clear_sse_sessions():
    await sse.close_all_sse_sessions()
    yield
    await sse.close_all_sse_sessions()


@pytest.mark.asyncio
async def test_sse_queue_is_bounded_and_overflow_closes_session(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_SSE_QUEUE_SIZE", "2")
    session = sse.Session("synthetic-session")

    session._enqueue({"id": 1})
    session._enqueue({"id": 2})
    session._enqueue({"id": 3})

    assert session.queue.qsize() == 2
    assert session.overflowed is True
    queued = [session.queue.get_nowait(), session.queue.get_nowait()]
    assert queued[-1]["error"]["code"] == -32000


@pytest.mark.asyncio
async def test_sse_session_capacity_is_bounded(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_SSE_MAX_SESSIONS", "1")

    first = sse._create_session()
    second = sse._create_session()

    assert first is not None
    assert second is None
    assert sse.sse_transport_status()["active_sessions"] == 1


@pytest.mark.asyncio
async def test_sse_dispatch_tasks_are_bounded(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_SSE_MAX_INFLIGHT", "1")
    session_id = sse._create_session()
    session = sse._active_sessions[session_id]
    release = asyncio.Event()

    async def blocked_dispatch(_session_id, _message):
        await release.wait()

    monkeypatch.setattr(sse, "_dispatch_async", blocked_dispatch)

    assert sse._schedule_dispatch(session_id, {"id": 1}) is True
    await asyncio.sleep(0)
    assert sse._schedule_dispatch(session_id, {"id": 2}) is False
    release.set()
    await asyncio.gather(*tuple(session.inflight_tasks))


@pytest.mark.asyncio
async def test_sse_shutdown_cancels_tasks_and_clears_session_state():
    session_id = sse._create_session()
    session = sse._active_sessions[session_id]
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS[session_id] = {"initialized": True}
    session.inflight_tasks.add(asyncio.create_task(asyncio.sleep(60)))

    closed = await sse.close_all_sse_sessions()

    assert closed == 1
    assert not sse._active_sessions
    assert not session.inflight_tasks
    with _SESSION_CONTEXTS_LOCK:
        assert session_id not in _SESSION_CONTEXTS


@pytest.mark.asyncio
async def test_sse_chunked_request_body_stops_at_configured_bound(monkeypatch):
    monkeypatch.setenv("MUNINN_MCP_SSE_MAX_REQUEST_BYTES", "1024")
    session_id = sse._create_session()
    chunks = iter((b"x" * 700, b"y" * 700))

    async def receive():
        try:
            chunk = next(chunks)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)

    with pytest.raises(HTTPException) as exc_info:
        await sse.messages_endpoint(request, session_id)

    assert exc_info.value.status_code == 413
