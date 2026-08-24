import asyncio
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette.sse import EventSourceResponse

from muninn.core.security import is_security_enabled
from muninn.core.security import verify_token as core_verify_token
from muninn.mcp.handlers import (
    handle_call_tool as _handle_call_tool,
)
from muninn.mcp.handlers import (
    handle_call_tool_with_task as _handle_call_tool_with_task,
)
from muninn.mcp.handlers import (
    handle_cancel_task as _handle_cancel_task,
)
from muninn.mcp.handlers import (
    handle_get_task as _handle_get_task,
)
from muninn.mcp.handlers import (
    handle_get_task_result as _handle_get_task_result,
)
from muninn.mcp.handlers import (
    handle_initialize as _handle_initialize,
)
from muninn.mcp.handlers import (
    handle_list_tasks as _handle_list_tasks,
)
from muninn.mcp.handlers import (
    handle_list_tools as _handle_list_tools,
)
from muninn.mcp.state import (
    _SESSION_CONTEXTS,
    _SESSION_CONTEXTS_LOCK,
    _SESSION_STATE,
    _thread_local,
)

logger = logging.getLogger("Muninn.mcp.sse")


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def _session_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_SSE_MAX_SESSIONS", 128, minimum=1, maximum=4096)


def _queue_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_SSE_QUEUE_SIZE", 256, minimum=1, maximum=4096)


def _inflight_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_SSE_MAX_INFLIGHT", 8, minimum=1, maximum=128)


def _request_bytes_limit() -> int:
    return _bounded_int_env(
        "MUNINN_MCP_SSE_MAX_REQUEST_BYTES",
        1024 * 1024,
        minimum=1024,
        maximum=16 * 1024 * 1024,
    )


_sse_security = HTTPBearer(auto_error=False)


async def _verify_sse_credentials(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_sse_security),
) -> Optional[HTTPAuthorizationCredentials]:
    if is_security_enabled():
        token = credentials.credentials if credentials else None
        if not core_verify_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials


sse_router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
    dependencies=[Security(_verify_sse_credentials)],
)


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=_queue_limit())
        self.loop = asyncio.get_running_loop()
        self.inflight_tasks: set[asyncio.Task] = set()
        self.overflowed = False
        self.closed = False

    def _enqueue(self, message: Dict[str, Any]) -> None:
        if self.closed:
            return
        if self.queue.full():
            # A client that does not consume responses cannot retain process
            # memory indefinitely. Replace the oldest item with a terminal,
            # content-free overflow error and let the event stream close.
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.overflowed = True
            message = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": "SSE response queue capacity reached"},
            }
        self.queue.put_nowait(message)

    def send_rpc_threadsafe(self, message: Dict[str, Any]) -> None:
        if not self.closed:
            self.loop.call_soon_threadsafe(self._enqueue, message)


_active_sessions: Dict[str, Session] = {}
_active_sessions_lock = threading.RLock()


def _cleanup_session_context(session_id: str) -> None:
    with _SESSION_CONTEXTS_LOCK:
        _SESSION_CONTEXTS.pop(session_id, None)


def _create_session() -> Optional[str]:
    with _active_sessions_lock:
        if len(_active_sessions) >= _session_limit():
            return None
        session_id = str(uuid.uuid4())
        _active_sessions[session_id] = Session(session_id)
        return session_id


async def _close_session(session_id: str) -> bool:
    with _active_sessions_lock:
        session = _active_sessions.pop(session_id, None)
    _cleanup_session_context(session_id)
    if session is None:
        return False
    session.closed = True
    tasks = tuple(session.inflight_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    session.inflight_tasks.clear()
    return True


async def close_all_sse_sessions() -> int:
    """Cancel in-flight legacy transport work and release session state."""
    with _active_sessions_lock:
        session_ids = list(_active_sessions)
    closed = 0
    for session_id in session_ids:
        closed += int(await _close_session(session_id))
    if closed:
        logger.info("Closed %d MCP SSE sessions", closed)
    return closed


def sse_transport_status() -> Dict[str, int]:
    """Return content-free transport bounds and current utilization."""
    with _active_sessions_lock:
        active = len(_active_sessions)
        inflight = sum(len(session.inflight_tasks) for session in _active_sessions.values())
        queued = sum(session.queue.qsize() for session in _active_sessions.values())
    return {
        "active_sessions": active,
        "inflight_tasks": inflight,
        "queued_responses": queued,
        "max_sessions": _session_limit(),
        "queue_size": _queue_limit(),
        "max_inflight_per_session": _inflight_limit(),
        "max_request_bytes": _request_bytes_limit(),
    }


def send_result(session_id: str, msg_id: Any, result: Any):
    with _active_sessions_lock:
        session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe({"jsonrpc": "2.0", "id": msg_id, "result": result})


def send_error(session_id: str, msg_id: Any, code: int, message: str):
    with _active_sessions_lock:
        session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def send_notification(session_id: str, message: Dict[str, Any]):
    with _active_sessions_lock:
        session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe(message)


# Handlers wrappers to inject session_id
def handle_initialize(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_initialize(msg_id, params, s_err, s_res)


def handle_list_tools(session_id: str, msg_id: Any):
    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_list_tools(msg_id, s_res)


def handle_call_tool(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_call_tool(msg_id, params, s_err, s_res)


def handle_call_tool_with_task(
    session_id: str, msg_id: Any, name: str, args: Dict[str, Any], task_request: Dict[str, Any]
):
    def s_res(mid, r):
        send_result(session_id, mid, r)

    def s_notif(msg):
        send_notification(session_id, msg)

    return _handle_call_tool_with_task(
        session_id, msg_id, name, args, task_request, s_res, send_notification_fn=s_notif
    )


def handle_list_tasks(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_list_tasks(msg_id, params, s_err, s_res)


def handle_get_task(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_get_task(msg_id, params, s_err, s_res)


def handle_get_task_result(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    return _handle_get_task_result(msg_id, params, s_err, s_res)


def handle_cancel_task(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m):
        send_error(session_id, mid, c, m)

    def s_res(mid, r):
        send_result(session_id, mid, r)

    def s_notif(msg):
        send_notification(session_id, msg)

    return _handle_cancel_task(msg_id, params, s_err, s_res, send_notification_fn=s_notif)


OPTIONAL_CAPS = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
}


def _dispatch_sync(session_id: str, msg: Dict[str, Any]):
    # Crucial security and isolation mechanism: inject session_id into thread-local state.
    # state.py DynamicProxy reads this to map _SESSION_STATE lookup exclusively to this connection.
    _thread_local.mcp_session_id = session_id

    try:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "initialize":
            handle_initialize(session_id, msg_id, params)
        elif method == "notifications/initialized":
            if _SESSION_STATE.get("negotiated"):
                _SESSION_STATE["initialized"] = True
                logger.info("Client initialized connection for session %s", session_id)
        elif method == "tools/list":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_list_tools(session_id, msg_id)
        elif method == "tools/call":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            task_request = params.get("task")
            if isinstance(task_request, dict):
                handle_call_tool_with_task(
                    session_id, msg_id, params.get("name", ""), params.get("arguments", {}), task_request
                )
            else:
                handle_call_tool(session_id, msg_id, params)
        elif method == "tasks/list":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_list_tasks(session_id, msg_id, params)
        elif method == "tasks/get":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_get_task(session_id, msg_id, params)
        elif method == "tasks/cancel":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_cancel_task(session_id, msg_id, params)
        elif method == "tasks/result":
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_get_task_result(session_id, msg_id, params)
        elif method in OPTIONAL_CAPS:
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            if msg_id:
                send_result(session_id, msg_id, OPTIONAL_CAPS[method])
        elif method in ("resources/read", "prompts/get"):
            if not _SESSION_STATE.get("initialized"):
                if msg_id:
                    send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            if not isinstance(params, dict):
                if msg_id:
                    send_error(session_id, msg_id, -32602, f"{method} params must be an object")
                return
            if msg_id:
                res = {"contents": []} if method == "resources/read" else {"messages": []}
                send_result(session_id, msg_id, res)
        elif method == "ping":
            if msg_id:
                send_result(session_id, msg_id, {})
        else:
            if msg_id:
                send_error(session_id, msg_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        logger.exception("Error dispatching message: %s", exc)
        msg_id = msg.get("id")
        if msg_id:
            send_error(session_id, msg_id, -32603, f"Internal error during dispatch: {exc}")
    finally:
        if getattr(_thread_local, "mcp_session_id", None) == session_id:
            delattr(_thread_local, "mcp_session_id")


async def _dispatch_async(session_id: str, msg: Dict[str, Any]) -> None:
    await asyncio.to_thread(_dispatch_sync, session_id, msg)


def _schedule_dispatch(session_id: str, msg: Dict[str, Any]) -> bool:
    with _active_sessions_lock:
        session = _active_sessions.get(session_id)
        if session is None or session.closed or len(session.inflight_tasks) >= _inflight_limit():
            return False
        task = asyncio.create_task(_dispatch_async(session_id, msg))
        session.inflight_tasks.add(task)
        task.add_done_callback(session.inflight_tasks.discard)
        return True


@sse_router.get("/sse")
async def sse_endpoint(request: Request):
    """Establish an SSE connection."""
    session_id = _create_session()
    if session_id is None:
        raise HTTPException(status_code=503, detail="MCP SSE session capacity reached")

    # We must yield an endpoint URL to let MCP clients know where to POST messages
    post_uri = str(request.base_url).rstrip("/") + f"/mcp/messages?session_id={session_id}"
    logger.info("New MCP SSE connection: %s at %s", session_id, post_uri)

    async def event_generator():
        try:
            yield {"event": "endpoint", "data": post_uri}

            with _active_sessions_lock:
                session = _active_sessions[session_id]
            while True:
                if await request.is_disconnected():
                    break
                # Wait for the next message from the queue
                msg = await session.queue.get()
                yield {"event": "message", "data": json.dumps(msg)}
                if session.overflowed:
                    break
        finally:
            logger.info("Closed MCP SSE connection: %s", session_id)
            await _close_session(session_id)

    return EventSourceResponse(event_generator())


@sse_router.post("/messages")
async def messages_endpoint(request: Request, session_id: str):
    """Receive JSON-RPC messages from the client."""
    with _active_sessions_lock:
        session_exists = session_id in _active_sessions
    if not session_exists:
        raise HTTPException(status_code=404, detail="Session not found")

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _request_bytes_limit():
                raise HTTPException(status_code=413, detail="MCP request body too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")

    try:
        body_chunks: list[bytes] = []
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > _request_bytes_limit():
                raise HTTPException(status_code=413, detail="MCP request body too large")
            body_chunks.append(chunk)
        body = b"".join(body_chunks)
        msg = json.loads(body)
    except HTTPException:
        raise
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(msg, dict):
        raise HTTPException(status_code=400, detail="JSON-RPC message must be an object")

    # Dispatch in a bounded task set so slow tools cannot accumulate unbounded
    # request bodies, thread work, or task references.
    if not _schedule_dispatch(session_id, msg):
        raise HTTPException(status_code=429, detail="MCP SSE in-flight capacity reached")
    return "Accepted"
