import asyncio
import json
import logging
import uuid
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException
from sse_starlette.sse import EventSourceResponse

from muninn.mcp.handlers import (
    handle_initialize as _handle_initialize,
    handle_list_tools as _handle_list_tools,
    handle_call_tool as _handle_call_tool,
    handle_call_tool_with_task as _handle_call_tool_with_task,
    handle_list_tasks as _handle_list_tasks,
    handle_get_task as _handle_get_task,
    handle_get_task_result as _handle_get_task_result,
    handle_cancel_task as _handle_cancel_task,
)
from muninn.mcp.state import _SESSION_STATE, _thread_local

logger = logging.getLogger("Muninn.mcp.sse")

sse_router = APIRouter(prefix="/mcp", tags=["mcp"])

class Session:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()
        
    def send_rpc_threadsafe(self, message: Dict[str, Any]) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, message)

_active_sessions: Dict[str, Session] = {}

def send_result(session_id: str, msg_id: Any, result: Any):
    session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        })

def send_error(session_id: str, msg_id: Any, code: int, message: str):
    session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message}
        })

def send_notification(session_id: str, message: Dict[str, Any]):
    session = _active_sessions.get(session_id)
    if session:
        session.send_rpc_threadsafe(message)


# Handlers wrappers to inject session_id
def handle_initialize(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_initialize(msg_id, params, s_err, s_res)

def handle_list_tools(session_id: str, msg_id: Any):
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_list_tools(msg_id, s_res)

def handle_call_tool(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_call_tool(msg_id, params, s_err, s_res)

def handle_call_tool_with_task(session_id: str, msg_id: Any, name: str, args: Dict[str, Any], task_request: Dict[str, Any]):
    def s_res(mid, r): send_result(session_id, mid, r)
    def s_notif(msg): send_notification(session_id, msg)
    return _handle_call_tool_with_task(session_id, msg_id, name, args, task_request, s_res, send_notification_fn=s_notif)

def handle_list_tasks(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_list_tasks(msg_id, params, s_err, s_res)

def handle_get_task(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_get_task(msg_id, params, s_err, s_res)

def handle_get_task_result(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    return _handle_get_task_result(msg_id, params, s_err, s_res)

def handle_cancel_task(session_id: str, msg_id: Any, params: Dict[str, Any]):
    def s_err(mid, c, m): send_error(session_id, mid, c, m)
    def s_res(mid, r): send_result(session_id, mid, r)
    def s_notif(msg): send_notification(session_id, msg)
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
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_list_tools(session_id, msg_id)
        elif method == "tools/call":
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            task_request = params.get("task")
            if isinstance(task_request, dict):
                handle_call_tool_with_task(session_id, msg_id, params.get("name", ""), params.get("arguments", {}), task_request)
            else:
                handle_call_tool(session_id, msg_id, params)
        elif method == "tasks/list":
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_list_tasks(session_id, msg_id, params)
        elif method == "tasks/get":
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_get_task(session_id, msg_id, params)
        elif method == "tasks/cancel":
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_cancel_task(session_id, msg_id, params)
        elif method == "tasks/result":
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            handle_get_task_result(session_id, msg_id, params)
        elif method in OPTIONAL_CAPS:
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            if msg_id:
                send_result(session_id, msg_id, OPTIONAL_CAPS[method])
        elif method in ("resources/read", "prompts/get"):
            if not _SESSION_STATE.get("initialized"):
                if msg_id: send_error(session_id, msg_id, -32600, "Server not initialized")
                return
            if not isinstance(params, dict):
                if msg_id: send_error(session_id, msg_id, -32602, f"{method} params must be an object")
                return
            if msg_id:
                res = {"contents": []} if method == "resources/read" else {"messages": []}
                send_result(session_id, msg_id, res)
        elif method == "ping":
            if msg_id: send_result(session_id, msg_id, {})
        else:
            if msg_id: send_error(session_id, msg_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        logger.exception("Error dispatching message: %s", exc)
        msg_id = msg.get("id")
        if msg_id:
            send_error(session_id, msg_id, -32603, f"Internal error during dispatch: {exc}")


@sse_router.get("/sse")
async def sse_endpoint(request: Request):
    """Establish an SSE connection."""
    session_id = str(uuid.uuid4())
    _active_sessions[session_id] = Session()
    
    # We must yield an endpoint URL to let MCP clients know where to POST messages
    post_uri = str(request.base_url).rstrip("/") + f"/mcp/messages?session_id={session_id}"
    logger.info("New MCP SSE connection: %s at %s", session_id, post_uri)
    
    async def event_generator():
        try:
            yield {
                "event": "endpoint",
                "data": post_uri
            }
            
            session = _active_sessions[session_id]
            while True:
                if await request.is_disconnected():
                    break
                # Wait for the next message from the queue
                msg = await session.queue.get()
                yield {
                    "event": "message",
                    "data": json.dumps(msg)
                }
        finally:
            logger.info("Closed MCP SSE connection: %s", session_id)
            _active_sessions.pop(session_id, None)
            
            # Clean up the global isolated session state dictionary for this id
            from muninn.mcp.state import _SESSION_CONTEXTS, _SESSION_CONTEXTS_LOCK
            with _SESSION_CONTEXTS_LOCK:
                _SESSION_CONTEXTS.pop(session_id, None)

    return EventSourceResponse(event_generator())


@sse_router.post("/messages")
async def messages_endpoint(request: Request, session_id: str):
    """Receive JSON-RPC messages from the client."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        msg = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    # Dispatch in a thread so we don't block the ASGI loop handling other connections
    asyncio.create_task(asyncio.to_thread(_dispatch_sync, session_id, msg))
    return "Accepted"
