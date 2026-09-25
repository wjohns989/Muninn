"""Bounded MCP Streamable HTTP transport for the shared Muninn server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, Response, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from muninn.core.security import is_security_enabled
from muninn.core.security import verify_token as core_verify_token
from muninn.mcp.definitions import SUPPORTED_PROTOCOL_VERSIONS
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

logger = logging.getLogger("Muninn.mcp.http")

MCP_SESSION_ID_HEADER = "Mcp-Session-Id"
MCP_PROTOCOL_VERSION_HEADER = "Mcp-Protocol-Version"
JSON_CONTENT_TYPE = "application/json"

OPTIONAL_CAPS = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
}


def _bounded_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def _session_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_HTTP_MAX_SESSIONS", 128, minimum=1, maximum=4096)


def _session_ttl_seconds() -> int:
    return _bounded_int_env("MUNINN_MCP_HTTP_SESSION_TTL_SEC", 3600, minimum=30, maximum=86400)


def _batch_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_HTTP_MAX_BATCH_SIZE", 100, minimum=1, maximum=1000)


def _request_bytes_limit() -> int:
    return _bounded_int_env(
        "MUNINN_MCP_HTTP_MAX_REQUEST_BYTES",
        1024 * 1024,
        minimum=1024,
        maximum=64 * 1024 * 1024,
    )


def _inflight_limit() -> int:
    return _bounded_int_env("MUNINN_MCP_HTTP_MAX_INFLIGHT", 64, minimum=1, maximum=4096)


def _session_inflight_limit() -> int:
    return _bounded_int_env(
        "MUNINN_MCP_HTTP_MAX_INFLIGHT_PER_SESSION", 8, minimum=1, maximum=256
    )


@dataclass
class HttpSession:
    created_at: float
    last_seen_at: float
    active_dispatches: int = 0
    dispatch_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_ACTIVE_HTTP_SESSIONS: Dict[str, HttpSession] = {}
_ACTIVE_HTTP_SESSIONS_LOCK = threading.Lock()
_ACTIVE_HTTP_DISPATCHES = 0


def _cleanup_session_contexts(session_ids: List[str]) -> None:
    if not session_ids:
        return
    with _SESSION_CONTEXTS_LOCK:
        for session_id in session_ids:
            _SESSION_CONTEXTS.pop(session_id, None)


def _purge_expired_locked(now: float) -> List[str]:
    cutoff = now - _session_ttl_seconds()
    expired = [
        session_id
        for session_id, session in _ACTIVE_HTTP_SESSIONS.items()
        if session.last_seen_at < cutoff and session.active_dispatches == 0
    ]
    for session_id in expired:
        _ACTIVE_HTTP_SESSIONS.pop(session_id, None)
    return expired


def _create_session() -> str | None:
    now = time.time()
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        expired = _purge_expired_locked(now)
        if len(_ACTIVE_HTTP_SESSIONS) >= _session_limit():
            session_id = None
        else:
            session_id = uuid.uuid4().hex
            _ACTIVE_HTTP_SESSIONS[session_id] = HttpSession(now, now)
    _cleanup_session_contexts(expired)
    return session_id


def _get_session(session_id: str, *, touch: bool = True) -> HttpSession | None:
    now = time.time()
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        expired = _purge_expired_locked(now)
        session = _ACTIVE_HTTP_SESSIONS.get(session_id)
        if session is not None and touch:
            session.last_seen_at = now
    _cleanup_session_contexts(expired)
    return session


def _destroy_session(session_id: str) -> None:
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        _ACTIVE_HTTP_SESSIONS.pop(session_id, None)
    _cleanup_session_contexts([session_id])


def _begin_dispatch(session_id: str) -> HttpSession | None:
    """Acquire bounded global and per-session dispatch capacity."""
    global _ACTIVE_HTTP_DISPATCHES
    now = time.time()
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        expired = _purge_expired_locked(now)
        session = _ACTIVE_HTTP_SESSIONS.get(session_id)
        if (
            session is None
            or session.active_dispatches >= _session_inflight_limit()
            or _ACTIVE_HTTP_DISPATCHES >= _inflight_limit()
        ):
            lease = None
        else:
            session.last_seen_at = now
            session.active_dispatches += 1
            _ACTIVE_HTTP_DISPATCHES += 1
            lease = session
    _cleanup_session_contexts(expired)
    return lease


def _finish_dispatch(session: HttpSession) -> None:
    """Release capacity acquired by :func:`_begin_dispatch`."""
    global _ACTIVE_HTTP_DISPATCHES
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        if session.active_dispatches > 0:
            session.active_dispatches -= 1
            _ACTIVE_HTTP_DISPATCHES = max(0, _ACTIVE_HTTP_DISPATCHES - 1)


def close_all_http_sessions() -> int:
    """Release all HTTP transport and session-state references on shutdown."""
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        session_ids = list(_ACTIVE_HTTP_SESSIONS)
        _ACTIVE_HTTP_SESSIONS.clear()
    _cleanup_session_contexts(session_ids)
    if session_ids:
        logger.info("Closed %d MCP HTTP sessions", len(session_ids))
    return len(session_ids)


def http_transport_status() -> Dict[str, int]:
    """Return content-free transport bounds and current session count."""
    with _ACTIVE_HTTP_SESSIONS_LOCK:
        count = len(_ACTIVE_HTTP_SESSIONS)
        active_dispatches = _ACTIVE_HTTP_DISPATCHES
    return {
        "active_sessions": count,
        "max_sessions": _session_limit(),
        "session_ttl_seconds": _session_ttl_seconds(),
        "max_batch_size": _batch_limit(),
        "max_request_bytes": _request_bytes_limit(),
        "active_dispatches": active_dispatches,
        "max_inflight": _inflight_limit(),
        "max_inflight_per_session": _session_inflight_limit(),
    }


def _json_error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _json_response(
    status_code: int,
    payload: Any = None,
    *,
    session_id: str | None = None,
    extra_headers: Dict[str, str] | None = None,
) -> Response:
    headers: Dict[str, str] = {}
    if session_id:
        headers[MCP_SESSION_ID_HEADER] = session_id
    if extra_headers:
        headers.update(extra_headers)
    if payload is None:
        return Response(status_code=status_code, headers=headers)
    return Response(
        content=json.dumps(payload),
        status_code=status_code,
        headers=headers,
        media_type=JSON_CONTENT_TYPE,
    )


def _accepts_json(request: Request) -> bool:
    accept = request.headers.get("accept", "*/*")
    media_types = [part.strip().lower() for part in accept.split(",")]
    return "*/*" in media_types or any(part.startswith(JSON_CONTENT_TYPE) for part in media_types)


def _has_json_content_type(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip().lower() == JSON_CONTENT_TYPE


def _request_has_id(msg: Dict[str, Any]) -> bool:
    return isinstance(msg.get("method"), str) and msg.get("id") is not None


def _notification_only(msg: Dict[str, Any]) -> bool:
    return isinstance(msg.get("method"), str) and msg.get("id") is None


def _dispatch_http_message(session_id: str, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    session = _get_session(session_id)
    if session is None:
        return [_json_error(msg.get("id"), -32001, "Session not found")]
    responses: List[Dict[str, Any]] = []

    def send_result(msg_id: Any, result: Any) -> None:
        if msg_id is not None:
            responses.append({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def send_error(msg_id: Any, code: int, message: str) -> None:
        if msg_id is not None:
            responses.append(_json_error(msg_id, code, message))

    def send_notification(_message: Dict[str, Any]) -> None:
        # JSON-only Streamable HTTP has no server event stream. Do not retain or
        # log notification payloads because tool output may contain memory text.
        return

    with session.dispatch_lock:
        _thread_local.mcp_session_id = session_id
        try:
            msg_id = msg.get("id")
            method = msg.get("method")
            params = msg.get("params", {})
            if method == "initialize":
                _handle_initialize(msg_id, params, send_error, send_result, startup_warnings=[])
            elif method == "notifications/initialized":
                if _SESSION_STATE.get("negotiated"):
                    _SESSION_STATE["initialized"] = True
            elif method == "tools/list":
                if not _SESSION_STATE.get("initialized"):
                    send_error(msg_id, -32600, "Server not initialized")
                else:
                    _handle_list_tools(msg_id, send_result)
            elif method == "tools/call":
                if not _SESSION_STATE.get("initialized"):
                    send_error(msg_id, -32600, "Server not initialized")
                elif isinstance(params.get("task"), dict):
                    _handle_call_tool_with_task(
                        session_id,
                        msg_id,
                        params.get("name", ""),
                        params.get("arguments", {}),
                        params["task"],
                        send_result,
                        send_notification_fn=send_notification,
                    )
                else:
                    _handle_call_tool(msg_id, params, send_error, send_result)
            elif method == "tasks/list":
                _handle_list_tasks(msg_id, params, send_error, send_result)
            elif method == "tasks/get":
                _handle_get_task(msg_id, params, send_error, send_result)
            elif method == "tasks/cancel":
                _handle_cancel_task(
                    msg_id,
                    params,
                    send_error,
                    send_result,
                    send_notification_fn=send_notification,
                )
            elif method == "tasks/result":
                _handle_get_task_result(msg_id, params, send_error, send_result)
            elif method in OPTIONAL_CAPS:
                send_result(msg_id, OPTIONAL_CAPS[method])
            elif method in ("resources/read", "prompts/get"):
                if not isinstance(params, dict):
                    send_error(msg_id, -32602, f"{method} params must be an object")
                else:
                    send_result(
                        msg_id,
                        {"contents": []} if method == "resources/read" else {"messages": []},
                    )
            elif method == "ping":
                send_result(msg_id, {})
            elif msg_id is not None:
                send_error(msg_id, -32601, f"Method not found: {method}")
        except Exception as exc:
            logger.exception("HTTP MCP dispatch failed for session %s: %s", session_id, exc)
            if msg.get("id") is not None:
                send_error(msg.get("id"), -32603, "Internal error during dispatch")
        finally:
            if getattr(_thread_local, "mcp_session_id", None) == session_id:
                delattr(_thread_local, "mcp_session_id")
    return responses


async def _handle_post(request: Request) -> Response:
    if not _accepts_json(request):
        return _json_response(
            HTTPStatus.NOT_ACCEPTABLE,
            _json_error("server-error", -32600, "Not Acceptable: Client must accept application/json"),
        )
    if not _has_json_content_type(request):
        return _json_response(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            _json_error("server-error", -32600, "Content-Type must be application/json"),
        )
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _request_bytes_limit():
                return _json_response(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    _json_error("server-error", -32600, "MCP request body too large"),
                )
        except ValueError:
            return _json_response(
                HTTPStatus.BAD_REQUEST,
                _json_error("server-error", -32600, "Invalid Content-Length"),
            )
    body_chunks: list[bytes] = []
    body_size = 0
    async for chunk in request.stream():
        body_size += len(chunk)
        if body_size > _request_bytes_limit():
            return _json_response(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                _json_error("server-error", -32600, "MCP request body too large"),
            )
        body_chunks.append(chunk)
    body = b"".join(body_chunks)
    try:
        raw_body = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return _json_response(
            HTTPStatus.BAD_REQUEST, _json_error("server-error", -32700, "Parse error: Invalid JSON")
        )

    is_batch = isinstance(raw_body, list)
    messages = raw_body if is_batch else [raw_body]
    if (
        not messages
        or len(messages) > _batch_limit()
        or not all(isinstance(msg, dict) for msg in messages)
    ):
        return _json_response(
            HTTPStatus.BAD_REQUEST,
            _json_error("server-error", -32600, "Invalid or oversized JSON-RPC batch"),
        )

    initialize_messages = [msg for msg in messages if msg.get("method") == "initialize"]
    if len(initialize_messages) > 1 or (initialize_messages and len(messages) > 1):
        return _json_response(
            HTTPStatus.BAD_REQUEST,
            _json_error("server-error", -32600, "Only one initialization request is allowed"),
        )
    if not all(_request_has_id(msg) or _notification_only(msg) for msg in messages):
        return _json_response(
            HTTPStatus.BAD_REQUEST,
            _json_error("server-error", -32600, "Unsupported JSON-RPC envelope"),
        )

    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    is_initialize = bool(initialize_messages)
    if is_initialize:
        if session_id:
            code = HTTPStatus.BAD_REQUEST if _get_session(session_id) else HTTPStatus.NOT_FOUND
            return _json_response(code, _json_error("server-error", -32600, "Invalid session"))
        session_id = _create_session()
        if session_id is None:
            return _json_response(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _json_error("server-error", -32000, "MCP session capacity reached"),
            )
    else:
        if not session_id:
            return _json_response(
                HTTPStatus.BAD_REQUEST,
                _json_error("server-error", -32600, "Mcp-Session-Id header is required"),
            )
        if _get_session(session_id) is None:
            return _json_response(
                HTTPStatus.NOT_FOUND, _json_error("server-error", -32001, "Session not found")
            )
        protocol_version = request.headers.get(MCP_PROTOCOL_VERSION_HEADER)
        if protocol_version and protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            return _json_response(
                HTTPStatus.BAD_REQUEST,
                _json_error("server-error", -32600, "Unsupported protocol version"),
                session_id=session_id,
            )

    dispatch_lease = _begin_dispatch(session_id)
    if dispatch_lease is None:
        if is_initialize:
            _destroy_session(session_id)
        return _json_response(
            HTTPStatus.TOO_MANY_REQUESTS,
            _json_error("server-error", -32000, "MCP dispatch capacity reached"),
            session_id=session_id,
        )

    responses: List[Dict[str, Any]] = []
    try:
        for msg in messages:
            responses.extend(await asyncio.to_thread(_dispatch_http_message, session_id, msg))
    finally:
        _finish_dispatch(dispatch_lease)
    if not any(_request_has_id(msg) for msg in messages):
        return _json_response(HTTPStatus.ACCEPTED, session_id=session_id)
    if not responses:
        if is_initialize:
            _destroy_session(session_id)
        return _json_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            _json_error("server-error", -32603, "Internal error: No response generated"),
            session_id=session_id,
        )
    if is_initialize and "error" in responses[0]:
        _destroy_session(session_id)
    return _json_response(HTTPStatus.OK, responses if is_batch else responses[0], session_id=session_id)


async def _handle_delete(request: Request) -> Response:
    session_id = request.headers.get(MCP_SESSION_ID_HEADER)
    if not session_id:
        return _json_response(
            HTTPStatus.BAD_REQUEST,
            _json_error("server-error", -32600, "Mcp-Session-Id header is required"),
        )
    if _get_session(session_id) is None:
        return _json_response(
            HTTPStatus.NOT_FOUND, _json_error("server-error", -32001, "Session not found")
        )
    _destroy_session(session_id)
    return _json_response(HTTPStatus.OK, session_id=session_id)


_http_security = HTTPBearer(auto_error=False)


async def _verify_http_credentials(
    credentials: HTTPAuthorizationCredentials | None = Security(_http_security),
) -> HTTPAuthorizationCredentials | None:
    if is_security_enabled():
        token = credentials.credentials if credentials else None
        if not core_verify_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials


streamable_http_router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
    dependencies=[Security(_verify_http_credentials)],
)


@streamable_http_router.api_route("", methods=["GET", "POST", "DELETE"])
@streamable_http_router.api_route("/", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def streamable_http_endpoint(request: Request) -> Response:
    if request.method == "POST":
        return await _handle_post(request)
    if request.method == "DELETE":
        return await _handle_delete(request)
    return _json_response(
        HTTPStatus.METHOD_NOT_ALLOWED,
        _json_error("server-error", -32600, "Method Not Allowed"),
        session_id=request.headers.get(MCP_SESSION_ID_HEADER),
        extra_headers={"Allow": "GET, POST, DELETE"},
    )
