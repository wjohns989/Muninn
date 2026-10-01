import sys
import os
import logging
import threading
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from .state import (
    _REAL_SESSION_STATE,
    _TRANSPORT_CLOSED,
    _RPC_WRITE_LOCK,
    _thread_local,
)
from .utils import (
    env_flag as _env_flag,
    get_host_safe_tool_call_budget_seconds
)
from .lifecycle import (
    ensure_server_running,
    check_and_start_ollama,
)
from .handlers import (
    handle_initialize as _handle_initialize,
    handle_list_tools as _handle_list_tools,
    handle_call_tool as _handle_call_tool,
    handle_call_tool_with_task as _handle_call_tool_with_task,
    handle_list_tasks as _handle_list_tasks,
    handle_get_task as _handle_get_task,
    handle_get_task_result as _handle_get_task_result,
    handle_cancel_task as _handle_cancel_task,
)
from .server import McpServer
from .protocol import negotiate_protocol_version as _negotiate_protocol_version
from ..core.security import initialize_security

logger = logging.getLogger("Muninn.mcp.main")
_SESSION_STATE = _REAL_SESSION_STATE
_INITIALIZE_TRACE_PATH = os.path.join(os.path.dirname(__file__), "mcp_initialize_trace.jsonl")
_TOOL_CALL_TRACE_PATH = os.path.join(os.path.dirname(__file__), "mcp_tool_call_trace.jsonl")

def _append_initialize_trace(event: str, payload: Dict[str, Any]) -> None:
    """Compatibility diagnostic; supplied payloads are never persisted."""
    safe_event = event if event in {"initialize_request", "initialize_result"} else "other"
    logger.debug("MCP initialize event: %s", safe_event)

def _append_tool_call_trace(event: str, payload: Dict[str, Any]) -> None:
    """Fixed diagnostic events only, without arguments, identifiers or paths."""
    safe_event = event if event in {"tools_call_dispatch", "main_loop_stdin_eof", "main_loop_exit"} else "other"
    logger.debug("MCP transport event: %s", safe_event)

def _legacy_send_result(mid, result):
    if isinstance(result, dict) and "protocolVersion" in result and "serverInfo" in result:
        _append_initialize_trace("initialize_result", {"id": mid, "result": result})
    send_json_rpc({"jsonrpc": "2.0", "id": mid, "result": result})

def _legacy_send_error(mid, code, message):
    send_json_rpc({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}})

def handle_initialize(msg_id: Any, params: Dict[str, Any]):
    if _env_flag("MUNINN_MCP_INCLUDE_STARTUP_WARNINGS_IN_INITIALIZE", False):
        warnings = _collect_startup_warnings()
    else:
        warnings = []
    _append_initialize_trace("initialize_request", {"id": msg_id, "params": params, "warnings": warnings})
    return _handle_initialize(msg_id, params, _legacy_send_error, _legacy_send_result, startup_warnings=warnings)

def send_json_rpc(message: Dict[str, Any]) -> None:
    _server.send_rpc(message)

def _send_json_rpc_error(msg_id: Any, code: int, message: str) -> None:
    send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})

_OPTIONAL_CAPABILITY_METHOD_RESULTS: Dict[str, Dict[str, Any]] = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
}

def _handle_optional_capability_method(msg_id: Any, method: str, params: Any) -> bool:
    if method in _OPTIONAL_CAPABILITY_METHOD_RESULTS:
        if msg_id is not None:
            send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": _OPTIONAL_CAPABILITY_METHOD_RESULTS[method]})
        return True
    if method == "resources/read":
        if msg_id is not None:
            send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {"contents": []}})
        return True
    if method == "prompts/get":
        if msg_id is not None:
            send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {"messages": []}})
        return True
    return False

def _dispatch_rpc_message(msg: Dict[str, Any]) -> None:
    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if method == "initialize":
        handle_initialize(msg_id, params)
    elif method == "notifications/initialized":
        if _SESSION_STATE.get("negotiated"):
            _SESSION_STATE["initialized"] = True
            logger.info("Client initialized connection")
    elif method == "tools/list":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized.")
            return
        _handle_list_tools(msg_id, _legacy_send_result)
    elif method == "tools/call":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized.")
            return
        task_request = params.get("task")
        # Trace every tools/call to diagnose EOF / async-task-path routing (see mcp_tool_call_trace.jsonl)
        _append_tool_call_trace("tools_call_dispatch", {
            "id": msg_id,
            "name": params.get("name"),
            "has_task_key": "task" in params,
            "task_value": task_request,
            "task_is_dict": isinstance(task_request, dict),
            "route": "async_task" if isinstance(task_request, dict) else "sync",
            "raw_params_keys": list(params.keys()) if isinstance(params, dict) else None,
        })
        if isinstance(task_request, dict):
            _handle_call_tool_with_task("stdio", msg_id, params.get("name"), params.get("arguments"), task_request, _legacy_send_result, send_notification_fn=send_json_rpc, worker_fn=_run_tool_call_task_worker)
        else:
            _handle_call_tool(msg_id, params, _legacy_send_error, _legacy_send_result)
    elif method == "tasks/list":
        _handle_list_tasks(msg_id, params, _legacy_send_error, _legacy_send_result)
    elif method == "tasks/get":
        _handle_get_task(msg_id, params, _legacy_send_error, _legacy_send_result)
    elif method == "tasks/cancel":
        _handle_cancel_task(msg_id, params, _legacy_send_error, _legacy_send_result, send_notification_fn=send_json_rpc)
    elif method == "tasks/result":
        _handle_get_task_result(msg_id, params, _legacy_send_error, _legacy_send_result)
    elif method in ("resources/list", "resources/templates/list", "resources/read", "prompts/list", "prompts/get"):
        _handle_optional_capability_method(msg_id, method, params)
    elif method == "ping":
        if msg_id: send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {}})
    else:
        if msg_id: _send_json_rpc_error(msg_id, -32601, f"Method not found: {method}")

def _run_tool_call_task_worker(session_id: str, task_id: str, name: str, arguments: Dict[str, Any], *args) -> None:
    from .handlers import _run_tool_call_task_worker as _internal_worker
    send_notif = args[0] if args else _legacy_send_result
    return _internal_worker(session_id, task_id, name, arguments, send_notif)

def _dispatch_rpc_message_guarded(msg: Dict[str, Any]) -> None:
    try:
        _dispatch_rpc_message(msg)
    except Exception as exc:
        logger.error("Internal error during dispatch: %s", type(exc).__name__)
        if msg.get("id") is not None and not _TRANSPORT_CLOSED.is_set():
            _send_json_rpc_error(msg["id"], -32603, "Internal error during request dispatch.")

_server = McpServer(dispatch_fn=_dispatch_rpc_message_guarded)

def _should_dispatch_in_background(msg: Dict[str, Any]) -> bool:
    method = msg.get("method", "")
    return method in ("tasks/result", "notifications/tasks/status")

def _collect_startup_warnings(autostart_server: bool = False, autostart_ollama: bool = False) -> list:
    from mcp_wrapper import _collect_startup_warnings as collect
    return collect(autostart_server, autostart_ollama)

def _bootstrap_dependencies_on_launch():
    from mcp_wrapper import _bootstrap_dependencies_on_launch as bootstrap
    return bootstrap()

def main():
    """Package execution uses the same transport as configured stdio clients."""
    from mcp_wrapper import main as run_canonical_bridge
    return run_canonical_bridge()

if __name__ == "__main__":
    main()
