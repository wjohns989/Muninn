"""
Muninn MCP Wrapper - Facade for the muninn.mcp modular package.
Maintains backward compatibility for tests and legacy orchestrators.
"""

import sys
import os
import logging
import threading
from typing import Any, Dict, Optional, List
import requests

# Global state for legacy test monkeypatching. 
# These are kept in sync with muninn.mcp via Dynamic State Resolvers in state.py.
from muninn.mcp.state import (
    _REAL_SESSION_STATE,
    _TRANSPORT_CLOSED,
    _BACKEND_CIRCUIT_LOCK,
    _BACKEND_CIRCUIT_STATE,
    _RPC_WRITE_LOCK,
    _thread_local,
    _DISPATCH_EXECUTOR_LOCK,
    is_backend_circuit_open
)

_SESSION_STATE = _REAL_SESSION_STATE
_TASKS = _SESSION_STATE["tasks"]
_TASKS_LOCK = threading.RLock()
_TASKS_CONDITION = threading.Condition(_TASKS_LOCK)

# Legacy naming for some constants
from muninn.mcp.utils import (
    get_git_info,
    inject_operator_profile_metadata as _inject_operator_profile_metadata,
    env_flag as _env_flag,
    format_tool_result_text as _format_tool_result_text,
    truncate_tool_text as _truncate_tool_text,
    public_tool_error_message as _public_tool_error_message,
    _get_tool_call_warn_ms,
    _get_tool_response_preview_max_string_chars,
    _safe_json_dumps,
    _truncate_preview_string,
    _compact_tool_response_payload,
    get_host_safe_tool_call_budget_seconds
)

def _record_tool_call_response_metrics(message: Dict[str, Any], serialized: str) -> None:
    """Legacy metrics recorder for protocol tests."""
    metrics = getattr(_thread_local, "tool_call_metrics", None)
    if not isinstance(metrics, dict):
        return
    if metrics.get("msg_id") != message.get("id"):
        return
    payload_size_bytes = len(serialized.encode("utf-8")) + 1
    metrics["response_count"] = int(metrics.get("response_count", 0)) + 1
    metrics["response_bytes_total"] = int(metrics.get("response_bytes_total", 0)) + payload_size_bytes
    metrics["response_bytes_max"] = max(int(metrics.get("response_bytes_max", 0)), payload_size_bytes)
    if isinstance(message.get("error"), dict):
        metrics["saw_error"] = True
from muninn.mcp.requests import (
    make_request_with_retry,
    get_remaining_deadline_seconds as _remaining_deadline_seconds
)
from muninn.mcp.lifecycle import (
    ensure_server_running,
    check_and_start_ollama,
    is_circuit_open as _backend_circuit_open
)
from muninn.mcp.definitions import (
    JSON_SCHEMA_2020_12,
    SUPPORTED_PROTOCOL_VERSIONS,
    SUPPORTED_MODEL_PROFILES,
    TOOLS_SCHEMAS,
    READ_ONLY_TOOLS,
    DESTRUCTIVE_TOOLS,
    IDEMPOTENT_TOOLS
)
from muninn.mcp.tasks import (
    get_registry as _task_registry,
    utc_now_iso as _utc_now_iso,
    task_now_epoch as _task_now_epoch,
    sanitize_task_ttl_ms as _sanitize_task_ttl_ms,
    get_task_worker_start_delay_ms as _get_task_worker_start_delay_ms,
    public_task as _public_task,
    encode_task_cursor as _encode_task_cursor,
    decode_task_cursor as _decode_task_cursor,
    set_task_state_locked as _set_task_state_locked,
    purge_and_retain_tasks_locked as _purge_and_retain_tasks_locked,
    lookup_task_locked as _lookup_task_locked,
    emit_task_status_notification as _modular_emit_task_status_notification,
    TASK_POLL_INTERVAL_MS as _TASK_POLL_INTERVAL_MS,
    TASKS_DEFAULT_TTL_MS as _TASKS_DEFAULT_TTL_MS,
    TASKS_MAX_TTL_MS as _TASKS_MAX_TTL_MS,
    TASKS_LIST_PAGE_SIZE as _TASKS_LIST_PAGE_SIZE,
    TASKS_MAX_RETAINED as _TASKS_MAX_RETAINED,
    TASK_TERMINAL_STATUSES as _TASK_TERMINAL_STATUSES,
    get_task_result_mode as _get_task_result_mode
)

def _emit_task_status_notification(task_snapshot: Dict[str, Any]) -> None:
    """Legacy 1-arg facade for emitting task status notifications."""
    return _modular_emit_task_status_notification(task_snapshot, send_json_rpc)
from muninn.mcp.handlers import (
    handle_initialize as _handle_initialize,
    handle_list_tools as _handle_list_tools,
    handle_call_tool as _handle_call_tool,
    handle_call_tool_with_task as _handle_call_tool_with_task,
    handle_list_tasks as _handle_list_tasks,
    handle_get_task as _handle_get_task,
    handle_get_task_result as _handle_get_task_result,
    handle_cancel_task as _handle_cancel_task,
    get_tool_call_deadline_epoch as _get_tool_call_deadline_epoch,
    startup_recovery_allowed as _startup_recovery_allowed
)
from muninn.mcp.server import McpServer

# Logging configuration (test-transparent)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Muninn.mcp_wrapper")

# Internal constants for compatibility
_TASKS_DEFAULT_TTL_MS = 600000
_TASKS_MAX_TTL_MS = 86400000
_TASKS_LIST_PAGE_SIZE = 50
_TASKS_MAX_RETAINED = 500
_TASK_CURSOR_PREFIX = "task_offset_"
_TASK_TERMINAL_STATUSES = {"cancelled", "completed", "failed"}
_BACKEND_CIRCUIT_FAILURE_THRESHOLD = 5

# Exceptions for tests
from muninn.mcp.lifecycle import BackendCircuitOpenError as _BackendCircuitOpenError
from muninn.mcp.requests import _RequestDeadlineExceededError

# Facade Functions (Legacy Signature Support)

def _negotiate_protocol_version(requested: Optional[str]) -> Optional[str]:
    """Return requested protocol version only when explicitly supported."""
    if requested and requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    if requested is None:
        return SUPPORTED_PROTOCOL_VERSIONS[0]
    return None

def _extract_client_elicitation_modes(capabilities: Any) -> tuple:
    """Read declared client elicitation modes with 2025-11-25 defaults."""
    elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
    if not isinstance(elicitation, dict):
        return tuple()
    if not elicitation:
        return ("form",)
    modes = []
    if isinstance(elicitation.get("form"), dict):
        modes.append("form")
    if isinstance(elicitation.get("url"), dict):
        modes.append("url")
    return tuple(modes)

def _read_operator_model_profile(env_var: str) -> Optional[str]:
    profile = os.environ.get(env_var, "").strip()
    if not profile:
        return None
    if profile in SUPPORTED_MODEL_PROFILES:
        return profile
    logger.warning(
        "Ignoring unsupported %s='%s'; expected one of %s",
        env_var,
        profile,
        SUPPORTED_MODEL_PROFILES,
    )
    return None

from muninn.version import __version__

def _build_initialize_instructions(startup_warnings: Optional[List[str]] = None) -> str:
    base_instructions = (
        "Muninn MCP server. Set project goals, store/search memories, and use handoff tools "
        "for cross-assistant continuity."
    )
    session_profile = _read_operator_model_profile("MUNINN_OPERATOR_MODEL_PROFILE")
    if session_profile:
        base_instructions = (
            f"{base_instructions}\n\nSession model profile: {session_profile} "
            "(from MUNINN_OPERATOR_MODEL_PROFILE)."
        )
    if not startup_warnings:
        return base_instructions
    bullet_list = "\n".join(f"- {warning}" for warning in startup_warnings)
    return f"{base_instructions}\n\nStartup checks:\n{bullet_list}"

# Adapter Functions for Modular Handlers
def _legacy_send_result(mid, result):
    send_json_rpc({"jsonrpc": "2.0", "id": mid, "result": result})

def _legacy_send_error(mid, code, message):
    send_json_rpc({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}})

def handle_initialize(msg_id: Any, params: Dict[str, Any]):
    # Use the module-local _collect_startup_warnings so tests can monkeypatch it
    warnings = _collect_startup_warnings()
    return _handle_initialize(msg_id, params, _legacy_send_error, _legacy_send_result, startup_warnings=warnings)

def handle_list_tools(msg_id: Any):
    return _handle_list_tools(msg_id, _legacy_send_result)

def handle_call_tool(msg_id: Any, params: Dict[str, Any]):
    return _handle_call_tool(msg_id, params, _legacy_send_error, _legacy_send_result)

def handle_list_tasks(msg_id: Any, params: Dict[str, Any]):
    return _handle_list_tasks(msg_id, params, _legacy_send_error, _legacy_send_result)

def handle_get_task(msg_id: Any, params: Dict[str, Any]):
    return _handle_get_task(msg_id, params, _legacy_send_error, _legacy_send_result)

def handle_get_task_result(msg_id: Any, params: Dict[str, Any]):
    """Facade for tasks/result with blocking support."""
    return _handle_get_task_result(msg_id, params, _legacy_send_error, _legacy_send_result)

def handle_cancel_task(msg_id: Any, params: Dict[str, Any]):
    return _handle_cancel_task(msg_id, params, _legacy_send_error, _legacy_send_result, send_notification_fn=send_json_rpc)

def handle_call_tool_with_task(msg_id: Any, name: str, args: Dict[str, Any], task_request: Dict[str, Any]):
    """Facade for task-backed tool calls."""
    return _handle_call_tool_with_task(
        msg_id, name, args, task_request, _legacy_send_result, 
        send_notification_fn=send_json_rpc,
        worker_fn=_run_tool_call_task_worker
    )

def _run_tool_call_task_worker(task_id: str, name: str, arguments: Dict[str, Any], *args) -> None:
    """Facade for background worker, allowing test monkeypatching."""
    from muninn.mcp.handlers import _run_tool_call_task_worker as _internal_worker
    # If 4th arg is provided, pass it. Otherwise use _legacy_send_result
    send_notif = args[0] if args else _legacy_send_result
    return _internal_worker(task_id, name, arguments, send_notif)

def send_json_rpc(message: Dict[str, Any]) -> None:
    _server.send_rpc(message)

def _send_json_rpc_error(msg_id: Any, code: int, message: str) -> None:
    send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})

def _read_rpc_message(stream) -> Optional[Dict[str, Any]]:
    return _server.read_message(stream)

def _should_dispatch_in_background(msg: Dict[str, Any]) -> bool:
    method = msg.get("method", "")
    if method in ("tasks/result", "notifications/tasks/status"):
        return True
    if method == "tools/call" and _env_flag("MUNINN_MCP_BACKGROUND_TOOLS_CALL", False):
        return True
    return False

_OPTIONAL_CAPABILITY_METHOD_RESULTS: Dict[str, Dict[str, Any]] = {
    "resources/list": {"resources": []},
    "resources/templates/list": {"resourceTemplates": []},
    "prompts/list": {"prompts": []},
}


def _handle_optional_capability_method(msg_id: Any, method: str, params: Any) -> bool:
    """
    Handle optional MCP methods with graceful fallbacks.
    This avoids hard method-not-found failures for clients that probe
    optional capability surfaces unconditionally.
    """
    if method in _OPTIONAL_CAPABILITY_METHOD_RESULTS:
        if msg_id is not None:
            send_json_rpc(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": _OPTIONAL_CAPABILITY_METHOD_RESULTS[method],
                }
            )
        return True

    if method == "resources/read":
        if not isinstance(params, dict):
            if msg_id is not None:
                _send_json_rpc_error(msg_id, -32602, "resources/read params must be an object.")
            return True
        if msg_id is not None:
            send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {"contents": []}})
        return True

    if method == "prompts/get":
        if not isinstance(params, dict):
            if msg_id is not None:
                _send_json_rpc_error(msg_id, -32602, "prompts/get params must be an object.")
            return True
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
        else:
            logger.warning("Ignored notifications/initialized before successful initialize")
    elif method == "tools/list":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        handle_list_tools(msg_id)
    elif method == "tools/call":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        task_request = params.get("task")
        if isinstance(task_request, dict):
            handle_call_tool_with_task(msg_id, params.get("name"), params.get("arguments"), task_request)
        else:
            handle_call_tool(msg_id, params)
    elif method == "tasks/list":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        handle_list_tasks(msg_id, params)
    elif method == "tasks/get":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        handle_get_task(msg_id, params)
    elif method == "tasks/cancel":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        handle_cancel_task(msg_id, params)
    elif method == "tasks/result":
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        handle_get_task_result(msg_id, params)
    elif method in ("resources/list", "resources/templates/list", "resources/read", "prompts/list", "prompts/get"):
        if not _SESSION_STATE.get("initialized"):
            if msg_id: _send_json_rpc_error(msg_id, -32600, "Server not initialized. Send initialize then notifications/initialized.")
            return
        _handle_optional_capability_method(msg_id, method, params)
    elif method == "ping":
        if msg_id: send_json_rpc({"jsonrpc": "2.0", "id": msg_id, "result": {}})
    else:
        if msg_id: _send_json_rpc_error(msg_id, -32601, f"Method not found: {method}")

def _dispatch_rpc_message_guarded(msg: Dict[str, Any]) -> None:
    try:
        _dispatch_rpc_message(msg)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        logger.exception("Internal error during dispatch: %s", exc)
        if msg.get("id") and not _TRANSPORT_CLOSED.is_set():
            _send_json_rpc_error(msg["id"], -32603, f"Internal error during request dispatch: {exc}")

# Instantiate Singleton Server
_server = McpServer(dispatch_fn=_dispatch_rpc_message_guarded)

# Legacy naming for executor/semaphore for tests
def _get_dispatch_executor():
    return _server.get_executor()

_DISPATCH_EXECUTOR = _server.get_executor()
_DISPATCH_QUEUE_SEMAPHORE = _server._queue_semaphore

def _submit_background_dispatch(msg: Dict[str, Any]) -> None:
    if not _server.submit_dispatch(msg):
        if msg.get("id"):
            _send_json_rpc_error(msg["id"], -32001, "Server busy: dispatch queue is saturated.")

def _collect_startup_warnings(autostart_server=True, autostart_ollama=True) -> list:
    warnings = []
    if autostart_server and not ensure_server_running():
        warnings.append("Muninn server is not reachable")
    if autostart_ollama and not check_and_start_ollama():
        warnings.append("Ollama is not reachable")
    return warnings

def _bootstrap_dependencies_on_launch():
    if _env_flag("MUNINN_MCP_AUTOSTART_ON_LAUNCH", True):
        srv = _env_flag("MUNINN_MCP_AUTOSTART_SERVER", True)
        olm = _env_flag("MUNINN_MCP_AUTOSTART_OLLAMA", False)
        _collect_startup_warnings(srv, olm)

def _get_tool_call_deadline_seconds() -> Optional[float]:
    # Logic matching utils.py get_tool_call_deadline_epoch but returning seconds duration
    # Priority 1: Explicit duration
    raw = os.environ.get("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC")
    if raw == "0": return None
    
    safe_budget = get_host_safe_tool_call_budget_seconds()
    
    if raw:
        try:
            explicit = float(raw)
            if not _env_flag("MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN", False):
                explicit = min(explicit, safe_budget)
            return explicit
        except ValueError:
            pass
            
    # Priority 2: Derived
    return safe_budget

def _get_task_result_max_wait_seconds() -> Optional[float]:
    raw = os.environ.get("MUNINN_MCP_TASK_RESULT_MAX_WAIT_SEC")
    if raw == "0": return None
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return get_host_safe_tool_call_budget_seconds()

def main():
    from muninn.core.security import initialize_security
    initialize_security()
    logger.info("Muninn MCP Modular Wrapper starting...")
    threading.Thread(target=_bootstrap_dependencies_on_launch, daemon=True).start()
    
    try:
        while not _TRANSPORT_CLOSED.is_set():
            msg = _server.read_message(sys.stdin.buffer)
            if msg is None: break
            if _should_dispatch_in_background(msg):
                _submit_background_dispatch(msg)
            else:
                _dispatch_rpc_message_guarded(msg)
    finally:
        _server.stop()

if __name__ == "__main__":
    main()
