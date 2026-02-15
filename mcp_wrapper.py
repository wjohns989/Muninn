#!/usr/bin/env python3
"""
Muninn MCP Wrapper
------------------
Acts as a bridge between MCP clients (Claude Desktop, etc.) and the Muninn Memory Server.
Auto-starts the server if it's not running.
"""

import sys
import os
import time
import json
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO, Callable
from muninn.version import __version__

# Configure logging to file since stdout is used for MCP protocol
GLOBAL_MEMORY_DIR = Path(__file__).parent.resolve()
LOG_FILE = GLOBAL_MEMORY_DIR / "mcp_wrapper.log"

import functools

SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2024-11-05")
JSON_SCHEMA_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_MODEL_PROFILES = ("low_latency", "balanced", "high_reasoning")
_SESSION_STATE = {
    "negotiated": False,
    "initialized": False,
    "protocol_version": SUPPORTED_PROTOCOL_VERSIONS[0],
    "client_capabilities": {},
    "client_elicitation_modes": tuple(),
    "tasks": {},
}
_TASK_TERMINAL_STATUSES = {"cancelled", "completed", "failed"}
_TASKS_LOCK = threading.RLock()
_TASKS_CONDITION = threading.Condition(_TASKS_LOCK)
_TASKS_DEFAULT_TTL_MS = max(1, int(os.environ.get("MUNINN_MCP_TASK_TTL_MS", "600000")))
_TASKS_MAX_TTL_MS = max(_TASKS_DEFAULT_TTL_MS, int(os.environ.get("MUNINN_MCP_TASK_MAX_TTL_MS", "86400000")))
_TASKS_LIST_PAGE_SIZE = max(1, int(os.environ.get("MUNINN_MCP_TASKS_LIST_PAGE_SIZE", "50")))
_TASKS_MAX_RETAINED = max(_TASKS_LIST_PAGE_SIZE, int(os.environ.get("MUNINN_MCP_TASKS_MAX_RETAINED", "500")))
_thread_local = threading.local()
_RPC_WRITE_LOCK = threading.Lock()
_DISPATCH_MAX_WORKERS = max(1, int(os.environ.get("MUNINN_MCP_DISPATCH_MAX_WORKERS", "8")))
_DISPATCH_EXECUTOR: Optional[ThreadPoolExecutor] = None
_DISPATCH_EXECUTOR_LOCK = threading.Lock()

def get_git_info() -> Dict[str, str]:
    """Get project name and git branch in real-time."""
    info = {"project": "global", "branch": "none"}
    cwd = os.getcwd()
    try:
        # Find git root to avoid issues in subdirectories
        root_res = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=1)
        if root_res.returncode == 0:
            git_root = root_res.stdout.strip()
            info["project"] = os.path.basename(git_root)
            
            # Get branch
            branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=1)
            if branch_res.returncode == 0:
                info["branch"] = branch_res.stdout.strip()
        else:
            # Fallback to CWD basename if not a git repo
            info["project"] = os.path.basename(cwd)
    except Exception:
        info["project"] = os.path.basename(cwd)
    return info

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(LOG_FILE),
    filemode='a'
)
logger = logging.getLogger("Muninn")

SERVER_SCRIPT = GLOBAL_MEMORY_DIR / "server.py"

# Critical: Ensure we use the neutral data directory env var if needed, 
# though server.py now hardcodes ~/.muninn/data. 
# We maintain the relative path for the script execution.
SERVER_URL = os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069")
HEALTH_URL = f"{SERVER_URL}/health"
OLLAMA_URL = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434")


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


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


def _get_operator_model_profile_for_operation(operation: str) -> Optional[str]:
    env_map = {
        "add": "MUNINN_OPERATOR_RUNTIME_MODEL_PROFILE",
        "ingest": "MUNINN_OPERATOR_INGESTION_MODEL_PROFILE",
        "legacy_ingest": "MUNINN_OPERATOR_LEGACY_INGESTION_MODEL_PROFILE",
    }
    operation_env = env_map.get(operation)
    if operation_env:
        scoped = _read_operator_model_profile(operation_env)
        if scoped:
            return scoped
    return _read_operator_model_profile("MUNINN_OPERATOR_MODEL_PROFILE")


def _inject_operator_profile_metadata(
    metadata: Optional[Dict[str, Any]],
    operation: str = "add",
) -> Dict[str, Any]:
    scoped = dict(metadata or {})
    session_profile = _get_operator_model_profile_for_operation(operation)
    if session_profile and "operator_model_profile" not in scoped:
        scoped["operator_model_profile"] = session_profile
    return scoped

def make_request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Make HTTP request with exponential backoff retry and server auto-restart."""
    max_retries = 3
    base_delay = 0.5
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            logger.warning(f"Connection failed (attempt {attempt+1}/{max_retries}): {e}")
            
            # If server might be down, ensure it's running
            if attempt < max_retries - 1:
                ensure_server_running()
                time.sleep(base_delay * (2 ** attempt))
            
    if last_error:
        raise last_error
    raise requests.RequestException("Unknown connection error after retries")

def is_server_running() -> bool:
    """Check if the Muninn server is running."""
    try:
        response = requests.get(HEALTH_URL, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False


def is_ollama_running() -> bool:
    """Check if Ollama is reachable."""
    try:
        response = requests.get(OLLAMA_URL, timeout=0.5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_and_start_ollama():
    """Check if Ollama is running, and start it if not."""
    from muninn.platform import spawn_detached_process, find_ollama_executable

    if is_ollama_running():
        logger.info("Ollama is already running (responsive).")
        return True
    logger.warning("Ollama is not responding. Attempting to start...")

    try:
        ollama_path = find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found on this system.")
            return False

        spawn_detached_process([ollama_path, "serve"])

        # Wait for Ollama to become responsive
        for _ in range(20):  # 10 seconds wait
            if is_ollama_running():
                logger.info("Ollama started successfully.")
                return True
            time.sleep(0.5)

        logger.error("Timed out waiting for Ollama to start.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch Ollama: {e}")
        return False

def start_server():
    """Start the Muninn server in a detached process (cross-platform)."""
    from muninn.platform import spawn_detached_process, find_python_executable

    logger.info("Starting Muninn server...")

    python_executable = find_python_executable()

    try:
        spawn_detached_process(
            [python_executable, str(SERVER_SCRIPT)],
            cwd=str(GLOBAL_MEMORY_DIR),
        )
        # Don't block waiting for full startup here.
        # The first tool call will trigger a retry loop if needed.
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def ensure_server_running():
    """Ensure server is up, starting it if necessary."""
    if is_server_running():
        return True
    if not start_server():
        return False
    for _ in range(20):
        if is_server_running():
            return True
        time.sleep(0.25)
    return False


def _collect_startup_warnings(
    autostart_server: Optional[bool] = None,
    autostart_ollama: Optional[bool] = None,
) -> List[str]:
    """
    Validate startup dependencies for assistant/IDE sessions.

    On initialize, we either auto-start missing dependencies or return explicit
    startup prompts so the user can fix the environment immediately.
    """
    warnings: List[str] = []
    autostart_server_enabled = (
        _env_flag("MUNINN_MCP_AUTOSTART_SERVER", True)
        if autostart_server is None
        else autostart_server
    )
    autostart_ollama_enabled = (
        _env_flag("MUNINN_MCP_AUTOSTART_OLLAMA", True)
        if autostart_ollama is None
        else autostart_ollama
    )

    server_ok = ensure_server_running() if autostart_server_enabled else is_server_running()
    if not server_ok:
        warnings.append(
            f"Muninn server is not reachable at {SERVER_URL}. "
            f"Start it with: python {SERVER_SCRIPT.name}"
        )

    ollama_ok = check_and_start_ollama() if autostart_ollama_enabled else is_ollama_running()
    if not ollama_ok:
        warnings.append(
            f"Ollama is not reachable at {OLLAMA_URL}. "
            "Start it with: ollama serve"
        )

    return warnings


def _bootstrap_dependencies_on_launch() -> None:
    """
    Best-effort dependency bootstrap when the MCP wrapper process starts.

    This runs independently from initialize-time checks so users who enable
    autostart do not have to wait for first MCP tool usage to trigger startup.
    """
    if not _env_flag("MUNINN_MCP_AUTOSTART_ON_LAUNCH", True):
        logger.info("Launch-time dependency bootstrap disabled by MUNINN_MCP_AUTOSTART_ON_LAUNCH.")
        return

    autostart_server_enabled = _env_flag("MUNINN_MCP_AUTOSTART_SERVER", True)
    autostart_ollama_enabled = _env_flag("MUNINN_MCP_AUTOSTART_OLLAMA", True)

    if autostart_server_enabled:
        if ensure_server_running():
            logger.info("Launch-time bootstrap: Muninn server is ready.")
        else:
            logger.warning("Launch-time bootstrap: Muninn server is not reachable.")

    if autostart_ollama_enabled:
        if check_and_start_ollama():
            logger.info("Launch-time bootstrap: Ollama is ready.")
        else:
            logger.warning("Launch-time bootstrap: Ollama is not reachable.")


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

# --- MCP Protocol Implementation ---

def _read_rpc_message(stream: BinaryIO) -> Optional[Dict[str, Any]]:
    """
    Read one inbound JSON-RPC message from stdin.

    Supports both:
    - newline-delimited JSON (legacy/simple stdio clients)
    - Content-Length framed JSON-RPC (LSP/MCP-style clients)
    """
    while True:
        first_line = stream.readline()
        if not first_line:
            return None
        if not first_line.strip():
            continue

        lowered = first_line.lower()
        if lowered.startswith(b"content-length:"):
            try:
                content_length = int(first_line.split(b":", 1)[1].strip())
            except Exception:
                logger.warning("Invalid Content-Length header: %r", first_line)
                return None

            # Consume remaining headers until blank line.
            while True:
                header_line = stream.readline()
                if not header_line:
                    return None
                if header_line in (b"\r\n", b"\n"):
                    break

            payload = stream.read(content_length)
            if not payload:
                return None
            try:
                msg = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid framed JSON payload")
                return None
            if isinstance(msg, dict):
                return msg
            return None

        try:
            msg = json.loads(first_line.decode("utf-8"))
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON line on stdio transport")
            continue
        if isinstance(msg, dict):
            return msg
        return None


def send_json_rpc(message: Dict[str, Any]):
    """Send JSON-RPC message to stdout."""
    emitter = getattr(_thread_local, "rpc_emitter", None)
    if callable(emitter):
        emitter(message)
        return
    with _RPC_WRITE_LOCK:
        print(json.dumps(message))
        sys.stdout.flush()


def _send_json_rpc_error(msg_id: Any, code: int, message: str):
    """Send a JSON-RPC error response."""
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": code,
            "message": message,
        },
    })

def _negotiate_protocol_version(requested: Optional[str]) -> Optional[str]:
    """Return requested protocol version only when explicitly supported."""
    if requested and requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    if requested is None:
        return SUPPORTED_PROTOCOL_VERSIONS[0]
    return None


def _extract_client_elicitation_modes(capabilities: Any) -> tuple[str, ...]:
    """Read declared client elicitation modes with 2025-11-25 defaults."""
    elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
    if not isinstance(elicitation, dict):
        return tuple()
    if not elicitation:
        # Spec compatibility: empty elicitation object implies form mode.
        return ("form",)
    modes: list[str] = []
    if isinstance(elicitation.get("form"), dict):
        modes.append("form")
    if isinstance(elicitation.get("url"), dict):
        modes.append("url")
    return tuple(modes)


def handle_initialize(msg_id: Any, params: Optional[Dict[str, Any]] = None):
    """Handle the initialize request from the client."""
    requested = None
    if params:
        requested = params.get("protocolVersion")
    negotiated = _negotiate_protocol_version(requested)
    if negotiated is None:
        send_json_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32602,
                "message": (
                    f"Unsupported protocol version: {requested}. "
                    f"Supported versions: {', '.join(SUPPORTED_PROTOCOL_VERSIONS)}"
                ),
            },
        })
        return
    _SESSION_STATE["negotiated"] = True
    _SESSION_STATE["initialized"] = False
    _SESSION_STATE["protocol_version"] = negotiated
    raw_capabilities = params.get("capabilities") if isinstance(params, dict) else None
    client_capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    _SESSION_STATE["client_capabilities"] = client_capabilities
    _SESSION_STATE["client_elicitation_modes"] = _extract_client_elicitation_modes(client_capabilities)
    if not isinstance(_SESSION_STATE.get("tasks"), dict):
        _SESSION_STATE["tasks"] = {}
    startup_warnings = _collect_startup_warnings()

    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "tasks": {
                    "list": {},
                    "cancel": {},
                    "requests": {
                        "tools/call": {},
                    },
                    "notifications": {
                        "status": {},
                    },
                }
            },
            "serverInfo": {
                "name": "muninn-mcp",
                "version": __version__
            },
            "instructions": _build_initialize_instructions(startup_warnings),
        }
    })


def _dispatch_rpc_message(msg: Dict[str, Any]) -> None:
    """
    Handle a single parsed JSON-RPC message.

    Conformance notes:
    - Unknown request methods (with id) return -32601.
    - Unknown notifications (no id) are ignored.
    - notifications/initialized is only accepted after successful initialize.
    """
    msg_id = msg.get("id")
    method = msg.get("method")
    params = msg.get("params", {})

    if not isinstance(method, str):
        if msg_id is not None:
            _send_json_rpc_error(msg_id, -32600, "Invalid Request: missing method")
        return

    if method == "initialize":
        if params is None:
            params = {}
        if not isinstance(params, dict):
            _send_json_rpc_error(msg_id, -32602, "Invalid params: initialize params must be an object")
            return
        handle_initialize(msg_id, params)
        return

    if method == "notifications/initialized":
        if _SESSION_STATE["negotiated"]:
            _SESSION_STATE["initialized"] = True
            logger.info("Client initialized connection")
        else:
            logger.warning("Ignored notifications/initialized before successful initialize")
        return

    if method == "ping":
        if msg_id is not None:
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {}
            })
        return

    if method == "tools/list":
        if not _SESSION_STATE["initialized"]:
            if msg_id is not None:
                _send_json_rpc_error(
                    msg_id,
                    -32600,
                    "Server not initialized. Send initialize then notifications/initialized.",
                )
            return
        if msg_id is None:
            logger.debug("Ignoring tools/list notification without id")
            return
        handle_list_tools(msg_id)
        return

    if method == "tasks/list":
        validated = _validate_initialized_rpc_params(msg_id, method, params)
        if validated is None:
            return
        handle_list_tasks(msg_id, validated)
        return

    if method in ("tasks/get", "tasks/result", "tasks/cancel"):
        validated = _validate_initialized_rpc_params(msg_id, method, params)
        if validated is None:
            return
        if method == "tasks/get":
            handle_get_task(msg_id, validated)
        elif method == "tasks/result":
            handle_get_task_result(msg_id, validated)
        else:
            handle_cancel_task(msg_id, validated)
        return

    if method == "tools/call":
        validated = _validate_initialized_rpc_params(msg_id, method, params)
        if validated is None:
            return
        parsed = _validate_tools_call_params(msg_id, validated)
        if parsed is None:
            return
        name = parsed["name"]
        arguments = parsed["arguments"]
        task_request = parsed.get("task")
        if isinstance(task_request, dict):
            handle_call_tool_with_task(msg_id, name, arguments, task_request)
            return
        handle_call_tool(msg_id, {"name": name, "arguments": arguments})
        return

    if msg_id is not None:
        _send_json_rpc_error(msg_id, -32601, f"Method not found: {method}")
    else:
        logger.debug(f"Ignoring unknown notification method: {method}")


def _validate_initialized_rpc_params(
    msg_id: Any,
    method: str,
    params: Any,
) -> Optional[Dict[str, Any]]:
    """Validate initialized lifecycle and dict params for request methods."""
    if not _SESSION_STATE["initialized"]:
        if msg_id is not None:
            _send_json_rpc_error(
                msg_id,
                -32600,
                "Server not initialized. Send initialize then notifications/initialized.",
            )
        return None
    if msg_id is None:
        logger.debug(f"Ignoring {method} notification without id")
        return None
    validated = {} if params is None else params
    if not isinstance(validated, dict):
        _send_json_rpc_error(msg_id, -32602, f"Invalid params: {method} params must be an object")
        return None
    return validated


def _validate_tools_call_params(msg_id: Any, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = params.get("name")
    if not isinstance(name, str) or not name.strip():
        _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call requires non-empty string name")
        return None
    arguments = params.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call arguments must be an object")
        return None
    task_request = params.get("task")
    if task_request is not None and not isinstance(task_request, dict):
        _send_json_rpc_error(msg_id, -32602, "Invalid params: tools/call task must be an object")
        return None
    return {
        "name": name.strip(),
        "arguments": arguments,
        "task": task_request,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _task_now_epoch() -> float:
    return time.time()


def _sanitize_task_ttl_ms(raw_ttl: Any) -> int:
    if raw_ttl is None:
        return _TASKS_DEFAULT_TTL_MS
    if not isinstance(raw_ttl, int):
        raise ValueError("tools/call task ttl must be an integer in milliseconds")
    if raw_ttl <= 0:
        raise ValueError("tools/call task ttl must be greater than zero")
    return min(raw_ttl, _TASKS_MAX_TTL_MS)


def _public_task(task: Dict[str, Any]) -> Dict[str, Any]:
    public = {}
    for key, value in task.items():
        if key.startswith("_"):
            continue
        public[key] = value
    return public


def _related_task_meta(task_id: str) -> Dict[str, Any]:
    return {"io.modelcontextprotocol/related-task": {"id": task_id}}


def _emit_task_status_notification(task: Dict[str, Any]) -> None:
    if not _SESSION_STATE.get("initialized"):
        return
    send_json_rpc({
        "jsonrpc": "2.0",
        "method": "notifications/tasks/status",
        "params": {
            "task": _public_task(task),
        },
    })


def _set_task_state_locked(
    task: Dict[str, Any],
    *,
    status: str,
    status_message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    task["status"] = status
    task["lastUpdatedAt"] = _utc_now_iso()
    if status_message:
        task["statusMessage"] = status_message
    if result is not None:
        task["result"] = result
        task.pop("error", None)
    elif error is not None:
        task["error"] = error
        task.pop("result", None)


def _purge_and_retain_tasks_locked(now_epoch: float) -> None:
    tasks = _task_registry()
    expired_ids: List[str] = []
    for task_id, task in tasks.items():
        expires_at = task.get("_expiresAtEpoch")
        if isinstance(expires_at, (int, float)) and now_epoch >= float(expires_at):
            expired_ids.append(task_id)
    for task_id in expired_ids:
        tasks.pop(task_id, None)

    if len(tasks) <= _TASKS_MAX_RETAINED:
        return

    ordered = sorted(
        tasks.items(),
        key=lambda item: (
            item[1].get("status") not in _TASK_TERMINAL_STATUSES,
            item[1].get("lastUpdatedAt") or "",
            item[0],
        ),
    )
    for task_id, _ in ordered:
        if len(tasks) <= _TASKS_MAX_RETAINED:
            break
        tasks.pop(task_id, None)


def _lookup_task_locked(task_id: str) -> Optional[Dict[str, Any]]:
    _purge_and_retain_tasks_locked(_task_now_epoch())
    return _task_registry().get(task_id)


def handle_list_tasks(msg_id: Any, params: Optional[Dict[str, Any]] = None):
    """Return retained tasks with deterministic cursor pagination."""
    params = params or {}
    cursor = params.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        _send_json_rpc_error(msg_id, -32602, "Invalid params: tasks/list cursor must be a string")
        return
    limit = params.get("limit")
    page_size = _TASKS_LIST_PAGE_SIZE
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tasks/list limit must be a positive integer")
            return
        page_size = min(limit, _TASKS_LIST_PAGE_SIZE)

    start = 0
    if isinstance(cursor, str) and cursor:
        try:
            start = int(cursor)
        except ValueError:
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tasks/list cursor must be a decimal offset string")
            return
        if start < 0:
            _send_json_rpc_error(msg_id, -32602, "Invalid params: tasks/list cursor must be non-negative")
            return

    with _TASKS_LOCK:
        _purge_and_retain_tasks_locked(_task_now_epoch())
        tasks = [_public_task(task) for task in _task_registry().values()]
    tasks.sort(key=lambda item: (item.get("lastUpdatedAt") or "", item.get("taskId") or ""), reverse=True)
    page = tasks[start:start + page_size]
    next_cursor = start + page_size if start + page_size < len(tasks) else None
    result = {"tasks": page}
    if next_cursor is not None:
        result["nextCursor"] = str(next_cursor)
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": result,
    })


def _task_id_or_error(msg_id: Any, params: Dict[str, Any], method: str) -> Optional[str]:
    task_id = params.get("taskId")
    if not isinstance(task_id, str) or not task_id.strip():
        _send_json_rpc_error(msg_id, -32602, f"Invalid params: {method} requires non-empty string taskId")
        return None
    return task_id.strip()


def _task_registry() -> Dict[str, Dict[str, Any]]:
    tasks = _SESSION_STATE.get("tasks")
    if isinstance(tasks, dict):
        return tasks
    reset: Dict[str, Dict[str, Any]] = {}
    _SESSION_STATE["tasks"] = reset
    return reset


def handle_get_task(msg_id: Any, params: Dict[str, Any]) -> None:
    task_id = _task_id_or_error(msg_id, params, "tasks/get")
    if task_id is None:
        return
    with _TASKS_LOCK:
        task = _lookup_task_locked(task_id)
        payload = _public_task(task) if isinstance(task, dict) else None
    if not isinstance(payload, dict):
        _send_json_rpc_error(msg_id, -32602, "Invalid params: unknown taskId")
        return
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": payload,
    })


def handle_get_task_result(msg_id: Any, params: Dict[str, Any]) -> None:
    task_id = _task_id_or_error(msg_id, params, "tasks/result")
    if task_id is None:
        return
    with _TASKS_CONDITION:
        while True:
            task = _lookup_task_locked(task_id)
            if not isinstance(task, dict):
                _send_json_rpc_error(msg_id, -32602, "Invalid params: unknown taskId")
                return
            status = str(task.get("status") or "")
            if status in _TASK_TERMINAL_STATUSES or status == "input_required":
                break
            _TASKS_CONDITION.wait(timeout=0.1)
        payload = task.get("result")
        error = task.get("error")

    if isinstance(error, dict):
        send_json_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": error,
        })
        return
    if not isinstance(payload, dict):
        _send_json_rpc_error(msg_id, -32603, "Task reached terminal state without a result payload.")
        return
    payload_with_meta = dict(payload)
    result_meta = payload_with_meta.get("_meta")
    if isinstance(result_meta, dict):
        merged_meta = dict(result_meta)
        merged_meta.update(_related_task_meta(task_id))
        payload_with_meta["_meta"] = merged_meta
    else:
        payload_with_meta["_meta"] = _related_task_meta(task_id)
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": payload_with_meta,
    })


def handle_cancel_task(msg_id: Any, params: Dict[str, Any]) -> None:
    task_id = _task_id_or_error(msg_id, params, "tasks/cancel")
    if task_id is None:
        return
    with _TASKS_CONDITION:
        task = _lookup_task_locked(task_id)
        if not isinstance(task, dict):
            _send_json_rpc_error(msg_id, -32602, "Invalid params: unknown taskId")
            return
        status = str(task.get("status") or "")
        if status in _TASK_TERMINAL_STATUSES:
            _send_json_rpc_error(
                msg_id,
                -32602,
                "Invalid params: task is already in a terminal state.",
            )
            return
        _set_task_state_locked(
            task,
            status="cancelled",
            status_message="Task cancelled by client request.",
            error={"code": -32800, "message": "Task was cancelled by client request."},
        )
        task["_cancelRequested"] = True
        response_task = _public_task(task)
        _TASKS_CONDITION.notify_all()
    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": response_task,
    })
    _emit_task_status_notification(response_task)


def handle_call_tool_with_task(
    msg_id: Any,
    name: str,
    arguments: Dict[str, Any],
    task_request: Dict[str, Any],
) -> None:
    try:
        ttl_ms = _sanitize_task_ttl_ms(task_request.get("ttl"))
    except ValueError as exc:
        _send_json_rpc_error(msg_id, -32602, f"Invalid params: {exc}")
        return

    now_iso = _utc_now_iso()
    now_epoch = _task_now_epoch()
    task_id = str(uuid4())
    task: Dict[str, Any] = {
        "taskId": task_id,
        "status": "working",
        "statusMessage": f"Tool '{name}' is running.",
        "createdAt": now_iso,
        "lastUpdatedAt": now_iso,
        "ttl": ttl_ms,
        "_expiresAtEpoch": now_epoch + (ttl_ms / 1000.0),
        "_cancelRequested": False,
    }
    with _TASKS_CONDITION:
        _task_registry()[task_id] = task
        _purge_and_retain_tasks_locked(now_epoch)
        response_task = _public_task(task)
        _TASKS_CONDITION.notify_all()

    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "task": response_task,
            "_meta": {
                "io.modelcontextprotocol/model-immediate-response": (
                    f"Task accepted for tool '{name}'. Monitor with tasks/get or tasks/result."
                )
            },
        },
    })
    _emit_task_status_notification(response_task)

    worker = threading.Thread(
        target=_run_tool_call_task_worker,
        args=(task_id, name, arguments),
        daemon=True,
    )
    worker.start()


def _run_tool_call_task_worker(task_id: str, name: str, arguments: Dict[str, Any]) -> None:
    captured: List[Dict[str, Any]] = []

    def _capture_rpc(message: Dict[str, Any]) -> None:
        captured.append(message)

    setattr(_thread_local, "rpc_emitter", _capture_rpc)
    try:
        handle_call_tool(task_id, {"name": name, "arguments": arguments})
    except Exception:
        logger.exception("Unhandled exception while executing task-backed tool call")
        captured.append({
            "jsonrpc": "2.0",
            "id": task_id,
            "error": {"code": -32603, "message": "Internal error during task execution."},
        })
    finally:
        setattr(_thread_local, "rpc_emitter", None)

    message = captured[-1] if captured else {
        "jsonrpc": "2.0",
        "id": task_id,
        "error": {"code": -32603, "message": "Task execution finished without a response payload."},
    }

    status = "failed"
    status_message = f"Tool '{name}' failed."
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    if isinstance(message.get("error"), dict):
        error = message["error"]
    else:
        payload = message.get("result")
        if isinstance(payload, dict):
            result = payload
            if payload.get("isError") is True:
                status = "failed"
                status_message = f"Tool '{name}' returned an error result."
            else:
                status = "completed"
                status_message = f"Tool '{name}' completed."
        else:
            error = {"code": -32603, "message": "Task execution produced an invalid result payload."}

    with _TASKS_CONDITION:
        task = _lookup_task_locked(task_id)
        if not isinstance(task, dict):
            return
        if task.get("status") == "cancelled":
            _TASKS_CONDITION.notify_all()
            return
        _set_task_state_locked(task, status=status, status_message=status_message, result=result, error=error)
        response_task = _public_task(task)
        _TASKS_CONDITION.notify_all()
    _emit_task_status_notification(response_task)


def handle_list_tools(msg_id: Any):
    """Return list of available tools."""
    tools = [
        {
            "name": "add_memory",
            "description": "Add a new memory to the global knowledge base. Use this to store facts, preferences, or important information that should be remembered across sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata tags (e.g., {'project': 'phoenix', 'category': 'api'})."
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search for memories relevant to a query. Uses hybrid search with optional reranking for precision.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 5)",
                        "default": 5
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable SOTA reranking for precision (default true)",
                        "default": True
                    },
                    "explain": {
                        "type": "boolean",
                        "description": "Include per-result recall trace explaining retrieval signals (v3.1.0)",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_all_memories",
            "description": "Retrieve all stored memories, optionally filtered by user/agent.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results (default 100)",
                        "default": 100
                    }
                }
            }
        },
        {
            "name": "update_memory",
            "description": "Update an existing memory by ID. Use to correct or enhance stored information.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to update."
                    },
                    "content": {
                        "type": "string",
                        "description": "New content to replace the existing memory."
                    }
                },
                "required": ["memory_id", "content"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Delete a specific memory by ID. Cannot be undone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to delete."
                    }
                },
                "required": ["memory_id"]
            }
        },
        {
            "name": "delete_all_memories",
            "description": "Delete ALL memories for a user. DANGEROUS - use with extreme caution. Cannot be undone.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm deletion of all memories.",
                        "default": False
                    }
                },
                "required": ["confirm"]
            }
        },
        {
            "name": "set_project_goal",
            "description": "Set or update project north-star goal/constraints for drift checks and goal-aware retrieval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal_statement": {
                        "type": "string",
                        "description": "Canonical objective statement for this project."
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional constraints/non-goals that must be preserved."
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace scope (default: global).",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override; defaults to current git repo."
                    }
                },
                "required": ["goal_statement"]
            }
        },
        {
            "name": "get_project_goal",
            "description": "Fetch the active project goal for current repository/namespace scope.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    }
                }
            }
        },
        {
            "name": "get_model_profiles",
            "description": "Fetch active runtime extraction profile policy for helper/ingestion routing.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "set_model_profiles",
            "description": "Update runtime extraction profile policy without restarting the server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_profile": {
                        "type": "string",
                        "enum": list(SUPPORTED_MODEL_PROFILES),
                        "description": "Default extraction profile fallback."
                    },
                    "runtime_model_profile": {
                        "type": "string",
                        "enum": list(SUPPORTED_MODEL_PROFILES),
                        "description": "Profile for add/update helper extraction."
                    },
                    "ingestion_model_profile": {
                        "type": "string",
                        "enum": list(SUPPORTED_MODEL_PROFILES),
                        "description": "Profile for source ingestion extraction."
                    },
                    "legacy_ingestion_model_profile": {
                        "type": "string",
                        "enum": list(SUPPORTED_MODEL_PROFILES),
                        "description": "Profile for legacy source ingestion extraction."
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional mutation source tag for audit trail.",
                        "default": "mcp_tool"
                    }
                }
            }
        },
        {
            "name": "get_model_profile_events",
            "description": "Fetch recent runtime model profile policy mutation events for auditability.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 25,
                        "description": "Maximum number of recent events to return."
                    }
                }
            }
        },
        {
            "name": "export_handoff",
            "description": "Export deterministic cross-assistant handoff bundle for this project.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "description": "Number of top memories to include."
                    }
                }
            }
        },
        {
            "name": "import_handoff",
            "description": "Import a handoff bundle idempotently using event ledger checks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bundle": {
                        "type": "object",
                        "description": "Handoff bundle produced by export_handoff."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "source": {
                        "type": "string",
                        "default": "mcp_import"
                    }
                },
                "required": ["bundle"]
            }
        },
        {
            "name": "record_retrieval_feedback",
            "description": "Record retrieval outcome feedback to improve adaptive signal weighting over time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that produced the retrieved memory."
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "Identifier of the memory being rated."
                    },
                    "outcome": {
                        "type": "number",
                        "description": "Feedback score in [0,1] where 1 means helpful/accepted."
                    },
                    "rank": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based displayed rank position for counterfactual calibration."
                    },
                    "sampling_prob": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                        "description": "Optional probability that this result was shown/clicked under the logging policy."
                    },
                    "signals": {
                        "type": "object",
                        "description": "Optional signal contribution map, e.g. {\"vector\":0.8,\"bm25\":0.1}."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "source": {
                        "type": "string",
                        "default": "mcp_feedback"
                    }
                },
                "required": ["query", "memory_id", "outcome"]
            }
        },
        {
            "name": "ingest_sources",
            "description": "Ingest local files/directories into memory with fail-open parsing and per-source provenance metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of local file or directory paths to ingest."
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Recursively traverse directory sources."
                    },
                    "chronological_order": {
                        "type": "string",
                        "enum": ["none", "oldest_first", "newest_first"],
                        "default": "none",
                        "description": "Process source files in deterministic path order or by file modification time."
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata merged into each ingested chunk."
                    },
                    "max_file_size_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional maximum source file size."
                    },
                    "chunk_size_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional chunk size override."
                    },
                    "chunk_overlap_chars": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional chunk overlap override."
                    },
                    "min_chunk_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional minimum chunk length."
                    }
                },
                "required": ["sources"]
            }
        },
        {
            "name": "discover_legacy_sources",
            "description": "Discover local legacy assistant/MCP memory sources (Codex, Claude Code, Serena, Cursor/VS Code stores, etc.) available for import.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "roots": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional additional root directories to scan."
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional provider allowlist, e.g. ['codex_cli','serena_memory']."
                    },
                    "include_unsupported": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include files not currently supported by ingestion parsers."
                    },
                    "max_results_per_provider": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum files returned per provider."
                    }
                }
            }
        },
        {
            "name": "ingest_legacy_sources",
            "description": "Ingest user-selected legacy sources discovered from assistant logs and MCP memory programs with contextual metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "selected_source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source IDs selected from discover_legacy_sources."
                    },
                    "selected_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit local paths to include."
                    },
                    "roots": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "include_unsupported": {
                        "type": "boolean",
                        "default": False
                    },
                    "max_results_per_provider": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False
                    },
                    "chronological_order": {
                        "type": "string",
                        "enum": ["none", "oldest_first", "newest_first"],
                        "default": "none"
                    },
                    "namespace": {
                        "type": "string",
                        "default": "global"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project override."
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata merged into each ingested chunk."
                    },
                    "max_file_size_bytes": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "chunk_size_chars": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "chunk_overlap_chars": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "min_chunk_chars": {
                        "type": "integer",
                        "minimum": 1
                    }
                }
            }
        }
    ]
    
    read_only_tools = {
        "search_memory",
        "get_all_memories",
        "get_project_goal",
        "get_model_profiles",
        "get_model_profile_events",
        "export_handoff",
        "discover_legacy_sources",
    }
    destructive_tools = {"delete_memory", "delete_all_memories"}
    idempotent_tools = read_only_tools.union({
        "update_memory",
        "delete_memory",
        "delete_all_memories",
        "set_project_goal",
        "set_model_profiles",
        "import_handoff",
    })
    for tool in tools:
        schema = tool.get("inputSchema", {})
        if isinstance(schema, dict) and "$schema" not in schema:
            schema["$schema"] = JSON_SCHEMA_2020_12
        name = tool.get("name")
        read_only = name in read_only_tools
        annotations = dict(tool.get("annotations") or {})
        annotations.update({
            "readOnlyHint": read_only,
            "destructiveHint": name in destructive_tools,
            "idempotentHint": name in idempotent_tools,
            "openWorldHint": True,
        })
        tool["annotations"] = annotations
        execution = dict(tool.get("execution") or {})
        execution.setdefault("taskSupport", "optional")
        tool["execution"] = execution

    send_json_rpc({
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "tools": tools
        }
    })

def handle_call_tool(msg_id: Any, params: Dict[str, Any]):
    """Handle tool execution requests."""
    name = params.get("name")
    arguments = params.get("arguments", {})
    
    ensure_server_running()
    
    try:
        if name == "add_memory":
            # SOTA: Inject current working directory as 'project' metadata
            # This allows the memory system to automatically anchor memories to the workspace
            metadata = _inject_operator_profile_metadata(arguments.get("metadata", {}), operation="add")
            git_info = get_git_info()
            if "project" not in metadata:
                metadata["project"] = git_info["project"]
            if "branch" not in metadata:
                metadata["branch"] = git_info["branch"]
            
            payload = {
                "content": arguments.get("content"),
                "metadata": metadata,
                "user_id": "global_user"
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/add", json=payload, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
            
        elif name == "search_memory":
            git_info = get_git_info()
            filters = arguments.get("filters", {})
            if "project" not in filters:
                filters["project"] = git_info["project"]

            explain = arguments.get("explain", False)
            payload = {
                "query": arguments.get("query"),
                "limit": arguments.get("limit", 5),
                "rerank": arguments.get("rerank", True),
                "user_id": "global_user",
                "filters": filters,
                "explain": explain,
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/search", json=payload, timeout=10)
            result = resp.json()

            formatted_results = []
            if result.get("success") and result.get("data"):
                for item in result["data"]:
                    content = str(item.get('content', item.get('memory', 'Unknown content')))
                    score = item.get('score', '')
                    mem_type = item.get('memory_type', '')
                    prefix = f"[{mem_type}:{score:.2f}] " if score and mem_type else ""
                    line = f"- {prefix}{content}"

                    # Append recall trace explanation if present (v3.1.0)
                    trace = item.get("trace")
                    if trace and explain:
                        explanation = trace.get("explanation", "")
                        dominant = trace.get("dominant_signal", "")
                        if explanation:
                            line += f"\n  Why: {explanation}"
                        elif dominant:
                            line += f"\n  Dominant signal: {dominant}"

                    formatted_results.append(line)
                text_response = "\n".join(formatted_results) if formatted_results else "No relevant memories found."
            else:
                text_response = f"Error or no data: {result}"

            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": text_response
                    }]
                }
            })
            
        elif name == "get_all_memories":
            limit = arguments.get("limit", 100)
            resp = make_request_with_retry("GET", f"{SERVER_URL}/get_all", params={"user_id": "global_user", "limit": limit}, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "update_memory":
            payload = {
                "memory_id": arguments.get("memory_id"),
                "data": arguments.get("content")
            }
            resp = make_request_with_retry("PUT", f"{SERVER_URL}/update", json=payload, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "delete_memory":
            memory_id = arguments.get("memory_id")
            resp = make_request_with_retry("DELETE", f"{SERVER_URL}/delete/{memory_id}", timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        
        elif name == "delete_all_memories":
            confirm = arguments.get("confirm", False)
            if not confirm:
                send_json_rpc({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": "Must set 'confirm: true' to delete all memories"
                    }
                })
                return

            # Muninn v3 uses POST /delete_all with JSON body
            resp = make_request_with_retry("POST", f"{SERVER_URL}/delete_all", json={"user_id": "global_user"}, timeout=10)
            result = resp.json()
            
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "set_project_goal":
            git_info = get_git_info()
            payload = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "goal_statement": arguments.get("goal_statement"),
                "constraints": arguments.get("constraints", []),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/goal/set", json=payload, timeout=15)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "get_project_goal":
            git_info = get_git_info()
            params = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
            }
            resp = make_request_with_retry("GET", f"{SERVER_URL}/goal/get", params=params, timeout=10)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "get_model_profiles":
            resp = make_request_with_retry("GET", f"{SERVER_URL}/profiles/model", timeout=10)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "set_model_profiles":
            payload = {}
            for key in (
                "model_profile",
                "runtime_model_profile",
                "ingestion_model_profile",
                "legacy_ingestion_model_profile",
            ):
                value = arguments.get(key)
                if value is not None:
                    payload[key] = value
            if not payload:
                raise ValueError("set_model_profiles requires at least one profile field")
            payload["source"] = arguments.get("source", "mcp_tool")
            resp = make_request_with_retry("POST", f"{SERVER_URL}/profiles/model", json=payload, timeout=10)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "get_model_profile_events":
            params = {
                "limit": arguments.get("limit", 25),
            }
            resp = make_request_with_retry(
                "GET",
                f"{SERVER_URL}/profiles/model/events",
                params=params,
                timeout=10,
            )
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "export_handoff":
            git_info = get_git_info()
            payload = {
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "limit": arguments.get("limit", 25),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/export", json=payload, timeout=30)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "import_handoff":
            git_info = get_git_info()
            payload = {
                "bundle": arguments.get("bundle"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "source": arguments.get("source", "mcp_import"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/import", json=payload, timeout=30)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "record_retrieval_feedback":
            git_info = get_git_info()
            payload = {
                "query": arguments.get("query"),
                "memory_id": arguments.get("memory_id"),
                "outcome": arguments.get("outcome"),
                "rank": arguments.get("rank"),
                "sampling_prob": arguments.get("sampling_prob"),
                "signals": arguments.get("signals", {}),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "source": arguments.get("source", "mcp_feedback"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/feedback/retrieval", json=payload, timeout=15)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "ingest_sources":
            git_info = get_git_info()
            payload = {
                "sources": arguments.get("sources", []),
                "recursive": arguments.get("recursive", False),
                "chronological_order": arguments.get("chronological_order", "none"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "metadata": _inject_operator_profile_metadata(arguments.get("metadata", {}), operation="ingest"),
                "max_file_size_bytes": arguments.get("max_file_size_bytes"),
                "chunk_size_chars": arguments.get("chunk_size_chars"),
                "chunk_overlap_chars": arguments.get("chunk_overlap_chars"),
                "min_chunk_chars": arguments.get("min_chunk_chars"),
            }
            resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest", json=payload, timeout=60)
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "discover_legacy_sources":
            payload = {
                "roots": arguments.get("roots", []),
                "providers": arguments.get("providers", []),
                "include_unsupported": arguments.get("include_unsupported", False),
                "max_results_per_provider": arguments.get("max_results_per_provider", 100),
            }
            resp = make_request_with_retry(
                "POST",
                f"{SERVER_URL}/ingest/legacy/discover",
                json=payload,
                timeout=60,
            )
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
        elif name == "ingest_legacy_sources":
            git_info = get_git_info()
            payload = {
                "selected_source_ids": arguments.get("selected_source_ids", []),
                "selected_paths": arguments.get("selected_paths", []),
                "roots": arguments.get("roots", []),
                "providers": arguments.get("providers", []),
                "include_unsupported": arguments.get("include_unsupported", False),
                "max_results_per_provider": arguments.get("max_results_per_provider", 100),
                "recursive": arguments.get("recursive", False),
                "chronological_order": arguments.get("chronological_order", "none"),
                "user_id": "global_user",
                "namespace": arguments.get("namespace", "global"),
                "project": arguments.get("project", git_info["project"]),
                "metadata": _inject_operator_profile_metadata(arguments.get("metadata", {}), operation="legacy_ingest"),
                "max_file_size_bytes": arguments.get("max_file_size_bytes"),
                "chunk_size_chars": arguments.get("chunk_size_chars"),
                "chunk_overlap_chars": arguments.get("chunk_overlap_chars"),
                "min_chunk_chars": arguments.get("min_chunk_chars"),
            }
            resp = make_request_with_retry(
                "POST",
                f"{SERVER_URL}/ingest/legacy/import",
                json=payload,
                timeout=120,
            )
            result = resp.json()
            send_json_rpc({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }]
                }
            })
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        send_json_rpc({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32603,
                "message": str(e)
            }
        })


def _dispatch_rpc_message_guarded(msg: Dict[str, Any]) -> None:
    try:
        _dispatch_rpc_message(msg)
    except Exception:
        logger.error("An unexpected error occurred during RPC dispatch.")


def _get_dispatch_executor() -> ThreadPoolExecutor:
    global _DISPATCH_EXECUTOR
    executor = _DISPATCH_EXECUTOR
    if executor is not None:
        return executor
    with _DISPATCH_EXECUTOR_LOCK:
        if _DISPATCH_EXECUTOR is None:
            _DISPATCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=_DISPATCH_MAX_WORKERS,
                thread_name_prefix="muninn-mcp-dispatch",
            )
        return _DISPATCH_EXECUTOR


def _submit_background_dispatch(msg: Dict[str, Any]) -> None:
    _get_dispatch_executor().submit(_dispatch_rpc_message_guarded, msg)


def _should_dispatch_in_background(msg: Dict[str, Any]) -> bool:
    method = msg.get("method")
    return method in {"tasks/result", "tools/call"}

def main():
    logger.info("Muninn MCP Wrapper started")

    # Trigger dependency startup immediately (best effort) so MCP sessions
    # can attach to warm backends when user has autostart enabled.
    threading.Thread(
        target=_bootstrap_dependencies_on_launch,
        name="muninn-mcp-bootstrap",
        daemon=True,
    ).start()
    
    # Standard input loop
    try:
        while True:
            try:
                msg = _read_rpc_message(sys.stdin.buffer)
                if msg is None:
                    break
                if _should_dispatch_in_background(msg):
                    _submit_background_dispatch(msg)
                else:
                    _dispatch_rpc_message(msg)
            except Exception as e:
                logger.error(f"Loop error: {e}")
    finally:
        executor = _DISPATCH_EXECUTOR
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    main()
