import os
import time
import json
import base64
import binascii
import logging
import threading
import math
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable

from .state import (
    _SESSION_STATE,
    get_tasks_lock,
    get_tasks_condition
)

logger = logging.getLogger("Muninn.mcp.tasks")

# Task Management Constants
TASK_TERMINAL_STATUSES = {"cancelled", "completed", "failed"}

# Configurable limits from environment
# ... existing limits ...
TASKS_DEFAULT_TTL_MS = max(1, int(os.environ.get("MUNINN_MCP_TASK_TTL_MS", "600000")))
TASKS_MAX_TTL_MS = max(TASKS_DEFAULT_TTL_MS, int(os.environ.get("MUNINN_MCP_TASK_MAX_TTL_MS", "86400000")))
TASKS_LIST_PAGE_SIZE = max(1, int(os.environ.get("MUNINN_MCP_TASKS_LIST_PAGE_SIZE", "50")))
TASKS_MAX_RETAINED = max(TASKS_LIST_PAGE_SIZE, int(os.environ.get("MUNINN_MCP_TASKS_MAX_RETAINED", "500")))
TASK_POLL_INTERVAL_MS = max(1, int(os.environ.get("MUNINN_MCP_TASK_POLL_INTERVAL_MS", "250")))
TASK_CURSOR_PREFIX = "tasks:v1:"

def get_registry() -> Dict[str, Any]:
    with get_tasks_lock():
        return _SESSION_STATE["tasks"]

def set_registry(tasks: Dict[str, Any]) -> None:
    # This might be tricky because we can't easily set the facade's dict if it was monkeypatched.
    # But for now, we'll try to update the internal state to match if possible.
    reg = _SESSION_STATE["tasks"]
    if reg is not tasks:
        reg.clear()
        reg.update(tasks)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def task_now_epoch() -> float:
    return time.time()

def sanitize_task_ttl_ms(raw_ttl: Any) -> int:
    if raw_ttl is None:
        return TASKS_DEFAULT_TTL_MS
    if not isinstance(raw_ttl, int):
        raise ValueError("tools/call task ttl must be an integer in milliseconds")
    if raw_ttl <= 0:
        raise ValueError("tools/call task ttl must be greater than zero")
    return min(raw_ttl, TASKS_MAX_TTL_MS)

def get_task_worker_start_delay_ms() -> float:
    raw_value = os.environ.get("MUNINN_MCP_TASK_WORKER_START_DELAY_MS", "0")
    try:
        parsed = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid MUNINN_MCP_TASK_WORKER_START_DELAY_MS=%r; using default of 0ms.",
            raw_value,
        )
        return 0.0
    if not math.isfinite(parsed) or parsed < 0:
        logger.warning(
            "Non-finite/negative MUNINN_MCP_TASK_WORKER_START_DELAY_MS=%r; using default of 0ms.",
            raw_value,
        )
        return 0.0
    return min(parsed, 60000.0)

def public_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Filter internal keys from a task object for public visibility."""
    return {k: v for k, v in task.items() if not k.startswith("_")}

def related_task_meta(task_id: str) -> Dict[str, Any]:
    return {"io.modelcontextprotocol/related-task": {"taskId": task_id}}

def encode_task_cursor(offset: int) -> str:
    payload = f"{TASK_CURSOR_PREFIX}{offset}".encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

def decode_task_cursor(cursor: str) -> int:
    if cursor.isdigit():
        return int(cursor)
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("Invalid params: tasks/list cursor must be an opaque cursor token") from exc
    if not decoded.startswith(TASK_CURSOR_PREFIX):
        raise ValueError("Invalid params: tasks/list cursor must be an opaque cursor token")
    raw_offset = decoded[len(TASK_CURSOR_PREFIX):]
    if not raw_offset.isdigit():
        raise ValueError("Invalid params: tasks/list cursor must be an opaque cursor token")
    return int(raw_offset)

def set_task_state_locked(
    task: Dict[str, Any],
    *,
    status: str,
    status_message: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    task["status"] = status
    task["lastUpdatedAt"] = utc_now_iso()
    if status_message:
        task["statusMessage"] = status_message
    if result is not None:
        task["result"] = result
        task.pop("error", None)
    elif error is not None:
        task["error"] = error
        task.pop("result", None)

def purge_and_retain_tasks_locked(now_epoch: float) -> None:
    registry = get_registry()
    expired_ids: List[str] = []
    for task_id, task in registry.items():
        expires_at = task.get("_expiresAtEpoch")
        if isinstance(expires_at, (int, float)) and now_epoch >= float(expires_at):
            expired_ids.append(task_id)
    for task_id in expired_ids:
        registry.pop(task_id, None)

    if len(registry) <= TASKS_MAX_RETAINED:
        return

    ordered = sorted(
        registry.items(),
        key=lambda item: (
            item[1].get("status") not in TASK_TERMINAL_STATUSES,
            item[1].get("lastUpdatedAt") or "",
            item[0],
        ),
    )
    for task_id, _ in ordered:
        if len(registry) <= TASKS_MAX_RETAINED:
            break
        registry.pop(task_id, None)

def lookup_task_locked(task_id: str) -> Optional[Dict[str, Any]]:
    purge_and_retain_tasks_locked(task_now_epoch())
    return get_registry().get(task_id)

def create_task(name: str, ttl_ms: int) -> Dict[str, Any]:
    ttl_ms = sanitize_task_ttl_ms(ttl_ms)
    now_iso = utc_now_iso()
    now_epoch = task_now_epoch()
    task_id = str(uuid4())
    task: Dict[str, Any] = {
        "taskId": task_id,
        "status": "working",
        "statusMessage": f"Tool '{name}' is running.",
        "createdAt": now_iso,
        "lastUpdatedAt": now_iso,
        "ttl": ttl_ms,
        "pollInterval": TASK_POLL_INTERVAL_MS,
        "_expiresAtEpoch": now_epoch + (ttl_ms / 1000.0),
        "_cancelRequested": False,
    }
    with get_tasks_lock():
        get_registry()[task_id] = task
        purge_and_retain_tasks_locked(now_epoch)
    return task

def get_task_result_mode() -> str:
    val = os.environ.get("MUNINN_MCP_TASK_RESULT_MODE", "auto")
    if val not in ("auto", "blocking", "immediate_retry"):
        return "auto"
    return val

def get_task_result_max_wait_seconds() -> Optional[float]:
    raw = os.environ.get("MUNINN_MCP_TASK_RESULT_MAX_WAIT_SEC")
    if raw == "0":
        return None
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    
    from .utils import get_host_safe_tool_call_budget_seconds
    return get_host_safe_tool_call_budget_seconds()

def task_result_should_block(params: Dict[str, Any]) -> bool:
    # Check explicit 'wait' override first (Phase 5A.5)
    wait = params.get("wait")
    if isinstance(wait, bool):
        return wait
    
    mode = params.get("mode", get_task_result_mode())
    if mode == "blocking":
        return True
    if mode == "auto":
        # Check if the client is one that typically prefers/needs blocking
        client_name = (_SESSION_STATE.get("client_info") or {}).get("name", "").lower()
        if "claude" in client_name or "gemini" in client_name:
            # SOTA: Claude/Gemini prefer immediate retry to manage their own polling loops
            return False 
    return False

def emit_task_status_notification(task: Dict[str, Any], send_rpc_fn: Callable) -> None:
    """Emit a status notification for a task using the provided RPC sender."""
    send_rpc_fn({
        "jsonrpc": "2.0",
        "method": "notifications/tasks/status",
        "params": {"task": public_task(task)}
    })