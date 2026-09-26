import os
import uuid

DEFAULT_HTTP_TIMEOUT = float(os.getenv("MUNINN_MCP_HTTP_TIMEOUT_SEC", "40"))
import time
import logging
import threading
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
from urllib.parse import quote

from muninn.version import __version__ as _MUNINN_VERSION

from .state import _SESSION_STATE
from .definitions import (
    SUPPORTED_PROTOCOL_VERSIONS, TOOLS_SCHEMAS, JSON_SCHEMA_2020_12,
    READ_ONLY_TOOLS, DESTRUCTIVE_TOOLS, IDEMPOTENT_TOOLS, OPEN_WORLD_TOOLS,
    SUPPORTED_MODEL_PROFILES, MIMIR_TOOLS, TOOLSETS, resolve_toolset, toolset_schemas, tool_title,
)
from .tasks import (
    create_task, lookup_task_locked, purge_and_retain_tasks_locked, 
    get_registry, public_task, encode_task_cursor, decode_task_cursor,
    TASK_TERMINAL_STATUSES, TASKS_LIST_PAGE_SIZE, TASK_CURSOR_PREFIX,
    set_task_state_locked, emit_task_status_notification
)
from .utils import (
    get_git_info, inject_operator_profile_metadata, format_tool_result_text, 
    truncate_tool_text, env_flag, negotiated_protocol_version, build_initialize_instructions,
    _read_operator_model_profile, get_tool_call_deadline_epoch, remaining_deadline_seconds,
    startup_recovery_allowed
)
from .lifecycle import ensure_server_running, is_circuit_open, SERVER_URL, check_and_start_ollama
from .requests import make_request_with_retry

logger = logging.getLogger("Muninn.mcp.handlers")

_thread_local = threading.local()


def _parse_positive_float_env(name: str, default: float, *, lower: float, upper: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if not (lower <= value <= upper):
        return default
    return value


def _write_timeout_seconds() -> float:
    return _parse_positive_float_env("MUNINN_MCP_WRITE_TIMEOUT_SEC", 40.0, lower=0.1, upper=180.0)


def _initialize_capabilities_for_protocol(protocol_version: str) -> Dict[str, Any]:
    """Advertise only capabilities defined for the negotiated MCP version."""
    capabilities: Dict[str, Any] = {"tools": {"listChanged": False}, "prompts": {"listChanged": False}}
    if protocol_version == "2025-11-25":
        capabilities["io.modelcontextprotocol.elicitation"] = {"modes": ["form"]}
        capabilities["tasks"] = {
            "list": {},
            "cancel": {},
            "requests": {"tools/call": {}},
            "notifications": {"status": {}},
        }
    return capabilities


def handle_initialize(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn, startup_warnings: Optional[List[str]] = None):
    """Handle protocol negotiation and server initialization."""
    if not isinstance(params, dict):
        send_error_fn(msg_id, -32602, "initialize params must be an object")
        return

    requested_version = params.get("protocolVersion")
    negotiated_version = negotiated_protocol_version(requested_version)
    
    if not negotiated_version:
        send_error_fn(msg_id, -32602, f"Unsupported protocol version {requested_version}")
        return

    _SESSION_STATE["negotiated"] = True
    # Several current hosts treat the initialize response as readiness and do
    # not send notifications/initialized before their first tools/list probe.
    _SESSION_STATE["initialized"] = True
    _SESSION_STATE["protocol_version"] = negotiated_version
    _SESSION_STATE["client_capabilities"] = params.get("capabilities", {})
    _SESSION_STATE["client_info"] = params.get("clientInfo", {})
    
    # Elicitation modes
    client_meta = params.get("_meta", {})
    if isinstance(client_meta, dict):
        modes = client_meta.get("io.modelcontextprotocol/elicitation-modes")
        if isinstance(modes, (list, tuple)):
            _SESSION_STATE["client_elicitation_modes"] = tuple(modes)

    if startup_warnings is None:
        startup_warnings = []
        if env_flag("MUNINN_MCP_AUTOSTART_ON_LAUNCH", True):
            srv = env_flag("MUNINN_MCP_AUTOSTART_SERVER", True)
            olm = env_flag("MUNINN_MCP_AUTOSTART_OLLAMA", False)
            if srv and not ensure_server_running():
                startup_warnings.append("Muninn server is not reachable")
            if olm and not check_and_start_ollama():
                startup_warnings.append("Ollama is not reachable")

    # Elicitation modes
    capabilities = params.get("capabilities", {})
    elicitation = capabilities.get("elicitation") if isinstance(capabilities, dict) else None
    if isinstance(elicitation, dict):
        if not elicitation:
            _SESSION_STATE["client_elicitation_modes"] = ("form",)
        else:
            modes = []
            if isinstance(elicitation.get("form"), dict): modes.append("form")
            if isinstance(elicitation.get("url"), dict): modes.append("url")
            _SESSION_STATE["client_elicitation_modes"] = tuple(modes)
    
    # Operator model profile
    session_profile = _read_operator_model_profile("MUNINN_OPERATOR_MODEL_PROFILE")
    _SESSION_STATE["operator_model_profile"] = session_profile
    
    # Send result
    result = {
        "protocolVersion": negotiated_version,
        "capabilities": _initialize_capabilities_for_protocol(negotiated_version),
        "serverInfo": {"name": "muninn-mcp", "version": _MUNINN_VERSION},
        "instructions": build_initialize_instructions(startup_warnings, toolset=active_toolset())
    }
    send_result_fn(msg_id, result)

def active_toolset() -> str:
    """Toolset for this session: the transport's choice, else MUNINN_MCP_TOOLSET, else full."""
    chosen = _SESSION_STATE.get("toolset")
    return resolve_toolset(chosen or os.environ.get("MUNINN_MCP_TOOLSET"))


def tool_definition(schema_def: Dict[str, Any]) -> Dict[str, Any]:
    name = schema_def["name"]
    input_schema = dict(schema_def["inputSchema"])
    input_schema.setdefault("$schema", JSON_SCHEMA_2020_12)
    read_only = name in READ_ONLY_TOOLS
    return {
        "name": name,
        "title": tool_title(name),
        "description": schema_def["description"],
        "inputSchema": input_schema,
        "annotations": {
            "title": tool_title(name),
            "readOnlyHint": read_only,
            "destructiveHint": name in DESTRUCTIVE_TOOLS,
            "idempotentHint": name in IDEMPOTENT_TOOLS or read_only,
            # Everything but mimir_relay stays inside the local memory store.
            "openWorldHint": name in OPEN_WORLD_TOOLS,
        },
        "execution": {"taskSupport": "optional"},
        **({"outputSchema": schema_def["outputSchema"]} if "outputSchema" in schema_def else {}),
    }


def handle_list_tools(msg_id: Any, send_result_fn):
    """List the tools in the session's toolset with schemas and hints."""
    tools_list = [tool_definition(schema_def) for schema_def in toolset_schemas(active_toolset())]
    send_result_fn(msg_id, {"tools": tools_list})

def handle_list_prompts(msg_id: Any, send_result_fn):
    from .prompts import list_prompts

    send_result_fn(msg_id, {"prompts": list_prompts(active_toolset())})


def handle_get_prompt(msg_id: Any, params: Any, send_error_fn, send_result_fn):
    from .prompts import get_prompt

    if not isinstance(params, dict):
        send_error_fn(msg_id, -32602, "prompts/get params must be an object.")
        return
    try:
        result = get_prompt(params.get("name"), params.get("arguments"), active_toolset())
    except ValueError as exc:
        send_error_fn(msg_id, -32602, str(exc))
        return
    send_result_fn(msg_id, result)


def handle_call_tool_with_task(session_id: str, msg_id: Any, name: str, arguments: Dict[str, Any], task_request: Dict[str, Any], send_result_fn, send_notification_fn=None, worker_fn=None):
    """Create a task for a tool call and return immediately."""
    if worker_fn is None:
        worker_fn = _run_tool_call_task_worker
        
    if send_notification_fn is None:
        # Fallback for environments where notification sender isn't explicitly provided
        send_notification_fn = lambda msg: send_result_fn(None, msg)
        
    ttl = task_request.get("ttl")
    task = create_task(name, ttl_ms=ttl)
    
    # Capture snapshot before starting thread to ensure 'working' status in immediate response
    task_snapshot = public_task(task)
    
    worker_args = (session_id, task["taskId"], name, arguments, send_notification_fn)
    # Support legacy 3-arg workers (from tests) by dropping notification sender
    try:
        import inspect
        sig = inspect.signature(worker_fn)
        if len(sig.parameters) == 3:
            worker_args = (task["taskId"], name, arguments)
    except Exception:
        pass

    threading.Thread(
        target=worker_fn,
        args=worker_args,
        daemon=True
    ).start()

    logger.debug("Task accepted: tool=%s task_id=%s", name, task["taskId"])

    payload = {
        "task": task_snapshot,
        "_meta": {
            "io.modelcontextprotocol/related-task": {"taskId": task["taskId"]},
            "io.modelcontextprotocol/model-immediate-response": (
                f"Task accepted for tool '{name}'. Monitor with tasks/get or tasks/result."
            )
        }
    }
    send_result_fn(msg_id, payload)

def _run_tool_call_task_worker(session_id: str, task_id: str, name: str, arguments: Dict[str, Any], send_notification_fn):
    """Background worker for task-backed tool calls."""
    # Crucial Fix: A spawned thread has an empty thread-local storage mapping.
    # We must forcibly re-inject the session ID so DynamicProxy can route state lookups.
    from .state import _thread_local
    _thread_local.mcp_session_id = session_id

    # Delay if requested (SOTA pattern)
    delay = float(os.environ.get("MUNINN_MCP_TASK_WORKER_START_DELAY_MS", "0")) / 1000.0
    if delay > 0:
        time.sleep(delay)
        
    task = lookup_task_locked(task_id)
    if not task:
        logger.error("Task worker started for unknown task: %s", task_id)
        return

    try:
        deadline = time.time() + (task["ttl"] / 1000.0)
        res = _do_call_tool_logic(name, arguments, deadline)
        if res is None:
             set_task_state_locked(task, status="failed", error={"code": -32601, "message": f"Method not found: {name}"})
        else:
             set_task_state_locked(task, status="completed", result=res)
    except Exception as e:
        logger.exception("Task execution failed: %s", task_id)
        set_task_state_locked(task, status="failed", error={"code": -32603, "message": str(e)})
    finally:
        emit_task_status_notification(task, send_notification_fn)

def _task_id_or_error(msg_id: Any, params: Dict[str, Any], method: str, send_error_fn) -> Optional[str]:
    """Extract taskId or send -32602 error."""
    task_id = params.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        send_error_fn(msg_id, -32602, f"Invalid params: 'taskId' is required for {method}")
        return None
    return task_id

def _task_result_should_block(params: Dict[str, Any]) -> bool:
    """Determine if a tasks/result request should block until completion."""
    from .tasks import task_result_should_block
    return task_result_should_block(params)

def _get_task_result_max_wait_seconds() -> float:
    """Host-safe budget for blocking on task results."""
    from .tasks import get_task_result_max_wait_seconds
    return get_task_result_max_wait_seconds()

def handle_get_task_result(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn):
    """Handle tasks/result requests with optional blocking wait."""
    task_id = _task_id_or_error(msg_id, params, "tasks/result", send_error_fn)
    if not task_id:
        return

    # Validate wait param
    if "wait" in params and not isinstance(params["wait"], bool):
        send_error_fn(msg_id, -32602, "Invalid params: wait must be a boolean")
        return

    should_block = _task_result_should_block(params)
    max_wait_seconds = _get_task_result_max_wait_seconds()
    wait_started_at = time.monotonic()

    from .state import _TASKS_CONDITION
    from .tasks import lookup_task_locked, TASK_TERMINAL_STATUSES, related_task_meta

    with _TASKS_CONDITION:
        while True:
            task = lookup_task_locked(task_id)
            if not task:
                send_error_fn(msg_id, -32602, "Invalid params: unknown taskId")
                return
            
            status = task.get("status")
            if status in TASK_TERMINAL_STATUSES or status == "input_required":
                break
            
            if not should_block:
                send_error_fn(msg_id, -32002, f"Task result is not ready (status: {status}). Poll or use blocking mode.")
                return

            wait_timeout = 0.1
            if max_wait_seconds > 0:
                elapsed = time.monotonic() - wait_started_at
                remaining = max_wait_seconds - elapsed
                if remaining <= 0:
                    send_error_fn(msg_id, -32002, "Task result not ready within host-safe budget.")
                    return
                wait_timeout = min(wait_timeout, remaining)
            
            _TASKS_CONDITION.wait(timeout=wait_timeout)

        # Terminal state reached
        payload = task.get("result")
        error = task.get("error")

    if error:
        # Relay task error through the JSON-RPC error channel, not the result channel.
        # error dict contains {"code": int, "message": str} from set_task_state_locked.
        err_code = error.get("code", -32603) if isinstance(error, dict) else -32603
        err_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        send_error_fn(msg_id, err_code, err_msg)
        return

    if not isinstance(payload, dict):
        send_error_fn(msg_id, -32603, "Task reached terminal state without a result payload.")
        return

    # Add related-task metadata
    result_with_meta = payload.copy()
    meta = result_with_meta.get("_meta", {})
    if isinstance(meta, dict):
        meta.update(related_task_meta(task_id))
        result_with_meta["_meta"] = meta
    else:
        result_with_meta["_meta"] = related_task_meta(task_id)
    
    send_result_fn(msg_id, result_with_meta)

def handle_call_tool(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn):
    """Execute a single tool call."""
    from muninn.core.security import verify_token
    # Security Gate: Stdio transport defaults to environmental trust (the MCP process
    # is started locally by the authorized user).  When MUNINN_MCP_STRICT_AUTH=1 is
    # set, an explicit bearer token must be present in _meta.token for defense-in-depth.
    if env_flag("MUNINN_MCP_STRICT_AUTH", False):
        meta = params.get("_meta") or {}
        if not verify_token(meta.get("token")):
            send_error_fn(msg_id, -32600, "Unauthorized: valid bearer token required in _meta.token")
            return
    name = params.get("name")
    if not isinstance(name, str) or not name:
        send_error_fn(msg_id, -32602, "Invalid params: tools/call requires non-empty string name")
        return
        
    task_request = params.get("task")
    if task_request is not None and not isinstance(task_request, dict):
        send_error_fn(msg_id, -32602, "Invalid params: task must be an object")
        return

    # Auto-defer long tools to task mode (Phase 5A.1)
    if task_request is None and env_flag("MUNINN_MCP_AUTO_TASK_FOR_LONG_TOOLS", False):
        long_tools = os.environ.get("MUNINN_MCP_AUTO_TASK_TOOL_NAMES", "").split(",")
        if name in long_tools:
            require_cap = env_flag("MUNINN_MCP_AUTO_TASK_REQUIRE_CLIENT_CAP", False)
            client_caps = _SESSION_STATE.get("client_capabilities", {})
            # If gate is enabled, check if client supports tasks
            if not require_cap or client_caps.get("tasks"):
                # Redirect to task handler — grab session_id from thread-local (SSE) or default to stdio
                from .state import _thread_local as _state_tl
                sid = getattr(_state_tl, "mcp_session_id", "stdio")
                handle_call_tool_with_task(sid, msg_id, name, params.get("arguments", {}), {}, send_result_fn)
                return

    arguments = params.get("arguments", {})
    deadline = get_tool_call_deadline_epoch()
    
    # Validation for specific tools that need early exit
    if name == "delete_all_memories" and not arguments.get("confirm", False):
        send_result_fn(msg_id, tool_error_result("Must set 'confirm: true' to delete all memories"))
        return

    # Metrics tracking (integrated into threading)
    tool_metrics = {
        "msg_id": msg_id,
        "name": name,
        "response_count": 0,
        "response_bytes_total": 0,
        "response_bytes_max": 0,
        "saw_error": False,
    }
    setattr(_thread_local, "tool_call_metrics", tool_metrics)
    tool_call_started_monotonic = time.monotonic()

    try:
        if env_flag("MUNINN_MCP_AUTOSTART_SERVER", True) and not is_circuit_open():
            if startup_recovery_allowed(deadline):
                ensure_server_running()

        res = _do_call_tool_logic(name, arguments, deadline)
        if res is None: # Tool not found
            send_error_fn(msg_id, -32601, f"Method not found: {name}")
            return

        structured = None
        if name in EXACT_JSON_TOOLS and res.get("success"):
            # Exact JSON: the preview compactor cuts anything nested past two levels.
            text_response = json.dumps(res["data"], indent=1, default=str)
            if name in STRUCTURED_TOOLS:
                structured = res["data"]
        else:
            text_response = format_tool_result_text(res, name)
        truncated_text = truncate_tool_text(text_response, name)

        result = {"content": [{"type": "text", "text": truncated_text}]}
        if structured is not None:
            result["structuredContent"] = structured
        if not res.get("success"):
            result["isError"] = True
            tool_metrics["saw_error"] = True
        send_result_fn(msg_id, result)
        
        # Update metrics for success
        tool_metrics["response_count"] += 1
        # Simplified size tracking
        tool_metrics["response_bytes_total"] += len(truncated_text)
        tool_metrics["response_bytes_max"] = max(tool_metrics["response_bytes_max"], len(truncated_text))

    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        tool_metrics["saw_error"] = True
        # MCP 2025-11-25: execution and input-validation failures are tool
        # results with isError so the model can read them and self-correct.
        send_result_fn(msg_id, tool_error_result(str(e) or type(e).__name__))
    finally:
        # Telemetry logging matching the original wrapper
        elapsed_ms = (time.monotonic() - tool_call_started_monotonic) * 1000.0
        outcome = "error" if tool_metrics["saw_error"] else ("success" if tool_metrics["response_count"] > 0 else "no_response")
        
        logger.info(
            "Tool call telemetry: name=%s id=%r outcome=%s elapsed_ms=%.1f responses=%d "
            "response_bytes_total=%d response_bytes_max=%d",
            name, msg_id, outcome, elapsed_ms, 
            tool_metrics["response_count"], tool_metrics["response_bytes_total"], 
            tool_metrics["response_bytes_max"]
        )
        setattr(_thread_local, "tool_call_metrics", None)
def tool_error_result(message: str) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}


def _do_call_tool_logic(name: str, arguments: Dict[str, Any], deadline: Optional[float]) -> Optional[Dict[str, Any]]:
    """Dispatch to internal tool implementations."""
    if name not in TOOLSETS[active_toolset()]:
        return None
    dispatch = {
        "add_memory": _do_add_memory,
        "add_image_memory": _do_add_image_memory,
        "search_memory": _do_search_memory,
        "hunt_memory": _do_hunt_memory,
        "get_all_memories": _do_get_all_memories,
        "update_memory": _do_update_memory,
        "delete_memory": _do_delete_memory,
        "delete_all_memories": _do_delete_all_memories,
        "set_project_instruction": _do_set_project_instruction,
        "set_project_goal": _do_set_project_goal,
        "get_project_goal": _do_get_project_goal,
        "set_user_profile": _do_set_user_profile,
        "get_user_profile": _do_get_user_profile,
        "get_model_profiles": _do_get_model_profiles,
        "set_model_profiles": _do_set_model_profiles,
        "get_model_profile_events": _do_get_model_profile_events,
        "get_model_profile_alerts": _do_get_model_profile_alerts,
        "export_handoff": _do_export_handoff,
        "import_handoff": _do_import_handoff,
        "record_retrieval_feedback": _do_record_retrieval_feedback,
        "ingest_sources": _do_ingest_sources,
        "discover_legacy_sources": _do_discover_legacy_sources,
        "ingest_legacy_sources": _do_ingest_legacy_sources,
        "get_periodic_ingestion_status": _do_get_periodic_ingestion_status,
        "run_periodic_ingestion": _do_run_periodic_ingestion,
        "start_periodic_ingestion": _do_start_periodic_ingestion,
        "stop_periodic_ingestion": _do_stop_periodic_ingestion,
        "get_temporal_knowledge": _do_get_temporal_knowledge,
        "create_federation_manifest": _do_create_federation_manifest,
        "calculate_federation_delta": _do_calculate_federation_delta,
        "create_federation_bundle": _do_create_federation_bundle,
        "apply_federation_bundle": _do_apply_federation_bundle,
        "mimir_relay": _do_mimir_relay,
        "detect_information_gaps": _do_detect_information_gaps,
        "trigger_distillation": _do_trigger_distillation,
        "correct_fact": _do_correct_fact,
        "forage_knowledge": _do_forage_knowledge,
        "get_project_context": _do_get_project_context,
        "create_handoff": _do_create_handoff,
        "resume_handoff": _do_resume_handoff,
        "complete_handoff": _do_complete_handoff,
        "search": _do_chatgpt_search,
        "fetch": _do_chatgpt_fetch,
    }
    
    handler = dispatch.get(name)
    if not handler:
        return None
    return handler(arguments, deadline)

# --- Who is calling, and for which project -------------------------------------
# Several agents (Claude Code, Claude Desktop, Codex in the ChatGPT app, Gemini,
# Cursor...) share one store, so every write records the agent and the project.
# Over HTTP the MCP code runs inside the server process, whose working directory
# says nothing about the client's repository, so the project must come from the
# tool arguments there; stdio wrappers run in the client's directory and may
# fall back to git.

_AGENT_ALIASES = {
    "claude-ai": "claude-desktop",
    "codex-mcp-client": "codex",
    "gemini-cli-mcp-client": "gemini-cli",
    "cursor-vscode": "cursor",
    "visual-studio-code": "vscode",
}


def _network_session() -> bool:
    from .state import get_current_session_id

    return get_current_session_id() not in ("default", "stdio")


def _normalize_agent(name: Any) -> str:
    text = "-".join(str(name or "").strip().lower().split())
    return _AGENT_ALIASES.get(text, text)[:64]


def client_agent() -> str:
    """Agent label: ?agent= on the MCP URL, MUNINN_AGENT_NAME (stdio), else the MCP clientInfo name."""
    chosen = _SESSION_STATE.get("agent_name")
    if not chosen and not _network_session():
        chosen = os.environ.get("MUNINN_AGENT_NAME")
    if not chosen:
        chosen = (_SESSION_STATE.get("client_info") or {}).get("name")
    return _normalize_agent(chosen) or "unknown"


def client_project(args: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
    """Project for this call: explicit argument, then (stdio only) MUNINN_PROJECT or the git repo."""
    args = args if isinstance(args, dict) else {}
    explicit = args.get("project")
    if isinstance(explicit, str) and explicit.strip():
        branch = args.get("branch")
        return {"project": explicit.strip(), "branch": branch if isinstance(branch, str) and branch else None}
    if _network_session():
        return {"project": None, "branch": None}
    pinned = os.environ.get("MUNINN_PROJECT", "").strip()
    if pinned:
        return {"project": pinned, "branch": None}
    git = get_git_info()
    if git.get("branch") in (None, "", "unknown"):
        return {"project": None, "branch": None}  # not inside a repository
    return {"project": git.get("project") or None, "branch": git.get("branch")}


def _do_add_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    metadata = inject_operator_profile_metadata(args.get("metadata", {}), operation="add")
    git = client_project(args)
    metadata.setdefault("project", git["project"] or "global")
    if git["branch"]:
        metadata.setdefault("branch", git["branch"])
    agent = client_agent()
    metadata.setdefault("agent", agent)

    # v3.11.0: Pass scope so the server persists it correctly
    scope = args.get("scope", "project")
    if scope not in ("project", "global"):
        scope = "project"

    # Phase 20: Multimodal support
    media_type = args.get("media_type", "text")

    payload = {
        "content": args.get("content"),
        "metadata": metadata,
        "user_id": "global_user",
        "agent_id": agent,
        "scope": scope,
        "media_type": media_type,
    }
    resp = make_request_with_retry(
        "POST",
        f"{SERVER_URL}/add",
        deadline_epoch=deadline,
        json=payload,
        timeout=_write_timeout_seconds(),
    )
    return resp.json()


def _do_add_image_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    image_path = args.get("image_path")
    description = args.get("description")
    if not image_path or not description:
        return {"success": False, "error": "image_path and description are required"}

    git = client_project(args)
    metadata = inject_operator_profile_metadata(args.get("metadata", {}), operation="add")
    metadata.setdefault("project", git["project"] or "global")
    if git["branch"]:
        metadata.setdefault("branch", git["branch"])
    metadata.setdefault("agent", client_agent())
    scope = args.get("scope", "project")
    if scope not in ("project", "global"):
        scope = "project"
    payload = {
        "image_path": image_path,
        "description": description,
        "project": git["project"] or "global",
        "namespace": args.get("namespace", "global"),
        "scope": scope,
        "metadata": metadata,
        "linked_memory_ids": args.get("linked_memory_ids"),
        "user_id": "global_user",
    }
    response = make_request_with_retry(
        "POST",
        f"{SERVER_URL}/add-image",
        deadline_epoch=deadline,
        json=payload,
        timeout=_write_timeout_seconds(),
    )
    return response.json()

def _do_set_project_instruction(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    """Convenience tool: create a scope='project' memory tagged with the current git project.

    The resulting memory will NEVER be returned when working in a different repo,
    preventing cross-project instruction bleed.
    """
    instruction = args.get("instruction", "")
    category = args.get("category", "project_instruction")
    if not instruction:
        return {"success": False, "error": "instruction is required"}

    git = client_project(args)
    agent = client_agent()
    metadata = inject_operator_profile_metadata(
        {"project": git["project"] or "global", "category": category, "agent": agent},
        operation="add",
    )
    if git["branch"]:
        metadata["branch"] = git["branch"]

    payload = {
        "content": instruction,
        "metadata": metadata,
        "user_id": "global_user",
        "agent_id": agent,
        "scope": "project",
    }
    resp = make_request_with_retry(
        "POST",
        f"{SERVER_URL}/add",
        deadline_epoch=deadline,
        json=payload,
        timeout=_write_timeout_seconds(),
    )
    return resp.json()


# Legacy stdio sessions all report "default"; give this process its own key so
# separate agents sharing one backend do not inhibit each other's results.
_STDIO_SEARCH_SESSION_ID = f"stdio-{uuid.uuid4().hex}"


def _search_session_id() -> Optional[str]:
    from .state import get_current_session_id

    session_id = get_current_session_id()
    if session_id.startswith("stateless-"):
        # MCP 2026-07-28 has no sessions; each request's context is throwaway.
        return None
    return _STDIO_SEARCH_SESSION_ID if session_id in ("default", "stdio") else session_id


def _do_search_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    filters = dict(args.get("filters") or {})
    auto_project = False
    if "project" not in filters and git["project"]:
        filters["project"] = git["project"]
        auto_project = True

    payload = {
        "query": args.get("query"),
        "limit": args.get("limit", 5),
        "rerank": args.get("rerank", True),
        "user_id": "global_user",
        "filters": filters,
        "explain": args.get("explain", False),
        "media_type": args.get("media_type"),
        "session_id": args.get("session_id") or _search_session_id(),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/search", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    result = resp.json()

    from muninn.core.feature_flags import get_flags
    _flags = get_flags()
    # v3.11.0: project_scope_strict disables fallback entirely — zero cross-project leakage
    fallback_enabled = (
        auto_project
        and result.get("success")
        and not result.get("data")
        and env_flag("MUNINN_MCP_SEARCH_PROJECT_FALLBACK", True)
        and not _flags.project_scope_strict
    )
    if fallback_enabled:
        logger.info("Retrying search without project filter (scope=global only)")
        # Copy filters to avoid mutating the original object referenced by test mocks.
        # Critical: add scope=global so we NEVER leak scope=project memories from other projects.
        fallback_filters = filters.copy()
        fallback_filters.pop("project")
        fallback_filters["scope"] = "global"
        payload["filters"] = fallback_filters
        resp = make_request_with_retry("POST", f"{SERVER_URL}/search", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
        result = resp.json()

    return result

def _do_hunt_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "query": args.get("query"),
        "limit": args.get("limit", 5),
        "depth": args.get("depth", 2),
        "user_id": "global_user",
        "namespaces": args.get("namespaces"),
        "synthesize": True,  # v3.18.1: always request LLM synthesis for richer tool output
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/search/hunt", deadline_epoch=deadline, json=payload, timeout=20)
    result = resp.json()
    # Surface synthesis narrative: format_tool_result_text only reads result["data"],
    # so inject synthesis as the first element of the list for LLM visibility.
    synthesis = result.get("synthesis", "")
    if synthesis and isinstance(result.get("data"), list):
        result["data"] = [{"synthesis_narrative": synthesis}] + result["data"]
    return result

def _do_get_all_memories(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"user_id": "global_user", "limit": args.get("limit", 100)}
    resp = make_request_with_retry("GET", f"{SERVER_URL}/get_all", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_update_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {"memory_id": args.get("memory_id"), "data": args.get("content")}
    resp = make_request_with_retry("PUT", f"{SERVER_URL}/update", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_delete_memory(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    mid = quote(str(args.get("memory_id")), safe="")
    resp = make_request_with_retry("DELETE", f"{SERVER_URL}/delete/{mid}", deadline_epoch=deadline, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_delete_all_memories(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {"user_id": "global_user"}
    resp = make_request_with_retry("POST", f"{SERVER_URL}/delete_all", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_set_project_goal(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "goal_statement": args.get("goal_statement"),
        "constraints": args.get("constraints", []),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/goal/set", deadline_epoch=deadline, json=payload, timeout=15)
    return resp.json()

def _do_get_project_goal(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    params = {
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
    }
    resp = make_request_with_retry("GET", f"{SERVER_URL}/goal/get", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_set_user_profile(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "user_id": "global_user",
        "profile": args.get("profile", {}),
        "merge": bool(args.get("merge", True)),
        "source": args.get("source", "mcp_tool"),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/profile/user/set", deadline_epoch=deadline, json=payload, timeout=15)
    return resp.json()

def _do_get_user_profile(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"user_id": "global_user"}
    resp = make_request_with_retry("GET", f"{SERVER_URL}/profile/user/get", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_get_model_profiles(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    resp = make_request_with_retry("GET", f"{SERVER_URL}/profiles/model", deadline_epoch=deadline, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_set_model_profiles(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {}
    for key in ("model_profile", "runtime_model_profile", "ingestion_model_profile", "legacy_ingestion_model_profile"):
        val = args.get(key)
        if val is not None:
            payload[key] = val
    if not payload:
        raise ValueError("set_model_profiles requires at least one profile field")
    payload["source"] = args.get("source", "mcp_tool")
    resp = make_request_with_retry("POST", f"{SERVER_URL}/profiles/model", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_get_model_profile_events(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"limit": args.get("limit", 25)}
    resp = make_request_with_retry("GET", f"{SERVER_URL}/profiles/model/events", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_get_model_profile_alerts(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {}
    for key in ("window_seconds", "churn_threshold", "source_churn_threshold", "distinct_sources_threshold"):
        value = args.get(key)
        if value is not None:
            params[key] = value
    resp = make_request_with_retry(
        "GET",
        f"{SERVER_URL}/profiles/model/alerts",
        deadline_epoch=deadline,
        params=params,
        timeout=DEFAULT_HTTP_TIMEOUT,
    )
    return resp.json()

def _do_export_handoff(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "limit": args.get("limit", 25),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/export", deadline_epoch=deadline, json=payload, timeout=30)
    return resp.json()

def _do_import_handoff(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "bundle": args.get("bundle"),
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "source": args.get("source", "mcp_import"),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/handoff/import", deadline_epoch=deadline, json=payload, timeout=30)
    return resp.json()

def _do_record_retrieval_feedback(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "query": args.get("query"),
        "memory_id": args.get("memory_id"),
        "outcome": args.get("outcome"),
        "rank": args.get("rank"),
        "sampling_prob": args.get("sampling_prob"),
        "signals": args.get("signals", {}),
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "source": args.get("source", "mcp_feedback"),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/feedback/retrieval", deadline_epoch=deadline, json=payload, timeout=15)
    return resp.json()

def _do_ingest_sources(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "sources": args.get("sources", []),
        "recursive": args.get("recursive", False),
        "chronological_order": args.get("chronological_order", "none"),
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "metadata": inject_operator_profile_metadata(args.get("metadata", {}), operation="ingest"),
        "max_file_size_bytes": args.get("max_file_size_bytes"),
        "chunk_size_chars": args.get("chunk_size_chars"),
        "chunk_overlap_chars": args.get("chunk_overlap_chars"),
        "min_chunk_chars": args.get("min_chunk_chars"),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest", deadline_epoch=deadline, json=payload, timeout=60)
    return resp.json()

def _do_discover_legacy_sources(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "roots": args.get("roots", []),
        "providers": args.get("providers", []),
        "include_unsupported": args.get("include_unsupported", False),
        "max_results_per_provider": args.get("max_results_per_provider", 100),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest/legacy/discover", deadline_epoch=deadline, json=payload, timeout=60)
    return resp.json()

def _do_ingest_legacy_sources(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    payload = {
        "selected_source_ids": args.get("selected_source_ids", []),
        "selected_paths": args.get("selected_paths", []),
        "roots": args.get("roots", []),
        "providers": args.get("providers", []),
        "include_unsupported": args.get("include_unsupported", False),
        "max_results_per_provider": args.get("max_results_per_provider", 100),
        "recursive": args.get("recursive", False),
        "chronological_order": args.get("chronological_order", "none"),
        "user_id": "global_user",
        "namespace": args.get("namespace", "global"),
        "project": git["project"] or "global",
        "metadata": inject_operator_profile_metadata(args.get("metadata", {}), operation="legacy_ingest"),
        "max_file_size_bytes": args.get("max_file_size_bytes"),
        "chunk_size_chars": args.get("chunk_size_chars"),
        "chunk_overlap_chars": args.get("chunk_overlap_chars"),
        "min_chunk_chars": args.get("min_chunk_chars"),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest/legacy/import", deadline_epoch=deadline, json=payload, timeout=120)
    return resp.json()

def _do_get_periodic_ingestion_status(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    resp = make_request_with_retry("GET", f"{SERVER_URL}/ingest/periodic/status", deadline_epoch=deadline, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_run_periodic_ingestion(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest/periodic/run", deadline_epoch=deadline, json={}, timeout=120)
    return resp.json()

def _do_start_periodic_ingestion(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest/periodic/start", deadline_epoch=deadline, json={}, timeout=15)
    return resp.json()

def _do_stop_periodic_ingestion(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    resp = make_request_with_retry("POST", f"{SERVER_URL}/ingest/periodic/stop", deadline_epoch=deadline, json={}, timeout=15)
    return resp.json()

def _do_get_temporal_knowledge(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"timestamp": args.get("timestamp"), "limit": args.get("limit", 50)}
    resp = make_request_with_retry("GET", f"{SERVER_URL}/knowledge/temporal", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_create_federation_manifest(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"project": args.get("project", "global")}
    resp = make_request_with_retry("POST", f"{SERVER_URL}/federation/manifest", deadline_epoch=deadline, params=params, timeout=15)
    return resp.json()

def _do_calculate_federation_delta(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "local": args.get("local", {}),
        "remote": args.get("remote", {}),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/federation/delta", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_create_federation_bundle(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = args.get("memory_ids", [])
    # Correctly send as body list? No, server endpoint expects list in body directly if declared as List[str],
    # but FastAPI usually wraps simple types or expects pydantic.
    # Check server.py: `async def create_federation_bundle_endpoint(memory_ids: List[str]):`
    # FastAPI expects a JSON body which is the list itself if not embedded in a dict key?
    # Actually, for List[str] body, it expects `["id1", "id2"]`.
    resp = make_request_with_retry("POST", f"{SERVER_URL}/federation/bundle", deadline_epoch=deadline, json=payload, timeout=30)
    return resp.json()

def _do_apply_federation_bundle(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    # server.py: `async def apply_federation_bundle_endpoint(bundle: Dict[str, Any]):`
    # Argument name is `bundle`, but in body it's just the dict.
    # FastAPI body logic: if it's a Pydantic model or Dict, it expects the JSON body to match.
    payload = args.get("bundle", {})
    resp = make_request_with_retry("POST", f"{SERVER_URL}/federation/apply", deadline_epoch=deadline, json=payload, timeout=30)
    return resp.json()


def _do_detect_information_gaps(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "query": args.get("query"),
        "context": args.get("context"),
        "user_id": "global_user",
        "limit": args.get("limit", 10),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/reasoning/detect-gaps", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_trigger_distillation(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {"force": bool(args.get("force", True))}
    resp = make_request_with_retry("POST", f"{SERVER_URL}/optimization/distill", deadline_epoch=deadline, json=payload, timeout=_write_timeout_seconds())
    return resp.json()

def _do_correct_fact(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {"memory_id": args.get("memory_id"), "correction": args.get("correction")}
    resp = make_request_with_retry("POST", f"{SERVER_URL}/optimization/correct", deadline_epoch=deadline, json=payload, timeout=_write_timeout_seconds())
    return resp.json()

def _do_forage_knowledge(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "query": args.get("query"),
        "ambiguity_threshold": args.get("ambiguity_threshold", 0.7),
        "user_id": "global_user",
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/optimization/forage", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    return resp.json()

def _do_get_project_context(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    params = {"user_id": "global_user", "recent_limit": args.get("recent_limit", 10)}
    project = client_project(args)["project"]
    if project:
        params["project"] = project
    resp = make_request_with_retry(
        "GET", f"{SERVER_URL}/context", deadline_epoch=deadline, params=params, timeout=DEFAULT_HTTP_TIMEOUT
    )
    return resp.json()

def _do_create_handoff(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    git = client_project(args)
    if not git["project"]:
        raise ValueError("create_handoff needs project: the repository or folder name you are working in")
    details = {field: args.get(field) for field in ("next_steps", "open_questions", "decisions", "files")}
    details["branch"] = args.get("branch") or git["branch"]
    payload = {
        "project": git["project"],
        "summary": args.get("summary"),
        "title": args.get("title"),
        "to_agent": args.get("to_agent"),
        "from_agent": client_agent(),
        "details": {key: value for key, value in details.items() if value},
        "user_id": "global_user",
    }
    resp = make_request_with_retry(
        "POST", f"{SERVER_URL}/handoffs", deadline_epoch=deadline, json=payload, timeout=_write_timeout_seconds()
    )
    return resp.json()

def _do_resume_handoff(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    payload = {
        "agent": client_agent(),
        "project": client_project(args)["project"],
        "handoff_id": args.get("handoff_id"),
        "claim": bool(args.get("claim", True)),
        "user_id": "global_user",
    }
    resp = make_request_with_retry(
        "POST", f"{SERVER_URL}/handoffs/resume", deadline_epoch=deadline, json=payload, timeout=_write_timeout_seconds()
    )
    return resp.json()

def _do_complete_handoff(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    handoff_id = str(args.get("handoff_id") or "")
    if not handoff_id:
        raise ValueError("complete_handoff requires handoff_id")
    payload = {
        "agent": client_agent(),
        "status": args.get("status", "done"),
        "note": args.get("note"),
        "user_id": "global_user",
    }
    resp = make_request_with_retry(
        "POST", f"{SERVER_URL}/handoffs/{quote(handoff_id, safe='')}/finish",
        deadline_epoch=deadline, json=payload, timeout=_write_timeout_seconds(),
    )
    return resp.json()

CHATGPT_SEARCH_LIMIT = 10
# Tools with an outputSchema return structuredContent alongside the text block.
STRUCTURED_TOOLS = {"search", "fetch"}
# Nested briefings and handoffs are returned whole rather than preview-compacted.
EXACT_JSON_TOOLS = STRUCTURED_TOOLS | {"get_project_context", "create_handoff", "resume_handoff", "complete_handoff"}
_TITLE_CHARS = 80


def _memory_url(memory_id: str) -> str:
    return f"muninn://memory/{quote(memory_id, safe='')}"


def _memory_title(content: str) -> str:
    first_line = " ".join((content or "").split())
    return first_line if len(first_line) <= _TITLE_CHARS else first_line[: _TITLE_CHARS - 1] + "…"


def _do_chatgpt_search(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    """ChatGPT connector search: {results: [{id, title, url}]} across all projects."""
    payload = {
        "query": args.get("query"),
        "limit": CHATGPT_SEARCH_LIMIT,
        "rerank": True,
        "user_id": "global_user",
        "session_id": _search_session_id(),
    }
    resp = make_request_with_retry("POST", f"{SERVER_URL}/search", deadline_epoch=deadline, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
    result = resp.json()
    if not result.get("success"):
        return result
    hits = [
        {"id": item["id"], "title": _memory_title(item.get("memory", "")), "url": _memory_url(item["id"])}
        for item in result.get("data") or []
        if isinstance(item, dict) and item.get("id")
    ]
    return {"success": True, "data": {"results": hits}}


def _do_chatgpt_fetch(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    """ChatGPT connector fetch: {id, title, text, url, metadata} for one memory."""
    memory_id = str(args.get("id") or "")
    if not memory_id:
        raise ValueError("fetch requires an 'id' returned by search")
    resp = make_request_with_retry("GET", f"{SERVER_URL}/memory/{quote(memory_id, safe='')}", deadline_epoch=deadline, timeout=DEFAULT_HTTP_TIMEOUT)
    result = resp.json()
    if not result.get("success"):
        return result
    record = result["data"]
    metadata = {
        key: record.get(key)
        for key in ("memory_type", "project", "namespace", "importance", "created_at", "archived")
    }
    return {
        "success": True,
        "data": {
            "id": record["id"],
            "title": _memory_title(record.get("memory", "")),
            "text": record.get("memory", ""),
            "url": _memory_url(record["id"]),
            "metadata": metadata,
        },
    }

def _do_mimir_relay(args: Dict[str, Any], deadline: Optional[float]) -> Dict[str, Any]:
    """Relay an instruction to a remote AI agent via the IRP/1 protocol.

    Calls POST /mimir/relay on the Muninn FastAPI server, which delegates to
    MimirRelay.  The HTTP timeout is derived from the IRP/1 policy.timeout_seconds
    value (default 30 s) plus a 10-second transport overhead budget.

    The server returns a RelayResult-shaped dict:
        {
            "run_id": str,
            "status": "success" | "failed" | "policy_blocked" | ...,
            "output": str | None,
            "error_code": str | None,
            "error_message": str | None,
            "latency_ms": float,
            "provider_used": str | None,
            "redaction_count": int,
        }
    """
    policy_raw: Dict[str, Any] = args.get("policy") or {}

    # Build the MimirRelayRequest body that the FastAPI endpoint expects.
    payload: Dict[str, Any] = {
        "instruction": args.get("instruction"),
        "mode": args.get("mode", "A"),
        "target": args.get("provider", "auto"),
        "user_id": args.get("user_id", "global_user"),
        "max_tokens": args.get("max_tokens", 4096),
    }

    # Only include optional fields when the caller supplied them so that the
    # server-side defaults in IRPPolicy remain authoritative.
    if args.get("context") is not None:
        payload["context"] = args["context"]

    if policy_raw:
        policy: Dict[str, Any] = {}
        if "max_hops" in policy_raw:
            policy["max_hops"] = int(policy_raw["max_hops"])
        if "timeout_seconds" in policy_raw:
            policy["timeout_seconds"] = float(policy_raw["timeout_seconds"])
        if "allow_redaction" in policy_raw:
            policy["allow_redaction"] = bool(policy_raw["allow_redaction"])
        if "require_consent" in policy_raw:
            policy["require_consent"] = bool(policy_raw["require_consent"])
        if "network" in policy_raw:
            policy["network"] = policy_raw["network"]
        if policy:
            payload["policy"] = policy

    # Give the HTTP call policy.timeout_seconds + 10 s for transport overhead.
    # If the caller omitted a policy timeout, fall back to 40 s (30 + 10).
    http_timeout = float(policy_raw.get("timeout_seconds", 30.0)) + 10.0

    resp = make_request_with_retry(
        "POST",
        f"{SERVER_URL}/mimir/relay",
        deadline_epoch=deadline,
        json=payload,
        timeout=http_timeout,
    )
    return resp.json()


# Task Handlers

def handle_list_tasks(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn):
    if not isinstance(params, dict):
        send_error_fn(msg_id, -32602, "tasks/list params must be an object")
        return

    cursor = params.get("cursor")
    limit = min(params.get("limit", TASKS_LIST_PAGE_SIZE), TASKS_LIST_PAGE_SIZE)
    
    try:
        offset = decode_task_cursor(cursor) if cursor else 0
    except ValueError as e:
        send_error_fn(msg_id, -32602, str(e))
        return

    registry = get_registry()
    purge_and_retain_tasks_locked(time.time())
    
    all_tasks = sorted(registry.values(), key=lambda t: t.get("createdAt", ""), reverse=True)
    page = all_tasks[offset : offset + limit]
    
    next_cursor = encode_task_cursor(offset + limit) if offset + limit < len(all_tasks) else None
    send_result = {
        "tasks": [public_task(t) for t in page],
    }
    if next_cursor:
        send_result["nextCursor"] = next_cursor
        
    send_result_fn(msg_id, send_result)

def handle_get_task(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn):
    if not isinstance(params, dict):
        send_error_fn(msg_id, -32602, "tasks/get params must be an object")
        return

    task_id = params.get("taskId")
    task = lookup_task_locked(task_id)
    if not task:
        # Sanitize ID in case of XSS attempt
        safe_id = str(task_id).replace("<", "&lt;").replace(">", "&gt;")
        send_error_fn(msg_id, -32002, f"Task not found: {safe_id}")
        return
    send_result_fn(msg_id, {"task": public_task(task)})

def handle_cancel_task(msg_id: Any, params: Dict[str, Any], send_error_fn, send_result_fn, send_notification_fn=None):
    if not isinstance(params, dict):
        send_error_fn(msg_id, -32602, "tasks/cancel params must be an object")
        return

    task_id = params.get("taskId")
    task = lookup_task_locked(task_id)
    if not task:
        send_error_fn(msg_id, -32002, f"Task not found: {task_id}")
        return
    
    if task["status"] in TASK_TERMINAL_STATUSES:
        send_error_fn(msg_id, -32602, f"Task {task_id} is already in terminal state {task['status']}")
        return

    if task["status"] not in TASK_TERMINAL_STATUSES:
        set_task_state_locked(task, status="cancelled", status_message="Cancelled by user request.")
        logger.info("Cancelled task %s", task_id)
        if send_notification_fn:
            emit_task_status_notification(task, send_notification_fn)
        
    send_result_fn(msg_id, {"task": public_task(task)})
