import os
import json
import logging
import subprocess
import datetime
from typing import Any, Dict, Optional, List

from .definitions import SUPPORTED_MODEL_PROFILES

logger = logging.getLogger("Muninn.mcp.utils")

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

def _do_get_git_info() -> Dict[str, str]:
    """Retrieve Git branch and repository name for contextual metadata."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        
        repo_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        
        project = repo_url.split("/")[-1].replace(".git", "") if repo_url else "unknown"
        return {"branch": branch, "project": project}
    except Exception:
        return {"branch": "unknown", "project": os.path.basename(os.getcwd())}

def get_git_info() -> Dict[str, str]:
    """Facade-aware get_git_info wrapper."""
    try:
        import mcp_wrapper
        if hasattr(mcp_wrapper, "get_git_info") and mcp_wrapper.get_git_info != get_git_info:
            return mcp_wrapper.get_git_info()
    except (ImportError, AttributeError):
        pass
    return _do_get_git_info()

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

def inject_operator_profile_metadata(metadata: Dict[str, Any], operation: str = "add") -> Dict[str, Any]:
    """Incorporate operator and environment details into metadata."""
    metadata = dict(metadata or {})
    metadata.setdefault("operator", os.environ.get("USER", os.environ.get("USERNAME", "unknown")))
    metadata.setdefault("operation", operation)
    # Using the same utcnow pattern as legacy for compatibility if needed, 
    # but datetime.datetime.utcnow().isoformat() is standard
    metadata.setdefault("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
    
    session_profile = _get_operator_model_profile_for_operation(operation)
    if session_profile:
        metadata.setdefault("operator_model_profile", session_profile)
        
    return metadata

def truncate_tool_text(text: str, name: str) -> str:
    """Apply global and per-tool length constraints to responses."""
    max_chars = int(os.environ.get("MUNINN_MCP_TOOL_RESPONSE_MAX_CHARS", "32768"))
    if len(text) > max_chars:
        logger.info("Truncating response for tool '%s' (%d -> %d chars)", name, len(text), max_chars)
        suffix = "\n\n[Response truncated due to size limits]"
        cutoff = max(0, max_chars - len(suffix))
        return text[:cutoff] + suffix
    return text

def format_tool_result_text(result: Dict[str, Any], name: str) -> str:
    """Convert backend JSON result to standard text representation for tool output."""
    if not result.get("success"):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    data = result.get("data")
    if data is None:
        return "Success"
        
    # Phase 5A.8: Compact large payloads
    compacted, _ = _compact_tool_response_payload(data)
    
    if isinstance(compacted, (dict, list)):
        formatted = json.dumps(compacted, indent=2)
    else:
        formatted = str(compacted)
        
    return truncate_tool_text(formatted, name)

def env_flag(key: str, default: bool) -> bool:
    """Helper to parse boolean flags from environment variables."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

def negotiated_protocol_version(requested: Optional[str]) -> Optional[str]:
    """Negotiate the protocol version with the client."""
    from .definitions import SUPPORTED_PROTOCOL_VERSIONS
    if not requested:
        return SUPPORTED_PROTOCOL_VERSIONS[0]
    if requested in SUPPORTED_PROTOCOL_VERSIONS:
        return requested
    return None

def build_initialize_instructions(startup_warnings: Optional[List[str]] = None) -> str:
    """Build a set of instructions for the client during initialization."""
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

def get_host_safe_tool_call_budget_seconds() -> float:
    """Calculate the maximum safe time an RPC call can block before the host likely timeouts."""
    import math
    import time
    host_timeout = 120.0
    host_timeout_raw = os.environ.get("MUNINN_MCP_HOST_TOOLS_CALL_TIMEOUT_SEC", "120")
    try:
        parsed_host_timeout = float(host_timeout_raw)
        if math.isfinite(parsed_host_timeout) and parsed_host_timeout > 0:
            host_timeout = parsed_host_timeout
    except ValueError:
        pass

    margin = 10.0
    margin_raw = os.environ.get("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "10")
    try:
        parsed_margin = float(margin_raw)
        if math.isfinite(parsed_margin) and parsed_margin >= 0:
            margin = parsed_margin
    except ValueError:
        pass

    derived = host_timeout - margin
    return max(1.0, derived)

def get_tool_call_deadline_epoch() -> float:
    """Calculate the absolute epoch deadline for a tool call starting now."""
    import time
    
    # Priority 1: Explicit epoch from environment (for distributed context)
    raw = os.environ.get("MUNINN_MCP_TOOL_CALL_DEADLINE_EPOCH")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass

    safe_budget = get_host_safe_tool_call_budget_seconds()

    # Priority 2: Explicit duration from environment
    raw_sec = os.environ.get("MUNINN_MCP_TOOL_CALL_DEADLINE_SEC")
    if raw_sec and raw_sec != "0":
        try:
            explicit = float(raw_sec)
            # Clamp if overrun not allowed
            if not env_flag("MUNINN_MCP_TOOL_CALL_DEADLINE_ALLOW_OVERRUN", False):
                explicit = min(explicit, safe_budget)
            return time.monotonic() + explicit
        except ValueError:
            pass
            
    # Priority 3: Derived from host timeout
    return time.monotonic() + safe_budget

def remaining_deadline_seconds(deadline_epoch: Optional[float]) -> Optional[float]:
    import time
    if deadline_epoch is None:
        return None
    return max(0.0, deadline_epoch - time.time())

def startup_recovery_allowed(deadline_epoch: Optional[float]) -> bool:
    if deadline_epoch is None:
        return True
    margin = float(os.environ.get("MUNINN_MCP_TOOL_CALL_DEADLINE_MARGIN_SEC", "1.5"))
    remaining = remaining_deadline_seconds(deadline_epoch)
    return remaining > margin if remaining is not None else True

def _get_tool_call_warn_ms() -> float:
    """Legacy helper for tool call warning threshold."""
    import math
    raw_value = os.environ.get("MUNINN_MCP_TOOL_CALL_WARN_MS", "90000")
    try:
        parsed = float(raw_value)
    except ValueError:
        return 90000.0
    if not math.isfinite(parsed) or parsed < 0:
        return 90000.0
    return parsed

def _get_tool_response_preview_max_string_chars() -> int:
    """Legacy helper for response preview limits."""
    raw_value = os.environ.get("MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_STRING_CHARS", "2000")
    try:
        parsed = int(raw_value)
    except ValueError:
        return 2000
    return max(32, min(parsed, 500000))

def _safe_json_dumps(payload: Any) -> str:
    """Legacy helper for robust JSON serialization."""
    try:
        return json.dumps(payload, indent=2)
    except TypeError:
        return json.dumps(str(payload), indent=2)

def _truncate_preview_string(value: str, max_chars: int) -> tuple[str, bool]:
    """Legacy helper for string truncation with notification."""
    if len(value) <= max_chars:
        return value, False
    omitted = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {omitted} chars]", True

def public_tool_error_message(error: Exception) -> str:
    """Sanitize error messages for public tool output."""
    import re
    msg = str(error)
    # Redact IP addresses
    msg = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED_IP]", msg)
    if "socket timeout" in msg or "ConnectionError" in msg:
        msg = "Unable to reach backend service"
    return msg

def _compact_tool_response_payload(payload: Any) -> tuple[Any, bool]:
    """Legacy helper for bounding payload shape before serialization."""
    max_chars = _get_tool_response_preview_max_string_chars()
    max_items = int(os.environ.get("MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_ITEMS", "20"))
    max_depth = int(os.environ.get("MUNINN_MCP_TOOL_RESPONSE_PREVIEW_MAX_DEPTH", "2"))

    def _compact(obj, depth):
        if depth > max_depth:
            return "[max preview depth reached]", True
        
        if isinstance(obj, str):
            return _truncate_preview_string(obj, max_chars)
            
        if isinstance(obj, list):
            if len(obj) > max_items:
                truncated_len = len(obj) - max_items
                # Recurse on kept items
                compacted_items = [_compact(x, depth+1)[0] for x in obj[:max_items]]
                return compacted_items + [f"_muninn_truncated_items: ... truncated {truncated_len} items"], True
            
            res = []
            any_ch = False
            for x in obj:
                v, ch = _compact(x, depth+1)
                res.append(v)
                if ch: any_ch = True
            return res, any_ch
            
        if isinstance(obj, dict):
            res = {}
            any_ch = False
            for k, v in obj.items():
                val, ch = _compact(v, depth+1)
                res[k] = val
                if ch: any_ch = True
            return res, any_ch
            
        return obj, False

    return _compact(payload, 0)