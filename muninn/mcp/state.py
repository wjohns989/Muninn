import threading
import time
from typing import Dict, Any, Tuple, Optional

# Global Session Contexts mapping session_id -> Session State
_SESSION_CONTEXTS: Dict[str, Dict[str, Any]] = {}
_SESSION_CONTEXTS_LOCK = threading.Lock()

# Thread-local storage for metrics/tracing and session routing
_thread_local = threading.local()

def _create_default_session_state() -> Dict[str, Any]:
    return {
        "negotiated": False,
        "initialized": False,
        "protocol_version": "2025-11-25",
        "client_capabilities": {},
        "client_info": {},
        "client_elicitation_modes": (),
        "tasks": {},
    }

def get_current_session_id() -> str:
    """Return the active SSE session_id for this thread, or 'default' for legacy transports."""
    return getattr(_thread_local, "mcp_session_id", "default")

def get_session_state() -> Dict[str, Any]:
    """Retrieve the isolated session state for the current thread context."""
    session_id = get_current_session_id()
    
    with _SESSION_CONTEXTS_LOCK:
        if session_id not in _SESSION_CONTEXTS:
            _SESSION_CONTEXTS[session_id] = _create_default_session_state()
        return _SESSION_CONTEXTS[session_id]

# Helper for Dynamic State Resolution (DSR) to support monkeypatching in tests
class _DynamicProxy(dict):
    def __init__(self, fallback_dict_getter, facade_attr_name):
        # We now accept a getter function (get_session_state) instead of a static dict
        self._fallback_getter = fallback_dict_getter
        self._attr = facade_attr_name
    
    def _resolve(self):
        try:
            import mcp_wrapper
            # If wrapper exists, it likely monkeypatched something locally.
            # In Phase 10 we don't strictly care, but we check if it exported _SESSION_STATE.
            # However, wrapper often aliases back to muninn.mcp.state._SESSION_STATE which is this proxy.
            # To avoid infinite recursion, we just use the getter.
            if hasattr(mcp_wrapper, self._attr) and getattr(mcp_wrapper, self._attr) is not self:
                 return getattr(mcp_wrapper, self._attr)
        except (ImportError, AttributeError):
            pass
        return self._fallback_getter()

    def __getitem__(self, k): return self._resolve()[k]
    def __setitem__(self, k, v): self._resolve()[k] = v
    def __delitem__(self, k): del self._resolve()[k]
    def __contains__(self, k): return k in self._resolve()
    def __len__(self): return len(self._resolve())
    def __iter__(self): return iter(self._resolve())
    def get(self, k, default=None): return self._resolve().get(k, default)
    def update(self, *args, **kwargs): self._resolve().update(*args, **kwargs)
    def clear(self): self._resolve().clear()
    def items(self): return self._resolve().items()
    def keys(self): return self._resolve().keys()
    def values(self): return self._resolve().values()
    def pop(self, *args): return self._resolve().pop(*args)
    def __repr__(self): return f"DynamicProxy({self._attr}, {self._resolve()})"

# Forward-declare for imports that depend on this specific variable being exported
_SESSION_STATE = _DynamicProxy(get_session_state, "_SESSION_STATE")

# For older tests explicitly modifying _REAL_SESSION_STATE, map it to 'default'
_REAL_SESSION_STATE = get_session_state() 

from .definitions import SUPPORTED_PROTOCOL_VERSIONS

# Transport & Circuit State
_TRANSPORT_CLOSED = threading.Event()

# Backend Circuit Breaker State
_BACKEND_CIRCUIT_LOCK = threading.RLock()
_BACKEND_CIRCUIT_STATE = {
    "consecutive_failures": 0,
    "open_until_epoch": 0.0,
}
_BACKEND_CIRCUIT_FAILURE_THRESHOLD = 5

# Task management locks
_TASKS_LOCK = threading.RLock()
_TASKS_CONDITION = threading.Condition(_TASKS_LOCK)

# RPC I/O locks
_RPC_WRITE_LOCK = threading.Lock()

# NOTE: _thread_local is defined once at line 10 for the entire module.
# Do NOT re-declare it here — that would orphan the _DynamicProxy binding.

# Dispatch locks
_DISPATCH_EXECUTOR_LOCK = threading.Lock()

def get_tasks_lock() -> threading.RLock:
    try:
        import mcp_wrapper
        if hasattr(mcp_wrapper, "_TASKS_LOCK"):
            return mcp_wrapper._TASKS_LOCK
    except (ImportError, AttributeError):
        pass
    return _TASKS_LOCK

def get_tasks_condition() -> threading.Condition:
    try:
        import mcp_wrapper
        if hasattr(mcp_wrapper, "_TASKS_CONDITION"):
            return mcp_wrapper._TASKS_CONDITION
    except (ImportError, AttributeError):
        pass
    return _TASKS_CONDITION

def get_rpc_write_lock() -> threading.Lock:
    return _RPC_WRITE_LOCK

def is_backend_circuit_open(now_epoch: Optional[float] = None) -> bool:
    if now_epoch is None:
        now_epoch = time.time()
    with _BACKEND_CIRCUIT_LOCK:
        return float(_BACKEND_CIRCUIT_STATE["open_until_epoch"]) > now_epoch
