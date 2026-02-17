import threading
import time
from typing import Dict, Any, Tuple, Optional

# Helper for Dynamic State Resolution (DSR) to support monkeypatching in tests
class _DynamicProxy(dict):
    def __init__(self, fallback_dict, facade_attr_name):
        self._fallback = fallback_dict
        self._attr = facade_attr_name
    
    def _resolve(self):
        try:
            import mcp_wrapper
            return getattr(mcp_wrapper, self._attr, self._fallback)
        except (ImportError, AttributeError):
            return self._fallback

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

# Global Session State
_REAL_SESSION_STATE = {
    "negotiated": False,
    "initialized": False,
    "protocol_version": "2025-11-25",
    "client_capabilities": {},
    "client_info": {},
    "client_elicitation_modes": (),
    "tasks": {},
}

_SESSION_STATE = _DynamicProxy(_REAL_SESSION_STATE, "_SESSION_STATE")

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

# Thread-local storage for metrics/tracing
_thread_local = threading.local()

# Dispatch locks
_DISPATCH_EXECUTOR_LOCK = threading.Lock()

def get_session_state() -> Dict[str, Any]:
    return _SESSION_STATE

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
