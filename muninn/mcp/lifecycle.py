import os
import time
import math
import logging
import requests
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from .state import _BACKEND_CIRCUIT_STATE, _BACKEND_CIRCUIT_LOCK, is_backend_circuit_open

logger = logging.getLogger("Muninn.mcp.lifecycle")

# Path discovery (assuming we are in muninn/mcp/lifecycle.py)
MOD_DIR = Path(__file__).parent.resolve()
GLOBAL_MEMORY_DIR = MOD_DIR.parent.parent.resolve()
SERVER_SCRIPT = GLOBAL_MEMORY_DIR / "server.py"

# URLs
SERVER_URL = os.environ.get("MUNINN_SERVER_URL", "http://localhost:42069")
HEALTH_URL = f"{SERVER_URL}/health"
OLLAMA_URL = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434")

# Backend Circuit Breaker State (Linked to state.py)
_FAILURE_THRESHOLD = max(1, int(os.environ.get("MUNINN_MCP_BACKEND_FAILURE_THRESHOLD", "3")))
_COOLDOWN_SEC = max(1.0, float(os.environ.get("MUNINN_MCP_BACKEND_COOLDOWN_SEC", "30")))

class BackendCircuitOpenError(requests.ConnectionError):
    """Raised when backend requests are short-circuited during cooldown."""

def is_circuit_open(now_epoch: Optional[float] = None) -> bool:
    return is_backend_circuit_open(now_epoch)

def mark_success() -> None:
    with _BACKEND_CIRCUIT_LOCK:
        _BACKEND_CIRCUIT_STATE["consecutive_failures"] = 0
        _BACKEND_CIRCUIT_STATE["open_until_epoch"] = 0.0

def mark_failure(error: Exception) -> None:
    now = time.time()
    with _BACKEND_CIRCUIT_LOCK:
        failures = int(_BACKEND_CIRCUIT_STATE["consecutive_failures"]) + 1
        _BACKEND_CIRCUIT_STATE["consecutive_failures"] = failures
        if failures >= _FAILURE_THRESHOLD:
            _BACKEND_CIRCUIT_STATE["open_until_epoch"] = now + _COOLDOWN_SEC
            logger.warning(
                "Backend circuit opened for %.1fs after %d consecutive failures: %s",
                _COOLDOWN_SEC,
                failures,
                error,
            )

# Lifecycle Management

def is_server_running() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def is_ollama_running() -> bool:
    try:
        response = requests.get(OLLAMA_URL, timeout=0.5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_and_start_ollama() -> bool:
    from muninn.platform import spawn_detached_process, find_ollama_executable
    if is_ollama_running():
        return True
    
    ollama_path = find_ollama_executable()
    if not ollama_path:
        logger.error("Ollama executable not found.")
        return False

    try:
        spawn_detached_process([ollama_path, "serve"])
        for _ in range(20):
            if is_ollama_running():
                return True
            time.sleep(0.5)
        return False
    except Exception as e:
        logger.error(f"Failed to launch Ollama: {e}")
        return False

def start_server() -> bool:
    from muninn.platform import spawn_detached_process, find_python_executable
    python_executable = find_python_executable()
    try:
        spawn_detached_process(
            [python_executable, str(SERVER_SCRIPT)],
            cwd=str(GLOBAL_MEMORY_DIR),
        )
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def ensure_server_running() -> bool:
    if is_server_running():
        return True
    if not start_server():
        return False
    for _ in range(20):
        if is_server_running():
            return True
        time.sleep(0.25)
    return False