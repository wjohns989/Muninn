import os
import time
import math
import json
import hashlib
import tempfile
import logging
import requests
import subprocess
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from .state import _BACKEND_CIRCUIT_STATE, _BACKEND_CIRCUIT_LOCK, is_backend_circuit_open

logger = logging.getLogger("Muninn.mcp.lifecycle")

# Path discovery (assuming we are in muninn/mcp/lifecycle.py)
MOD_DIR = Path(__file__).parent.resolve()
MUNINN_DIR = MOD_DIR.parent.parent.resolve()
SERVER_SCRIPT = MUNINN_DIR / "server.py"

# URLs
SERVER_URL = os.environ.get("MUNINN_SERVER_URL", "http://127.0.0.1:42069")
HEALTH_URL = f"{SERVER_URL}/health"
OLLAMA_URL = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434")

# Backend Circuit Breaker State (Linked to state.py)
_FAILURE_THRESHOLD = max(1, int(os.environ.get("MUNINN_MCP_BACKEND_FAILURE_THRESHOLD", "3")))
_COOLDOWN_SEC = max(1.0, float(os.environ.get("MUNINN_MCP_BACKEND_COOLDOWN_SEC", "30")))
_STARTUP_LOCK_TIMEOUT_SEC = max(
    0.1,
    float(os.environ.get("MUNINN_MCP_STARTUP_LOCK_TIMEOUT_SEC", "8.0")),
)
_SERVER_STARTUP_MAX_WAIT_SEC = max(
    1.0,
    float(os.environ.get("MUNINN_MCP_SERVER_STARTUP_MAX_WAIT_SEC", "30.0")),
)
_SERVER_STARTUP_POLL_SEC = max(
    0.05,
    float(os.environ.get("MUNINN_MCP_SERVER_STARTUP_POLL_SEC", "0.25")),
)
_STARTUP_FALLBACK_THREAD_LOCK = threading.Lock()

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
    from muninn.core.security import get_token
    python_executable = find_python_executable()
    try:
        # Propagate the auth token to the spawned server so both processes
        # share the same token. This is critical when MUNINN_AUTH_TOKEN is
        # not in the system environment (e.g., injected via the MCP -e config
        # for the wrapper only) — without it each process generates a different
        # random token and every tool call returns HTTP 401.
        token = os.environ.get("MUNINN_AUTH_TOKEN") or get_token()
        proc = spawn_detached_process(
            [python_executable, str(SERVER_SCRIPT)],
            cwd=str(MUNINN_DIR),
            env={"MUNINN_AUTH_TOKEN": token},
        )
        pid = getattr(proc, "pid", None)
        if isinstance(pid, int) and pid > 0:
            _write_server_pid_lease(pid)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def _server_pid_lease_path() -> Path:
    """Path for best-effort spawned-server PID lease metadata."""
    data_dir = os.environ.get("MUNINN_DATA_DIR")
    if data_dir:
        return Path(data_dir) / ".muninn_server.pid"
    return MUNINN_DIR / ".muninn_server.pid"

def _write_server_pid_lease(pid: int) -> None:
    """
    Best-effort lease file for local diagnostics.

    This is advisory metadata only; lock coordination is still done by
    `_startup_spawn_lock()` and backend liveness is determined by `/health`.
    """
    try:
        lease_path = _server_pid_lease_path()
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": int(pid),
            "server_url": SERVER_URL,
            "data_dir": os.environ.get("MUNINN_DATA_DIR"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        lease_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Unable to write server PID lease: %s", exc)

def _startup_lock_path() -> Path:
    """
    Resolve the cross-process startup lock path.

    Lock scope is keyed by SERVER_URL (not data_dir) so multiple wrappers that
    target the same endpoint coordinate startup even if their environment
    differs (for example, mixed `MUNINN_DATA_DIR` overrides).
    """
    url_hash = hashlib.sha1(SERVER_URL.encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"muninn_server_start_{url_hash}.lock"

@contextmanager
def _startup_spawn_lock():
    """
    Serialize backend startup attempts across MCP wrapper processes.

    Without this, multiple wrappers starting simultaneously can all fail the
    initial health probe and race to spawn duplicate `server.py` processes.
    """
    try:
        import portalocker

        lock_path = _startup_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(
            str(lock_path),
            mode="a",
            timeout=_STARTUP_LOCK_TIMEOUT_SEC,
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        ):
            yield
        return
    except Exception as exc:
        logger.debug("Cross-process startup lock unavailable; using in-process fallback: %s", exc)

    with _STARTUP_FALLBACK_THREAD_LOCK:
        yield

def ensure_server_running() -> bool:
    def _wait_for_ready(max_wait_sec: float) -> bool:
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            if is_server_running():
                return True
            time.sleep(_SERVER_STARTUP_POLL_SEC)
        return is_server_running()

    if is_server_running():
        return True
    try:
        with _startup_spawn_lock():
            if is_server_running():
                return True
            if not start_server():
                # Another coordinator may have started concurrently but we missed
                # health in the tiny window before this check.
                return _wait_for_ready(min(2.0, _SERVER_STARTUP_MAX_WAIT_SEC))
            # Keep lock while waiting so concurrent wrappers do not race-spawn
            # a second backend during slow cold starts.
            return _wait_for_ready(_SERVER_STARTUP_MAX_WAIT_SEC)
    except Exception as exc:
        logger.warning(
            "Startup lock acquisition failed; waiting for existing coordinator: %s",
            exc,
        )
        # Another process may currently hold the startup lock and be booting the
        # server; wait instead of immediately competing with an additional spawn.
        return _wait_for_ready(_SERVER_STARTUP_MAX_WAIT_SEC)
