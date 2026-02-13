"""
Muninn Platform Abstraction
----------------------------
Cross-platform path resolution, process management, and detection utilities.

Abstracts all OS-specific behavior behind a clean API so Muninn runs
identically on Windows, Linux, macOS, and Docker.

Uses platformdirs (MIT) when available, with manual fallbacks for
XDG/Windows/macOS conventions when the library is not installed.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("Muninn.Platform")

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

_APP_NAME = "muninn"
_APP_AUTHOR = "AntigravityLabs"


def is_running_in_docker() -> bool:
    """Detect if we're running inside a Docker container."""
    if os.environ.get("MUNINN_DOCKER") == "1":
        return True
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except (FileNotFoundError, PermissionError):
        return False


def get_platform_info() -> Dict[str, Any]:
    """Return platform diagnostic information for health endpoints."""
    return {
        "os": sys.platform,
        "python": sys.version,
        "is_windows": IS_WINDOWS,
        "is_macos": IS_MACOS,
        "is_linux": IS_LINUX,
        "is_docker": is_running_in_docker(),
        "data_dir": str(get_data_dir()),
        "config_dir": str(get_config_dir()),
        "log_dir": str(get_log_dir()),
    }


# ---------------------------------------------------------------------------
# Directory resolution (platformdirs with manual fallback)
# ---------------------------------------------------------------------------

def _fallback_user_data_dir() -> str:
    """Manual XDG / Windows / macOS data dir resolution."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return str(Path(base) / _APP_NAME)
    elif IS_MACOS:
        return str(Path.home() / "Library" / "Application Support" / _APP_NAME)
    else:
        xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        return str(Path(xdg) / _APP_NAME)


def _fallback_user_config_dir() -> str:
    """Manual XDG / Windows / macOS config dir resolution."""
    if IS_WINDOWS:
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return str(Path(base) / _APP_NAME)
    elif IS_MACOS:
        return str(Path.home() / "Library" / "Preferences" / _APP_NAME)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return str(Path(xdg) / _APP_NAME)


def _fallback_user_log_dir() -> str:
    """Manual XDG / Windows / macOS log dir resolution."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return str(Path(base) / _APP_NAME / "Logs")
    elif IS_MACOS:
        return str(Path.home() / "Library" / "Logs" / _APP_NAME)
    else:
        xdg = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
        return str(Path(xdg) / _APP_NAME / "log")


def _resolve_dir(env_var: str, platformdirs_fn: str, fallback_fn) -> Path:
    """
    Resolve a directory path with priority:
    1. Environment variable override (highest)
    2. platformdirs library (if installed)
    3. Manual OS-specific fallback (always works)
    """
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)

    try:
        import platformdirs
        fn = getattr(platformdirs, platformdirs_fn)
        return Path(fn(_APP_NAME, _APP_AUTHOR))
    except ImportError:
        return Path(fallback_fn())


def get_data_dir() -> Path:
    """
    Get the Muninn data directory.

    Priority: MUNINN_DATA_DIR env var > platformdirs > OS fallback.
    Docker override: /data when MUNINN_DOCKER=1.

    Contains: vectors/, graph/, metadata.db, bm25_index/
    """
    if is_running_in_docker():
        return Path(os.environ.get("MUNINN_DATA_DIR", "/data"))
    return _resolve_dir("MUNINN_DATA_DIR", "user_data_dir", _fallback_user_data_dir)


def get_config_dir() -> Path:
    """
    Get the Muninn configuration directory.

    Priority: MUNINN_CONFIG_DIR env var > platformdirs > OS fallback.
    Contains: config.yaml, feature overrides
    """
    if is_running_in_docker():
        return Path(os.environ.get("MUNINN_CONFIG_DIR", "/config"))
    return _resolve_dir("MUNINN_CONFIG_DIR", "user_config_dir", _fallback_user_config_dir)


def get_log_dir() -> Path:
    """
    Get the Muninn log directory.

    Priority: MUNINN_LOG_DIR env var > platformdirs > OS fallback.
    Contains: muninn.log, consolidation.log
    """
    if is_running_in_docker():
        return Path(os.environ.get("MUNINN_LOG_DIR", "/data/logs"))
    return _resolve_dir("MUNINN_LOG_DIR", "user_log_dir", _fallback_user_log_dir)


def get_legacy_data_dir() -> Path:
    """
    Return the legacy ~/.muninn/data path for migration detection.
    If data exists here but not in the new location, we can offer migration.
    """
    return Path.home() / ".muninn" / "data"


# ---------------------------------------------------------------------------
# Process management (cross-platform)
# ---------------------------------------------------------------------------

def get_process_creation_flags() -> int:
    """
    Get subprocess creation flags for detached background processes.

    Windows: CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    Unix/Docker: 0 (no special flags needed)
    """
    if IS_WINDOWS:
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        return CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    return 0


def spawn_detached_process(
    args: list,
    cwd: str | Path | None = None,
    env: dict | None = None,
) -> subprocess.Popen:
    """
    Launch a subprocess fully detached from the parent.

    Cross-platform: uses creationflags on Windows, preexec_fn on Unix.

    Args:
        args: Command and arguments list.
        cwd: Working directory for the subprocess.
        env: Optional environment variable overrides (merged with os.environ).

    Returns:
        The Popen object for the detached process.
    """
    merged_env = {**os.environ, **(env or {})}
    kwargs = {
        "args": args,
        "cwd": str(cwd) if cwd else None,
        "env": merged_env,
        "close_fds": True,
        "shell": False,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }

    if IS_WINDOWS:
        kwargs["creationflags"] = get_process_creation_flags()
    else:
        # On Unix, start new session to fully detach
        kwargs["start_new_session"] = True

    return subprocess.Popen(**kwargs)


def find_python_executable() -> str:
    """
    Find the best Python executable for spawning subprocesses.

    Windows: prefers pythonw.exe (no console window), falls back to python.exe.
    Unix: uses sys.executable directly.
    """
    if IS_WINDOWS:
        python_dir = Path(sys.executable).parent
        pythonw = python_dir / "pythonw.exe"
        if pythonw.exists():
            return str(pythonw)
    return sys.executable


# ---------------------------------------------------------------------------
# Ollama detection (cross-platform)
# ---------------------------------------------------------------------------

def find_ollama_executable() -> str | None:
    """
    Locate the Ollama executable across platforms.

    Returns:
        Full path to ollama binary, or None if not found.
    """
    import shutil

    # Check PATH first (works on all platforms)
    ollama_path = shutil.which("ollama")
    if ollama_path:
        return ollama_path

    # Platform-specific common locations
    candidates = []
    if IS_WINDOWS:
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidates.append(Path(local_app) / "Programs" / "Ollama" / "ollama.exe")
        candidates.append(Path("C:/Program Files/Ollama/ollama.exe"))
    elif IS_MACOS:
        candidates.append(Path("/usr/local/bin/ollama"))
        candidates.append(Path.home() / ".ollama" / "ollama")
    else:
        candidates.append(Path("/usr/local/bin/ollama"))
        candidates.append(Path("/usr/bin/ollama"))
        candidates.append(Path.home() / ".local" / "bin" / "ollama")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> Dict[str, Path]:
    """
    Create all required Muninn directories and return their paths.

    Returns:
        Dict mapping directory purpose to resolved Path.
    """
    dirs = {
        "data": get_data_dir(),
        "config": get_config_dir(),
        "logs": get_log_dir(),
        "vectors": get_data_dir() / "qdrant_v8",
        "graph": get_data_dir() / "kuzu_v12",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    logger.info("Platform: %s | Data: %s", sys.platform, dirs["data"])
    return dirs


def log_platform_summary() -> None:
    """Log a one-line platform summary at startup."""
    info = get_platform_info()
    docker_tag = " [Docker]" if info["is_docker"] else ""
    logger.info(
        "Muninn on %s%s | Python %s | Data: %s",
        info["os"],
        docker_tag,
        sys.version.split()[0],
        info["data_dir"],
    )
