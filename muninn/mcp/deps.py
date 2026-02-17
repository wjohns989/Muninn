"""
Muninn MCP Dependency & Startup Management
"""

import requests
import logging
import os
import time
from pathlib import Path
from muninn.platform import spawn_detached_process, find_python_executable, find_ollama_executable

logger = logging.getLogger("Muninn.MCP.Deps")

def is_server_running(url: str) -> bool:
    try:
        response = requests.get(f"{url}/health", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False

def is_ollama_running(url: str) -> bool:
    try:
        response = requests.get(url, timeout=0.5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def check_and_start_ollama(url: str):
    if is_ollama_running(url):
        return True
    
    try:
        ollama_path = find_ollama_executable()
        if not ollama_path:
            return False

        spawn_detached_process([ollama_path, "serve"])
        for _ in range(20):
            if is_ollama_running(url):
                return True
            time.sleep(0.5)
        return False
    except Exception as e:
        logger.error(f"Failed to launch Ollama: {e}")
        return False

def start_server(script_path: Path, cwd: Path):
    python_executable = find_python_executable()
    try:
        spawn_detached_process(
            [python_executable, str(script_path)],
            cwd=str(cwd),
        )
        time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def ensure_server_running(url: str, script_path: Path, cwd: Path):
    if is_server_running(url):
        return True
    if not start_server(script_path, cwd):
        return False
    for _ in range(20):
        if is_server_running(url):
            return True
        time.sleep(0.25)
    return False
