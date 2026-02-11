"""
Muninn System Tray Application
-------------------------------
Windows system tray wrapper for Muninn Memory Server.
Provides server lifecycle management, status monitoring, and quick access to dashboard.
"""

import pystray
from PIL import Image, ImageDraw, ImageFont
import subprocess
import webbrowser
import os
import sys
import threading
import time
import socket
import json
import logging
import winreg
from pathlib import Path
from typing import Optional

import requests

# --- Configuration ---
PROJECT_NAME = "Muninn"
SERVER_URL = "http://localhost:8000"
HEALTH_URL = f"{SERVER_URL}/health"
OLLAMA_URL = "http://localhost:11434"
CWD = Path(__file__).parent.resolve()
SERVER_SCRIPT = CWD / "server.py"
LOG_FILE = CWD / "tray_app.log"
PYTHON_EXE = sys.executable

# Startup registry key
STARTUP_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
STARTUP_REG_NAME = "MuninnMemory"

# Windows process creation flags
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# Status polling interval (seconds)
POLL_INTERVAL = 10

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=str(LOG_FILE),
    filemode='a'
)
logger = logging.getLogger("Muninn.Tray")


class ServerStatus:
    """Tracks the current state of Muninn backend services."""
    STARTING = "starting"
    ONLINE = "online"
    DEGRADED = "degraded"  # Server up but Ollama down
    OFFLINE = "offline"
    ERROR = "error"


class MuninnTrayApp:
    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.icon: Optional[pystray.Icon] = None
        self.running = True
        self.status = ServerStatus.OFFLINE
        self.memory_count = 0
        self.graph_nodes = 0
        self.ollama_ok = False
        self._status_lock = threading.Lock()

    # --- Icon Generation ---

    def _create_icon(self, status: str) -> Image.Image:
        """Generate a status-colored Mannaz rune icon."""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        dc = ImageDraw.Draw(image)

        # Status-dependent colors
        colors = {
            ServerStatus.ONLINE: (212, 175, 55, 255),     # Gold
            ServerStatus.STARTING: (100, 149, 237, 255),   # Cornflower blue
            ServerStatus.DEGRADED: (255, 165, 0, 255),     # Orange
            ServerStatus.OFFLINE: (128, 128, 128, 255),    # Gray
            ServerStatus.ERROR: (220, 50, 50, 255),        # Red
        }
        bg_colors = {
            ServerStatus.ONLINE: (20, 20, 30, 255),
            ServerStatus.STARTING: (20, 20, 30, 255),
            ServerStatus.DEGRADED: (30, 25, 15, 255),
            ServerStatus.OFFLINE: (30, 30, 30, 255),
            ServerStatus.ERROR: (35, 15, 15, 255),
        }

        fg = colors.get(status, colors[ServerStatus.OFFLINE])
        bg = bg_colors.get(status, bg_colors[ServerStatus.OFFLINE])

        # Draw rounded rectangle background
        dc.rounded_rectangle([0, 0, size - 1, size - 1], radius=8, fill=bg, outline=fg, width=2)

        # Draw Mannaz rune (ᛗ) - stylized M
        pad = 14
        mid_x = size // 2
        mid_y = size // 2
        w = 3

        # Left vertical
        dc.line([pad, pad, pad, size - pad], fill=fg, width=w)
        # Right vertical
        dc.line([size - pad, pad, size - pad, size - pad], fill=fg, width=w)
        # Upper-left diagonal to center
        dc.line([pad, pad, mid_x, mid_y], fill=fg, width=w)
        # Upper-right diagonal to center
        dc.line([size - pad, pad, mid_x, mid_y], fill=fg, width=w)

        # Status indicator dot (bottom-right corner)
        dot_r = 6
        dot_x = size - dot_r - 4
        dot_y = size - dot_r - 4
        dot_colors = {
            ServerStatus.ONLINE: (0, 200, 0, 255),
            ServerStatus.STARTING: (100, 149, 237, 255),
            ServerStatus.DEGRADED: (255, 165, 0, 255),
            ServerStatus.OFFLINE: (128, 128, 128, 255),
            ServerStatus.ERROR: (220, 50, 50, 255),
        }
        dot_fg = dot_colors.get(status, (128, 128, 128, 255))
        dc.ellipse(
            [dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r],
            fill=dot_fg, outline=(0, 0, 0, 200), width=1
        )

        return image

    def _load_custom_icon(self) -> Optional[Image.Image]:
        """Try to load a custom icon file if one exists."""
        for name in ("muninn.ico", "muninn.png", "muninn_raven.png", "muninn_raven.ico"):
            icon_path = CWD / name
            if icon_path.exists():
                try:
                    return Image.open(icon_path)
                except Exception:
                    pass
        return None

    # --- Server Lifecycle ---

    def _check_port(self, port: int) -> bool:
        """Check if a TCP port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(('127.0.0.1', port)) == 0
        except Exception:
            return False

    def _check_server_health(self) -> dict:
        """Query the Muninn /health endpoint."""
        try:
            resp = requests.get(HEALTH_URL, timeout=3)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}

    def _check_ollama(self) -> bool:
        """Check if Ollama is responsive."""
        try:
            resp = requests.get(OLLAMA_URL, timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def _update_status(self):
        """Poll server and Ollama status, update icon accordingly."""
        health = self._check_server_health()
        ollama_ok = self._check_ollama()

        with self._status_lock:
            self.ollama_ok = ollama_ok

            if health.get("status") == "ok" or health.get("status") == "healthy":
                self.memory_count = health.get("memory_count", 0)
                self.graph_nodes = health.get("graph_nodes", 0)
                if ollama_ok:
                    self.status = ServerStatus.ONLINE
                else:
                    self.status = ServerStatus.DEGRADED
            elif self._check_port(8000):
                # Port open but health check failed — starting or error
                self.status = ServerStatus.STARTING
            else:
                self.status = ServerStatus.OFFLINE

        # Update icon and tooltip
        if self.icon:
            custom = self._load_custom_icon()
            self.icon.icon = custom if custom else self._create_icon(self.status)
            self.icon.title = self._build_tooltip()

    def _build_tooltip(self) -> str:
        """Build a multi-line tooltip string."""
        with self._status_lock:
            status_label = {
                ServerStatus.ONLINE: "Online",
                ServerStatus.STARTING: "Starting...",
                ServerStatus.DEGRADED: "Degraded (Ollama down)",
                ServerStatus.OFFLINE: "Offline",
                ServerStatus.ERROR: "Error",
            }.get(self.status, "Unknown")

            lines = [f"Muninn - {status_label}"]
            if self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED):
                lines.append(f"Memories: {self.memory_count}")
                if self.graph_nodes:
                    lines.append(f"Graph nodes: {self.graph_nodes}")
            if not self.ollama_ok:
                lines.append("Ollama: Not responding")
            return "\n".join(lines)

    def start_server(self):
        """Start the Muninn server as a detached process."""
        if self._check_port(8000):
            logger.info("Server already running on port 8000")
            self._update_status()
            return

        logger.info("Starting Muninn server...")
        with self._status_lock:
            self.status = ServerStatus.STARTING

        # Use pythonw.exe for windowless execution if available
        python_exe = PYTHON_EXE
        pythonw = PYTHON_EXE.replace("python.exe", "pythonw.exe")
        if os.path.exists(pythonw):
            python_exe = pythonw

        try:
            self.server_process = subprocess.Popen(
                [python_exe, str(SERVER_SCRIPT)],
                cwd=str(CWD),
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
                close_fds=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Server process started (PID: {self.server_process.pid})")

            # Wait for server to become responsive (up to 30s)
            for i in range(60):
                if self._check_port(8000):
                    logger.info("Server is now accepting connections")
                    break
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            with self._status_lock:
                self.status = ServerStatus.ERROR

        self._update_status()

    def stop_server(self):
        """Stop the Muninn server process."""
        logger.info("Stopping server...")
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            finally:
                self.server_process = None

        with self._status_lock:
            self.status = ServerStatus.OFFLINE
        self._update_status()

    # --- Auto-Start (Windows Registry) ---

    def _is_autostart_enabled(self) -> bool:
        """Check if Muninn is registered for Windows auto-start."""
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_REG_KEY, 0, winreg.KEY_READ) as key:
                winreg.QueryValueEx(key, STARTUP_REG_NAME)
                return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _toggle_autostart(self, icon, item):
        """Toggle Windows auto-start registration."""
        if self._is_autostart_enabled():
            # Remove
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_REG_KEY, 0, winreg.KEY_WRITE) as key:
                    winreg.DeleteValue(key, STARTUP_REG_NAME)
                logger.info("Auto-start disabled")
            except Exception as e:
                logger.error(f"Failed to disable auto-start: {e}")
        else:
            # Add — use pythonw.exe for silent startup
            python_exe = PYTHON_EXE.replace("python.exe", "pythonw.exe")
            if not os.path.exists(python_exe):
                python_exe = PYTHON_EXE
            tray_script = str(CWD / "tray_app.py")
            cmd = f'"{python_exe}" "{tray_script}"'
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, STARTUP_REG_KEY, 0, winreg.KEY_WRITE) as key:
                    winreg.SetValueEx(key, STARTUP_REG_NAME, 0, winreg.REG_SZ, cmd)
                logger.info(f"Auto-start enabled: {cmd}")
            except Exception as e:
                logger.error(f"Failed to enable auto-start: {e}")

    # --- Menu Actions ---

    def _on_open_dashboard(self, icon, item):
        webbrowser.open(SERVER_URL)

    def _on_start(self, icon, item):
        threading.Thread(target=self.start_server, daemon=True).start()

    def _on_stop(self, icon, item):
        threading.Thread(target=self.stop_server, daemon=True).start()

    def _on_restart(self, icon, item):
        def _restart():
            self.stop_server()
            time.sleep(2)
            self.start_server()
        threading.Thread(target=_restart, daemon=True).start()

    def _on_view_logs(self, icon, item):
        """Open the server log file in the default text editor."""
        server_log = CWD / "server.log"
        if server_log.exists():
            os.startfile(str(server_log))
        else:
            # Fallback to tray log
            if LOG_FILE.exists():
                os.startfile(str(LOG_FILE))

    def _on_open_folder(self, icon, item):
        """Open the Muninn installation folder."""
        os.startfile(str(CWD))

    def _on_exit(self, icon, item):
        logger.info("Tray app exiting...")
        self.running = False
        self.stop_server()
        if self.icon:
            self.icon.stop()

    # --- Status Monitor Thread ---

    def _status_monitor(self):
        """Background thread that periodically polls service health."""
        while self.running:
            try:
                self._update_status()
            except Exception as e:
                logger.error(f"Status monitor error: {e}")
            time.sleep(POLL_INTERVAL)

    # --- Tray Setup ---

    def _build_menu(self) -> pystray.Menu:
        """Build the context menu with dynamic state."""
        return pystray.Menu(
            pystray.MenuItem(
                lambda text: f"Muninn ({self._status_text()})",
                lambda: None,
                enabled=False
            ),
            pystray.MenuItem(
                lambda text: f"Memories: {self.memory_count}" if self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED) else "",
                lambda: None,
                enabled=False,
                visible=lambda item: self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED)
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Dashboard", self._on_open_dashboard,
                             enabled=lambda item: self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Start Server", self._on_start,
                             visible=lambda item: self.status == ServerStatus.OFFLINE),
            pystray.MenuItem("Stop Server", self._on_stop,
                             visible=lambda item: self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED, ServerStatus.STARTING)),
            pystray.MenuItem("Restart Server", self._on_restart,
                             visible=lambda item: self.status in (ServerStatus.ONLINE, ServerStatus.DEGRADED)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("View Logs", self._on_view_logs),
            pystray.MenuItem("Open Folder", self._on_open_folder),
            pystray.MenuItem(
                "Start with Windows",
                self._toggle_autostart,
                checked=lambda item: self._is_autostart_enabled()
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._on_exit),
        )

    def _status_text(self) -> str:
        """Human-readable status for menu header."""
        return {
            ServerStatus.ONLINE: "Online",
            ServerStatus.STARTING: "Starting...",
            ServerStatus.DEGRADED: "Degraded",
            ServerStatus.OFFLINE: "Offline",
            ServerStatus.ERROR: "Error",
        }.get(self.status, "Unknown")

    def run(self):
        """Main entry point — sets up tray and starts server."""
        logger.info("Muninn tray app starting...")

        # Create initial icon
        custom = self._load_custom_icon()
        icon_image = custom if custom else self._create_icon(ServerStatus.OFFLINE)

        self.icon = pystray.Icon(
            PROJECT_NAME,
            icon_image,
            self._build_tooltip(),
            self._build_menu()
        )

        # Start server in background
        threading.Thread(target=self.start_server, daemon=True).start()

        # Start status monitor
        threading.Thread(target=self._status_monitor, daemon=True).start()

        # Run tray icon (blocking)
        logger.info("Tray icon running")
        self.icon.run()


def main():
    app = MuninnTrayApp()
    app.run()


if __name__ == "__main__":
    main()
