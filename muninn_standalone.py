#!/usr/bin/env python3
"""
Standalone launcher for Huginn (Muninn browser-first mode).

This entrypoint is intended for direct execution or packaging into an
executable (e.g., with PyInstaller) so users can run Muninn without an
assistant/IDE bridge and ingest/search via the built-in browser UI.
"""

from __future__ import annotations

import argparse
import webbrowser
import uvicorn
import threading
from muninn_tray import create_tray_icon


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Huginn (Muninn standalone browser-first memory service)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the API/UI server.")
    parser.add_argument("--port", default=42069, type=int, help="Bind port for the API/UI server.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser UI.",
    )
    parser.add_argument(
        "--browser-delay-sec",
        default=1.2,
        type=float,
        help="Delay before opening browser UI (seconds).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Uvicorn log level.",
    )
    return parser


def _schedule_browser_launch(url: str, delay_sec: float) -> None:
    delay = max(0.0, float(delay_sec))
    timer = threading.Timer(delay, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        _schedule_browser_launch(url, args.browser_delay_sec)

    # Launch tray icon in a separate thread
    def stop_uvicorn():
        # This is a bit hacky but works for a standalone script
        import os
        os._exit(0)

    tray_thread = threading.Thread(
        target=create_tray_icon, 
        args=(url, stop_uvicorn),
        daemon=True
    )
    tray_thread.start()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=int(args.port),
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
