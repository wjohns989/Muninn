#!/usr/bin/env python3
"""
Build helper for packaging Huginn (Muninn standalone launcher) with PyInstaller.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Huginn standalone executable (Muninn browser mode) with PyInstaller."
    )
    parser.add_argument(
        "--name",
        default="HuginnControlCenter",
        help="Output executable/app name.",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build single-file executable instead of onedir layout.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Disable console window (recommended for desktop-click UX).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean previous build/dist directories before build.",
    )
    return parser


def _run(cmd: list[str]) -> int:
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    entrypoint = repo_root / "muninn_standalone.py"
    dashboard_html = repo_root / "dashboard.html"

    if not entrypoint.exists():
        print(f"error: entrypoint not found: {entrypoint}", file=sys.stderr)
        return 2
    if not dashboard_html.exists():
        print(f"error: dashboard HTML not found: {dashboard_html}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        args.name,
    ]
    if args.clean:
        cmd.append("--clean")
    if args.onefile:
        cmd.append("--onefile")
    if args.windowed:
        cmd.append("--noconsole")

    add_data_sep = ";" if os.name == "nt" else ":"
    cmd.extend(
        [
            "--add-data",
            f"{dashboard_html}{add_data_sep}.",
            str(entrypoint),
        ]
    )

    print("Running:", " ".join(cmd))
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
