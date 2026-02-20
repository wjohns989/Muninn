"""
Muninn CLI — operational utilities for the Muninn server (Phase 18).

Usage:
    python -m muninn.cli rotate-token [options]
    python -m muninn.cli --help

Commands:
    rotate-token    Generate a new auth token, write it to .muninn_token, and
                    print platform-specific instructions for applying it.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Token file resolution
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOKEN_FILE = Path(".muninn_token")

# Well-known MCP host configuration paths (cross-platform).
_MCP_CONFIG_PATHS: list[Path] = [
    # Claude Desktop (Windows)
    Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json",
    # Claude Desktop (macOS)
    Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    # Claude Desktop (Linux)
    Path.home() / ".config" / "Claude" / "claude_desktop_config.json",
    # Cursor / VS Code MCP settings
    Path.home() / ".cursor" / "mcp.json",
    Path.home() / ".vscode" / "mcp.json",
]


def _resolve_token_file(token_file: Optional[Path]) -> Path:
    """Return the canonical token file path.

    Resolution order:
      1. Explicit --token-file argument
      2. MUNINN_TOKEN_FILE environment variable
      3. .muninn_token in the current working directory (default)
    """
    if token_file is not None:
        return token_file
    env_path = os.environ.get("MUNINN_TOKEN_FILE")
    if env_path:
        return Path(env_path)
    return _DEFAULT_TOKEN_FILE


# ─────────────────────────────────────────────────────────────────────────────
# MCP config patching
# ─────────────────────────────────────────────────────────────────────────────

def _patch_mcp_config(config_path: Path, new_token: str, *, dry_run: bool = False) -> bool:
    """Update MUNINN_AUTH_TOKEN in a Claude Desktop / Cursor MCP JSON config.

    Returns True if the file was modified (or would be modified in dry-run mode).
    Returns False if the file doesn't exist, has no muninn server entry, or
    already uses a different env-var injection strategy (e.g. env var reference).
    """
    if not config_path.exists():
        return False

    try:
        raw = config_path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return False

    mcp_servers = cfg.get("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        return False

    patched = False
    for server_name, server_cfg in mcp_servers.items():
        if not isinstance(server_cfg, dict):
            continue
        # Match any server whose name contains "muninn" (case-insensitive)
        if "muninn" not in server_name.lower():
            continue
        env = server_cfg.setdefault("env", {})
        if not isinstance(env, dict):
            continue
        if "MUNINN_AUTH_TOKEN" in env:
            env["MUNINN_AUTH_TOKEN"] = new_token
            patched = True

    if not patched:
        return False

    if not dry_run:
        config_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# rotate-token command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_rotate_token(args: argparse.Namespace) -> int:
    """
    Generate a new auth token, persist it, and emit update instructions.

    Steps:
      1. Generate a 32-byte URL-safe random token
      2. Write to token file (default: .muninn_token)
      3. Patch MUNINN_AUTH_TOKEN in any auto-detected MCP host config files
      4. Print platform-specific instructions for applying the new token
    """
    token_file = _resolve_token_file(args.token_file)
    dry_run: bool = args.dry_run

    # Step 1 — generate
    new_token = secrets.token_urlsafe(32)

    # Step 2 — persist to token file (owner-read-only on Unix)
    if not dry_run:
        try:
            token_file.write_text(new_token, encoding="utf-8")
            if sys.platform != "win32":
                token_file.chmod(0o600)
        except OSError as exc:
            print(f"Error: could not write token file {token_file}: {exc}", file=sys.stderr)
            return 1

    # Step 3 — patch MCP config files
    patched_configs: list[Path] = []
    skipped_configs: list[Path] = []
    for cfg_path in _MCP_CONFIG_PATHS:
        modified = _patch_mcp_config(cfg_path, new_token, dry_run=dry_run)
        if modified:
            patched_configs.append(cfg_path)
        elif cfg_path.exists():
            skipped_configs.append(cfg_path)

    # Step 4 — print output
    if args.token_only:
        # Machine-readable: just print the token
        print(new_token)
        return 0

    mode_tag = " [DRY RUN — no files written]" if dry_run else ""
    print(f"\nMuninn Token Rotation{mode_tag}")
    print("=" * 50)
    print(f"New token: {new_token}")
    print()

    if not dry_run:
        print(f"Token file written: {token_file.resolve()}")
    else:
        print(f"Would write token file: {token_file.resolve()}")
    print()

    if patched_configs:
        verb = "Would update" if dry_run else "Updated"
        print(f"{verb} MCP config MUNINN_AUTH_TOKEN in:")
        for p in patched_configs:
            print(f"  {p}")
        print()

    if skipped_configs:
        print("Skipped (no muninn server with MUNINN_AUTH_TOKEN env key):")
        for p in skipped_configs:
            print(f"  {p}")
        print()

    # Platform-specific apply instructions
    print("To apply the new token, restart the Muninn server with:")
    print()
    if sys.platform == "win32":
        print("  PowerShell:")
        print(f"    $env:MUNINN_AUTH_TOKEN = (Get-Content '{token_file}')")
        print(f"    python server.py")
        print()
        print("  To persist permanently (user-scope):")
        print(f"    setx MUNINN_AUTH_TOKEN \"{new_token}\"")
        print("    # Restart terminal for setx to take effect")
    else:
        print("  Bash/Zsh:")
        print(f"    MUNINN_AUTH_TOKEN=$(cat '{token_file}') python server.py")
        print()
        print("  To persist in shell profile (~/.bashrc / ~/.zshrc):")
        print(f"    echo 'export MUNINN_AUTH_TOKEN=\"{new_token}\"' >> ~/.bashrc")
        print("    source ~/.bashrc")

    print()
    if patched_configs:
        print("Restart your MCP host (Claude Desktop / Cursor / VS Code) to")
        print("pick up the updated token in the config file(s) above.")
    else:
        print("No MCP host config was auto-patched. If your MCP host config")
        print("sets MUNINN_AUTH_TOKEN in the server env, update it manually.")

    print()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="muninn.cli",
        description="Muninn CLI — operational utilities for the Muninn server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python -m muninn.cli rotate-token\n"
               "  python -m muninn.cli rotate-token --dry-run\n"
               "  python -m muninn.cli rotate-token --token-only\n"
               "  python -m muninn.cli rotate-token --token-file /etc/muninn/.token\n",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rotate-token sub-command
    rotate = subparsers.add_parser(
        "rotate-token",
        help="Generate a new auth token and persist it.",
        description=(
            "Generates a cryptographically secure 32-byte URL-safe token,\n"
            "writes it to .muninn_token (or the specified file), patches\n"
            "any auto-detected MCP host config files, and prints instructions\n"
            "for restarting the server with the new token."
        ),
    )
    rotate.add_argument(
        "--token-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to write the new token (default: .muninn_token in cwd, "
            "or MUNINN_TOKEN_FILE env var)."
        ),
    )
    rotate.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without writing any files.",
    )
    rotate.add_argument(
        "--token-only",
        action="store_true",
        default=False,
        help="Print only the new token to stdout (machine-readable, no instructions).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "rotate-token":
        return cmd_rotate_token(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
