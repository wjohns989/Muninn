"""
Muninn CLI — operational utilities for the Muninn server (Phase 18).

Usage:
    python -m muninn.cli rotate-token [options]
    python -m muninn.cli doctor [options]
    python -m muninn.cli --help

Commands:
    rotate-token    Generate a new auth token, write it to .muninn_token, and
                    print platform-specific instructions for applying it.
    doctor          Validate/repair MCP host Muninn URL+token convergence and
                    verify server health with the expected token.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Token / URL resolution
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOKEN_FILE = Path(".muninn_token")
_DEFAULT_SERVER_URL = "http://127.0.0.1:42069"

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
    Path(os.environ.get("APPDATA", "")) / "Code" / "User" / "mcp.json",
    # Gemini / Antigravity
    Path.home() / ".gemini" / "settings.json",
    Path.home() / ".gemini" / "antigravity" / "mcp_config.json",
]
_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


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


def _resolve_server_url(server_url: Optional[str]) -> str:
    """
    Return canonical server URL.

    Resolution order:
      1. Explicit --server-url argument
      2. MUNINN_SERVER_URL environment variable
      3. http://127.0.0.1:42069
    """
    if server_url:
        return server_url.strip()
    env_url = os.environ.get("MUNINN_SERVER_URL")
    if env_url and env_url.strip():
        return env_url.strip()
    return _DEFAULT_SERVER_URL


# ─────────────────────────────────────────────────────────────────────────────
# MCP config patching
# ─────────────────────────────────────────────────────────────────────────────

def _iter_server_blocks(cfg: dict) -> list[dict]:
    server_blocks = []
    mcp_servers = cfg.get("mcpServers")
    if isinstance(mcp_servers, dict):
        server_blocks.append(mcp_servers)
    servers = cfg.get("servers")
    if isinstance(servers, dict):
        server_blocks.append(servers)
    return server_blocks


def _patch_mcp_config_env(
    config_path: Path,
    *,
    new_token: Optional[str] = None,
    new_server_url: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Patch Muninn MCP env fields in a host config JSON.

    Returns True if the file was modified (or would be modified in dry-run mode).
    Returns False if the file doesn't exist, has no muninn server entry, has no
    server blocks, or no effective value change is needed.
    """
    if new_token is None and new_server_url is None:
        return False
    if not config_path.exists():
        return False

    try:
        raw = config_path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return False

    # Support both legacy and newer MCP config schemas.
    server_blocks = _iter_server_blocks(cfg)
    if not server_blocks:
        return False

    patched = False
    for block in server_blocks:
        for server_name, server_cfg in block.items():
            if not isinstance(server_cfg, dict):
                continue
            # Match any server whose name contains "muninn" (case-insensitive)
            if "muninn" not in server_name.lower():
                continue
            env = server_cfg.setdefault("env", {})
            if not isinstance(env, dict):
                continue
            if new_token is not None and env.get("MUNINN_AUTH_TOKEN") != new_token:
                env["MUNINN_AUTH_TOKEN"] = new_token
                patched = True
            if new_server_url is not None and env.get("MUNINN_SERVER_URL") != new_server_url:
                env["MUNINN_SERVER_URL"] = new_server_url
                patched = True

    if not patched:
        return False

    if not dry_run:
        config_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return True


def _patch_mcp_config(config_path: Path, new_token: str, *, dry_run: bool = False) -> bool:
    """
    Backward-compatible token-only patching helper used by existing tests/code.
    """
    return _patch_mcp_config_env(
        config_path,
        new_token=new_token,
        dry_run=dry_run,
    )


def _patch_codex_toml(
    config_path: Path,
    *,
    new_token: Optional[str] = None,
    new_server_url: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """Patch Codex config.toml with Muninn MCP env values."""
    if new_token is None and new_server_url is None:
        return False
    if not config_path.exists():
        return False

    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _find_section(start_idx: int, header: str) -> Optional[int]:
        for idx in range(start_idx, len(lines)):
            if lines[idx].strip() == header:
                return idx
        return None

    muninn_idx = _find_section(0, "[mcp_servers.muninn]")
    if muninn_idx is None:
        return False

    def _section_end(from_idx: int) -> int:
        for idx in range(from_idx + 1, len(lines)):
            if lines[idx].strip().startswith("["):
                return idx
        return len(lines)

    muninn_end = _section_end(muninn_idx)
    env_idx = _find_section(0, "[mcp_servers.muninn.env]")

    def _ensure_key(value: Optional[str]) -> bool:
        return value is not None

    def _upsert_env_line(existing: list[str], key: str, value: str) -> tuple[list[str], bool]:
        updated = False
        replaced = False
        prefix = f"{key} = "
        target = f'{prefix}"{value}"'
        for i, line in enumerate(existing):
            if line.strip().startswith(prefix):
                if line.strip() != target:
                    existing[i] = target
                    updated = True
                replaced = True
                break
        if not replaced:
            existing.append(target)
            updated = True
        return existing, updated

    changed = False
    if env_idx is None:
        env_lines: list[str] = ["[mcp_servers.muninn.env]"]
        if _ensure_key(new_token):
            env_lines.append(f'MUNINN_AUTH_TOKEN = "{new_token}"')
        if _ensure_key(new_server_url):
            env_lines.append(f'MUNINN_SERVER_URL = "{new_server_url}"')
        if len(env_lines) > 1:
            if not dry_run:
                lines[muninn_end:muninn_end] = [""] + env_lines
            changed = True
    else:
        env_end = _section_end(env_idx)
        existing = lines[env_idx + 1 : env_end]
        if _ensure_key(new_token):
            existing, updated = _upsert_env_line(existing, "MUNINN_AUTH_TOKEN", new_token or "")
            changed = changed or updated
        if _ensure_key(new_server_url):
            existing, updated = _upsert_env_line(existing, "MUNINN_SERVER_URL", new_server_url or "")
            changed = changed or updated
        if changed and not dry_run:
            lines[env_idx + 1 : env_end] = existing

    if changed and not dry_run:
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


@dataclass
class _DoctorServerEntry:
    config_path: Path
    server_name: str
    token: Optional[str]
    server_url: Optional[str]


def _collect_muninn_server_entries(config_path: Path) -> list[_DoctorServerEntry]:
    if not config_path.exists():
        return []
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    entries: list[_DoctorServerEntry] = []
    for block in _iter_server_blocks(cfg):
        for server_name, server_cfg in block.items():
            if not isinstance(server_name, str) or "muninn" not in server_name.lower():
                continue
            env = server_cfg.get("env", {}) if isinstance(server_cfg, dict) else {}
            if not isinstance(env, dict):
                env = {}
            token = env.get("MUNINN_AUTH_TOKEN")
            server_url = env.get("MUNINN_SERVER_URL")
            entries.append(
                _DoctorServerEntry(
                    config_path=config_path,
                    server_name=server_name,
                    token=str(token).strip() if token is not None else None,
                    server_url=str(server_url).strip() if server_url is not None else None,
                )
            )
    return entries


def _collect_codex_muninn_entries(config_path: Path) -> list[_DoctorServerEntry]:
    if not config_path.exists():
        return []

    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _find_section(start_idx: int, header: str) -> Optional[int]:
        for idx in range(start_idx, len(lines)):
            if lines[idx].strip() == header:
                return idx
        return None

    muninn_idx = _find_section(0, "[mcp_servers.muninn]")
    if muninn_idx is None:
        return []

    def _section_end(from_idx: int) -> int:
        for idx in range(from_idx + 1, len(lines)):
            if lines[idx].strip().startswith("["):
                return idx
        return len(lines)

    env_idx = _find_section(0, "[mcp_servers.muninn.env]")
    if env_idx is None:
        return [
            _DoctorServerEntry(
                config_path=config_path,
                server_name="codex.muninn",
                token=None,
                server_url=None,
            )
        ]

    env_end = _section_end(env_idx)
    token = None
    server_url = None
    for line in lines[env_idx + 1 : env_end]:
        stripped = line.strip()
        if stripped.startswith("MUNINN_AUTH_TOKEN"):
            token = stripped.split("=", 1)[1].strip().strip('"')
        if stripped.startswith("MUNINN_SERVER_URL"):
            server_url = stripped.split("=", 1)[1].strip().strip('"')

    return [
        _DoctorServerEntry(
            config_path=config_path,
            server_name="codex.muninn",
            token=token or None,
            server_url=server_url or None,
        )
    ]


def _read_token_from_file(token_file: Path) -> Optional[str]:
    if not token_file.exists():
        return None
    try:
        value = token_file.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _check_server_health(url: str, token: Optional[str], timeout_seconds: float) -> tuple[bool, str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get(f"{url}/health", headers=headers, timeout=timeout_seconds)
        if response.status_code == 200:
            return True, "ok"
        return False, f"http_{response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


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
    target_server_url = _resolve_server_url(None)
    patched_configs: list[Path] = []
    skipped_configs: list[Path] = []
    for cfg_path in _MCP_CONFIG_PATHS:
        modified = _patch_mcp_config_env(
            cfg_path,
            new_token=new_token,
            new_server_url=target_server_url,
            dry_run=dry_run,
        )
        if modified:
            patched_configs.append(cfg_path)
        elif cfg_path.exists():
            skipped_configs.append(cfg_path)
    codex_modified = _patch_codex_toml(
        _CODEX_CONFIG_PATH,
        new_token=new_token,
        new_server_url=target_server_url,
        dry_run=dry_run,
    )
    if codex_modified:
        patched_configs.append(_CODEX_CONFIG_PATH)
    elif _CODEX_CONFIG_PATH.exists():
        skipped_configs.append(_CODEX_CONFIG_PATH)

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
        print("Restart your MCP host (Codex / Claude Desktop / Cursor / VS Code) to")
        print("pick up the updated token in the config file(s) above.")
    else:
        print("No MCP host config was auto-patched. If your MCP host config")
        print("sets MUNINN_AUTH_TOKEN in the server env, update it manually.")

    print()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# doctor command
# ─────────────────────────────────────────────────────────────────────────────

def cmd_doctor(args: argparse.Namespace) -> int:
    """
    Validate and optionally repair Muninn MCP host config convergence.

    Exit codes:
      0 = healthy/converged
      1 = warnings/drift present
      2 = critical failure (cannot authenticate or no expected token)
    """
    token_file = _resolve_token_file(args.token_file)
    target_url = _resolve_server_url(args.server_url)
    timeout = max(0.1, float(args.timeout_seconds))

    token_from_file = _read_token_from_file(token_file)
    token_from_env = os.environ.get("MUNINN_AUTH_TOKEN")
    token_from_env = token_from_env.strip() if token_from_env else None
    expected_token = token_from_file or token_from_env

    issues: list[str] = []
    warnings: list[str] = []
    critical = False

    if expected_token is None:
        critical = True
        issues.append(
            f"No expected auth token found (missing token file '{token_file}' and MUNINN_AUTH_TOKEN env)."
        )

    health_ok = False
    health_detail = "skipped"
    if expected_token is not None:
        health_ok, health_detail = _check_server_health(target_url, expected_token, timeout)
        if not health_ok:
            critical = True
            issues.append(
                f"Server health/auth check failed at {target_url}/health using expected token ({health_detail})."
            )

    entries: list[_DoctorServerEntry] = []
    bad_config_paths: list[Path] = []
    for cfg_path in _MCP_CONFIG_PATHS:
        if not cfg_path.exists():
            continue
        try:
            entries.extend(_collect_muninn_server_entries(cfg_path))
        except Exception:
            bad_config_paths.append(cfg_path)
    entries.extend(_collect_codex_muninn_entries(_CODEX_CONFIG_PATH))

    token_mismatches = []
    url_mismatches = []
    missing_url_entries = []

    for entry in entries:
        if expected_token is not None and entry.token != expected_token:
            token_mismatches.append(entry)
        if entry.server_url is None:
            missing_url_entries.append(entry)
        elif entry.server_url != target_url:
            url_mismatches.append(entry)

    if bad_config_paths:
        warnings.append(f"{len(bad_config_paths)} MCP config file(s) could not be parsed.")
    if token_mismatches:
        warnings.append(f"{len(token_mismatches)} Muninn MCP server entry/entries have token drift.")
    if url_mismatches:
        warnings.append(f"{len(url_mismatches)} Muninn MCP server entry/entries have URL drift.")
    if missing_url_entries:
        warnings.append(f"{len(missing_url_entries)} Muninn MCP server entry/entries do not pin MUNINN_SERVER_URL.")

    repaired_paths: list[Path] = []
    if args.repair:
        for cfg_path in _MCP_CONFIG_PATHS:
            patched = _patch_mcp_config_env(
                cfg_path,
                new_token=expected_token,
                new_server_url=target_url,
                dry_run=False,
            )
            if patched:
                repaired_paths.append(cfg_path)
        if _patch_codex_toml(
            _CODEX_CONFIG_PATH,
            new_token=expected_token,
            new_server_url=target_url,
            dry_run=False,
        ):
            repaired_paths.append(_CODEX_CONFIG_PATH)

        # Re-evaluate drift after repair.
        refreshed_entries: list[_DoctorServerEntry] = []
        for cfg_path in _MCP_CONFIG_PATHS:
            if cfg_path.exists():
                refreshed_entries.extend(_collect_muninn_server_entries(cfg_path))
        refreshed_entries.extend(_collect_codex_muninn_entries(_CODEX_CONFIG_PATH))
        entries = refreshed_entries
        token_mismatches = [
            e for e in entries if expected_token is not None and e.token != expected_token
        ]
        url_mismatches = [e for e in entries if e.server_url is not None and e.server_url != target_url]
        missing_url_entries = [e for e in entries if e.server_url is None]
        warnings = [w for w in warnings if "drift" not in w and "do not pin" not in w]
        if token_mismatches:
            warnings.append(f"{len(token_mismatches)} token mismatches remain after repair.")
        if url_mismatches:
            warnings.append(f"{len(url_mismatches)} URL mismatches remain after repair.")
        if missing_url_entries:
            warnings.append(f"{len(missing_url_entries)} entries still missing pinned MUNINN_SERVER_URL.")

    print("\nMuninn Doctor")
    print("=" * 50)
    print(f"Target server URL: {target_url}")
    print(f"Token file: {token_file.resolve()}")
    print(f"Token source: {'file' if token_from_file else ('env' if token_from_env else 'none')}")
    print(f"Health/auth check: {'PASS' if health_ok else 'FAIL'} ({health_detail})")
    print(f"Muninn MCP entries discovered: {len(entries)}")
    if repaired_paths:
        print("Repaired config files:")
        for p in repaired_paths:
            print(f"  {p}")
    if issues:
        print("Critical issues:")
        for issue in issues:
            print(f"  - {issue}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print()

    if critical:
        return 2
    if warnings:
        return 1
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
               "  python -m muninn.cli doctor\n"
               "  python -m muninn.cli doctor --repair\n"
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

    # doctor sub-command
    doctor = subparsers.add_parser(
        "doctor",
        help="Validate/repair Muninn MCP server/token convergence.",
        description=(
            "Validates that local Muninn MCP host configs converge on one\n"
            "MUNINN_SERVER_URL + MUNINN_AUTH_TOKEN and checks server health\n"
            "with the expected token. Optionally repairs discovered drift."
        ),
    )
    doctor.add_argument(
        "--token-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to token file (default: .muninn_token or MUNINN_TOKEN_FILE).",
    )
    doctor.add_argument(
        "--server-url",
        type=str,
        default=None,
        metavar="URL",
        help="Expected Muninn server URL (default: MUNINN_SERVER_URL or local default).",
    )
    doctor.add_argument(
        "--timeout-seconds",
        type=float,
        default=3.0,
        help="HTTP timeout for health/auth check.",
    )
    doctor.add_argument(
        "--repair",
        action="store_true",
        default=False,
        help="Rewrite discovered Muninn MCP host entries to expected token + URL.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "rotate-token":
        return cmd_rotate_token(args)
    if args.command == "doctor":
        return cmd_doctor(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
