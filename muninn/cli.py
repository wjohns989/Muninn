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
import re
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
    """Patch Codex config.toml without mixing stdio and HTTP schemas."""
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

    def _upsert_env_line(existing: list[str], key: str, value: str) -> tuple[list[str], bool]:
        updated = False
        replaced = False
        target = f'{key} = "{value}"'
        assignment = re.compile(rf"^\s*{re.escape(key)}\s*=")
        for i, line in enumerate(existing):
            if assignment.match(line):
                if line.strip() != target:
                    existing[i] = target
                    updated = True
                replaced = True
                break
        if not replaced:
            existing.append(target)
            updated = True
        return existing, updated

    muninn_body = lines[muninn_idx + 1 : muninn_end]
    is_streamable_http = any(re.match(r"^\s*url\s*=", line) for line in muninn_body)

    if is_streamable_http:
        changed = False
        if new_server_url is not None:
            mcp_url = new_server_url.rstrip("/")
            if not mcp_url.endswith("/mcp"):
                mcp_url += "/mcp"
            muninn_body, updated = _upsert_env_line(muninn_body, "url", mcp_url)
            changed = changed or updated
        if new_token is not None:
            muninn_body, updated = _upsert_env_line(
                muninn_body,
                "bearer_token_env_var",
                "MUNINN_AUTH_TOKEN",
            )
            changed = changed or updated

        lines[muninn_idx + 1 : muninn_end] = muninn_body

        # ``env`` is a valid child table for stdio servers only.  Older Muninn
        # releases added it to HTTP configs, which prevents Codex from loading
        # the entire config file.  Remove the complete legacy table, including
        # any serialized bearer token.
        env_idx = _find_section(0, "[mcp_servers.muninn.env]")
        if env_idx is not None:
            env_end = _section_end(env_idx)
            del lines[env_idx:env_end]
            changed = True

        if changed and not dry_run:
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return changed

    def _ensure_key(value: Optional[str]) -> bool:
        return value is not None

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

    muninn_end = _section_end(muninn_idx)
    server_url = None
    bearer_token_env_var = None
    for line in lines[muninn_idx + 1 : muninn_end]:
        stripped = line.strip()
        url_match = re.match(r'^url\s*=\s*"([^"]*)"', stripped)
        bearer_match = re.match(r'^bearer_token_env_var\s*=\s*"([^"]*)"', stripped)
        if url_match:
            server_url = url_match.group(1)
        elif bearer_match:
            bearer_token_env_var = bearer_match.group(1)

    if server_url is not None:
        normalized_url = server_url.rstrip("/")
        if normalized_url.endswith("/mcp"):
            normalized_url = normalized_url[:-4]
        return [
            _DoctorServerEntry(
                config_path=config_path,
                server_name="codex.muninn",
                token=os.environ.get(bearer_token_env_var) if bearer_token_env_var else None,
                server_url=normalized_url or None,
            )
        ]

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
        # Token rotation must not silently replace a custom HTTP endpoint.
        # ``doctor --repair`` remains the explicit URL convergence operation.
        new_server_url=None,
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
        print("    python server.py")
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

def _admin_request(args: argparse.Namespace, method: str, path: str, **kwargs) -> dict:
    """Call an endpoint on the running Muninn server, with the token when one is configured."""
    token = _read_token_from_file(_resolve_token_file(args.token_file)) or (
        os.environ.get("MUNINN_AUTH_TOKEN") or ""
    ).strip()
    response = requests.request(
        method,
        f"{_resolve_server_url(args.server_url)}{path}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=args.timeout_seconds,
        **kwargs,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise SystemExit(f"{method} {path} failed ({response.status_code}): {detail}")
    return response.json().get("data", {})


def _admin_post(args: argparse.Namespace, path: str, payload: dict) -> dict:
    """POST to an admin endpoint on the running Muninn server."""
    return _admin_request(args, "POST", path, json=payload)


def cmd_hooks(args: argparse.Namespace) -> int:
    """Install Muninn's session hooks into Claude Code and Codex (dry run unless --apply)."""
    from muninn.history import hook_install

    apps = args.app or ["claude", "codex"]
    install = args.action == "install"
    server_url = _resolve_server_url(args.server_url)
    plans = []
    if "claude" in apps:
        plans.append(hook_install.claude_plan(server_url, install=install))
    if "codex" in apps:
        plans.append(hook_install.codex_plan(install=install))
    for plan in plans:
        present = hook_install.installed(plan)
        print(f"{plan.app}: {plan.path}")
        print(f"  Muninn hooks now: {', '.join(present) or 'none'}")
        if args.action == "status":
            continue
        if not plan.changed:
            print("  nothing to change")
        elif args.apply:
            backup = hook_install.apply_plan(plan)
            print(f"  {'installed' if install else 'removed'}" + (f" (backup: {backup.name})" if backup else ""))
        else:
            print("  would write:")
            print("    " + json.dumps(plan.after.get("hooks", {}), indent=2).replace("\n", "\n    "))
    if args.action != "status" and not args.apply:
        print("Dry run only. Re-run with --apply to write the settings.")
    return 0


def _mask(key: str) -> str:
    return key[:8] + "…" + key[-4:] if len(key) > 16 else "…"


def _prompt_openrouter_key(*, first_run: bool) -> bool:
    """Ask for an OpenRouter key in an interactive terminal; returns True when one was saved."""
    import getpass

    from muninn.history import llm_settings

    if first_run:
        print(
            "\nMuninn can use OpenRouter to understand your imported conversations: pull out decisions,\n"
            "preferences, fixes and open items, and summarize each thread. Every request requires\n"
            "zero data retention (no storage, no training), and secrets are redacted before sending.\n"
            f"Default model: {llm_settings.DEFAULT_MODEL} (about $2 per 1,000 threads).\n"
            f"Get a key at {llm_settings.KEYS_PAGE}. Press Enter to skip and use local Ollama instead;\n"
            "you can add a key later with: python -m muninn.cli openrouter set\n"
        )
    for _ in range(3):
        key = getpass.getpass("OpenRouter API key (input hidden; Enter to skip): ").strip()
        if not key:
            if first_run:
                llm_settings.decline()
                print("Using local Ollama for thread analysis. Change this any time with `openrouter set`.")
            return False
        ok, message = llm_settings.verify_key(key)
        print(("✓ " if ok else "✗ ") + message)
        if ok:
            path = llm_settings.save_key(key)
            print(f"Saved to {path} (readable only by you).")
            return True
    return False


def _maybe_first_run_openrouter() -> None:
    from muninn.history import llm_settings

    if sys.stdin.isatty() and sys.stdout.isatty() and llm_settings.should_prompt():
        _prompt_openrouter_key(first_run=True)


def cmd_openrouter(args: argparse.Namespace) -> int:
    """Show, set or clear the OpenRouter key and model used for thread analysis."""
    from muninn.history import llm_settings

    if args.action == "clear":
        llm_settings.forget()
        print(f"Removed {llm_settings.settings_path()}. Thread analysis will use local Ollama.")
        return 0
    if args.action == "set":
        if args.key:
            ok, message = (True, "not verified (--no-verify)") if args.no_verify else llm_settings.verify_key(args.key)
            print(("✓ " if ok else "✗ ") + message)
            if not ok:
                return 1
            llm_settings.save_key(args.key, args.model)
        elif args.model and llm_settings.api_key():
            llm_settings.save_model(args.model)
        elif not sys.stdin.isatty():
            raise SystemExit("Pass --key, or run this in an interactive terminal.")
        elif not _prompt_openrouter_key(first_run=False):
            return 1
        elif args.model:
            llm_settings.save_model(args.model)
    key = llm_settings.api_key()
    print(json.dumps({
        "key": _mask(key) if key else None,
        "key_from": llm_settings.key_source(),
        "models": llm_settings.models() if key else [],
        "local_ollama_chosen": bool(llm_settings.load().get("declined")),
        "settings_file": str(llm_settings.settings_path()),
    }, indent=2))
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Keep and import local AI conversation history (Claude Code/Desktop, Codex, Gemini CLI, exports)."""
    if args.action == "status":
        data = _admin_request(args, "GET", "/history/status")
    elif args.action == "sync":
        data = _admin_request(args, "POST", "/history/sync", json=[str(p) for p in args.path] or None)
    elif args.action == "import":
        _maybe_first_run_openrouter()
        payload = {"apply": args.apply, "providers": args.provider or None, "since": args.since,
                   "paths": [str(p) for p in args.path] or None}
        data = _admin_request(args, "POST", "/history/import", json=payload)
        if not args.apply:
            print(json.dumps(data, indent=2, default=str))
            print("Dry run only. Re-run with --apply to import (it runs in the background; see 'history status').")
            return 0
    elif args.action == "analyze":
        _maybe_first_run_openrouter()
        payload = {"apply": args.apply, "provider": args.llm, "model": args.model, "project": args.project,
                   "limit": args.limit}
        data = _admin_request(args, "POST", "/history/analyze", json=payload)
        if not args.apply:
            print(json.dumps(data, indent=2, default=str))
            print("Dry run only. Re-run with --apply. Uses OpenRouter (zero data retention) when a key is set "
                  "(`openrouter set`), else local Ollama; --llm and --model override.")
            return 0
    elif args.action == "threads":
        params = {"limit": args.limit}
        for key in ("project", "agent", "status", "topic", "since"):
            if getattr(args, key, None):
                params[key] = getattr(args, key)
        if args.q:
            params["q"] = args.q
        data = _admin_request(args, "GET", "/history/threads", params=params)
    elif args.action == "timeline":
        if not args.project:
            raise SystemExit("history timeline needs --project.")
        params = {"project": args.project, "offset": args.offset, "limit": args.limit}
        params.update({key: getattr(args, key) for key in ("since",) if getattr(args, key)})
        data = _admin_request(args, "GET", "/history/timeline", params=params)
    else:  # thread
        if not args.thread_id:
            raise SystemExit("history thread needs a thread id (see 'history threads').")
        data = _admin_request(args, "GET", f"/history/threads/{args.thread_id}",
                              params={"offset": args.offset, "limit": args.limit})
    print(json.dumps(data, indent=2, default=str))
    return 0


def cmd_reindex(args: argparse.Namespace) -> int:
    """Rebuild vectors/BM25 from metadata.db via the running server."""
    report = _admin_post(args, "/admin/reindex", {
        "vectors": not args.no_vectors,
        "bm25": not args.no_bm25,
        "recreate_vectors": args.recreate_vectors,
        "dry_run": not args.apply,
    })
    print(json.dumps(report, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to rebuild.")
    return 0


def _read_export(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{") and '"results"' in stripped[:200]:
        return json.loads(text)["results"]  # Mem0 GET /memories response
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cmd_import(args: argparse.Namespace) -> int:
    """Import exported memories (JSONL, JSON array, or a Mem0 /memories response)."""
    records = _read_export(args.file)
    totals: dict = {}
    for start in range(0, len(records), args.batch_size):
        report = _admin_post(args, "/admin/import", {
            "records": records[start:start + args.batch_size],
            "user_id": args.user_id,
            "namespace": args.namespace,
            "source": args.source,
            "dry_run": not args.apply,
        })
        for key, value in report.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    totals["dry_run"] = not args.apply
    print(json.dumps(totals, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply to import.")
    return 0


def _add_server_args(sub: argparse.ArgumentParser, timeout: float) -> None:
    sub.add_argument("--token-file", type=Path, default=None, metavar="PATH",
                     help="Path to token file (default: .muninn_token or MUNINN_TOKEN_FILE).")
    sub.add_argument("--server-url", type=str, default=None, metavar="URL",
                     help="Muninn server URL (default: MUNINN_SERVER_URL or local default).")
    sub.add_argument("--timeout-seconds", type=float, default=timeout, help="HTTP timeout.")
    sub.add_argument("--apply", action="store_true", default=False,
                     help="Perform the operation (default is a dry run that only reports).")


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
    reindex = subparsers.add_parser(
        "reindex",
        help="Rebuild vector and keyword indexes from metadata.db.",
        description=(
            "Re-embeds every live memory and rebuilds BM25 from the metadata store.\n"
            "Use after changing the embedding model (with --recreate-vectors if its\n"
            "dimensions changed) or after restoring a metadata.db into a fresh install."
        ),
    )
    _add_server_args(reindex, timeout=3600.0)
    reindex.add_argument("--no-vectors", action="store_true", help="Skip re-embedding.")
    reindex.add_argument("--no-bm25", action="store_true", help="Skip the BM25 rebuild.")
    reindex.add_argument("--recreate-vectors", action="store_true",
                         help="Drop and recreate the vector collection at the configured dimensions.")

    importer = subparsers.add_parser(
        "import",
        help="Import exported memories (Muninn, Mem0 or similar JSON).",
        description=(
            "Reads JSONL, a JSON array, or a Mem0 GET /memories response. Each record needs\n"
            "text in content/memory/text/data; created_at is preserved; exact duplicates are\n"
            "skipped. Memories import under --user-id so default searches find them."
        ),
    )
    _add_server_args(importer, timeout=600.0)
    importer.add_argument("file", type=Path, help="Export file to import.")
    importer.add_argument("--user-id", default="global_user")
    importer.add_argument("--namespace", default="global")
    importer.add_argument("--source", default="legacy", help="Recorded as metadata.import_source.")
    importer.add_argument("--batch-size", type=int, default=200)

    history = subparsers.add_parser(
        "history",
        help="Keep and import local AI conversation history as memories.",
        description=(
            "status   what the vault holds, where each app keeps history, retention warnings\n"
            "sync     copy new/changed transcripts into the vault now (runs every 30 min anyway)\n"
            "import   dry run of turning history into memories; --apply to import (then automatic)\n"
            "analyze  extract decisions, preferences, fixes, open items per thread (LLM; dry run first)\n"
            "threads  list imported conversation threads (--project to filter)\n"
            "thread   re-read one thread in order: history thread <thread-id>\n"
            "timeline one project's conversations from every app, interleaved in time order (--project)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_server_args(history, timeout=300.0)
    history.add_argument("action", choices=["status", "sync", "import", "analyze", "threads", "thread", "timeline"])
    history.add_argument("thread_id", nargs="?", help="Thread id for 'thread'.")
    history.add_argument("--provider", action="append",
                         choices=["claude_code", "codex", "gemini_cli", "chatgpt", "claude_ai"],
                         help="Only these sources (repeatable).")
    history.add_argument("--since", help="Only threads active since this date (YYYY-MM-DD).")
    history.add_argument("--path", action="append", type=Path, default=[],
                         help="Extra file, e.g. a ChatGPT/Claude export conversations.json or .zip (repeatable).")
    history.add_argument("--project", help="Project filter for 'threads' and 'analyze'.")
    history.add_argument("--agent", help="Agent filter for 'threads' (claude-code, codex, ...).")
    history.add_argument("--status", help="Status filter for 'threads' (completed, in_progress, ...).")
    history.add_argument("--topic", help="Topic filter for 'threads'.")
    history.add_argument("--q", help="Title text filter for 'threads'.")
    history.add_argument("--llm", choices=["openrouter", "ollama"], help="Model provider for 'analyze'.")
    history.add_argument("--model", help="Model for 'analyze' (e.g. an OpenRouter model id).")
    history.add_argument("--offset", type=int, default=0)
    history.add_argument("--limit", type=int, default=50)

    hooks = subparsers.add_parser(
        "hooks",
        help="Add Muninn session hooks to Claude Code and Codex.",
        description=(
            "Session start: the agent receives the project briefing (goal, open handoffs, earlier\n"
            "threads from every app). Before compaction and at session end: the transcript is copied\n"
            "to the vault and imported, so nothing compaction drops is lost. Dry run unless --apply;\n"
            "settings files are backed up; only Muninn's own entries are changed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    hooks.add_argument("action", choices=["status", "install", "uninstall"])
    hooks.add_argument("--app", action="append", choices=["claude", "codex"], help="Only this app (repeatable).")
    hooks.add_argument("--server-url", default=None, help="Muninn server URL the hooks call.")
    hooks.add_argument("--apply", action="store_true", help="Write the settings.")

    openrouter = subparsers.add_parser(
        "openrouter",
        help="Set up the OpenRouter key used to analyze imported conversations.",
        description=(
            "status  show which key and models are used (the key is masked)\n"
            "set     save a key (prompted, hidden, verified with OpenRouter) and optionally --model\n"
            "clear   remove the saved key; analysis falls back to local Ollama\n"
            "Every request requires zero data retention. The key is stored in Muninn's config\n"
            "directory with owner-only permissions; OPENROUTER_API_KEY in the environment wins."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    openrouter.add_argument("action", choices=["status", "set", "clear"])
    openrouter.add_argument("--key", help="Key to save (otherwise you are prompted without echo).")
    openrouter.add_argument("--model", help="Primary model, e.g. deepseek/deepseek-v4.1-flash.")
    openrouter.add_argument("--no-verify", action="store_true", help="Save without asking OpenRouter first.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "rotate-token":
        return cmd_rotate_token(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "reindex":
        return cmd_reindex(args)
    if args.command == "import":
        return cmd_import(args)
    if args.command == "history":
        return cmd_history(args)
    if args.command == "hooks":
        return cmd_hooks(args)
    if args.command == "openrouter":
        return cmd_openrouter(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
