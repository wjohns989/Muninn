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
            # HTTP MCP profiles authenticate in their own headers, not stdio
            # env. Injecting env here leaves the real auth unchanged and can
            # corrupt host-specific schemas. Disabled/no-auth profiles are
            # likewise not candidates for token rotation or doctor repair.
            existing_env = server_cfg.get("env")
            if ("url" in server_cfg or "serverUrl" in server_cfg
                    or "headers" in server_cfg
                    or server_cfg.get("disabled")
                    or not isinstance(server_cfg.get("command"), str)
                    or (isinstance(existing_env, dict)
                        and str(existing_env.get("MUNINN_NO_AUTH", "")).lower() in {"1", "true"})):
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
    # The line-oriented writer understands bare TOML keys only. Quoted keys
    # are valid TOML but appending a bare duplicate would invalidate the file.
    if any(re.match(r"^\s*[\"']", line) for line in muninn_body):
        return False
    is_streamable_http = any(re.match(r"^\s*url\s*=", line) for line in muninn_body)

    if is_streamable_http:
        # A custom bearer reference belongs to the host operator. Generic
        # rotation/repair must not silently replace it with our environment
        # variable, even when the URL itself could be rewritten.
        bearer_lines = [line for line in muninn_body
                        if re.match(r"^\s*bearer_token_env_var\s*=", line)]
        if any(not re.match(r'^\s*bearer_token_env_var\s*=\s*"MUNINN_AUTH_TOKEN"\s*$', line)
               for line in bearer_lines):
            return False
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
    token_check: bool = True
    url_check: bool = True
    note: Optional[str] = None


def _mcp_base_url(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value:
        return None
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(value)
    if parsed.path.rstrip("/") == "/mcp" and not parsed.username and not parsed.password:
        # Host-specific query parameters (for example agent identity) are not
        # the server origin; never include them in drift comparison or output.
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return value.rstrip("/")


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
            if not isinstance(server_cfg, dict):
                entries.append(_DoctorServerEntry(config_path, server_name, None, None,
                                                  token_check=False, url_check=False,
                                                  note="unsupported profile"))
                continue
            if server_cfg.get("disabled"):
                entries.append(_DoctorServerEntry(config_path, server_name, None, None,
                                                  token_check=False, url_check=False,
                                                  note="disabled profile"))
                continue
            http_url = server_cfg.get("url") or server_cfg.get("serverUrl")
            if http_url:
                if server_cfg.get("command") or ("url" in server_cfg and "serverUrl" in server_cfg):
                    entries.append(_DoctorServerEntry(config_path, server_name, None, None,
                                                      token_check=False, url_check=False,
                                                      note="unsupported mixed profile"))
                    continue
                headers = server_cfg.get("headers", {})
                header = headers.get("Authorization", "") if isinstance(headers, dict) else ""
                if not isinstance(header, str) or not header.startswith("Bearer "):
                    token, token_check = None, False
                elif "${" in header or "$MUNINN_AUTH_TOKEN" in header:
                    token, token_check = None, False
                else:
                    token, token_check = header[7:].strip(), True
                entries.append(_DoctorServerEntry(
                    config_path, server_name, token, _mcp_base_url(http_url),
                    token_check=token_check,
                    note=None if token_check else "runtime token unverified",
                ))
                continue
            if str(env.get("MUNINN_NO_AUTH", "")).lower() in {"1", "true"}:
                entries.append(_DoctorServerEntry(config_path, server_name, None, None,
                                                  token_check=False, url_check=False,
                                                  note="local no-auth stdio profile"))
                continue
            if "headers" in server_cfg or not isinstance(server_cfg.get("command"), str):
                entries.append(_DoctorServerEntry(config_path, server_name, None, None,
                                                  token_check=False, url_check=False,
                                                  note="unsupported profile"))
                continue
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
        runtime_token = os.environ.get(bearer_token_env_var) if bearer_token_env_var else None
        return [
            _DoctorServerEntry(
                config_path=config_path,
                server_name="codex.muninn",
                token=runtime_token,
                server_url=normalized_url or None,
                token_check=False,
                note="runtime token unverified" if bearer_token_env_var else "missing bearer reference",
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


def _read_windows_user_auth_token() -> Optional[str]:
    """Read a user-scoped token when this process predates a Windows env update."""
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value = winreg.QueryValueEx(key, "MUNINN_AUTH_TOKEN")[0]
        return (value.strip() or None) if isinstance(value, str) else None
    except (FileNotFoundError, OSError):
        return None


def _select_auth_token(token_file: Optional[Path]) -> tuple[Optional[str], str]:
    """Explicit file > process env > user env > default file; no explicit-file fallback."""
    if token_file is not None or os.environ.get("MUNINN_TOKEN_FILE"):
        return _read_token_from_file(_resolve_token_file(token_file)), "file"
    process = (os.environ.get("MUNINN_AUTH_TOKEN") or "").strip()
    if process:
        return process, "env"
    user = _read_windows_user_auth_token()
    if user:
        return user, "user-env"
    fallback = _read_token_from_file(_DEFAULT_TOKEN_FILE)
    return fallback, "file" if fallback else "none"


def _token_source_allowed_for_url(source: str, url: str) -> bool:
    """Never implicitly send a Windows user-registry token off this computer."""
    if source != "user-env":
        return True
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _check_server_health(url: str, token: Optional[str], timeout_seconds: float) -> tuple[bool, str]:
    """Prove token acceptance and auth enforcement; /health alone is public."""
    if not token:
        return False, "missing_token"
    try:
        response = requests.get(
            f"{url}/auth/check", headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_seconds, allow_redirects=False,
        )
        if response.status_code != 200:
            return False, f"http_{response.status_code}"
        invalid = secrets.token_urlsafe(32)
        while invalid == token:
            invalid = secrets.token_urlsafe(32)
        negative = requests.get(
            f"{url}/auth/check", headers={"Authorization": f"Bearer {invalid}"},
            timeout=timeout_seconds, allow_redirects=False,
        )
        if negative.status_code != 401:
            return False, "auth_not_enforced"
        return True, "ok"
    except requests.RequestException as exc:
        # Requests may include Authorization values in exception messages.
        return False, f"request_{type(exc).__name__}"


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

    expected_token, token_source = _select_auth_token(args.token_file)

    issues: list[str] = []
    warnings: list[str] = []
    critical = False

    target_allowed = _token_source_allowed_for_url(token_source, target_url)
    if not target_allowed:
        critical = True
        issues.append(
            "Windows user-environment token cannot be sent to a non-loopback server; "
            "select an explicit token file."
        )
    if expected_token is None:
        critical = True
        issues.append(
            f"No expected auth token found from selected source ({token_source}); token file path is '{token_file}'."
        )

    health_ok = False
    health_detail = "skipped"
    if expected_token is not None and target_allowed:
        health_ok, health_detail = _check_server_health(target_url, expected_token, timeout)
        if not health_ok:
            critical = True
            issues.append(
                f"Server authentication check failed at {target_url}/auth/check using expected token ({health_detail})."
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
        if entry.token_check and expected_token is not None and entry.token != expected_token:
            token_mismatches.append(entry)
        if not entry.url_check:
            continue
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
    unverified_entries = [e for e in entries if e.note]
    if unverified_entries:
        warnings.append(
            f"{len(unverified_entries)} Muninn MCP entry/entries are unverified or intentionally local-only."
        )

    repaired_paths: list[Path] = []
    if args.repair and health_ok and expected_token is not None and target_allowed:
        for cfg_path in _MCP_CONFIG_PATHS:
            patched = _patch_mcp_config_env(
                cfg_path,
                new_token=expected_token,
                new_server_url=target_url,
                dry_run=False,
            )
            if patched:
                repaired_paths.append(cfg_path)
        codex_entries = _collect_codex_muninn_entries(_CODEX_CONFIG_PATH)
        if not any(entry.note for entry in codex_entries):
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
            e for e in entries if e.token_check and expected_token is not None and e.token != expected_token
        ]
        url_mismatches = [e for e in entries if e.url_check and e.server_url is not None
                          and e.server_url != target_url]
        missing_url_entries = [e for e in entries if e.url_check and e.server_url is None]
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
    print(f"Token source: {token_source}")
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
    token, source = _select_auth_token(args.token_file)
    if source == "file" and not token:
        raise SystemExit("Selected token file is missing or empty; no fallback token was sent.")
    server_url = _resolve_server_url(args.server_url)
    if not _token_source_allowed_for_url(source, server_url):
        raise SystemExit("Windows user-environment token cannot be sent to a non-loopback server.")
    try:
        response = requests.request(
            method,
            f"{server_url}{path}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=args.timeout_seconds,
            allow_redirects=False,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise SystemExit(f"{method} {path} request failed ({type(exc).__name__})") from None
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
    """Install local session hooks (dry run unless --apply)."""
    from muninn.history import hook_install

    apps = args.app or ["claude", "codex", "gemini"]
    install = args.action == "install"
    server_url = _resolve_server_url(args.server_url)
    plans = []
    if "claude" in apps:
        plans.append(hook_install.claude_plan(server_url, install=install))
    if "codex" in apps:
        plans.append(hook_install.codex_plan(install=install))
    if "gemini" in apps:
        plans.append(hook_install.gemini_plan(server_url, install=install))
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
            llm_settings.save_key(key)
            print("Available for this process only; set MUNINN_OPENROUTER_API_KEY in your environment for future runs.")
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
                   "limit": args.limit, "retry_refused": args.retry_refused}
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


def cmd_credentials(args: argparse.Namespace) -> int:
    """Explicit local-only access to the separate encrypted credential vault."""
    import getpass

    from muninn.core.config import DEFAULT_DATA_DIR
    from muninn.history.credential_store import CredentialStore

    root = args.root or Path(os.environ.get("MUNINN_DATA_DIR", DEFAULT_DATA_DIR)) / "credential_vault"
    if args.action == "search":
        print(json.dumps(CredentialStore(root).search(args.query), indent=2))
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SystemExit("Credential unlock and reveal require an interactive local terminal.")
    passphrase = getpass.getpass("Credential vault passphrase (hidden): ")
    if args.action == "init":
        if passphrase != getpass.getpass("Confirm passphrase (hidden): "):
            raise SystemExit("Passphrases did not match; no vault created.")
        CredentialStore.create(root, passphrase)
        print(f"Encrypted credential vault created at {root}.")
    elif args.action == "reveal":
        # Reveal is intentionally printed only to an interactive local terminal.
        print(CredentialStore(root).reveal(args.record_id, passphrase=passphrase))
    elif args.action == "backup":
        count = CredentialStore(root).backup(args.destination, passphrase=passphrase)
        print(f"Validated encrypted backup at {args.destination} ({count} records).")
    elif args.action == "restore":
        CredentialStore.restore(args.source, root, passphrase=passphrase)
        print(f"Validated encrypted vault restored at {root}.")
    elif args.action == "scan":
        from muninn.history.credential_discovery import scan_archive, scan_project_files
        from muninn.history.secure_archive import SecureHistoryArchive

        store = CredentialStore(root)
        if args.backup_before:
            backup_count = store.backup(args.backup_before, passphrase=passphrase)
            print(json.dumps({"stage": "validated_pre_scan_backup", "records": backup_count},
                             sort_keys=True), flush=True)
        with store.scan_session(passphrase) as session:
            reports = []
            for project_index, project in enumerate(args.project_root or [], start=1):
                def project_progress(status):
                    print(json.dumps({"stage": "project_scan", "project_index": project_index,
                                      **status}, sort_keys=True), flush=True)

                reports.append(scan_project_files(
                    project, session, passphrase="", progress=project_progress,
                ))
            project_totals = {key: sum(int(report[key]) for report in reports) for key in (
                "files", "succeeded", "errors", "walk_errors", "ambiguous",
                "candidates", "inserted", "updated", "stale",
            )}
            project_totals["error_categories"] = {
                name: sum(int(report["error_categories"][name]) for report in reports)
                for name in ("root", "walk", "path", "metadata", "utf8", "io",
                             "source_changed", "other")
            }
            project_totals["complete"] = all(report["complete"] for report in reports)
            archive_report = None
            if args.archive_root:
                def progress(status):
                    print(json.dumps({"stage": "archive_scan", **status}, sort_keys=True), flush=True)

                archive_report = scan_archive(
                    SecureHistoryArchive(args.archive_root), session, passphrase="",
                    offset=args.archive_offset, max_snapshots=args.max_snapshots,
                    expected_generation=args.archive_generation, progress=progress,
                )
        complete = project_totals["complete"] and (archive_report is None or archive_report["complete"])
        print(json.dumps({"project": project_totals, "archive": archive_report, "complete": complete},
                         sort_keys=True))
        return 0 if complete else 2
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
    from muninn.core.maintenance import content_hash, normalize_legacy_record

    records = _read_export(args.file)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    totals: dict = {}
    dry_run_seen: set[str] = set()
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        cross_batch_duplicates = 0
        if not args.apply:
            filtered = []
            for raw in batch:
                item = normalize_legacy_record(
                    raw, user_id=args.user_id, namespace=args.namespace, source=args.source
                )
                if item is None:
                    filtered.append(raw)
                    continue
                digest = content_hash(item["content"])
                if digest in dry_run_seen:
                    cross_batch_duplicates += 1
                    continue
                dry_run_seen.add(digest)
                filtered.append(raw)
            batch = filtered
        report = _admin_post(args, "/admin/import", {
            "records": batch,
            "user_id": args.user_id,
            "namespace": args.namespace,
            "source": args.source,
            "dry_run": not args.apply,
        })
        if not args.apply:
            report["read"] += cross_batch_duplicates
            report["duplicates"] += cross_batch_duplicates
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
    history.add_argument("--retry-refused", action="store_true",
                         help="'analyze' also retries threads every model refused before (try another --model).")
    history.add_argument("--offset", type=int, default=0)
    history.add_argument("--limit", type=int, default=50)

    credentials = subparsers.add_parser(
        "credentials",
        help="Manage the separate encrypted credential vault and scan approved local project files.",
    )
    credentials.add_argument("action", choices=["init", "search", "reveal", "backup", "restore", "scan"])
    credentials.add_argument("query", nargs="?", help="Metadata-only query for 'search'.")
    credentials.add_argument("--root", type=Path, help="Vault location (default: MUNINN_DATA_DIR/credential_vault).")
    credentials.add_argument("--record-id", help="Record id for explicit 'reveal'.")
    credentials.add_argument("--destination", type=Path, help="New directory for 'backup'.")
    credentials.add_argument("--source", type=Path, help="Existing encrypted backup directory for 'restore'.")
    credentials.add_argument("--project-root", type=Path, action="append",
                             help="Approved project root for 'scan' (repeatable); streams supported text files including .env, config, source, and docs.")
    credentials.add_argument("--archive-root", type=Path,
                             help="Encrypted archive to scan as historical credential observations.")
    credentials.add_argument("--archive-offset", type=int, default=0,
                             help="Legacy fixed-generation range start. After manifest growth, restart at 0; authenticated receipts skip already-scanned snapshots.")
    credentials.add_argument("--archive-generation", type=int,
                             help="Required with nonzero --archive-offset; never reuse an offset after manifest growth.")
    credentials.add_argument("--max-snapshots", type=int,
                             help="Bound this archive scan; an unfinished range reports complete=false.")
    credentials.add_argument("--backup-before", type=Path,
                             help="For 'scan', make a new authenticated encrypted vault backup before any findings are written.")

    hooks = subparsers.add_parser(
        "hooks",
        help="Add Muninn session hooks to Claude Code, Codex, and Gemini CLI.",
        description=(
            "Session start: the agent receives the project briefing (goal, open handoffs, earlier\n"
            "threads from every app). Before compaction and at session end: a strict-mode hook\n"
            "durably queues transcript capture; archival and indexing finish asynchronously.\n"
            "A successful hook receipt does not guarantee that source bytes were copied.\n"
            "Dry run unless --apply;\n"
            "settings files are backed up; only Muninn's own entries are changed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    hooks.add_argument("action", choices=["status", "install", "uninstall"])
    hooks.add_argument("--app", action="append", choices=["claude", "codex", "gemini"],
                       help="Only this app (repeatable).")
    hooks.add_argument("--server-url", default=None, help="Muninn server URL the hooks call.")
    hooks.add_argument("--apply", action="store_true", help="Write the settings.")

    openrouter = subparsers.add_parser(
        "openrouter",
        help="Set up the OpenRouter key used to analyze imported conversations.",
        description=(
            "status  show which key and models are used (the key is masked)\n"
            "set     prompt for a key, verify it, use it in this process, and optionally save a model\n"
            "clear   remove saved nonsecret model settings; analysis falls back to local Ollama\n"
            "Every request requires zero data retention. Set MUNINN_OPENROUTER_API_KEY in the\n"
            "process or Windows user environment for persistent use; no key is saved to config."
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
    if args.command == "credentials":
        if args.action == "search" and not args.query:
            parser.error("credentials search requires a metadata query")
        if args.action == "reveal" and not args.record_id:
            parser.error("credentials reveal requires --record-id")
        if args.action == "backup" and not args.destination:
            parser.error("credentials backup requires --destination")
        if args.action == "restore" and not args.source:
            parser.error("credentials restore requires --source")
        if args.action == "scan" and not (args.project_root or args.archive_root):
            parser.error("credentials scan requires --project-root or --archive-root")
        return cmd_credentials(args)
    if args.command == "hooks":
        return cmd_hooks(args)
    if args.command == "openrouter":
        return cmd_openrouter(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
