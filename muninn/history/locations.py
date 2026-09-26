"""Where each AI app keeps its local conversation history, resolved from the apps' own settings.

Paths are never hard-coded to one layout: each app's relocation variable is
honoured (CLAUDE_CONFIG_DIR, CODEX_HOME), per-OS app-data folders are probed,
and versioned files (Codex ``state_*.sqlite``) are globbed. Each source also
reports the app's retention setting, because Claude Code and Gemini CLI delete
old transcripts on their own; the vault copies them before that happens.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Credential and settings files that must never be read as conversations.
NEVER_READ = {
    "auth.json", "oauth_creds.json", "google_accounts.json", "credentials.json", ".credentials.json",
    "claude_desktop_config.json", "settings.json", "settings.local.json", "config.toml", "config.json",
    "installation_id", "mcp-oauth-tokens.json", "keychain.json", "token.json", "tokens.json",
}


@dataclass
class HistorySource:
    """One app's history on this machine."""

    provider: str                      # claude_code | codex | gemini_cli
    home: Path                         # the app's data directory
    patterns: List[str]                # globs relative to home for transcripts
    prompt_histories: List[str] = field(default_factory=list)   # prompt logs kept after cleanup
    extras: Dict[str, List[Path]] = field(default_factory=dict)  # side metadata (titles, desktop sessions)
    retention: Dict[str, object] = field(default_factory=dict)
    relocated_by: Optional[str] = None

    @property
    def exists(self) -> bool:
        return self.home.is_dir()


def _env_dir(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def app_data_dirs(home: Path) -> List[Path]:
    """Per-OS roots where desktop apps (Claude, ChatGPT) keep application data."""
    dirs: List[Path] = []
    if sys.platform == "win32" or os.environ.get("APPDATA"):
        for var, fallback in (("APPDATA", home / "AppData" / "Roaming"), ("LOCALAPPDATA", home / "AppData" / "Local")):
            dirs.append(_env_dir(var) or fallback)
    dirs.append(home / "Library" / "Application Support")
    dirs.append(_env_dir("XDG_CONFIG_HOME") or home / ".config")
    seen, unique = set(), []
    for directory in dirs:
        if str(directory) not in seen:
            seen.add(str(directory))
            unique.append(directory)
    return unique


def _read_json(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def claude_code_source(home: Path) -> HistorySource:
    relocated = _env_dir("CLAUDE_CONFIG_DIR")
    base = relocated or home / ".claude"
    settings = _read_json(base / "settings.json")
    retention: Dict[str, object] = {
        "setting": "cleanupPeriodDays",
        "days": settings.get("cleanupPeriodDays", 30),
        "desktop_days": settings.get("desktopSessionCleanupPeriodDays"),
        "note": "Claude Code deletes transcripts older than cleanupPeriodDays (default 30) at startup.",
    }
    # Claude Desktop's Code tab and Cowork write transcripts here too; the
    # desktop app keeps titles and the CLI session id in its own app data.
    desktop_meta = [d / "Claude" / "claude-code-sessions" for d in app_data_dirs(home)]
    return HistorySource(
        provider="claude_code",
        home=base,
        patterns=["projects/**/*.jsonl"],
        prompt_histories=["history.jsonl"],
        extras={"desktop_sessions": [d for d in desktop_meta if d.is_dir()]},
        retention=retention,
        relocated_by="CLAUDE_CONFIG_DIR" if relocated else None,
    )


def codex_source(home: Path) -> HistorySource:
    """Codex CLI, IDE extension and the ChatGPT desktop app share CODEX_HOME."""
    relocated = _env_dir("CODEX_HOME")
    base = relocated or home / ".codex"
    return HistorySource(
        provider="codex",
        home=base,
        patterns=[
            "sessions/**/rollout-*.jsonl", "sessions/**/rollout-*.jsonl.zst",
            "archived_sessions/**/rollout-*.jsonl", "archived_sessions/**/rollout-*.jsonl.zst",
        ],
        prompt_histories=["history.jsonl"],
        extras={"state_db": sorted(base.glob("state_*.sqlite"), reverse=True) if base.is_dir() else []},
        retention={"note": "Codex compresses old and archived rollouts to .zst; deleting a thread removes its file."},
        relocated_by="CODEX_HOME" if relocated else None,
    )


def gemini_source(home: Path) -> HistorySource:
    base = home / ".gemini"
    settings = _read_json(base / "settings.json")
    general = settings.get("general") if isinstance(settings.get("general"), dict) else {}
    retention = general.get("sessionRetention") if isinstance(general, dict) else None
    return HistorySource(
        provider="gemini_cli",
        home=base,
        patterns=["tmp/*/chats/*.json", "tmp/*/chats/*.jsonl"],
        retention={
            "setting": "general.sessionRetention",
            "value": retention,
            "note": "Gemini CLI can delete sessions older than sessionRetention.maxAge (default 30d when enabled).",
        },
    )


def history_homes(home: Optional[Path] = None) -> List[Path]:
    """This user's home plus MUNINN_HISTORY_HOMES (e.g. /mnt/c/Users/me when Muninn runs in WSL)."""
    homes = [(home or Path.home()).expanduser()]
    for extra in os.environ.get("MUNINN_HISTORY_HOMES", "").split(os.pathsep):
        if extra.strip():
            path = Path(extra.strip()).expanduser()
            if path not in homes:
                homes.append(path)
    return homes


def history_sources(home: Optional[Path] = None) -> List[HistorySource]:
    user_home = (home or Path.home()).expanduser()
    return [claude_code_source(user_home), codex_source(user_home), gemini_source(user_home)]


def export_candidates(home: Optional[Path] = None) -> List[Path]:
    """ChatGPT and Claude data exports the user downloaded (conversations.json, or the export .zip)."""
    user_home = (home or Path.home()).expanduser()
    found: List[Path] = []
    for folder in (user_home / "Downloads", user_home / "Documents"):
        if not folder.is_dir():
            continue
        for pattern in ("conversations.json", "*/conversations.json", "*.zip"):
            for path in folder.glob(pattern):
                if path.suffix == ".zip" and not _zip_has_conversations(path):
                    continue
                found.append(path)
    return sorted(set(found))


def _zip_has_conversations(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            return any(Path(name).name == "conversations.json" for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
