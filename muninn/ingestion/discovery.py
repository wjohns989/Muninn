"""
Discovery of legacy assistant chats and external MCP-memory sources.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from muninn.ingestion.parser import SUPPORTED_EXTENSIONS


@dataclass
class DiscoveredLegacySource:
    source_id: str
    provider: str
    category: str
    path: str
    source_type: str
    parser_supported: bool
    confidence: str
    size_bytes: int
    notes: str = ""
    parent_path: str = ""
    path_depth: int = 0
    modified_at_epoch: Optional[float] = None
    modified_at_iso: Optional[str] = None
    relative_path_hint: str = ""


def _canonical_source_id(provider: str, category: str, path: str) -> str:
    key = f"{provider}|{category}|{path}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(key).hexdigest()
    return f"src_{digest[:16]}"


def _source_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    return "unsupported"


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return fallback


def _provider_specs(home: Path, roots: Sequence[Path]) -> List[Dict[str, object]]:
    appdata = _env_path("APPDATA", home / "AppData" / "Roaming")
    localapp = _env_path("LOCALAPPDATA", home / "AppData" / "Local")
    xdg_config = _env_path("XDG_CONFIG_HOME", home / ".config")
    mac_app_support = home / "Library" / "Application Support"

    specs: List[Dict[str, object]] = [
        {
            "provider": "codex_cli",
            "category": "assistant_chat",
            "patterns": [
                home / ".codex" / "history.jsonl",
                home / ".codex" / "sessions" / "**" / "*.jsonl",
            ],
            "confidence": "high",
            "notes": "Codex CLI session and history files",
        },
        {
            "provider": "claude_code",
            "category": "assistant_chat",
            "patterns": [home / ".claude" / "projects" / "**" / "*.jsonl"],
            "confidence": "high",
            "notes": "Claude Code project transcripts",
        },
        {
            "provider": "gemini_cli",
            "category": "assistant_chat",
            "patterns": [home / ".gemini" / "**" / "*.jsonl", home / ".gemini" / "**" / "*.json"],
            "confidence": "medium",
            "notes": "Gemini CLI settings/session artifacts",
        },
        {
            "provider": "antigravity_brain",
            "category": "assistant_chat",
            "patterns": [home / ".gemini" / "antigravity" / "brain" / "**" / "output.txt"],
            "confidence": "high",
            "notes": "Antigravity Brain output transcripts",
        },
        {
            "provider": "serena_memory",
            "category": "mcp_memory",
            "patterns": [home / ".serena" / "memories" / "**" / "*.md", home / ".serena" / "memories" / "**" / "*.json"],
            "confidence": "high",
            "notes": "Serena memory files",
        },
        {
            "provider": "loki_memory",
            "category": "mcp_memory",
            "patterns": [home / ".loki" / "memory" / "**" / "*.md", home / ".loki" / "memory" / "**" / "*.json"],
            "confidence": "medium",
            "notes": "Loki memory semantic/episodic artifacts",
        },
        {
            "provider": "vscode_chat",
            "category": "assistant_chat",
            "patterns": [
                appdata / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                appdata / "Code" / "User" / "workspaceStorage" / "**" / "state.vscdb",
                xdg_config / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                xdg_config / "Code" / "User" / "workspaceStorage" / "**" / "state.vscdb",
                mac_app_support / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                mac_app_support / "Code" / "User" / "workspaceStorage" / "**" / "state.vscdb",
            ],
            "confidence": "medium",
            "notes": "VS Code workspace chat/session state",
        },
        {
            "provider": "cursor_chat",
            "category": "assistant_chat",
            "patterns": [
                appdata / "Cursor" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                appdata / "Cursor" / "User" / "workspaceStorage" / "**" / "state.vscdb",
                xdg_config / "Cursor" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                xdg_config / "Cursor" / "User" / "workspaceStorage" / "**" / "state.vscdb",
                mac_app_support / "Cursor" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                mac_app_support / "Cursor" / "User" / "workspaceStorage" / "**" / "state.vscdb",
            ],
            "confidence": "medium",
            "notes": "Cursor workspace chat/session state",
        },
        {
            "provider": "copilot_chat",
            "category": "assistant_chat",
            "patterns": [
                appdata / "Code" / "User" / "globalStorage" / "github.copilot-chat" / "**" / "*.json",
                appdata / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                xdg_config / "Code" / "User" / "globalStorage" / "github.copilot-chat" / "**" / "*.json",
                xdg_config / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
                mac_app_support / "Code" / "User" / "globalStorage" / "github.copilot-chat" / "**" / "*.json",
                mac_app_support / "Code" / "User" / "workspaceStorage" / "**" / "chatSessions" / "*.json",
            ],
            "confidence": "low",
            "notes": "Copilot Chat artifacts (layout varies by VS Code version)",
        },
        {
            "provider": "claude_desktop",
            "category": "assistant_chat",
            "patterns": [
                appdata / "Claude" / "**" / "*.json",
                appdata / "Claude" / "**" / "*.jsonl",
                localapp / "Claude" / "**" / "*.json",
                localapp / "Claude" / "**" / "*.jsonl",
                home / ".config" / "Claude" / "**" / "*.json",
                home / ".config" / "Claude" / "**" / "*.jsonl",
                mac_app_support / "Claude" / "**" / "*.json",
                mac_app_support / "Claude" / "**" / "*.jsonl",
            ],
            "confidence": "low",
            "notes": "Claude Desktop local artifacts (layout may vary)",
        },
        {
            "provider": "chatgpt_desktop",
            "category": "assistant_chat",
            "patterns": [
                appdata / "ChatGPT" / "**" / "*.json",
                appdata / "ChatGPT" / "**" / "*.jsonl",
                localapp / "ChatGPT" / "**" / "*.json",
                localapp / "ChatGPT" / "**" / "*.jsonl",
                home / ".config" / "ChatGPT" / "**" / "*.json",
                home / ".config" / "ChatGPT" / "**" / "*.jsonl",
                mac_app_support / "ChatGPT" / "**" / "*.json",
                mac_app_support / "ChatGPT" / "**" / "*.jsonl",
            ],
            "confidence": "low",
            "notes": "ChatGPT desktop local artifacts (layout may vary)",
        },
    ]

    for root in roots:
        specs.append(
            {
                "provider": "custom_root",
                "category": "assistant_chat",
                "patterns": [
                    root / "**" / "*.jsonl",
                    root / "**" / "*.json",
                    root / "**" / "*.md",
                    root / "**" / "*.txt",
                    root / "**" / "*.sqlite",
                    root / "**" / "*.sqlite3",
                    root / "**" / "*.db",
                    root / "**" / "*.vscdb",
                ],
                "confidence": "manual",
                "notes": "User-supplied root scan",
            }
        )

    return specs


def _iter_paths(pattern: Path) -> Iterable[Path]:
    text = str(pattern)
    if "**" in text or "*" in text:
        base = pattern
        while any(tok in str(base) for tok in ("*", "?", "[")):
            base = base.parent
            if base == base.parent:
                break
        if not base.exists():
            return []
        pattern_parts = pattern.parts
        base_parts = base.parts
        glob_parts = pattern_parts[len(base_parts):]
        glob_expr = os.path.join(*glob_parts) if glob_parts else "*"
        return base.glob(glob_expr)

    if pattern.exists() and pattern.is_file():
        return [pattern]
    return []


def discover_legacy_sources(
    *,
    home: Optional[Path] = None,
    roots: Optional[Sequence[str]] = None,
    include_unsupported: bool = False,
    max_results_per_provider: int = 100,
) -> List[Dict[str, object]]:
    user_home = (home or Path.home()).expanduser().resolve()
    root_paths = [Path(r).expanduser().resolve() for r in (roots or [])]

    discovered: List[DiscoveredLegacySource] = []
    seen: set[str] = set()

    for spec in _provider_specs(user_home, root_paths):
        provider = str(spec["provider"])
        category = str(spec["category"])
        confidence = str(spec.get("confidence", "medium"))
        notes = str(spec.get("notes", ""))
        patterns: Sequence[Path] = spec.get("patterns", [])  # type: ignore[assignment]

        count = 0
        for pattern in patterns:
            for candidate in _iter_paths(pattern):
                if count >= max_results_per_provider:
                    break
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if not resolved.exists() or not resolved.is_file():
                    continue
                key = str(resolved)
                if key in seen:
                    continue

                source_type = _source_type_for_path(resolved)
                parser_supported = source_type != "unsupported"
                if not parser_supported and not include_unsupported:
                    continue

                size_bytes = 0
                modified_at_epoch: Optional[float] = None
                modified_at_iso: Optional[str] = None
                try:
                    stat = resolved.stat()
                    size_bytes = stat.st_size
                    modified_at_epoch = float(stat.st_mtime)
                    modified_at_iso = datetime.fromtimestamp(
                        modified_at_epoch, tz=timezone.utc
                    ).isoformat().replace("+00:00", "Z")
                except OSError:
                    pass

                relative_hint = ""
                for root in root_paths:
                    try:
                        relative_hint = str(resolved.relative_to(root))
                        break
                    except ValueError:
                        continue
                if not relative_hint:
                    try:
                        relative_hint = str(resolved.relative_to(user_home))
                    except ValueError:
                        relative_hint = resolved.name

                source_id = _canonical_source_id(provider, category, key)
                discovered.append(
                    DiscoveredLegacySource(
                        source_id=source_id,
                        provider=provider,
                        category=category,
                        path=key,
                        source_type=source_type,
                        parser_supported=parser_supported,
                        confidence=confidence,
                        size_bytes=size_bytes,
                        notes=notes,
                        parent_path=str(resolved.parent),
                        path_depth=len(resolved.parts),
                        modified_at_epoch=modified_at_epoch,
                        modified_at_iso=modified_at_iso,
                        relative_path_hint=relative_hint,
                    )
                )
                seen.add(key)
                count += 1

    discovered.sort(key=lambda item: (item.provider, item.path))
    return [
        {
            "source_id": item.source_id,
            "provider": item.provider,
            "category": item.category,
            "path": item.path,
            "source_type": item.source_type,
            "parser_supported": item.parser_supported,
            "confidence": item.confidence,
            "size_bytes": item.size_bytes,
            "notes": item.notes,
            "parent_path": item.parent_path,
            "path_depth": item.path_depth,
            "modified_at_epoch": item.modified_at_epoch,
            "modified_at_iso": item.modified_at_iso,
            "relative_path_hint": item.relative_path_hint,
        }
        for item in discovered
    ]
