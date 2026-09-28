"""Install Muninn's hooks into Claude Code, Codex, and Gemini CLI settings.

Claude Code (CLI, IDE and Claude Desktop's Code tab) reads ``settings.json``
under ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``. Both it and Codex run the
standard-library ``hook_client.py`` command bridge to reach the local server.
Gemini CLI uses the same bridge with its own event names and millisecond timeouts.
Only entries Muninn added are ever changed or removed.
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from muninn.history.locations import claude_code_source, codex_source, gemini_source

CLAUDE_EVENTS = {
    # event: (matcher, timeout seconds)
    "SessionStart": ("startup|resume|clear|compact", 10),
    "PreCompact": (None, 30),
    "Stop": (None, 5),
    "SessionEnd": (None, 2),
}
CODEX_EVENTS = {"SessionStart": 10, "PreCompact": 30, "Stop": 5, "SessionEnd": 1}
GEMINI_EVENTS_MS = {"SessionStart": 10000, "PreCompress": 30000,
                    "AfterAgent": 5000, "SessionEnd": 2000}


@dataclass
class HookPlan:
    app: str
    path: Path
    before: Dict[str, Any]
    after: Dict[str, Any]

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _read(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _is_muninn(handler: Dict[str, Any]) -> bool:
    if handler.get("name") == "muninn-local-memory":
        return True
    if str(handler.get("url", "")).endswith("/hooks/claude-code"):
        return True  # legacy HTTP hook installed by older Muninn versions
    return re.search(r'[/\\]muninn[/\\]hook_client\.py"?\s+(?:codex|claude-code|gemini-cli)(?:\s|$)',
                     str(handler.get("command", ""))) is not None


def _without_muninn(hooks: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for event, groups in hooks.items():
        kept_groups = []
        for group in groups if isinstance(groups, list) else []:
            handlers = [h for h in group.get("hooks", []) if not _is_muninn(h)]
            if handlers:
                kept_groups.append(dict(group, hooks=handlers))
        if kept_groups:
            cleaned[event] = kept_groups
    return cleaned


def _with_groups(settings: Dict[str, Any], groups: Dict[str, Dict[str, Any]], install: bool) -> Dict[str, Any]:
    updated = copy.deepcopy(settings)
    hooks = _without_muninn(updated.get("hooks") or {})
    if install:
        for event, group in groups.items():
            hooks.setdefault(event, []).append(group)
    if hooks:
        updated["hooks"] = hooks
    else:
        updated.pop("hooks", None)
    return updated


def claude_plan(server_url: str, install: bool = True, home: Optional[Path] = None,
                python: Optional[str] = None) -> HookPlan:
    path = claude_code_source(home or Path.home()).home / "settings.json"
    groups = {}
    for event, (matcher, timeout) in CLAUDE_EVENTS.items():
        # SessionStart cannot use HTTP handlers in Claude Code. Use the same
        # local bridge for every event so a process started before a Windows
        # User token was set can still authenticate without storing it here.
        client = Path(__file__).resolve().parent.parent / "hook_client.py"
        handler = {"type": "command",
                   "command": f'"{python or sys.executable}" "{client}" claude-code "{server_url}"',
                   "timeout": timeout}
        groups[event] = {**({"matcher": matcher} if matcher else {}), "hooks": [handler]}
    before = _read(path)
    return HookPlan("claude_code", path, before, _with_groups(before, groups, install))


def codex_plan(install: bool = True, home: Optional[Path] = None, python: Optional[str] = None) -> HookPlan:
    path = codex_source(home or Path.home()).home / "hooks.json"
    client = Path(__file__).resolve().parent.parent / "hook_client.py"
    command = f'"{python or sys.executable}" "{client}" codex'
    groups = {event: {"hooks": [{"type": "command", "command": command, "timeout": timeout}]}
              for event, timeout in CODEX_EVENTS.items()}
    before = _read(path)
    return HookPlan("codex", path, before, _with_groups(before, groups, install))


def gemini_plan(server_url: str, install: bool = True, home: Optional[Path] = None,
                python: Optional[str] = None) -> HookPlan:
    path = gemini_source(home or Path.home()).home / "settings.json"
    client = Path(__file__).resolve().parent.parent / "hook_client.py"
    command = f'"{python or sys.executable}" "{client}" gemini-cli "{server_url}"'
    groups = {event: {"hooks": [{"name": "muninn-local-memory", "type": "command",
                                 "command": command, "timeout": timeout}]}
              for event, timeout in GEMINI_EVENTS_MS.items()}
    before = _read(path)
    return HookPlan("gemini_cli", path, before, _with_groups(before, groups, install))


def apply_plan(plan: HookPlan) -> Optional[Path]:
    """Write the new settings, keeping a timestamped backup of the old file."""
    if not plan.changed:
        return None
    backup = None
    if plan.path.exists():
        backup = plan.path.with_name(f"{plan.path.name}.muninn-backup-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_bytes(plan.path.read_bytes())
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan.path.with_name(plan.path.name + ".tmp")
    tmp.write_text(json.dumps(plan.after, indent=2) + "\n", encoding="utf-8")
    tmp.replace(plan.path)
    return backup


def installed(plan: HookPlan) -> List[str]:
    return sorted(
        event for event, groups in (plan.before.get("hooks") or {}).items()
        if any(_is_muninn(h) for g in groups if isinstance(g, dict) for h in g.get("hooks", []))
    )
