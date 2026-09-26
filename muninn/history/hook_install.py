"""Install Muninn's hooks into Claude Code and Codex settings (dry run first, backup, idempotent).

Claude Code (CLI, IDE and Claude Desktop's Code tab) reads ``settings.json``
under ``$CLAUDE_CONFIG_DIR`` or ``~/.claude`` and supports ``http`` hooks, so it
posts straight to the server. Codex (CLI, IDE, ChatGPT desktop app) reads
``$CODEX_HOME/hooks.json`` and runs commands, so it calls ``hook_client.py``.
Only entries Muninn added are ever changed or removed.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from muninn.history.locations import claude_code_source, codex_source

CLAUDE_EVENTS = {
    # event: (matcher, timeout seconds)
    "SessionStart": ("startup|resume|clear|compact", 10),
    "PreCompact": (None, 30),
    "Stop": (None, 5),
    "SessionEnd": (None, 2),
}
CODEX_EVENTS = {"SessionStart": 10, "PreCompact": 30, "Stop": 5, "SessionEnd": 1}


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
    return "/hooks/claude-code" in str(handler.get("url", "")) or "hook_client.py" in str(handler.get("command", ""))


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


def claude_plan(server_url: str, install: bool = True, home: Optional[Path] = None) -> HookPlan:
    path = claude_code_source(home or Path.home()).home / "settings.json"
    groups = {}
    for event, (matcher, timeout) in CLAUDE_EVENTS.items():
        handler = {
            "type": "http",
            "url": f"{server_url.rstrip('/')}/hooks/claude-code",
            "timeout": timeout,
            "headers": {"Authorization": "Bearer $MUNINN_AUTH_TOKEN"},
            "allowedEnvVars": ["MUNINN_AUTH_TOKEN"],
        }
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
