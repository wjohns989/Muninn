"""Agent hooks: brief a new session, and capture transcripts before compaction or at the end.

Claude Code calls ``POST /hooks/claude-code`` directly (``type: "http"`` hooks);
Codex runs ``muninn/hook_client.py`` (command hooks), which posts the same
payload to ``/hooks/codex``. Both send ``hook_event_name``, ``session_id``,
``transcript_path`` and ``cwd``.

- SessionStart: returns the project briefing as ``additionalContext``, which
  both hosts add to the new session's context.
- PreCompact, PostCompact, SessionEnd: copy the transcript to the vault and
  import that thread now, so what compaction drops is saved at that moment.
- Stop (after every reply): the same, at most every couple of minutes per thread.

Hooks answer immediately; capture runs in the background.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from muninn.core import handoffs
from muninn.core.projects import project_for_directory

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory
    from muninn.history.service import HistoryService

PROVIDER_FOR_AGENT = {"claude-code": "claude_code", "codex": "codex"}
_CAPTURE_NOW = {"PreCompact", "PostCompact", "SessionEnd"}


async def handle_hook(
    agent: str, payload: Dict[str, Any], memory: "MuninnMemory", service: Optional["HistoryService"]
) -> Dict[str, Any]:
    event = str(payload.get("hook_event_name") or "")
    transcript = payload.get("transcript_path")
    provider = PROVIDER_FOR_AGENT.get(agent)
    started_from = payload.get("source") or payload.get("reason")
    if service is not None and provider and isinstance(transcript, str) and transcript:
        if event in _CAPTURE_NOW or (event == "SessionStart" and started_from in ("compact", "resume")):
            service.capture_later(transcript, provider, force=True)
        elif event == "Stop":
            service.capture_later(transcript, provider)
    if event != "SessionStart":
        return {}
    cwd = payload.get("cwd")
    project = project_for_directory(cwd) if isinstance(cwd, str) else None
    context = await handoffs.project_context(memory, project=project, recent_limit=8)
    return {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                   "additionalContext": handoffs.render_briefing(context)}}
