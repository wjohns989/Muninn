"""Agent-to-agent handoffs and the project briefing agents load at session start.

Every agent (Claude Code, Claude Desktop, Codex, Gemini CLI, Cursor, ...) talks
to the same store, so passing work on is a record in that store: one agent
writes a handoff for a project, the next one resumes (claims) it, and marks it
done. `project_context` gathers what a fresh session needs in one call.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

DETAIL_LIST_FIELDS = ("next_steps", "open_questions", "decisions", "files")
ACTIVE_STATUSES = ("open", "claimed")
_TITLE_CHARS = 80
_SNIPPET_CHARS = 400


def _title_from(summary: str) -> str:
    line = " ".join(summary.split())
    return line if len(line) <= _TITLE_CHARS else line[: _TITLE_CHARS - 1] + "…"


def _clean_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _snippet(text: str) -> str:
    return text if len(text) <= _SNIPPET_CHARS else text[: _SNIPPET_CHARS - 1] + "…"


async def create_handoff(
    memory: "MuninnMemory",
    *,
    project: str,
    summary: str,
    from_agent: str,
    title: Optional[str] = None,
    to_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    user_id: str = "global_user",
) -> Dict[str, Any]:
    if not project or not project.strip():
        raise ValueError("project is required")
    if not summary or not summary.strip():
        raise ValueError("summary is required: what was done and where things stand")
    details = details or {}
    clean: Dict[str, Any] = {field: _clean_list(details.get(field)) for field in DETAIL_LIST_FIELDS}
    if isinstance(details.get("branch"), str) and details["branch"].strip():
        clean["branch"] = details["branch"].strip()
    record = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "project": project.strip(),
        "title": (title or "").strip() or _title_from(summary),
        "summary": summary.strip(),
        "details": {key: value for key, value in clean.items() if value},
        "from_agent": from_agent or "unknown",
        "to_agent": (to_agent or "").strip() or None,
        "created_at": time.time(),
    }
    return await asyncio.to_thread(memory._metadata.add_handoff, record)


async def list_handoffs(
    memory: "MuninnMemory",
    *,
    project: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    limit: int = 20,
    user_id: str = "global_user",
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        memory._metadata.list_handoffs, user_id, project, statuses, max(1, min(int(limit), 100))
    )


async def resume_handoff(
    memory: "MuninnMemory",
    *,
    agent: str,
    project: Optional[str] = None,
    handoff_id: Optional[str] = None,
    claim: bool = True,
    user_id: str = "global_user",
) -> Dict[str, Any]:
    """Pick up a handoff: the given id, else the newest open one for the project.

    ``to_agent`` is a preference, not a lock: handoffs addressed to this agent come first, then
    unaddressed ones, then those meant for another agent (agent names differ between hosts).
    """
    store = memory._metadata
    if handoff_id:
        handoff = await asyncio.to_thread(store.get_handoff, handoff_id)
        if handoff is None or handoff["user_id"] != user_id:
            return {"handoff": None, "message": f"Handoff {handoff_id} not found"}
    else:
        candidates = await list_handoffs(memory, project=project, statuses=["open"], limit=50, user_id=user_id)
        ranked = sorted(candidates, key=lambda h: 0 if h["to_agent"] == agent else 1 if h["to_agent"] is None else 2)
        handoff = ranked[0] if ranked else None
        if handoff is None:
            where = f" for project '{project}'" if project else ""
            return {"handoff": None, "message": f"No open handoff{where}"}
    if claim and handoff["status"] in ACTIVE_STATUSES:
        previous = handoff["claimed_by"] if handoff["status"] == "claimed" else None
        claimed = await asyncio.to_thread(
            store.transition_handoff, handoff["id"], "claimed", agent=agent, allowed_from=ACTIVE_STATUSES
        )
        handoff = claimed or handoff
        if previous and previous != agent:
            handoff["previously_claimed_by"] = previous
    if handoff["to_agent"] not in (None, agent):
        handoff["note_to_agent"] = f"This handoff was addressed to {handoff['to_agent']}."
    return {"handoff": handoff}


async def finish_handoff(
    memory: "MuninnMemory",
    *,
    handoff_id: str,
    agent: str,
    status: str = "done",
    note: Optional[str] = None,
    user_id: str = "global_user",
) -> Optional[Dict[str, Any]]:
    if status not in ("done", "cancelled", "open"):
        raise ValueError("status must be 'done', 'cancelled' or 'open' (to release it)")
    existing = await asyncio.to_thread(memory._metadata.get_handoff, handoff_id)
    if existing is None or existing["user_id"] != user_id:
        return None
    return await asyncio.to_thread(
        memory._metadata.transition_handoff, handoff_id, status, agent=agent, note=note,
    )


def _memory_item(record: Any) -> Dict[str, Any]:
    return {
        "id": record.id,
        "memory": _snippet(record.content),
        "agent": record.source_agent,
        "created_at": record.created_at,
    }


async def project_context(
    memory: "MuninnMemory",
    *,
    project: Optional[str],
    recent_limit: int = 10,
    user_id: str = "global_user",
) -> Dict[str, Any]:
    """One-call briefing: goal, active handoffs, project rules, recent work and global preferences."""
    store = memory._metadata
    recent_limit = max(1, min(int(recent_limit), 50))
    global_records = await asyncio.to_thread(
        store.get_all, limit=10, user_id=user_id, scope="global", archived=False
    )
    context: Dict[str, Any] = {
        "project": project,
        "global_preferences": [_memory_item(r) for r in global_records],
    }
    profile = await asyncio.to_thread(store.get_user_profile, user_id=user_id)
    if profile and profile.get("profile"):
        context["user_profile"] = profile["profile"]
    if not project:
        context["projects_with_open_handoffs"] = sorted({
            h["project"] for h in await list_handoffs(memory, statuses=list(ACTIVE_STATUSES), limit=100)
        })
        context["hint"] = (
            "Pass project (your repository or folder name) to load that project's goal, handoffs and history."
        )
        return context

    # Read the stored goal directly: the goal compass would also embed it.
    goal = await asyncio.to_thread(store.get_project_goal, user_id=user_id, namespace="global", project=project)
    project_records = await asyncio.to_thread(
        store.get_all, limit=recent_limit + 30, project=project, user_id=user_id, archived=False
    )
    instructions = [r for r in project_records if "instruction" in str((r.metadata or {}).get("category", ""))]
    instruction_ids = {r.id for r in instructions}
    recent = [r for r in project_records if r.id not in instruction_ids][:recent_limit]
    handoffs = await list_handoffs(memory, project=project, statuses=list(ACTIVE_STATUSES), limit=5)
    context.update({
        "goal": {"goal_statement": goal.get("goal_statement"), "constraints": goal.get("constraints", [])}
        if goal else None,
        "active_handoffs": handoffs,
        "instructions": [_memory_item(r) for r in instructions[:10]],
        "recent_memories": [_memory_item(r) for r in recent],
        "agents": sorted({r.source_agent for r in project_records if r.source_agent not in ("", "unknown")}
                         | {h["from_agent"] for h in handoffs}),
    })
    if any(h["status"] == "open" for h in handoffs):
        context["hint"] = "An open handoff is waiting: call resume_handoff to claim it before starting."
    return context
