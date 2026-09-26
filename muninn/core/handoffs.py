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


async def _write(memory: "MuninnMemory", fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Serialize store writes with the engine's write lock (they share one SQLite connection)."""
    lock = getattr(memory, "_write_lock", None)
    if lock is None:  # no engine sharing the connection: write on this thread
        return fn(*args, **kwargs)
    async with lock:
        return await asyncio.to_thread(fn, *args, **kwargs)


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
    for key in ("branch", "thread_id"):  # thread_id: the imported conversation this came from
        if isinstance(details.get(key), str) and details[key].strip():
            clean[key] = details[key].strip()
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
    return await _write(memory, memory._metadata.add_handoff, record)


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
        claimed = await _write(
            memory, store.transition_handoff, handoff["id"], "claimed", agent=agent, allowed_from=ACTIVE_STATUSES
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
    return await _write(memory, memory._metadata.transition_handoff, handoff_id, status, agent=agent, note=note)


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
        store.get_all, limit=recent_limit + 200, project=project, user_id=user_id, archived=False
    )
    instructions = [r for r in project_records if "instruction" in str((r.metadata or {}).get("category", ""))]
    instruction_ids = {r.id for r in instructions}
    # Imported conversation turns are represented by recent_threads; list saved knowledge here.
    recent = [
        r for r in project_records
        if r.id not in instruction_ids and (r.metadata or {}).get("import_source") != "agent_history"
    ][:recent_limit]
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
    threads = await asyncio.to_thread(store.list_history_threads, project, 5)
    if threads:
        # Earlier conversations about this project in any app; read one with get_thread.
        context["recent_threads"] = [
            {"thread_id": t["thread_key"], "agent": t["agent"], "title": t["title"], "branch": t["branch"],
             "started_at": t["started_at"], "ended_at": t["ended_at"], "turns": t["turns_imported"]}
            for t in threads
        ]
    if any(h["status"] == "open" for h in handoffs):
        context["hint"] = "An open handoff is waiting: call resume_handoff to claim it before starting."
    return context


def _ago(at: Optional[float], now: Optional[float] = None) -> str:
    if not at:
        return "unknown time"
    minutes = max(0, int(((now or time.time()) - at) / 60))
    if minutes < 90:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60} h ago"
    return time.strftime("%Y-%m-%d", time.gmtime(at))


def render_briefing(context: Dict[str, Any], max_chars: int = 6000) -> str:
    """The project briefing as text an agent reads at session start (injected by hooks)."""
    project = context.get("project")
    lines: List[str] = []
    if project:
        lines.append(f'Muninn shared memory for project "{project}", shared with your other AI agents. '
                     "Use get_project_context, search_memory, get_thread and the handoff tools for more; "
                     f'pass project="{project}".')
    else:
        lines.append("Muninn shared memory: no project detected for this directory; pass project to Muninn tools.")
        waiting = context.get("projects_with_open_handoffs") or []
        if waiting:
            lines.append("Projects with open handoffs: " + ", ".join(waiting))
    goal = context.get("goal")
    if goal and goal.get("goal_statement"):
        lines.append(f"Goal: {goal['goal_statement']}")
        for constraint in goal.get("constraints") or []:
            lines.append(f"  constraint: {constraint}")
    for h in context.get("active_handoffs") or []:
        steps = "; ".join((h.get("details") or {}).get("next_steps", [])[:4])
        state = "OPEN" if h["status"] == "open" else f"claimed by {h.get('claimed_by')}"
        lines.append(f"Handoff {h['id']} ({state}) from {h['from_agent']}, {_ago(h['created_at'])}: {h['title']}"
                     + (f" Next: {steps}" if steps else "") + " -> resume_handoff to pick it up.")
    threads = context.get("recent_threads") or []
    if threads:
        lines.append("Earlier conversations (read with get_thread):")
        for t in threads:
            lines.append(f"  - {t['agent']} · {t.get('title') or 'untitled'} · {_ago(t.get('ended_at'))} · "
                         f"{t.get('turns', 0)} turns · thread_id {t['thread_id']}")
    rules = context.get("instructions") or []
    if rules:
        lines.append("Project rules:")
        lines.extend(f"  - {' '.join(r['memory'].split())[:300]}" for r in rules)
    recent = context.get("recent_memories") or []
    if recent:
        lines.append("Recent memories:")
        lines.extend(f"  - [{_ago(m.get('created_at'))}, {m.get('agent')}] {' '.join(m['memory'].split())[:200]}"
                     for m in recent[:8])
    prefs = context.get("global_preferences") or []
    if prefs:
        lines.append("User preferences:")
        lines.extend(f"  - {' '.join(p['memory'].split())[:200]}" for p in prefs[:6])
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
