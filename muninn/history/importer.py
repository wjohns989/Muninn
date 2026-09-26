"""Import local AI conversation history as memories, in order, under the right project.

Every turn of every thread (a user request, the assistant's reply, the files
it touched and the commands it ran) becomes one memory, split into ordered
parts when long, never truncated. Each memory keeps:

- ``created_at`` = when the turn happened, so memories interleave
  chronologically with everything else Muninn knows;
- ``project`` = the project the conversation ran in (git remote name, or the
  repository folder; worktrees resolve to their repository), the same name live
  agents use, so imported and current memories line up;
- ``thread_id``, ``turn_index`` and ``part``, so a whole thread can be re-read
  in order (``get_thread``), including the turns that compaction removed from
  the model's context;
- ``directory``, ``branch`` and ``agent`` (claude-code, claude-desktop, codex,
  gemini-cli, chatgpt, claude-ai).

Compaction summaries are kept as their own memories, and a thread summary
memory is refreshed whenever a thread grows. Re-running only adds what is new.
Secrets are redacted from memory text (the raw copy stays in the vault).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

from muninn.core.projects import name_from_remote, project_for_directory
from muninn.core.types import MemoryType, Provenance
from muninn.history import parsers
from muninn.history.vault import HistoryVault, read_bytes, read_text

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")

PART_CHARS = 6000          # memory text per part; long turns become several ordered parts
SUMMARY_REPLY_CHARS = 800
IMPORT_SOURCE = "agent_history"
AGENT_NAMES = {
    "claude-code": "Claude Code", "claude-desktop": "Claude Desktop", "codex": "Codex",
    "gemini-cli": "Gemini CLI", "chatgpt": "ChatGPT", "claude-ai": "Claude",
}


# --- redaction -----------------------------------------------------------------

def _redaction_patterns():
    import re

    from muninn.mimir import policy

    extra = [
        (r"(?i)(password|passwd|pwd|secret|token)\s*[:=]\s*['\"]?[^\s'\"]{8,}['\"]?", r"\1=[REDACTED]"),
        (r"(?i)(api[_\-]?key|apikey|access[_\-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9\-_.~+/]{16,}['\"]?", r"\1=[REDACTED]"),
        (r"(?i)(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/]+:[^@\s]+@", r"\1://[REDACTED]@"),
        (r"xox[baprs]-[A-Za-z0-9-]{10,}", "[REDACTED_SLACK_TOKEN]"),
        (r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_GOOGLE_KEY]"),
        (r"(?i)sk-ant-[A-Za-z0-9\-_]{20,}", "[REDACTED_API_KEY]"),
    ]
    compiled = [(re.compile(p, re.IGNORECASE | re.MULTILINE), r) for p, r in policy._BALANCED_RAW + extra]
    return compiled


_PATTERNS = None


def redact(text: str) -> str:
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = _redaction_patterns()
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# --- building memories -------------------------------------------------------------

def _stamp(at: Optional[float]) -> str:
    if not at:
        return "unknown time"
    return datetime.fromtimestamp(at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _split(text: str, size: int) -> List[str]:
    """Split into pieces of at most ``size`` characters, preferring paragraph boundaries."""
    if len(text) <= size:
        return [text]
    pieces: List[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= size:
            current = candidate
            continue
        if len(paragraph) <= size:
            pieces.append(current)
            current = paragraph
            continue
        # One paragraph longer than a part: fill the current part, then cut the rest.
        room = size - len(current) - 2 if current else size
        if room > size // 4:
            pieces.append(f"{current}\n\n{paragraph[:room]}" if current else paragraph[:room])
            paragraph = paragraph[room:]
        elif current:
            pieces.append(current)
        while len(paragraph) > size:
            pieces.append(paragraph[:size])
            paragraph = paragraph[size:]
        current = paragraph
    if current:
        pieces.append(current)
    return pieces


@dataclass
class Thread:
    session: parsers.Session
    project: Optional[str]
    source: str

    @property
    def key(self) -> str:
        return self.session.key

    @property
    def project_name(self) -> str:
        return self.project or "global"

    def header(self, turn_index: Optional[int] = None) -> str:
        s = self.session
        agent = AGENT_NAMES.get(s.agent, s.agent)
        where = self.project_name + (f" (branch {s.branch})" if s.branch else "")
        title = f' · thread "{s.title}"' if s.title else ""
        position = f" · turn {turn_index + 1}" if turn_index is not None else ""
        return f"{agent} · {where}{title}{position}"

    def base_metadata(self) -> Dict[str, Any]:
        s = self.session
        meta = {
            "import_source": IMPORT_SOURCE, "provider": s.provider, "agent": s.agent, "thread_id": s.key,
            "session_id": s.session_id, "thread_title": s.title, "project": self.project_name,
        }
        for key, value in (("directory", s.cwd), ("branch", s.branch), ("surface", s.surface)):
            if value:
                meta[key] = value
        return meta


def turn_memories(thread: Thread, index: int, turn: parsers.Turn) -> List[Dict[str, Any]]:
    """One turn as ordered memory parts: header, the user's words, the reply, what it did."""
    body = []
    if turn.user:
        body.append(f"User: {turn.user}")
    if turn.assistant:
        body.append(f"Assistant: {turn.assistant}")
    if turn.actions:
        body.append("Actions: " + "; ".join(dict.fromkeys(turn.actions)))
    pieces = _split(redact("\n\n".join(body)), PART_CHARS)
    header = f"[{_stamp(turn.at)}] {thread.header(index)}"
    memories = []
    for part, piece in enumerate(pieces, start=1):
        label = header if len(pieces) == 1 else f"{header} (part {part}/{len(pieces)})"
        meta = dict(thread.base_metadata(), kind="conversation_turn", turn_index=index, part=part, parts=len(pieces))
        if turn.files and part == 1:
            meta["files"] = list(dict.fromkeys(turn.files))[:25]
        memories.append({"content": f"{label}\n{piece}", "created_at": turn.at, "metadata": meta})
    return memories


def compaction_memories(thread: Thread, index: int, compaction: parsers.Compaction) -> List[Dict[str, Any]]:
    pieces = _split(redact(compaction.text), PART_CHARS)
    header = f"[{_stamp(compaction.at)}] Context compacted · {thread.header()}"
    return [
        {
            "content": f"{header}{'' if len(pieces) == 1 else f' (part {part}/{len(pieces)})'}\n{piece}",
            "created_at": compaction.at,
            "metadata": dict(thread.base_metadata(), kind="compaction_summary", compaction_index=index,
                             part=part, parts=len(pieces)),
        }
        for part, piece in enumerate(pieces, start=1)
    ]


def summary_memory(thread: Thread) -> Dict[str, Any]:
    s = thread.session
    turns = [t for t in s.turns if t.user]
    last_reply = next((t.assistant for t in reversed(s.turns) if t.assistant), "")
    lines = [
        f"Thread summary · {thread.header()}",
        f"{_stamp(s.started_at)} to {_stamp(s.ended_at)} · {len(turns)} requests"
        + (f" · {len(s.compactions)} compactions" if s.compactions else ""),
    ]
    if s.cwd:
        lines.append(f"Directory: {s.cwd}")
    if turns:
        lines.append("Requests: " + " | ".join(" ".join(t.user.split())[:140] for t in turns[:12]))
    if last_reply:
        lines.append("Last reply: " + " ".join(last_reply.split())[-SUMMARY_REPLY_CHARS:])
    files = s.files()
    if files:
        lines.append("Files: " + ", ".join(files[:20]))
    return {
        "content": redact("\n".join(lines)),
        "created_at": s.ended_at,
        "metadata": dict(thread.base_metadata(), kind="thread_summary", turn_count=len(turns),
                         started_at=s.started_at, ended_at=s.ended_at),
    }


# --- reading the vault -----------------------------------------------------------------

def _codex_titles(vault: HistoryVault) -> Dict[str, str]:
    import sqlite3

    titles: Dict[str, str] = {}
    for db in vault.files(provider="codex", kind="state_db"):
        try:
            conn = sqlite3.connect(f"file:{db.path}?mode=ro", uri=True)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(threads)")}
            if {"id", "title"} <= columns:
                titles.update({str(i): str(t) for i, t in conn.execute("SELECT id, title FROM threads") if t})
            conn.close()
        except sqlite3.Error:
            continue
    return titles


def _gemini_projects(known_directories: Iterable[str]) -> Dict[str, str]:
    """Gemini CLI names project folders by sha256 of the project root; match known directories."""
    return {hashlib.sha256(d.encode()).hexdigest(): d for d in known_directories if d}


@dataclass
class Collected:
    threads: List[Thread] = field(default_factory=list)
    prompts: List[parsers.PromptEntry] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def collect(
    vault: HistoryVault,
    providers: Optional[List[str]] = None,
    since: Optional[float] = None,
    sources: Optional[Iterable[str]] = None,
) -> Collected:
    """Parse the vault into threads; ``sources`` limits it to those original file paths."""
    wanted = {p.strip().lower() for p in providers or [] if p.strip()}
    only = {str(Path(s).resolve()) for s in sources} if sources else None

    def enabled(provider: str) -> bool:
        return not wanted or provider in wanted

    out = Collected()
    sessions: Dict[str, Tuple[parsers.Session, str]] = {}

    def keep(session: Optional[parsers.Session], source: str) -> None:
        if session is None or not session.session_id:
            return
        existing = sessions.get(session.key)
        # The same thread can exist twice (Codex archive move, kept versions): keep the fullest.
        if existing is None or len(session.turns) + len(session.compactions) > \
                len(existing[0].turns) + len(existing[0].compactions):
            sessions[session.key] = (session, source)

    desktop_titles = parsers.claude_desktop_titles(
        read_text(f.path) for f in vault.files(provider="claude_desktop"))
    codex_titles = _codex_titles(vault) if enabled("codex") else {}
    parsers_by_provider: Dict[str, Callable[[str, str], Optional[parsers.Session]]] = {
        "claude_code": lambda text, name: parsers.parse_claude_code(text, name, desktop_titles),
        "codex": lambda text, name: parsers.parse_codex(text, name, codex_titles),
        "gemini_cli": parsers.parse_gemini,
    }
    for provider, parse in parsers_by_provider.items():
        if not enabled(provider):
            continue
        for item in vault.files(provider=provider, kind="transcript"):
            if only is not None and item.source_path not in only:
                continue
            try:
                session = parse(read_text(item.path), Path(item.source_path).name)
                if session and provider == "gemini_cli" and not session.surface:
                    session.surface = Path(item.source_path).parent.parent.name  # tmp/<project hash>/chats/
                keep(session, item.source_path)
            except Exception as exc:  # one unreadable file must not stop the import
                out.errors.append(f"{provider}: {Path(item.source_path).name}: {exc}")
        for item in ([] if only is not None else vault.files(provider=provider, kind="prompt_history")):
            try:
                out.prompts.extend(parsers.parse_prompt_history(read_text(item.path), provider))
            except Exception as exc:
                out.errors.append(f"{provider}: prompt history: {exc}")
    if only is None and (enabled("chatgpt") or enabled("claude_ai") or enabled("export")):
        for item in vault.files(provider="export"):
            try:
                for payload in parsers.export_payloads(Path(item.source_path), read_bytes(item.path)):
                    for session in parsers.parse_export(payload):
                        if enabled(session.provider) or enabled("export"):
                            keep(session, item.source_path)
            except Exception as exc:
                out.errors.append(f"export: {Path(item.source_path).name}: {exc}")

    known_dirs = {s.cwd for s, _ in sessions.values() if s.cwd} | {p.cwd for p in out.prompts if p.cwd}
    gemini_roots = _gemini_projects(known_dirs)
    for session, source in sorted(sessions.values(), key=lambda pair: pair[0].started_at or 0):
        if since and (session.ended_at or 0) < since:
            continue
        if session.provider == "gemini_cli" and not session.cwd and session.surface in gemini_roots:
            session.cwd = gemini_roots[session.surface]
        if session.repo_url:
            project = name_from_remote(session.repo_url)
        else:
            project = session.project or project_for_directory(session.cwd)
        out.threads.append(Thread(session=session, project=project, source=source))
    if since:
        out.prompts = [p for p in out.prompts if p.at >= since]
    return out


# --- import ------------------------------------------------------------------------------

def _prompt_digest(entry: parsers.PromptEntry) -> str:
    return hashlib.sha256(f"{entry.provider}|{entry.at}|{entry.text}".encode()).hexdigest()


def recovered_prompts(collected: Collected) -> List[parsers.PromptEntry]:
    """Prompts whose transcript the app already deleted: their only remaining trace."""
    covered_ids = {t.session.session_id for t in collected.threads}
    spans: Dict[Tuple[str, Optional[str]], List[Tuple[float, float]]] = {}
    for thread in collected.threads:
        span = (thread.session.started_at or 0, (thread.session.ended_at or 0) + 60)
        spans.setdefault((thread.session.provider, thread.session.cwd), []).append(span)
    result = []
    for entry in collected.prompts:
        if entry.session_id and entry.session_id in covered_ids:
            continue
        if any(start - 60 <= entry.at <= end for start, end in spans.get((entry.provider, entry.cwd), [])):
            continue
        result.append(entry)
    return result


async def _write(memory: "MuninnMemory", fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Store writes share one SQLite connection with the engine; serialize them with its write lock."""
    lock = getattr(memory, "_write_lock", None)
    if lock is None:  # no engine sharing the connection: write on this thread
        return fn(*args, **kwargs)
    async with lock:
        return await asyncio.to_thread(fn, *args, **kwargs)


async def _add(memory: "MuninnMemory", item: Dict[str, Any], scope: str) -> Optional[str]:
    # Bulk history uses the fast rule-based entity pass, not a per-turn LLM call.
    metadata = dict(item["metadata"], operator_model_profile="low_latency", muninn_extraction_timeout_seconds=10)
    result = await memory.add(
        content=item["content"],
        user_id="global_user",
        agent_id=item["metadata"].get("agent"),
        metadata=metadata,
        memory_type=MemoryType.EPISODIC,
        provenance=Provenance.INGESTED,
        scope=scope,
    )
    memory_id = result.get("id")
    if memory_id and item.get("created_at"):
        await _write(memory, memory._metadata.update, memory_id, created_at=float(item["created_at"]))
    return memory_id


async def import_history(
    memory: "MuninnMemory",
    vault: HistoryVault,
    *,
    apply: bool = False,
    providers: Optional[List[str]] = None,
    since: Optional[float] = None,
    progress: Optional[Dict[str, Any]] = None,
    sources: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Dry run by default: report what would be imported. With ``apply`` write only what is new."""
    collected = await asyncio.to_thread(collect, vault, providers, since, sources)
    store = memory._metadata
    progress = progress if progress is not None else {}
    report: Dict[str, Any] = {
        "apply": apply, "threads": 0, "threads_new": 0, "threads_grown": 0, "turn_memories": 0,
        "compaction_memories": 0, "summaries": 0, "recovered_prompts": 0, "by_project": {}, "by_agent": {},
        "oldest": None, "newest": None, "errors": collected.errors,
    }
    progress.update({"threads_total": len(collected.threads), "threads_done": 0})
    for thread in collected.threads:
        session = thread.session
        state = await asyncio.to_thread(store.get_history_thread, thread.key)
        done_turns = state["turns_imported"] if state else 0
        done_compactions = state["compactions_imported"] if state else 0
        new_turns = list(enumerate(session.turns))[done_turns:]
        new_compactions = list(enumerate(session.compactions))[done_compactions:]
        report["threads"] += 1
        if state is None:
            report["threads_new"] += 1
        elif new_turns or new_compactions:
            report["threads_grown"] += 1
        items = [m for i, t in new_turns for m in turn_memories(thread, i, t)]
        compactions = [m for i, c in new_compactions for m in compaction_memories(thread, i, c)]
        report["turn_memories"] += len(items)
        report["compaction_memories"] += len(compactions)
        project_counts = report["by_project"].setdefault(thread.project_name, {"threads": 0, "memories": 0})
        project_counts["threads"] += 1
        project_counts["memories"] += len(items) + len(compactions)
        report["by_agent"][session.agent] = report["by_agent"].get(session.agent, 0) + 1
        for stamp in (session.started_at, session.ended_at):
            if stamp:
                report["oldest"] = min(report["oldest"] or stamp, stamp)
                report["newest"] = max(report["newest"] or stamp, stamp)
        needs_summary = bool(items or compactions or state is None)
        report["summaries"] += needs_summary
        if apply and needs_summary:
            # Always project-scoped: conversations outside any repository file under "global" and
            # show up in unfiltered searches, but do not leak into every project's results.
            scope = "project"
            semaphore = asyncio.Semaphore(4)

            async def add_one(item: Dict[str, Any]) -> None:
                async with semaphore:
                    await _add(memory, item, scope)

            await asyncio.gather(*(add_one(item) for item in items + compactions))
            summary = summary_memory(thread)
            summary_id = state["summary_memory_id"] if state else None
            if summary_id and await asyncio.to_thread(store.get, summary_id):
                await memory.update(summary_id, data=summary["content"], metadata_patch=summary["metadata"])
                await _write(memory, store.update, summary_id, created_at=float(summary["created_at"] or time.time()))
            else:
                summary_id = await _add(memory, summary, scope)
            await _write(memory, store.upsert_history_thread, {
                "thread_key": thread.key, "provider": session.provider, "agent": session.agent,
                "session_id": session.session_id, "project": thread.project_name, "directory": session.cwd,
                "branch": session.branch, "title": session.title, "started_at": session.started_at,
                "ended_at": session.ended_at, "turns_imported": len(session.turns),
                "compactions_imported": len(session.compactions), "summary_memory_id": summary_id,
                "updated_at": time.time(), "source_path": thread.source,
            })
        progress["threads_done"] = progress.get("threads_done", 0) + 1

    prompts = [p for p in recovered_prompts(collected)
               if not await asyncio.to_thread(store.history_prompt_seen, _prompt_digest(p))]
    report["recovered_prompts"] = len(prompts)
    if apply and prompts:
        for entry in prompts:
            project = project_for_directory(entry.cwd)
            agent = {"claude_code": "claude-code", "codex": "codex", "gemini_cli": "gemini-cli"}.get(
                entry.provider, entry.provider)
            item = {
                "content": f"[{_stamp(entry.at)}] Prompt to {AGENT_NAMES.get(agent, agent)} · {project or 'global'} "
                           f"(transcript no longer on disk)\nUser: {redact(entry.text)}",
                "created_at": entry.at,
                "metadata": {"import_source": IMPORT_SOURCE, "kind": "recovered_prompt", "provider": entry.provider,
                             "agent": agent, "project": project or "global",
                             **({"directory": entry.cwd} if entry.cwd else {}),
                             **({"session_id": entry.session_id} if entry.session_id else {})},
            }
            await _add(memory, item, "project")
        await _write(memory, store.mark_history_prompts, [_prompt_digest(p) for p in prompts])
    for key in ("oldest", "newest"):
        if report[key]:
            report[key] = _stamp(report[key])
    report["by_project"] = dict(sorted(report["by_project"].items(), key=lambda kv: -kv[1]["memories"]))
    return report


async def read_thread(memory: "MuninnMemory", thread_key: str, offset: int = 0, limit: int = 50) -> Dict[str, Any]:
    """A thread's turns and compaction summaries in conversation order."""
    store = memory._metadata
    state = await asyncio.to_thread(store.get_history_thread, thread_key)
    records = await asyncio.to_thread(store.get_thread_memories, thread_key, offset, limit)
    entries = [
        {"id": r.id, "kind": (r.metadata or {}).get("kind"), "turn_index": (r.metadata or {}).get("turn_index"),
         "part": (r.metadata or {}).get("part"), "created_at": r.created_at, "content": r.content}
        for r in records if (r.metadata or {}).get("kind") != "thread_summary"
    ]
    return {"thread": state, "offset": offset, "entries": entries,
            "next_offset": offset + len(records) if len(records) == limit else None}


def dumps(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=1, default=str)
