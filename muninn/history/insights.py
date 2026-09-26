"""Understand imported threads: what was decided, learned and left open.

For each imported conversation an LLM reads the (redacted) turns and returns
a summary, a status (completed / in_progress / abandoned / answered), topics,
and durable insights: decisions, preferences, conventions, facts, fixes and
open items. Each insight becomes a semantic memory dated at the turn it came
from and linked to the thread, so "why did we pick X?" finds the decision, not
just the chat. Recent unfinished threads can become handoffs.

Providers (opt-in; nothing is sent anywhere unless you run it):
- ``openrouter``: fast and strong. Every request sets ``provider.zdr = true``
  (and ``data_collection = deny``), so OpenRouter only routes it to endpoints
  that keep no data. Text is redacted before it is sent.
- ``ollama``: fully local, slower, weaker on long threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from muninn.core import handoffs
from muninn.core.types import MemoryType, Provenance
from muninn.history.importer import AGENT_NAMES, Thread, _stamp, _write, collect, redact
from muninn.history.vault import HistoryVault

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")

INSIGHT_KINDS = ("decision", "preference", "convention", "fact", "fix", "open_item")
STATUSES = ("completed", "in_progress", "abandoned", "answered")
WINDOW_CHARS = 60_000        # transcript text per LLM call
TURN_CHARS = 2_500           # per side of a turn inside the prompt
HANDOFF_MAX_AGE_DAYS = 14    # unfinished threads newer than this can become handoffs
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """You read a conversation between a user and an AI coding/chat assistant and extract \
what is worth remembering for future sessions. Reply with one JSON object:
{"summary": "3-6 sentences: goal, what was done, where it ended",
 "status": "completed | in_progress | abandoned | answered",
 "topics": ["2-6 short lowercase topic tags"],
 "insights": [{"kind": "decision | preference | convention | fact | fix | open_item",
               "text": "one self-contained sentence, understandable without the conversation",
               "turn": <index of the turn it comes from>,
               "scope": "project | global"}]}
Rules: only durable knowledge (choices and their reasons, user preferences, project conventions, facts \
about systems, bugs and their fixes, unfinished work); at most 15 insights; no secrets, tokens, \
passwords or personal data; "global" only for preferences that apply beyond this project; \
open_item only for work the conversation left unfinished."""


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls, name: Optional[str] = None, model: Optional[str] = None) -> "Provider":
        name = (name or os.environ.get("MUNINN_INSIGHTS_PROVIDER") or "").strip().lower()
        key = os.environ.get("MUNINN_OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not name:
            name = "openrouter" if key else "ollama"
        model = model or os.environ.get("MUNINN_INSIGHTS_MODEL")
        if name == "openrouter":
            if not key:
                raise ValueError("Set OPENROUTER_API_KEY (or MUNINN_OPENROUTER_API_KEY) to use OpenRouter")
            return cls("openrouter", "https://openrouter.ai/api/v1", model or DEFAULT_OPENROUTER_MODEL, key)
        if name == "ollama":
            base = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434").rstrip("/")
            return cls("ollama", f"{base}/v1", model or os.environ.get("MUNINN_OLLAMA_MODEL", "llama3.2:3b"))
        raise ValueError("provider must be 'openrouter' or 'ollama'")

    def request_body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.1,
                                "response_format": {"type": "json_object"}}
        if self.name == "openrouter":
            # Only endpoints that retain nothing; never allowed to train on it.
            body["provider"] = {"zdr": True, "data_collection": "deny"}
        return body

    async def complete(self, client: httpx.AsyncClient, messages: List[Dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json", "X-Title": "Muninn"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = await client.post(f"{self.base_url}/chat/completions", json=self.request_body(messages),
                                     headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(f"{self.name} {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"] or ""


def render_turns(thread: Thread, start: int = 0) -> List[str]:
    """Conversation text windows for the model, redacted, with turn indexes."""
    blocks = []
    for index, turn in enumerate(thread.session.turns[start:], start=start):
        parts = [f"### Turn {index} ({_stamp(turn.at)})"]
        if turn.user:
            parts.append("User: " + turn.user[:TURN_CHARS])
        if turn.assistant:
            reply = turn.assistant
            parts.append("Assistant: " + (reply if len(reply) <= TURN_CHARS else reply[:1000] + " [...] "
                                          + reply[-(TURN_CHARS - 1000):]))
        if turn.actions:
            parts.append("Actions: " + "; ".join(turn.actions[:12]))
        blocks.append(redact("\n".join(parts)))
    for compaction in thread.session.compactions:
        blocks.append(redact(f"### Compaction summary ({_stamp(compaction.at)})\n{compaction.text[:TURN_CHARS * 2]}"))
    windows, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) > WINDOW_CHARS:
            windows.append(current)
            current = ""
        current += block + "\n\n"
    if current:
        windows.append(current)
    return windows


def parse_reply(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1]) if start >= 0 else {}
    insights = []
    for item in data.get("insights") or []:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        kind = str(item.get("kind", "fact")).strip().lower()
        insights.append({
            "kind": kind if kind in INSIGHT_KINDS else "fact",
            "text": redact(" ".join(str(item["text"]).split()))[:600],
            "turn": item.get("turn") if isinstance(item.get("turn"), int) else None,
            "scope": "global" if item.get("scope") == "global" and kind == "preference" else "project",
        })
    status = str(data.get("status", "")).strip().lower()
    return {
        "summary": redact(" ".join(str(data.get("summary", "")).split()))[:1500],
        "status": status if status in STATUSES else "completed",
        "topics": [str(t).strip().lower()[:40] for t in data.get("topics") or [] if str(t).strip()][:6],
        "insights": insights[:15],
    }


async def understand(provider: Provider, client: httpx.AsyncClient, thread: Thread) -> Dict[str, Any]:
    session = thread.session
    header = (f"Agent: {AGENT_NAMES.get(session.agent, session.agent)}. Project: {thread.project_name}. "
              f"Title: {session.title or 'untitled'}. {len(session.turns)} turns.")
    results = []
    windows = render_turns(thread)
    for number, window in enumerate(windows, start=1):
        part = f" This is part {number} of {len(windows)} of the conversation." if len(windows) > 1 else ""
        reply = await provider.complete(client, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{header}{part}\n\n{window}"},
        ])
        results.append(parse_reply(reply))
    merged = {
        "summary": " ".join(r["summary"] for r in results if r["summary"]),
        "status": results[-1]["status"] if results else "completed",
        "topics": list(dict.fromkeys(t for r in results for t in r["topics"]))[:6],
        "insights": [i for r in results for i in r["insights"]],
    }
    return merged


def _turn_time(thread: Thread, turn: Optional[int]) -> float:
    turns = thread.session.turns
    if turn is not None and 0 <= turn < len(turns) and turns[turn].at:
        return turns[turn].at
    return thread.session.ended_at or time.time()


async def store_understanding(memory: "MuninnMemory", thread: Thread, result: Dict[str, Any],
                              create_handoffs: bool = True) -> Dict[str, int]:
    session = thread.session
    store = memory._metadata
    agent = AGENT_NAMES.get(session.agent, session.agent)
    written = 0
    for insight in result["insights"]:
        at = _turn_time(thread, insight["turn"])
        label = insight["kind"].replace("_", " ").capitalize()
        content = (f"{label}: {insight['text']}\n(from {agent} thread \"{session.title or 'untitled'}\" "
                   f"in {thread.project_name}, {_stamp(at)})")
        metadata = {
            "import_source": "thread_insight", "kind": "thread_insight", "insight_kind": insight["kind"],
            "thread_id": thread.key, "turn_index": insight["turn"], "project": thread.project_name,
            "agent": session.agent, "topics": result["topics"], "operator_model_profile": "low_latency",
        }
        added = await memory.add(content=content, user_id="global_user", agent_id=session.agent, metadata=metadata,
                                 memory_type=MemoryType.SEMANTIC, provenance=Provenance.AUTO_EXTRACTED,
                                 scope=insight["scope"])
        if added.get("id"):
            await _write(memory, store.update, added["id"], created_at=float(at))
            written += 1
    state = await asyncio.to_thread(store.get_history_thread, thread.key)
    summary_id = state.get("summary_memory_id") if state else None
    if summary_id and result["summary"]:
        text = (f"Thread summary · {agent} · {thread.project_name}"
                f"{f' (branch {session.branch})' if session.branch else ''} · \"{session.title or 'untitled'}\"\n"
                f"{_stamp(session.started_at)} to {_stamp(session.ended_at)} · status: {result['status']}"
                f" · topics: {', '.join(result['topics'])}\n{result['summary']}")
        await memory.update(summary_id, data=text, metadata_patch={"status": result["status"],
                                                                   "topics": result["topics"]})
    await _write(memory, store.set_history_analysis, thread.key, status=result["status"],
                 topics=result["topics"], analyzed_turns=len(session.turns))
    handoff = 0
    open_items = [i["text"] for i in result["insights"] if i["kind"] == "open_item"]
    recent = (session.ended_at or 0) > time.time() - HANDOFF_MAX_AGE_DAYS * 86400
    if create_handoffs and result["status"] == "in_progress" and open_items and recent and thread.project:
        existing = await handoffs.list_handoffs(memory, project=thread.project, statuses=["open", "claimed"])
        if not any((h.get("details") or {}).get("thread_id") == thread.key for h in existing):
            await handoffs.create_handoff(
                memory, project=thread.project, from_agent=session.agent,
                title=f"Unfinished: {session.title or 'conversation'}"[:80],
                summary=result["summary"] or f"Unfinished {agent} conversation",
                details={"next_steps": open_items, "branch": session.branch or "", "thread_id": thread.key},
            )
            handoff = 1
    return {"insights": written, "handoffs": handoff}


async def analyze_threads(
    memory: "MuninnMemory",
    vault: HistoryVault,
    *,
    apply: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50,
    concurrency: int = 4,
    create_handoffs: bool = True,
    progress: Optional[Dict[str, Any]] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> Dict[str, Any]:
    """Analyze imported threads that grew since their last analysis (dry run reports the volume)."""
    store = memory._metadata
    pending = await asyncio.to_thread(
        lambda: store.list_history_threads(project, limit, needs_analysis=True))
    sources = [t["source_path"] for t in pending if t.get("source_path")]
    collected = await asyncio.to_thread(collect, vault, None, None, sources) if sources else None
    by_key = {t.key: t for t in (collected.threads if collected else [])}
    threads = [by_key[t["thread_key"]] for t in pending if t["thread_key"] in by_key]
    chars = sum(len(w) for t in threads for w in render_turns(t))
    report: Dict[str, Any] = {
        "apply": apply, "threads": len(threads), "approx_input_tokens": chars // 4,
        "insights": 0, "handoffs": 0, "errors": [],
    }
    if not apply or not threads:
        return report
    chosen = Provider.from_env(provider, model)
    report.update({"provider": chosen.name, "model": chosen.model, "zero_data_retention": chosen.name == "openrouter"})
    progress = progress if progress is not None else {}
    progress.update({"threads_total": len(threads), "threads_done": 0})
    semaphore = asyncio.Semaphore(max(1, concurrency if chosen.name == "openrouter" else 1))

    async with httpx.AsyncClient(timeout=180.0, transport=transport) as client:
        async def one(thread: Thread) -> None:
            async with semaphore:
                try:
                    result = await understand(chosen, client, thread)
                    counts = await store_understanding(memory, thread, result, create_handoffs)
                    report["insights"] += counts["insights"]
                    report["handoffs"] += counts["handoffs"]
                except Exception as exc:
                    report["errors"].append(f"{thread.key}: {exc}")
                progress["threads_done"] = progress.get("threads_done", 0) + 1

        await asyncio.gather(*(one(t) for t in threads))
    return report

