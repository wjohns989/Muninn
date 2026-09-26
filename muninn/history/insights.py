"""Understand imported threads: what was decided, learned and left open.

For each imported conversation a model reads the (redacted) turns and returns
a summary, a status (completed / in_progress / abandoned / answered), topics,
and durable insights: decisions, preferences, conventions, facts, fixes and
open items. Each insight becomes a semantic memory dated at the turn it came
from and linked to the thread. Re-analysing a thread that grew replaces its
earlier insights. Recent unfinished threads can become handoffs.

Getting results stored correctly:
- the request carries a strict JSON Schema (``response_format.json_schema``) and,
  on OpenRouter, ``provider.require_parameters`` so only endpoints that enforce it
  are used;
- every reply is validated against the same schema before anything is stored;
  an invalid reply is sent back once with the error, then parsed leniently;
- whole conversations go in one call (1M-token models); only giant threads
  are split, and their parts are merged by a final call into one result.

Providers (opt-in; nothing is sent anywhere unless you run it):
- ``openrouter``: every request sets ``provider.zdr = true`` and
  ``data_collection = deny`` (endpoints that keep nothing and cannot train on it).
- ``ollama``: fully local, smaller context, slower.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from muninn.core import handoffs
from muninn.core.types import MemoryType, Provenance
from muninn.history import llm_settings
from muninn.history.importer import AGENT_NAMES, Thread, _stamp, _write, collect, redact
from muninn.history.vault import HistoryVault

if TYPE_CHECKING:
    from muninn.core.memory import MuninnMemory

logger = logging.getLogger("Muninn.history")

INSIGHT_KINDS = ("decision", "preference", "convention", "fact", "fix", "open_item")
STATUSES = ("completed", "in_progress", "abandoned", "answered")
CHARS_PER_TOKEN = 3.5          # conservative for code-heavy chats
TURN_CHARS = 40_000            # per side of one turn: keeps a pasted log from swamping a window
HANDOFF_MAX_AGE_DAYS = 14      # unfinished threads newer than this can become handoffs
MAX_INSIGHTS = 20

SYSTEM_PROMPT = """You read a conversation between a user and an AI coding/chat assistant and extract \
what is worth remembering for future sessions.
Return: a 3-6 sentence summary (goal, what was done, where it ended); the status of the work \
(completed, in_progress, abandoned, or answered for a question that got its answer); 2-6 short \
lowercase topic tags; and up to 20 insights. Each insight is one self-contained sentence someone \
could understand without the conversation, with the turn number it comes from.
Insight kinds: decision (a choice and its reason), preference (how the user likes things done), \
convention (a project rule or pattern), fact (about systems, APIs, environments), fix (a bug and \
what fixed it), open_item (work left unfinished).
Rules: only durable knowledge, not chit-chat or step-by-step narration; prefer the final state when \
something changed during the conversation; never include secrets, tokens, passwords or personal \
data; scope "global" only for preferences that apply beyond this project, otherwise "project".
Other agents may have worked on the same project in between (marked "Meanwhile" in the transcript), \
and notes already recorded from other conversations are listed with their time and id. Use them to \
read this conversation in order: set "current" to false for an insight that a later note or later \
work replaced, and list in "supersedes" the ids of earlier notes that this conversation replaced. \
Do not repeat what an existing note already says."""

MERGE_PROMPT = """These are analyses of consecutive parts of ONE long conversation. Merge them into a \
single analysis with the same fields: one summary of the whole conversation, the status at its end, \
up to 6 topics, and up to 20 insights with duplicates removed and anything superseded by a later \
part dropped. Keep each insight's turn number."""


# --- the shape the model must return -----------------------------------------------

class Insight(BaseModel):
    model_config = ConfigDict(extra="ignore")
    kind: Literal["decision", "preference", "convention", "fact", "fix", "open_item"]
    text: str = Field(min_length=3)
    turn: Optional[int] = None
    scope: Literal["project", "global"] = "project"
    current: bool = True

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        return redact(" ".join(value.split()))[:600]


class Understanding(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    status: Literal["completed", "in_progress", "abandoned", "answered"] = "completed"
    topics: List[str] = Field(default_factory=list)
    insights: List[Insight] = Field(default_factory=list)
    supersedes: List[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _clean_summary(cls, value: str) -> str:
        return redact(" ".join(value.split()))[:2000]

    @field_validator("topics")
    @classmethod
    def _clean_topics(cls, value: List[str]) -> List[str]:
        return list(dict.fromkeys(t.strip().lower()[:40] for t in value if t.strip()))[:6]


def _schema() -> Dict[str, Any]:
    """Strict JSON Schema (every property required, no extras), as OpenRouter/OpenAI expect."""
    insight = {
        "type": "object", "additionalProperties": False,
        "required": ["kind", "text", "turn", "scope", "current"],
        "properties": {
            "kind": {"type": "string", "enum": list(INSIGHT_KINDS)},
            "text": {"type": "string"},
            "turn": {"type": ["integer", "null"]},
            "scope": {"type": "string", "enum": ["project", "global"]},
            "current": {"type": "boolean"},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["summary", "status", "topics", "insights", "supersedes"],
        "properties": {
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": list(STATUSES)},
            "topics": {"type": "array", "items": {"type": "string"}},
            "insights": {"type": "array", "items": insight},
            "supersedes": {"type": "array", "items": {"type": "string"}},
        },
    }


RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "thread_understanding", "strict": True, "schema": _schema()}}


def _json_text(text: str) -> str:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 else text


def validate_reply(
    text: str, turn_count: Optional[int] = None, notes: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Parse and validate a model reply; raises ValueError/ValidationError when it does not fit."""
    parsed = Understanding.model_validate(json.loads(_json_text(text)))
    return _normalize(parsed, turn_count, notes)


def parse_reply(text: str, turn_count: Optional[int] = None, notes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Last resort for replies that fail validation: keep whatever is usable."""
    try:
        data = json.loads(_json_text(text))
    except ValueError:
        data = {}
    data = data if isinstance(data, dict) else {}
    insights = []
    for item in data.get("insights") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "fact")).strip().lower()
        try:
            insights.append(Insight(kind=kind if kind in INSIGHT_KINDS else "fact", text=str(item.get("text", "")),
                                    turn=item.get("turn") if isinstance(item.get("turn"), int) else None,
                                    scope="global" if item.get("scope") == "global" else "project",
                                    current=item.get("current") is not False))
        except ValidationError:
            continue
    status = str(data.get("status", "")).strip().lower()
    understanding = Understanding(
        summary=str(data.get("summary", "")), status=status if status in STATUSES else "completed",
        topics=[str(t) for t in data.get("topics") or []], insights=insights,
        supersedes=[str(x) for x in data.get("supersedes") or [] if isinstance(x, (str, int))])
    return _normalize(understanding, turn_count, notes)


def _normalize(
    parsed: Understanding, turn_count: Optional[int], notes: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    insights, seen = [], set()
    for item in parsed.insights:
        key = (item.kind, item.text.lower())
        if key in seen:
            continue
        seen.add(key)
        turn = item.turn if item.turn is not None and (turn_count is None or 0 <= item.turn < turn_count) else None
        insights.append({
            "kind": item.kind, "text": item.text, "turn": turn,
            # Only preferences may be global; everything else belongs to the project.
            "scope": "global" if item.scope == "global" and item.kind == "preference" else "project",
            "current": item.current,
        })
    # Only note ids the model was shown can be superseded (never a made-up id).
    known = notes or {}
    supersedes = list(dict.fromkeys(known[a.strip()] for a in parsed.supersedes if a.strip() in known))
    return {"summary": parsed.summary, "status": parsed.status, "topics": parsed.topics,
            "insights": insights[:MAX_INSIGHTS], "supersedes": supersedes}


# --- providers -----------------------------------------------------------------------

@dataclass
class Provider:
    name: str
    base_url: str
    models: List[str]
    api_key: Optional[str] = None
    window_tokens: int = 200_000

    @property
    def model(self) -> str:
        return self.models[0]

    @property
    def window_chars(self) -> int:
        return int(self.window_tokens * CHARS_PER_TOKEN)

    @classmethod
    def from_env(cls, name: Optional[str] = None, model: Optional[str] = None) -> "Provider":
        name = (name or os.environ.get("MUNINN_INSIGHTS_PROVIDER") or "").strip().lower()
        key = llm_settings.api_key()
        if not name:
            name = "openrouter" if key else "ollama"
        if name == "openrouter":
            if not key:
                raise ValueError(
                    "No OpenRouter key: run `python -m muninn.cli openrouter set`, or set OPENROUTER_API_KEY")
            models = llm_settings.models()
            if model:
                models = [model] + [m for m in models if m != model]
            return cls("openrouter", llm_settings.OPENROUTER_API, models, key,
                       _int_env("MUNINN_INSIGHTS_WINDOW_TOKENS", 200_000))
        if name == "ollama":
            base = os.environ.get("MUNINN_OLLAMA_URL", "http://localhost:11434").rstrip("/")
            local = model or os.environ.get("MUNINN_OLLAMA_MODEL", "llama3.2:3b")
            return cls("ollama", f"{base}/v1", [local], None, _int_env("MUNINN_OLLAMA_WINDOW_TOKENS", 24_000))
        raise ValueError("provider must be 'openrouter' or 'ollama'")

    def request_body(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "response_format": RESPONSE_FORMAT}
        if self.name == "openrouter":
            # Only parameters every chosen model's ZDR endpoints support: temperature and
            # max_tokens are absent on GPT-6 Luna's, and require_parameters would route around it.
            body.update({
                "models": self.models,
                "provider": {"zdr": True, "data_collection": "deny", "require_parameters": True},
                "reasoning": {"effort": "low", "exclude": True},
                "usage": {"include": True},
            })
        else:
            body["temperature"] = 0.1
        return body

    async def complete(self, client: httpx.AsyncClient, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        headers = {"Content-Type": "application/json", "X-Title": "Muninn"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = await client.post(f"{self.base_url}/chat/completions", json=self.request_body(messages),
                                     headers=headers)
        if response.status_code >= 400:
            detail = response.text[:300]
            if self.name == "openrouter" and response.status_code in (400, 404) and "endpoint" in detail.lower():
                detail += (f" -- no zero-data-retention endpoint for {', '.join(self.models)} supports strict JSON "
                           f"right now; pick another model with `openrouter set --model` "
                           f"(account setting: {llm_settings.PRIVACY_PAGE})")
            raise RuntimeError(f"{self.name} {response.status_code}: {detail}")
        data = response.json()
        message = (data.get("choices") or [{}])[0].get("message") or {}
        usage = data.get("usage") or {}
        meta = {"model": data.get("model") or self.model, "provider": data.get("provider"),
                "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                "cost": float(usage.get("cost") or 0.0)}
        return message.get("content") or "", meta


def _int_env(name: str, default: int) -> int:
    try:
        return max(2_000, int(os.environ.get(name, default)))
    except ValueError:
        return default


# --- building the prompt -------------------------------------------------------------------

def _cap(text: str, limit: int = TURN_CHARS) -> str:
    return text if len(text) <= limit else text[: limit // 2] + "\n[...]\n" + text[-limit // 2:]


def render_turns(thread: Thread, window_chars: int = 700_000,
                 meanwhile: Optional[List[Tuple[float, str]]] = None) -> List[str]:
    """The conversation as model input, redacted and numbered, in as few windows as fit.

    ``meanwhile`` are (time, text) markers for other agents' work on the project, placed where
    they happened between this conversation's turns.
    """
    blocks = [(at, f"--- Meanwhile: {text} ---") for at, text in meanwhile or []]
    for index, turn in enumerate(thread.session.turns):
        parts = [f"### Turn {index} ({_stamp(turn.at)})"]
        if turn.user:
            parts.append("User: " + _cap(turn.user))
        if turn.assistant:
            parts.append("Assistant: " + _cap(turn.assistant))
        if turn.actions:
            parts.append("Actions: " + "; ".join(turn.actions[:20]))
        blocks.append((turn.at or 0, redact("\n".join(parts))))
    for compaction in thread.session.compactions:
        blocks.append((compaction.at or 0, redact(f"### Compaction summary ({_stamp(compaction.at)})\n"
                                                  f"{_cap(compaction.text)}")))
    blocks.sort(key=lambda pair: pair[0])
    windows, current = [], ""
    for _, block in blocks:
        if current and len(current) + len(block) > window_chars:
            windows.append(current)
            current = ""
        current += block + "\n\n"
    if current:
        windows.append(current)
    return windows


@dataclass
class CallStats:
    calls: int = 0
    retries: int = 0
    lenient: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    models: Dict[str, int] = field(default_factory=dict)

    def merge(self, other: "CallStats") -> None:
        for name in ("calls", "retries", "lenient", "prompt_tokens", "completion_tokens", "cost"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for model, count in other.models.items():
            self.models[model] = self.models.get(model, 0) + count

    def add(self, meta: Dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += int(meta.get("prompt_tokens") or 0)
        self.completion_tokens += int(meta.get("completion_tokens") or 0)
        self.cost += float(meta.get("cost") or 0.0)
        self.models[meta.get("model") or "?"] = self.models.get(meta.get("model") or "?", 0) + 1


async def _structured_call(provider: Provider, client: httpx.AsyncClient, messages: List[Dict[str, str]],
                           turn_count: int, stats: CallStats, notes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Ask, validate, re-ask once with the validation error, then fall back to lenient parsing."""
    content, meta = await provider.complete(client, messages)
    stats.add(meta)
    try:
        return validate_reply(content, turn_count, notes)
    except (ValueError, ValidationError) as exc:
        stats.retries += 1
        retry = messages + [
            {"role": "assistant", "content": content[:8000]},
            {"role": "user", "content": f"That reply does not match the required JSON schema: {str(exc)[:500]}. "
                                        "Reply with only the corrected JSON object."},
        ]
        content, meta = await provider.complete(client, retry)
        stats.add(meta)
        try:
            return validate_reply(content, turn_count, notes)
        except (ValueError, ValidationError):
            stats.lenient += 1
            return parse_reply(content, turn_count, notes)


@dataclass
class ProjectContext:
    """What happened elsewhere in the project, for reading one thread in order."""
    meanwhile: List[Tuple[float, str]] = field(default_factory=list)
    notes: Dict[str, str] = field(default_factory=dict)      # alias -> memory id
    notes_text: str = ""


async def project_context_for(memory: "MuninnMemory", thread: Thread, max_notes: int = 40) -> ProjectContext:
    store = memory._metadata
    session = thread.session
    context = ProjectContext()
    if not thread.project:
        return context
    others = [t for t in await asyncio.to_thread(lambda: store.list_history_threads(thread.project, 500))
              if t["thread_key"] != thread.key and t.get("started_at") and t.get("ended_at")]
    stamps = [t.at for t in session.turns if t.at]
    for before, after in zip(stamps, stamps[1:]):
        for other in others:
            if other["started_at"] < after and other["ended_at"] > before:
                label = (f"{_stamp(max(other['started_at'], before))} to {_stamp(min(other['ended_at'], after))}, "
                         f"{AGENT_NAMES.get(other['agent'], other['agent'])} worked on this project in thread "
                         f"\"{other.get('title') or 'untitled'}\" ({other['thread_key']})")
                marker = (max(other["started_at"], before), label)
                if marker not in context.meanwhile:
                    context.meanwhile.append(marker)
    records = await asyncio.to_thread(
        lambda: store.get_all(limit=2000, project=thread.project, archived=False))
    notes = [r for r in records if (r.metadata or {}).get("kind") == "thread_insight"
             and (r.metadata or {}).get("thread_id") != thread.key]
    middle = ((session.started_at or 0) + (session.ended_at or 0)) / 2
    notes = sorted(sorted(notes, key=lambda r: abs(r.created_at - middle))[:max_notes], key=lambda r: r.created_at)
    lines = []
    for number, record in enumerate(notes, start=1):
        alias = f"N{number}"
        context.notes[alias] = record.id
        agent = AGENT_NAMES.get((record.metadata or {}).get("agent"), (record.metadata or {}).get("agent"))
        lines.append(f"[{alias}] {_stamp(record.created_at)} · {agent}: {record.content.splitlines()[0][:400]}")
    if lines:
        context.notes_text = "Notes already recorded from other conversations in this project:\n" + "\n".join(lines)
    return context


async def understand(provider: Provider, client: httpx.AsyncClient, thread: Thread,
                     stats: Optional[CallStats] = None, context: Optional[ProjectContext] = None) -> Dict[str, Any]:
    stats = stats or CallStats()
    context = context or ProjectContext()
    session = thread.session
    turn_count = len(session.turns)
    header = (f"Agent: {AGENT_NAMES.get(session.agent, session.agent)}. Project: {thread.project_name}. "
              f"Title: {session.title or 'untitled'}. {turn_count} turns.")
    if context.notes_text:
        header += "\n\n" + context.notes_text
    windows = render_turns(thread, provider.window_chars, context.meanwhile)
    parts = []
    for number, window in enumerate(windows, start=1):
        note = f" This is part {number} of {len(windows)} of the conversation." if len(windows) > 1 else ""
        parts.append(await _structured_call(provider, client, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{header}{note}\n\n{window}"},
        ], turn_count, stats, context.notes))
    if len(parts) == 1:
        return parts[0]
    # A thread too big for one call: merge the parts into one coherent result.
    merged = await _structured_call(provider, client, [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + MERGE_PROMPT},
        {"role": "user", "content": f"{header}\n\n" + json.dumps(
            {"parts": [dict(p, supersedes=[a for a, i in context.notes.items() if i in p["supersedes"]])
                       for p in parts]}, ensure_ascii=False)},
    ], turn_count, stats, context.notes)
    return merged


# --- storing -------------------------------------------------------------------------

def _turn_time(thread: Thread, turn: Optional[int]) -> float:
    turns = thread.session.turns
    if turn is not None and 0 <= turn < len(turns) and turns[turn].at:
        return turns[turn].at
    return thread.session.ended_at or time.time()


async def _drop_previous_insights(memory: "MuninnMemory", thread_key: str) -> int:
    """A re-analysis replaces the thread's earlier insights instead of piling up near-duplicates."""
    records = await asyncio.to_thread(memory._metadata.get_thread_memories, thread_key, 0, 10_000)
    old = [r.id for r in records if (r.metadata or {}).get("kind") == "thread_insight"]
    for memory_id in old:
        await memory.delete(memory_id)
    return len(old)


async def store_understanding(memory: "MuninnMemory", thread: Thread, result: Dict[str, Any],
                              create_handoffs: bool = True, model: Optional[str] = None) -> Dict[str, int]:
    session = thread.session
    store = memory._metadata
    agent = AGENT_NAMES.get(session.agent, session.agent)
    replaced = await _drop_previous_insights(memory, thread.key)
    written = 0
    for insight in result["insights"]:
        at = _turn_time(thread, insight["turn"])
        label = insight["kind"].replace("_", " ").capitalize()
        content = (f"{label}: {insight['text']}\n(from {agent} thread \"{session.title or 'untitled'}\" "
                   f"in {thread.project_name}, {_stamp(at)})")
        metadata = {
            "import_source": "thread_insight", "kind": "thread_insight", "insight_kind": insight["kind"],
            "thread_id": thread.key, "turn_index": insight["turn"], "part": 0, "project": thread.project_name,
            "agent": session.agent, "topics": result["topics"], "operator_model_profile": "low_latency",
            **({"insight_model": model} if model else {}),
        }
        if not insight.get("current", True):
            metadata["superseded"] = True   # replaced by later work: kept for the record, out of search
        added = await memory.add(content=content, user_id="global_user", agent_id=session.agent, metadata=metadata,
                                 memory_type=MemoryType.SEMANTIC, provenance=Provenance.AUTO_EXTRACTED,
                                 scope=insight["scope"])
        if added.get("id"):
            await _write(memory, store.update, added["id"], created_at=float(at))
            if not insight.get("current", True):
                await _write(memory, store.update, added["id"], archived=True)
            written += 1
    superseded = 0
    for memory_id in result.get("supersedes", []):
        # An earlier note from another agent that this conversation replaced: archived, restorable.
        await memory.update(memory_id, archived=True, metadata_patch={
            "superseded": True, "superseded_by_thread": thread.key, "superseded_at": time.time()})
        superseded += 1
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
    return {"insights": written, "replaced": replaced, "superseded": superseded, "handoffs": handoff}


async def estimate_cost(models: List[str], input_tokens: int, output_tokens: int,
                        transport: Optional[httpx.AsyncBaseTransport] = None) -> Optional[float]:
    """Price of the primary model's cheapest zero-data-retention endpoint (public list, no key needed)."""
    try:
        async with httpx.AsyncClient(timeout=10.0, transport=transport) as client:
            response = await client.get(f"{llm_settings.OPENROUTER_API}/endpoints/zdr")
        endpoints = [e for e in response.json().get("data", []) if e.get("model_id") == models[0]]
        prices = [float(e["pricing"]["prompt"]) * input_tokens + float(e["pricing"]["completion"]) * output_tokens
                  for e in endpoints if "structured_outputs" in (e.get("supported_parameters") or [])]
        return round(min(prices), 4) if prices else None
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None


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
    """Analyze imported threads that grew since their last analysis (dry run reports volume and cost)."""
    store = memory._metadata
    pending = await asyncio.to_thread(
        lambda: store.list_history_threads(project, limit, needs_analysis=True))
    sources = [t["source_path"] for t in pending if t.get("source_path")]
    collected = await asyncio.to_thread(collect, vault, None, None, sources) if sources else None
    by_key = {t.key: t for t in (collected.threads if collected else [])}
    threads = [by_key[t["thread_key"]] for t in pending if t["thread_key"] in by_key]
    try:
        chosen: Optional[Provider] = Provider.from_env(provider, model)
    except ValueError as exc:
        if apply:
            raise
        chosen, setup_note = None, str(exc)
    window = chosen.window_chars if chosen else 700_000
    chars = sum(len(w) for t in threads for w in render_turns(t, window))
    input_tokens = int(chars / CHARS_PER_TOKEN)
    report: Dict[str, Any] = {
        "apply": apply, "threads": len(threads), "approx_input_tokens": input_tokens,
        "insights": 0, "replaced_insights": 0, "handoffs": 0, "errors": [],
    }
    if chosen:
        report.update({"provider": chosen.name, "models": chosen.models,
                       "zero_data_retention": chosen.name == "openrouter"})
    else:
        report["setup"] = setup_note
    if not apply:
        if chosen and chosen.name == "openrouter" and threads:
            report["approx_cost_usd"] = await estimate_cost(chosen.models, input_tokens, 1_500 * len(threads),
                                                            transport)
        return report
    if not threads:
        return report
    progress = progress if progress is not None else {}
    progress.update({"threads_total": len(threads), "threads_done": 0})
    semaphore = asyncio.Semaphore(max(1, concurrency if chosen.name == "openrouter" else 1))
    stats = CallStats()
    report["superseded_insights"] = 0
    # Within a project, threads run oldest first so each sees what earlier ones recorded;
    # projects run in parallel.
    by_project: Dict[str, List[Thread]] = {}
    for thread in threads:
        by_project.setdefault(thread.project_name, []).append(thread)

    async with httpx.AsyncClient(timeout=300.0, transport=transport) as client:
        async def one(thread: Thread) -> None:
            local = CallStats()
            try:
                context = await project_context_for(memory, thread)
                result = await understand(chosen, client, thread, local, context)
                used = max(local.models, key=local.models.get) if local.models else None
                counts = await store_understanding(memory, thread, result, create_handoffs, model=used)
                report["insights"] += counts["insights"]
                report["replaced_insights"] += counts["replaced"]
                report["superseded_insights"] += counts["superseded"]
                report["handoffs"] += counts["handoffs"]
            except Exception as exc:
                report["errors"].append(f"{thread.key}: {exc}")
            stats.merge(local)
            progress["threads_done"] = progress.get("threads_done", 0) + 1

        async def project_run(group: List[Thread]) -> None:
            for thread in sorted(group, key=lambda t: t.session.started_at or 0):
                async with semaphore:
                    await one(thread)

        await asyncio.gather(*(project_run(group) for group in by_project.values()))
    report.update({"calls": stats.calls, "schema_retries": stats.retries, "lenient_parses": stats.lenient,
                   "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
                   "cost_usd": round(stats.cost, 4), "models_used": stats.models})
    return report
