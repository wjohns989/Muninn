"""Turn local AI transcripts into sessions of (user request, assistant reply) turns.

Each parser is tolerant of older and newer layouts and drops what is not
conversation: tool calls and results, reasoning, and messages the host injects
(environment context, hook output, task notifications, continuation summaries).
Slash commands are kept as "/name args" because they are what the user asked.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_MAX_ACTIONS_PER_TURN = 40


@dataclass
class Turn:
    at: float
    user: str
    assistant: str = ""
    files: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)  # "edit path", "run: pytest -q", ...


@dataclass
class Compaction:
    """A summary the host wrote when it compacted the conversation (kept: it is what the model saw next)."""

    at: float
    text: str


@dataclass
class Session:
    provider: str
    agent: str
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    cwd: Optional[str] = None
    branch: Optional[str] = None
    repo_url: Optional[str] = None
    title: Optional[str] = None
    surface: Optional[str] = None
    project: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    source_name: str = ""
    compactions: List[Compaction] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.session_id}"

    def files(self) -> List[str]:
        seen: Dict[str, None] = {}
        for turn in self.turns:
            for path in turn.files:
                seen.setdefault(path, None)
        return list(seen)


@dataclass
class PromptEntry:
    provider: str
    at: float
    text: str
    cwd: Optional[str] = None
    session_id: Optional[str] = None


def parse_time(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 1e11 else number
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d+(\.\d+)?", text):
            return parse_time(float(text))
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


_COMPACTION_PREFIX = "this session is being continued from a previous conversation"
_INJECTED_PREFIXES = (
    "<task-notification", "<local-command-stdout", "<local-command-caveat", "<local-command-stderr",
    "caveat: the messages below", _COMPACTION_PREFIX,
    "stop hook feedback", "[request interrupted", "<system-reminder", "<user-prompt-submit-hook",
    "base directory for this skill", "<environment_context", "<user_instructions", "# agents.md",
    "<permissions", "<turn_aborted", "<user_shell_command", "<subagent_notification", "<skill",
    "<bash-input", "<bash-stdout", "<bash-stderr",
)
_COMMAND = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.S)
_COMMAND_ARGS = re.compile(r"<command-args>\s*(.*?)\s*</command-args>", re.S)


def clean_user_text(text: str) -> str:
    """The user's own words, or '' for host-injected messages."""
    text = (text or "").strip()
    if not text:
        return ""
    command = _COMMAND.search(text)
    if command:
        args = _COMMAND_ARGS.search(text)
        name = command.group(1).strip()
        name = name if name.startswith("/") else "/" + name
        return f"{name} {args.group(1).strip()}".strip() if args else name
    lowered = text.lower()
    if lowered.startswith(_INJECTED_PREFIXES) or text.startswith("<command-message>"):
        return ""
    return text


def _append_reply(turn: Turn, text: str) -> None:
    text = (text or "").strip()
    if text:
        turn.assistant = f"{turn.assistant}\n\n{text}".strip() if turn.assistant else text


def _note_action(turn: Turn, name: str, tool_input: Any) -> None:
    """Record what a tool call did (files touched, commands run) without its output."""
    paths = _paths_from_tool_input(tool_input)
    turn.files.extend(paths)
    if len(turn.actions) >= _MAX_ACTIONS_PER_TURN:
        return
    data = tool_input
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            data = {}
    command = data.get("command") if isinstance(data, dict) else None
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    if isinstance(command, str) and command.strip():
        turn.actions.append(f"ran: {' '.join(command.split())[:160]}")
    elif paths:
        turn.actions.append(f"{name.lower()}: {', '.join(paths[:3])}")
    elif name:
        turn.actions.append(name)


_PATH_KEYS = ("file_path", "path", "notebook_path", "filePath", "target_file")
_PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.M)


def _paths_from_tool_input(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return _PATCH_FILE.findall(value)
    if not isinstance(value, dict):
        return []
    found = [str(value[key]) for key in _PATH_KEYS if isinstance(value.get(key), str)]
    for key in ("input", "patch", "command"):
        if isinstance(value.get(key), str):
            found.extend(_PATCH_FILE.findall(value[key]))
    return found


def _lines(text: str) -> Iterable[Dict[str, Any]]:
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except ValueError:
            continue
        if isinstance(item, dict):
            yield item


def _finish(session: Session) -> Optional[Session]:
    session.turns = [t for t in session.turns if t.user or t.assistant]
    if not session.turns and not session.compactions:
        return None
    stamps = [t.at for t in session.turns if t.at] + [c.at for c in session.compactions if c.at]
    session.started_at = session.started_at or (min(stamps) if stamps else None)
    session.ended_at = max(stamps) if stamps else session.started_at
    if not session.title:
        first = next((t.user for t in session.turns if t.user), "")
        session.title = " ".join(first.split())[:80] or None
    return session


# --- Claude Code (CLI, IDE, and Claude Desktop's Code tab / Cowork) --------------

def parse_claude_code(
    text: str, source_name: str = "", desktop_titles: Optional[Dict[str, str]] = None
) -> Optional[Session]:
    session: Optional[Session] = None
    current: Optional[Turn] = None
    for line in _lines(text):
        kind = line.get("type")
        if session is None and line.get("sessionId"):
            session = Session(provider="claude_code", agent="claude-code", session_id=str(line["sessionId"]),
                              source_name=source_name)
        if session is None:
            continue
        if kind in ("summary", "custom-title", "ai-title") and not session.title:
            session.title = line.get("customTitle") or line.get("summary") or line.get("title")
            continue
        if kind == "attachment" and not line.get("isSidechain"):
            # Messages typed while the agent was working arrive as queued commands.
            attached = line.get("attachment") if isinstance(line.get("attachment"), dict) else {}
            prompt = attached.get("prompt")
            queued = attached.get("type") == "queued_command" and isinstance(prompt, str)
            words = clean_user_text(prompt) if queued else ""
            if words and all(t.user != words for t in session.turns):
                current = Turn(at=parse_time(attached.get("timestamp") or line.get("timestamp")) or 0.0, user=words)
                session.turns.append(current)
            continue
        if line.get("isSidechain") or line.get("isMeta") or kind not in ("user", "assistant"):
            continue
        session.cwd = line.get("cwd") or session.cwd
        session.branch = line.get("gitBranch") or session.branch
        entry = line.get("entrypoint")
        if entry:
            session.surface = entry
            if "desktop" in str(entry):
                session.agent = "claude-desktop"
        at = parse_time(line.get("timestamp")) or 0.0
        message = line.get("message") if isinstance(line.get("message"), dict) else {}
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        if kind == "user":
            origin = line.get("origin") if isinstance(line.get("origin"), dict) else {}
            if line.get("promptSource") == "system" or origin.get("kind") not in (None, "human"):
                continue
            raw = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if line.get("isCompactSummary") or raw.lower().startswith(_COMPACTION_PREFIX):
                session.compactions.append(Compaction(at=at, text=raw))
                continue
            words = clean_user_text(raw)
            if words and not (session.turns and session.turns[-1].user == words and not session.turns[-1].assistant):
                current = Turn(at=at, user=words)
                session.turns.append(current)
            continue
        if current is None:
            current = Turn(at=at, user="")
            session.turns.append(current)
        for block in blocks:
            if block.get("type") == "text":
                _append_reply(current, block.get("text", ""))
            elif block.get("type") == "tool_use":
                _note_action(current, str(block.get("name") or ""), block.get("input"))
    if session and desktop_titles and session.session_id in desktop_titles:
        session.title = desktop_titles[session.session_id] or session.title
        session.agent = "claude-desktop"
    return _finish(session) if session else None


def claude_desktop_titles(metadata_files: Iterable[str]) -> Dict[str, str]:
    """Map Claude Code session ids to Claude Desktop titles (claude-code-sessions/*/local_*.json)."""
    titles: Dict[str, str] = {}
    for text in metadata_files:
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        cli_id = data.get("cliSessionId") or data.get("claudeSessionId") or data.get("sessionId")
        if cli_id:
            titles[str(cli_id)] = str(data.get("title") or data.get("name") or "")
    return titles


# --- Codex (CLI, IDE extension, ChatGPT desktop app) ----------------------------

def _codex_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
        )
    return ""


def parse_codex(text: str, source_name: str = "", titles: Optional[Dict[str, str]] = None) -> Optional[Session]:
    lines = list(_lines(text))
    session = Session(provider="codex", agent="codex", session_id="", source_name=source_name)
    has_events = any(
        line.get("type") == "event_msg" and (line.get("payload") or {}).get("type") in ("user_message", "agent_message")
        for line in lines
    )
    current: Optional[Turn] = None
    for index, line in enumerate(lines):
        kind = line.get("type")
        payload = line.get("payload") if isinstance(line.get("payload"), dict) else None
        at = parse_time(line.get("timestamp")) or 0.0
        if payload is None:
            if index == 0 and "id" in line and "type" not in line:  # early 2025 layout: bare meta line
                session.session_id = str(line["id"])
                session.started_at = parse_time(line.get("timestamp"))
                continue
            payload, kind = line, "response_item"  # early layout: bare response items
        if kind == "session_meta":
            if not session.session_id:  # forked sessions repeat the parent's meta later
                session.session_id = str(payload.get("id") or payload.get("session_id") or "")
                session.started_at = parse_time(payload.get("timestamp")) or at
                session.cwd = payload.get("cwd") or session.cwd
                session.surface = payload.get("originator") or payload.get("source")
                git = payload.get("git") if isinstance(payload.get("git"), dict) else {}
                session.branch = git.get("branch") or session.branch
                session.repo_url = git.get("repository_url") or session.repo_url
            continue
        if kind == "turn_context":
            session.cwd = payload.get("cwd") or session.cwd
            continue
        if kind == "compacted":
            if isinstance(payload.get("message"), str) and payload["message"].strip():
                session.compactions.append(Compaction(at=at, text=payload["message"].strip()))
            continue
        item_type = payload.get("type")
        if has_events:
            if kind != "event_msg":
                if kind == "response_item" and item_type in ("function_call", "custom_tool_call", "local_shell_call") \
                        and current:
                    _note_action(current, str(payload.get("name") or "shell"),
                                 payload.get("arguments") or payload.get("input") or payload.get("action"))
                continue
            if item_type == "user_message":
                words = clean_user_text(payload.get("message", ""))
                if words:
                    current = Turn(at=at, user=words)
                    session.turns.append(current)
            elif item_type == "agent_message":
                if current is None:
                    current = Turn(at=at, user="")
                    session.turns.append(current)
                _append_reply(current, payload.get("message", ""))
            continue
        if item_type == "message":
            role = payload.get("role")
            words = _codex_text(payload.get("content"))
            if role == "user":
                words = clean_user_text(words)
                if words:
                    current = Turn(at=at, user=words)
                    session.turns.append(current)
            elif role == "assistant":
                if current is None:
                    current = Turn(at=at, user="")
                    session.turns.append(current)
                _append_reply(current, words)
        elif item_type in ("function_call", "custom_tool_call", "local_shell_call") and current:
            _note_action(current, str(payload.get("name") or "shell"),
                         payload.get("arguments") or payload.get("input") or payload.get("action"))
    if not session.session_id:
        match = re.search(r"([0-9a-f]{8}-[0-9a-f-]{27,})", source_name)
        session.session_id = match.group(1) if match else source_name
    if titles and session.session_id in titles:
        session.title = titles[session.session_id]
    return _finish(session)


# --- Gemini CLI -----------------------------------------------------------------

def _gemini_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def parse_gemini(text: str, source_name: str = "") -> Optional[Session]:
    try:
        data: Any = json.loads(text)
        messages = data.get("messages", []) if isinstance(data, dict) else []
        header = data if isinstance(data, dict) else {}
    except ValueError:  # JSONL variant: header line plus one message per line
        rows = list(_lines(text))
        header = next((r for r in rows if "sessionId" in r and "messages" not in r and "type" not in r), {})
        messages = [r for r in rows if r.get("type") or r.get("role")]
    session = Session(provider="gemini_cli", agent="gemini-cli",
                      session_id=str(header.get("sessionId") or source_name), source_name=source_name)
    session.started_at = parse_time(header.get("startTime"))
    session.project = None
    session.surface = header.get("projectHash")
    current: Optional[Turn] = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("type") or message.get("role")
        at = parse_time(message.get("timestamp")) or session.started_at or 0.0
        words = _gemini_text(message.get("content") or message.get("parts"))
        if role == "user":
            words = clean_user_text(words)
            if words:
                current = Turn(at=at, user=words)
                session.turns.append(current)
        elif role in ("gemini", "model", "assistant"):
            if current is None:
                current = Turn(at=at, user="")
                session.turns.append(current)
            _append_reply(current, words)
            for call in message.get("toolCalls") or []:
                if isinstance(call, dict):
                    _note_action(current, str(call.get("name") or ""), call.get("args"))
    return _finish(session)


# --- Data exports (ChatGPT, Claude) --------------------------------------------------

def _chatgpt_conversation(conv: Dict[str, Any]) -> Optional[Session]:
    mapping = conv.get("mapping") if isinstance(conv.get("mapping"), dict) else {}
    node_id = conv.get("current_node")
    chain: List[Dict[str, Any]] = []
    while node_id and node_id in mapping:  # the visible branch: current node back to the root
        node = mapping[node_id]
        if isinstance(node.get("message"), dict):
            chain.append(node["message"])
        node_id = node.get("parent")
    chain.reverse()
    session = Session(provider="chatgpt", agent="chatgpt",
                      session_id=str(conv.get("conversation_id") or conv.get("id") or ""),
                      title=conv.get("title"), started_at=parse_time(conv.get("create_time")))
    gizmo = conv.get("gizmo_id") or conv.get("conversation_template_id")
    session.surface = f"chatgpt-project:{gizmo}" if isinstance(gizmo, str) and gizmo.startswith("g-p-") else None
    current: Optional[Turn] = None
    for message in chain:
        meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
        if meta.get("is_visually_hidden_from_conversation"):
            continue
        role = (message.get("author") or {}).get("role")
        content = message.get("content") if isinstance(message.get("content"), dict) else {}
        parts = content.get("parts") if isinstance(content.get("parts"), list) else []
        words = "\n".join(p for p in parts if isinstance(p, str)).strip()
        at = parse_time(message.get("create_time")) or session.started_at or 0.0
        if role == "user" and words:
            current = Turn(at=at, user=words)
            session.turns.append(current)
        elif role == "assistant" and words and content.get("content_type") in (None, "text", "multimodal_text"):
            if current is None:
                current = Turn(at=at, user="")
                session.turns.append(current)
            _append_reply(current, words)
    return _finish(session)


def _claude_conversation(conv: Dict[str, Any]) -> Optional[Session]:
    session = Session(provider="claude_ai", agent="claude-ai", session_id=str(conv.get("uuid") or ""),
                      title=conv.get("name") or None, started_at=parse_time(conv.get("created_at")))
    project = conv.get("project")
    if isinstance(project, dict) and project.get("name"):
        session.project = str(project["name"])
    current: Optional[Turn] = None
    for message in conv.get("chat_messages") or []:
        if not isinstance(message, dict):
            continue
        words = message.get("text") or ""
        if not words and isinstance(message.get("content"), list):
            words = "\n".join(
                b.get("text", "") for b in message["content"] if isinstance(b, dict) and b.get("type") == "text"
            )
        at = parse_time(message.get("created_at")) or session.started_at or 0.0
        if message.get("sender") == "human" and words.strip():
            current = Turn(at=at, user=words.strip())
            session.turns.append(current)
        elif message.get("sender") == "assistant" and words.strip():
            if current is None:
                current = Turn(at=at, user="")
                session.turns.append(current)
            _append_reply(current, words)
    return _finish(session)


def export_payloads(path: Path, data: bytes) -> List[Any]:
    """conversations.json contents from a file or an export .zip."""
    if path.name.endswith(".zip") or data[:2] == b"PK":
        import io

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return [json.loads(archive.read(name)) for name in archive.namelist()
                    if Path(name).name == "conversations.json"]
    return [json.loads(data)]


def parse_export(payload: Any) -> List[Session]:
    conversations = payload if isinstance(payload, list) else []
    sessions: List[Session] = []
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        parsed = _chatgpt_conversation(conv) if "mapping" in conv else (
            _claude_conversation(conv) if "chat_messages" in conv else None)
        if parsed:
            sessions.append(parsed)
    return sessions


# --- Prompt histories (kept by the apps after transcripts are deleted) ------------

def parse_prompt_history(text: str, provider: str) -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    for line in _lines(text):
        words = clean_user_text(str(line.get("display") or line.get("text") or ""))
        at = parse_time(line.get("timestamp") or line.get("ts"))
        if not words or at is None:
            continue
        entries.append(PromptEntry(
            provider=provider, at=at, text=words,
            cwd=line.get("project") or line.get("cwd"),
            session_id=line.get("sessionId") or line.get("session_id"),
        ))
    return entries
