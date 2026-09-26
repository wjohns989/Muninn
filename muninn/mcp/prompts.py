"""Built-in prompting: the memory protocol sent as server instructions, and MCP prompts.

Server instructions reach every client that reads the MCP `instructions` field
(Claude Code, Codex in the ChatGPT app and CLI, Claude Desktop, Gemini CLI), so
the protocol for sharing one store between agents lives there. MCP prompts are
extra entry points for hosts that surface them: Claude Code lists them as
/mcp__muninn__<name> slash commands and Claude Desktop in its prompt menu.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

PROTOCOL_INTRO = (
    "Muninn is the shared long-term memory for every AI agent this user runs (Claude Code, Claude "
    "Desktop, Codex in the ChatGPT app, Gemini CLI, Cursor and others). They all read and write the "
    "same store, so work can move from one agent to another.\n\n"
    "Always pass `project`: the repository or folder name you are working in (for example \"Muninn\"), "
    "spelled the same way every time. It is how other agents find this project's memories."
)

FULL_PROTOCOL = PROTOCOL_INTRO + """

1. Start of a session: call get_project_context(project) before other work and follow the project \
instructions it returns. If it shows an open handoff for you or for anyone, call resume_handoff and \
continue from its next steps. It also lists recent_threads: earlier conversations about this project \
in any app; read the relevant ones with get_thread instead of redoing work.
2. Before answering about earlier work, decisions or preferences, call search_memory.
3. Save durable knowledge with add_memory as you go: a decision and its reason, a convention, a fix \
for a recurring problem, a fact about the environment. Use scope="global" for user preferences that \
apply everywhere. One fact per memory, written so it makes sense on its own. Never store secrets, \
credentials, tokens or personal data the user has not asked you to keep.
4. When the user corrects a stored fact, use correct_fact or update_memory rather than adding a \
contradicting memory.
5. Handing off: when the user asks you to hand off, when you stop with work unfinished, or before a \
conversation ends mid-task, call create_handoff with a summary, next steps, decisions and files, \
written so an agent with no access to this conversation can continue. When you finish a handoff you \
resumed, call complete_handoff."""

READONLY_PROTOCOL = PROTOCOL_INTRO + """

This connection is read-only. Call get_project_context(project) at the start of a session, read \
relevant earlier conversations with get_thread, and call search_memory before answering about earlier \
work, decisions or preferences."""

CHATGPT_PROTOCOL = (
    "Muninn is the user's shared memory across their AI agents. Use search to find relevant memories "
    "and fetch to read one in full before relying on it."
)


def protocol_for(toolset: str) -> str:
    if toolset == "chatgpt":
        return CHATGPT_PROTOCOL
    if toolset == "readonly":
        return READONLY_PROTOCOL
    return FULL_PROTOCOL


_PROJECT_ARG = {
    "name": "project",
    "description": "Repository or folder name you are working in",
    "required": False,
}

PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "start",
        "title": "Start with Muninn context",
        "description": "Load the project's goal, handoffs, rules and recent memories before starting.",
        "arguments": [_PROJECT_ARG],
        "toolsets": ("full", "core", "readonly"),
    },
    {
        "name": "resume",
        "title": "Resume a handoff",
        "description": "Pick up the work another agent handed off for this project.",
        "arguments": [_PROJECT_ARG],
        "toolsets": ("full", "core"),
    },
    {
        "name": "handoff",
        "title": "Hand off to another agent",
        "description": "Write a handoff so another agent (or a later session) can continue this work.",
        "arguments": [
            _PROJECT_ARG,
            {"name": "to_agent", "description": "Intended agent, e.g. codex or claude-code", "required": False},
            {"name": "notes", "description": "Anything the handoff must mention", "required": False},
        ],
        "toolsets": ("full", "core"),
    },
    {
        "name": "remember",
        "title": "Remember this",
        "description": "Save a fact, decision or preference to Muninn.",
        "arguments": [
            {"name": "content", "description": "What to remember", "required": True},
            _PROJECT_ARG,
        ],
        "toolsets": ("full", "core"),
    },
]


def _project_phrase(arguments: Dict[str, str]) -> str:
    project = (arguments.get("project") or "").strip()
    return (
        f'project "{project}"'
        if project
        else "this project (use the repository or folder name you are working in as `project`)"
    )


def _render(name: str, arguments: Dict[str, str]) -> str:
    project = _project_phrase(arguments)
    if name == "start":
        return (
            f"Load my Muninn context for {project}: call get_project_context, then give me a short "
            "briefing: the goal, any open handoff (who left it and its next steps), project rules and "
            "the recent memories that matter. If a handoff is open, offer to resume it."
        )
    if name == "resume":
        return (
            f"Resume the open Muninn handoff for {project}: call resume_handoff, tell me what was done, "
            "which agent left it and the next steps, then continue from the first next step. Save new "
            "decisions with add_memory as you go and call complete_handoff when the work is finished."
        )
    if name == "handoff":
        recipient = (arguments.get("to_agent") or "").strip()
        notes = (arguments.get("notes") or "").strip()
        text = (
            f"Hand off the current work on {project}"
            + (f" to {recipient}" if recipient else "")
            + ". Call create_handoff with: a summary of what was done and the current state; concrete, "
            "ordered next steps; decisions made and why; open questions for me; and the relevant files "
            "and branch. Write it for an agent that cannot see this conversation. Save any durable "
            "facts from this session with add_memory first."
        )
        return text + (f"\n\nMake sure it covers: {notes}" if notes else "")
    if name == "remember":
        return (
            f"Save this to Muninn with add_memory for {project}: {arguments['content'].strip()}\n\n"
            'Use scope="global" if it is a preference or rule that applies across projects, otherwise '
            "scope=\"project\". Rewrite it as one self-contained fact if needed."
        )
    raise KeyError(name)


def list_prompts(toolset: str) -> List[Dict[str, Any]]:
    return [
        {key: value for key, value in prompt.items() if key != "toolsets"}
        for prompt in PROMPTS
        if toolset in prompt["toolsets"]
    ]


def get_prompt(name: Any, arguments: Optional[Dict[str, Any]], toolset: str) -> Dict[str, Any]:
    """Render a prompt; raises ValueError with a client-facing message when it cannot."""
    prompt = next((p for p in PROMPTS if p["name"] == name and toolset in p["toolsets"]), None)
    if prompt is None:
        raise ValueError(f"Unknown prompt: {name}")
    arguments = {key: str(value) for key, value in (arguments or {}).items() if value is not None}
    missing = [
        arg["name"] for arg in prompt["arguments"] if arg["required"] and not arguments.get(arg["name"], "").strip()
    ]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")
    return {
        "description": prompt["description"],
        "messages": [{"role": "user", "content": {"type": "text", "text": _render(name, arguments)}}],
    }
