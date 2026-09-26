# Connecting AI clients to Muninn

Muninn is one local server (`server.py`, port `42069`) that owns the memory
store. Every agent you run (Claude Code, Claude Desktop, Codex and ChatGPT Work
in the ChatGPT desktop app, Gemini CLI, Cursor, local models) connects to that
same server, so they all read and write the same memories and can hand work to
one another.

| Transport | Endpoint | Use it for |
|---|---|---|
| Streamable HTTP | `http://127.0.0.1:42069/mcp` | Any client that accepts an MCP URL (preferred) |
| stdio | `python /path/to/Muninn/mcp_wrapper.py` | Clients that only launch commands |

The stdio wrapper is a thin client of the same server; it never opens the
store itself. Start the server first (`python server.py`, the tray app, or the
service), then configure the clients below.

Replace `/path/to/Muninn` with your checkout and `python` with the interpreter
that has Muninn installed (for example `/path/to/Muninn/.venv/bin/python`).

## Handing work between agents

Muninn tells every agent how to share the store: the protocol travels in the MCP
server instructions, which Claude Code, Claude Desktop, Codex and Gemini CLI
read at the start of each session. In short:

1. **Start**: the agent calls `get_project_context(project)` and gets the
   project goal, open handoffs, project rules, recent memories (each labelled
   with the agent that wrote it) and your global preferences.
2. **Work**: it searches before answering about earlier work and saves decisions,
   conventions and fixes with `add_memory` as it goes.
3. **Hand off**: when you say "hand this off" (or it stops mid-task) it calls
   `create_handoff` with a summary, next steps, decisions, open questions,
   files and branch. Name a recipient ("hand off to Codex") or leave it open.
4. **Resume**: the next agent sees the open handoff in its briefing and calls
   `resume_handoff`, which marks it claimed so others know it is taken. When
   finished it calls `complete_handoff`.

You can also drive it yourself. In Claude Code the prompts are slash commands,
and in Claude Desktop they are in the **+** menu:

| Prompt | Claude Code | Does |
|---|---|---|
| start | `/mcp__muninn__start` | Load and summarize the project briefing |
| resume | `/mcp__muninn__resume` | Pick up the open handoff and continue |
| handoff | `/mcp__muninn__handoff` | Write a handoff for another agent |
| remember | `/mcp__muninn__remember` | Save a fact, decision or preference |

Codex does not show MCP prompts; ask it in plain words ("load the Muninn context
for this project", "hand this off to Claude") and it follows the server
instructions.

**Project names.** Agents pass `project` = the repository or folder name, so
the same project is filed under one name whichever app you use. Over HTTP the
server cannot see your folders, so the agent must supply it; over stdio Muninn
falls back to the git repository the client started in. Claude Desktop and the
ChatGPT app start stdio servers outside your repositories. Either let the agent
name the project, or pin it with `MUNINN_PROJECT` in that client's `env`. A
memory saved with no project goes to `global`.

**Agent names.** Each memory and handoff records which agent wrote it, taken
from the client's MCP identity (for example `claude-code`, `codex`,
`claude-desktop`). Override it with `?agent=<name>` on the URL or
`MUNINN_AGENT_NAME` for stdio when two apps would otherwise look the same.

## Tool profiles

Muninn exposes 41 tools. Some clients cap the total tool count across servers
(Cursor allows 40 active tools), and every tool schema costs context on each
request, so pick a profile per client:

| Profile | Tools | When |
|---|---|---|
| `full` (default) | all 41 | Claude Code, Claude Desktop, Codex, Gemini CLI |
| `core` | 15: context and handoffs, add, search, hunt, update, delete, feedback, goals, instructions, profile, correct | Cursor, VS Code, small local models |
| `readonly` | the read-only tools | Shared or untrusted agents |
| `chatgpt` | `search`, `fetch` | Web ChatGPT connectors and deep research |

Over HTTP add `?toolset=<profile>` to the URL (combine with `&agent=...`). For
stdio set `MUNINN_MCP_TOOLSET=<profile>` in the client's `env`. Profiles only
change what a client sees and may call; they are not an access-control boundary.

## Authentication and browser origins

- If `MUNINN_AUTH_TOKEN` (or `MUNINN_API_KEY`) is set for the server, HTTP
  clients must send `Authorization: Bearer <token>`, and the stdio wrapper needs
  the same variable in its `env`. With no token configured, Muninn accepts local
  requests without one.
- The server rejects browser requests whose `Origin` is not `localhost`,
  `127.0.0.1` or `[::1]`. This stops other web pages and DNS rebinding from
  reaching your memories. Add extra origins with
  `MUNINN_ALLOWED_ORIGINS=https://example.com` (comma-separated). Opening
  `dashboard.html` straight from disk sends `Origin: null`; include `null` to
  allow it, or use `http://localhost:42069/` instead.
- MCP hosts, the SDK, `curl` and tunnels send no `Origin` and are unaffected.

---

## ChatGPT desktop app (Codex and ChatGPT Work), Codex CLI and IDE extension

These share one MCP configuration, so setting Muninn up once covers all of them.
Both local stdio and local HTTP servers work; no tunnel is needed. Either use the
app (Settings → MCP servers → Add server) or edit `~/.codex/config.toml`:

```toml
[mcp_servers.muninn]
url = "http://127.0.0.1:42069/mcp?agent=codex"
# Only when the server has a token; Codex reads the variable at connect time.
bearer_token_env_var = "MUNINN_AUTH_TOKEN"
```

stdio alternative:

```toml
[mcp_servers.muninn]
command = "/path/to/Muninn/.venv/bin/python"
args = ["/path/to/Muninn/mcp_wrapper.py"]
env = { MUNINN_AGENT_NAME = "codex" }
```

Codex reads Muninn's server instructions, which is how it learns the start,
save and handoff routine.

## Claude Desktop (chat and the Code tab)

Claude Desktop launches local servers over stdio. Edit
`claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "muninn": {
      "command": "/path/to/Muninn/.venv/bin/python",
      "args": ["/path/to/Muninn/mcp_wrapper.py"],
      "env": {
        "MUNINN_AGENT_NAME": "claude-desktop",
        "MUNINN_AUTH_TOKEN": "only-if-the-server-has-one"
      }
    }
  }
}
```

On Windows use `silent_mcp.py` instead of `mcp_wrapper.py` to avoid a console
window. Restart Claude Desktop after editing.

The Code tab runs Claude Code, which also reads Claude Code's own configuration
(below). Current Desktop builds inject `claude_desktop_config.json` servers into
Code-tab sessions too, so use the same name `muninn` in both files. If the Code tab
lists Muninn twice, remove one of the entries.

Remote connectors on claude.ai (Settings → Connectors) are a different path and
need a public HTTPS URL; see [Remote access](#remote-access-web-chatgpt-claudeai-phones).

## Claude Code (CLI, IDE extensions, Desktop Code tab)

```bash
# HTTP (preferred)
claude mcp add --transport http -s user muninn "http://127.0.0.1:42069/mcp?agent=claude-code"
# with a token
claude mcp add --transport http -s user muninn "http://127.0.0.1:42069/mcp?agent=claude-code" \
  --header "Authorization: Bearer $MUNINN_AUTH_TOKEN"

# stdio
claude mcp add -s user muninn -- python /path/to/Muninn/mcp_wrapper.py
```

## Gemini CLI

`~/.gemini/settings.json` (or `.gemini/settings.json` in a project). `httpUrl`
selects Streamable HTTP:

```json
{
  "mcpServers": {
    "muninn": {
      "httpUrl": "http://127.0.0.1:42069/mcp?agent=gemini-cli",
      "headers": { "Authorization": "Bearer only-if-the-server-has-one" },
      "timeout": 40000
    }
  }
}
```

## Cursor

`~/.cursor/mcp.json` (or `.cursor/mcp.json`). Use the `core` profile so Muninn
fits inside Cursor's 40-tool limit alongside other servers:

```json
{
  "mcpServers": {
    "muninn": { "url": "http://127.0.0.1:42069/mcp?toolset=core&agent=cursor" }
  }
}
```

## VS Code (GitHub Copilot agent mode)

`.vscode/mcp.json` in a workspace, or **MCP: Add Server** from the command palette:

```json
{
  "servers": {
    "muninn": { "type": "http", "url": "http://127.0.0.1:42069/mcp?toolset=core&agent=vscode" }
  }
}
```

## Local models: LM Studio, Open WebUI, Ollama front ends

- **LM Studio** (0.3.17+): `~/.lmstudio/mcp.json` uses Cursor's format:
  `{"mcpServers": {"muninn": {"url": "http://127.0.0.1:42069/mcp?toolset=core&agent=lm-studio"}}}`.
  Small models choose tools better from the `core` profile.
- **Open WebUI**: Admin Settings → Integrations → External Tool Servers → Add
  Connection, set Type to **MCP (Streamable HTTP)** and the URL to
  `http://127.0.0.1:42069/mcp?toolset=core` (from Docker use
  `http://host.docker.internal:42069/mcp?toolset=core`). Choose Bearer auth only if
  the server has a token.
- Clients without MCP can call the REST API directly; the OpenAPI schema is at
  `http://127.0.0.1:42069/openapi.json`. Handoffs are under `/handoffs` and the
  session briefing is `GET /context?project=<name>`.

---

## Remote access: web ChatGPT, claude.ai, phones

Only clients that run in the cloud need this; the desktop apps above connect
locally. Web ChatGPT and claude.ai need an HTTPS URL they can reach, and
`127.0.0.1` will not work. ChatGPT connectors support OAuth or no
authentication, not a static bearer token, so **do not put Muninn on the public
internet with auth disabled**.

Recommended for ChatGPT: OpenAI's
[Secure MCP Tunnel](https://github.com/openai/tunnel-client). It runs an
outbound-only daemon next to Muninn, so nothing on your machine is exposed to
the internet. Point it at `http://127.0.0.1:42069/mcp?toolset=chatgpt` to give
ChatGPT read-only `search` and `fetch`.

In web ChatGPT:

1. Without Developer Mode (deep research, company knowledge), ChatGPT only
   calls `search` and `fetch`; the `chatgpt` profile provides exactly those.
2. With Developer Mode (Settings → Apps & Connectors → Advanced), ChatGPT can
   use any tool. Use `?toolset=core` to let it add memories and handoffs;
   ChatGPT asks for confirmation before write actions.

If a remote client gets `403 Origin not allowed`, add its origin to
`MUNINN_ALLOWED_ORIGINS`.

---

## Protocol support

| Client era | How it talks to `/mcp` |
|---|---|
| MCP 2024-11-05 to 2025-11-25 | `initialize` handshake plus `Mcp-Session-Id` |
| MCP 2026-07-28 | Stateless requests with `server/discover`; no session |

Both work on the same URL. The stdio wrapper speaks the handshake versions.
Every version is checked in CI against the official `mcp-types` schemas
(`tests/test_mcp_wire_schema.py`).

Tool results follow the current spec: failures come back as results with
`isError: true` so the model can read the message and retry, and `search` and
`fetch` also return `structuredContent` matching their `outputSchema`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` | Start the server; `curl http://127.0.0.1:42069/health` should answer |
| `401 Unauthorized` | The client's token does not match the server's `MUNINN_AUTH_TOKEN` |
| `403 Origin not allowed` | Browser or remote origin not allow-listed; set `MUNINN_ALLOWED_ORIGINS` |
| Too many tools, or the client ignores some | Use `?toolset=core` or `MUNINN_MCP_TOOLSET=core` |
| One agent cannot see another's project memories | They used different project names; check the `project` in `get_project_context`, or pin `MUNINN_PROJECT` for that client |
| Memories show agent `unknown` | Set `?agent=` on the URL or `MUNINN_AGENT_NAME` for stdio |
| Handoff not offered to the next agent | Handoffs are per project; resume with the same project name, or pass `handoff_id` |
