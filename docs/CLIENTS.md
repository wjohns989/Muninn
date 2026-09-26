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

## Bring in your existing conversations

Muninn can turn the conversations already on this machine into memories, filed
under the project and time they happened, so agents can search them and re-read
whole threads in order.

| App | Where Muninn reads it |
|---|---|
| Claude Code, and Claude Desktop's Code tab / Cowork | `~/.claude/projects/*/*.jsonl` (or `$CLAUDE_CONFIG_DIR`); Desktop titles from its `claude-code-sessions` folder |
| Codex CLI, IDE extension, ChatGPT desktop app (Codex) | `~/.codex/sessions` and `archived_sessions`, including compressed `.jsonl.zst` (or `$CODEX_HOME`); titles from `state_*.sqlite` |
| Gemini CLI | `~/.gemini/tmp/<project hash>/chats` |
| ChatGPT and Claude chats (web and desktop chat) | These live in the cloud. Download your data (ChatGPT: Settings → Data controls → Export; Claude: Settings → Privacy → Export data) and leave the `.zip` in Downloads, or pass it with `--path` |
| Prompts from deleted sessions | `~/.claude/history.jsonl` and `~/.codex/history.jsonl`, which the apps keep after deleting transcripts |

```bash
python -m muninn.cli history status          # what was found, retention warnings
python -m muninn.cli history import          # dry run: threads, projects, dates, memory counts
python -m muninn.cli history import --apply  # import (runs in the background)
python -m muninn.cli history threads --project Muninn   # also --agent, --status, --topic, --q, --since
python -m muninn.cli history thread <thread-id>
```

Agents can do the same with the `import_agent_history` and `get_thread` tools.

**What becomes a memory.** Every turn becomes one memory: your request, the
reply, the files it touched and the commands it ran. Tool output and hidden
reasoning are left out. Long turns are split into ordered parts, never cut short.
Each memory carries the time of the turn, the project, the working directory,
the branch, the agent and a thread id with the turn number, so:

- imported memories sit in the right place in time next to everything else;
- the project name matches what live agents use (the git remote name, or the
  repository folder; worktrees of a repository share its name), so imported
  history and current project memories line up;
- `get_thread` re-reads a conversation from start to finish.

Imported turns, compaction summaries and thread summaries are a record of what
was said, so Muninn's upkeep leaves them alone: they are never deduplicated,
merged, archived by decay, retyped, or used to resolve a conflict with another
memory. An old turn cannot retire a current decision. The insights drawn from
them (below) follow the normal memory lifecycle.

Compaction summaries are kept as memories of their own. The transcripts hold
the full conversation from before each compaction, so those turns come back
too. Each thread also gets a summary memory, refreshed as the thread grows.
Conversations outside any repository are filed under the project `global`.
Secrets (API keys, tokens, passwords, connection strings) are redacted from
memory text.

**One record per project, in time order, without copies.** Resuming or
forking a session (Claude Code `--resume`, Codex fork) writes a new transcript
that starts with a copy of the earlier conversation. Each turn is identified by
its original time and text, so a copy is recognized and stored once, and the
new thread is marked as continuing the old one. When agents pass a project back
and forth (Codex, then Claude Code, then Codex again in its first thread), the
project timeline reads it in the order it happened:

```bash
python -m muninn.cli history timeline --project Muninn     # every app, interleaved, with handoffs
```

Agents get the same view with `get_thread` and `timeline=true`. Entries show
which agent spoke, handoff events, and where the work switched from one agent
to another.

**Nothing the apps clean up is lost.** Claude Code deletes transcripts older
than 30 days by default (`cleanupPeriodDays`), Gemini CLI can expire sessions,
and deleting a Codex thread deletes its file. The server keeps its own
compressed copy of every transcript in `<data dir>/history_vault`, synced at
startup and every 30 minutes, and never removes anything from it. A file the
app deletes stays in the vault, and a file rewritten shorter keeps its previous
copy. Credential and settings files are never read. After your first
`--apply`, each sync also imports new turns, so live threads, including what
compaction drops, keep flowing in.

| Setting | Default | |
|---|---|---|
| `MUNINN_HISTORY_VAULT` | `1` | `0` turns the vault and automatic import off |
| `MUNINN_HISTORY_SYNC_MINUTES` | `30` | How often to copy and import new history |
| `MUNINN_HISTORY_AUTO_IMPORT` | after first import | `1`/`0` forces automatic import on or off |
| `MUNINN_HISTORY_HOMES` | none | Extra home folders to scan, separated by `:` (`;` on Windows), e.g. `/mnt/c/Users/me` when Muninn runs in WSL |

Muninn finds relocated data through the apps' own `CLAUDE_CONFIG_DIR` and
`CODEX_HOME`. It never changes app settings. To keep transcripts in Claude Code
itself for longer, raise `cleanupPeriodDays` in `~/.claude/settings.json`.

## Automatic briefing and capture (hooks)

Hooks make this automatic instead of something agents must remember:

```bash
python -m muninn.cli hooks install          # dry run: shows the settings it would write
python -m muninn.cli hooks install --apply  # write them (backups kept; your own hooks untouched)
python -m muninn.cli hooks status
python -m muninn.cli hooks uninstall --apply
```

| Event | What Muninn does |
|---|---|
| Session start (also after resume, clear, compaction) | Injects the project briefing into the new session: goal, open handoffs, earlier threads from every app, project rules, recent memories, preferences |
| Before compaction | Copies the transcript to the vault and imports the thread right away, so what compaction drops is saved at that moment |
| After each reply | Same, at most every two minutes per thread |
| Session end | Same, immediately |

- **Claude Code** (CLI, IDE extensions, Claude Desktop's Code tab): `http` hooks
  in `~/.claude/settings.json` (or `$CLAUDE_CONFIG_DIR`) that call
  `http://127.0.0.1:42069/hooks/claude-code`.
- **Codex** (CLI, IDE extension, ChatGPT desktop app): command hooks in
  `~/.codex/hooks.json` (or `$CODEX_HOME`) that run `muninn/hook_client.py`, a
  standard-library script that starts in about 50 ms (Codex gives session-end
  hooks one second) and never blocks Codex if the server is down.
- With a server token, export `MUNINN_AUTH_TOKEN` in the environment the apps
  start from; the hooks send it.

## Understanding imported threads (optional LLM step)

Importing keeps every turn. `analyze` goes further and has a model read each
imported thread and record what matters:

- **Insights:** decisions and why they were made, your preferences,
  project conventions, facts, bug fixes and open items. Each is saved as its
  own memory, dated to the turn it came from and linked to the thread.
- **Thread fields:** a written summary, a status (completed, in progress,
  abandoned, answered) and topic tags. The thread catalog can be filtered by
  these.
- **Handoffs:** threads from the last 14 days left in progress with open items
  become handoffs, so the next agent can pick them up.
- **Across apps, in order:** each project's threads are read oldest first.
  The model sees where another agent worked in between ("Meanwhile, Claude
  Code worked on …"), and it sees the insights other threads already recorded,
  with their times. When later work replaces an earlier decision, the old
  insight is archived (restorable), so search returns the current one.
  Re-analysing a thread that grew replaces its insights rather than adding
  more.

```bash
python -m muninn.cli history analyze                         # dry run: threads, tokens, estimated cost
python -m muninn.cli history analyze --apply                 # runs in the background
python -m muninn.cli history threads --status in_progress --topic auth
python -m muninn.cli history analyze --apply --retry-refused --model google/gemini-3.5-flash-lite
```

The cost estimate uses the rate OpenRouter actually billed for GPT-6 Luna Pro
(about 1.3 characters per prompt token), so it errs high for other models.

### Setting up OpenRouter

The first time you run `history import` or `history analyze` in a terminal,
Muninn asks for an OpenRouter key. The input is hidden, and the key is checked
with OpenRouter before it is saved to Muninn's config directory
(`openrouter.json`, readable only by you, never in a repository). Press Enter
to use local Ollama instead; you won't be asked again. Manage it any time:

```bash
python -m muninn.cli openrouter status                       # key (masked), models, where it is stored
python -m muninn.cli openrouter set                          # prompt for a key
python -m muninn.cli openrouter set --model deepseek/deepseek-v4.1-flash
python -m muninn.cli openrouter clear
```

`OPENROUTER_API_KEY` (or `MUNINN_OPENROUTER_API_KEY`) in the server's
environment takes precedence over the saved key.

### Which model

The default was chosen from OpenRouter's live list of zero-data-retention
endpoints (September 2026). Candidates had to offer strict structured outputs,
at least 128k context and healthy uptime; they were then compared on
independent extraction and summarization results and on cost per thread.

| Model | Role | Why |
|---|---|---|
| `openai/gpt-6-luna-pro` | Default | GPT-6 Luna's higher-reasoning tier on a zero-data-retention Azure host. 1.05M context. In live tests a 15.8 MB Claude Desktop transcript (68 turns, 6 compactions) took one call and $0.05 |
| `deepseek/deepseek-v4-flash` | First fallback | 1M context, 8 ZDR hosts with strict JSON, cheapest input. Keeps bulk runs moving if Luna Pro's single ZDR host (Azure) is busy |
| `google/gemini-3.5-flash-lite` | Second fallback | 1M context, Google-hosted ZDR, lowest hallucination rate of the three |

OpenRouter falls back down this list automatically when a model errors or is
unavailable; it accepts at most three models, so the list stops at three. A
model id ending in `:batch` is used as its direct model: those ids only work
through OpenRouter's asynchronous Batch API, which also keeps inputs and
results for 30 days. All three accept about a million tokens, so even very
long conversations go to the model whole: tool output is already stripped,
and a 15.8 MB transcript came to 431k characters of conversation. Only a thread beyond the window
(`MUNINN_INSIGHTS_WINDOW_TOKENS`, default 200k) is split. Its parts are
analyzed separately and then merged by one more call into a single summary,
status and deduplicated insight list. DeepSeek V4.1 Flash (released
2026-09-10) is one flag away; it is not the default because no independent
results for it were available yet.

### How results are stored correctly

- Requests carry a strict JSON Schema, and `provider.require_parameters`
  makes OpenRouter use only endpoints that enforce it.
- Only parameters every chosen model's ZDR endpoints accept are sent. For
  example, GPT-6 Luna's endpoints reject `temperature`, which would otherwise
  route around it.
- Every reply is validated before anything is written. An invalid reply is
  sent back once with the error, and if it still fails, whatever is usable is
  kept. Unknown turn numbers are dropped, and only preferences can be global.
  Only note ids the model was actually shown can be marked superseded.
- A refusal is never stored. A reply counts as a refusal when it has a
  `refusal` field, stops on a content filter or safety reason, is rejected by
  moderation or a provider content filter (such as Azure's), is empty, or its
  summary is the model declining ("I can't help with…"). OpenRouter only falls
  back on errors, so Muninn sends the same request to the next model itself.
  An insight worded as a refusal is dropped. If every model refuses, nothing
  is written: the thread keeps its imported turns, summary and status, and
  `analysis_error` says why. It is not retried on every run (that would pay
  for the same refusal again); use `--retry-refused`, for example with another
  `--model`. The prompt asks models to catalogue fiction and sensitive material
  neutrally and at a high level, which avoids most refusals and keeps explicit
  detail out of the insights.
- The report lists the calls, schema retries, refusals, refused threads,
  tokens, cost and the model that actually answered, and each insight records
  `insight_model`.

| Provider | When | Privacy |
|---|---|---|
| OpenRouter | A key is saved or set in the environment | Every request sets `provider.zdr = true` and `data_collection = deny`: only endpoints that retain nothing and cannot train on it. Text is redacted before sending |
| Ollama | No key, `--llm ollama`, or you skipped the prompt | Stays on your machine; smaller window (`MUNINN_OLLAMA_WINDOW_TOKENS`, default 24k), slower (`MUNINN_OLLAMA_MODEL`, default `llama3.2:3b`) |

Nothing is sent anywhere until you run `analyze --apply`. Set
`MUNINN_INSIGHTS_AUTO=1` to also analyze new threads after each automatic
import.

Why not use OpenRouter for embeddings to speed up the import itself? Memories
must all be embedded by the same model, so switching the embedding provider
means re-embedding the whole store (`muninn.cli reindex`), and every memory
you ever store afterwards goes to the cloud. On this project's test machine,
local embedding took about 0.13 s per turn (roughly 20 minutes per 10,000
turns, in the background), and batching did not help. The import already
skips per-turn LLM entity extraction. So the cloud pays off for understanding
threads, not for storing them.

## Tool profiles

Muninn exposes 43 tools. Some clients cap the total tool count across servers
(Cursor allows 40 active tools), and every tool schema costs context on each
request, so pick a profile per client:

| Profile | Tools | When |
|---|---|---|
| `full` (default) | all 43 | Claude Code, Claude Desktop, Codex, Gemini CLI |
| `core` | 16: context, handoffs and threads, add, search, hunt, update, delete, feedback, goals, instructions, profile, correct | Cursor, VS Code, small local models |
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
| Old conversations missing | Run `python -m muninn.cli history status`; for Claude Code, anything already past `cleanupPeriodDays` before the vault first ran is gone, but its prompts are recovered from `history.jsonl` |
| Handoff not offered to the next agent | Handoffs are per project; resume with the same project name, or pass `handoff_id` |
