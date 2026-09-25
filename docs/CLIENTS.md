# Connecting AI clients to Muninn

Muninn is one local server (`server.py`, port `42069`) that owns the memory
stores. Every client reaches it one of two ways:

| Transport | Endpoint | Use it for |
|---|---|---|
| Streamable HTTP | `http://127.0.0.1:42069/mcp` | Any client that accepts an MCP URL (preferred) |
| stdio | `python /path/to/Muninn/mcp_wrapper.py` | Clients that only launch commands |

The stdio wrapper is a thin client of the same server; it never opens the
stores itself. Start the server first (`python server.py`, the tray app, or the
service), then configure the clients below.

Replace `/path/to/Muninn` with your checkout and `python` with the interpreter
that has Muninn installed (for example `/path/to/Muninn/.venv/bin/python`).

## Tool profiles

Muninn exposes 37 tools. Some clients cap the total tool count across servers
(Cursor allows 40 active tools), and every tool schema costs context on each
request, so pick a profile per client:

| Profile | Tools | When |
|---|---|---|
| `full` (default) | all 37 | Claude Code, Codex, Gemini CLI |
| `core` | 11: add, search, hunt, update, delete, feedback, goals, instructions, profile, correct | Cursor, VS Code, small local models |
| `readonly` | the read-only tools | Shared or untrusted agents |
| `chatgpt` | `search`, `fetch` | ChatGPT connectors and deep research |

Over HTTP add `?toolset=<profile>` to the URL. For stdio set
`MUNINN_MCP_TOOLSET=<profile>` in the client's `env`. Profiles only change what
a client sees and may call; they are not an access-control boundary.

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

## Claude Code

```bash
# HTTP (preferred)
claude mcp add --transport http -s user muninn http://127.0.0.1:42069/mcp
# with a token
claude mcp add --transport http -s user muninn http://127.0.0.1:42069/mcp \
  --header "Authorization: Bearer $MUNINN_AUTH_TOKEN"

# stdio
claude mcp add -s user muninn -- python /path/to/Muninn/mcp_wrapper.py
```

## Claude Desktop

Claude Desktop launches local servers over stdio. Edit
`claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "muninn": {
      "command": "/path/to/Muninn/.venv/bin/python",
      "args": ["/path/to/Muninn/mcp_wrapper.py"],
      "env": { "MUNINN_AUTH_TOKEN": "only-if-the-server-has-one" }
    }
  }
}
```

On Windows use `silent_mcp.py` instead of `mcp_wrapper.py` to avoid a console
window. Restart Claude Desktop after editing.

Remote connectors on claude.ai (Settings → Connectors) need a public HTTPS URL;
see [Remote access](#remote-access-chatgpt-claudeai-phones).

## OpenAI Codex (CLI, IDE extension, app)

`~/.codex/config.toml` (or `.codex/config.toml` in a project):

```toml
[mcp_servers.muninn]
url = "http://127.0.0.1:42069/mcp"
# Only when the server has a token; Codex reads the variable at connect time.
bearer_token_env_var = "MUNINN_AUTH_TOKEN"
```

stdio alternative:

```toml
[mcp_servers.muninn]
command = "/path/to/Muninn/.venv/bin/python"
args = ["/path/to/Muninn/mcp_wrapper.py"]
```

## Gemini CLI

`~/.gemini/settings.json` (or `.gemini/settings.json` in a project). `httpUrl`
selects Streamable HTTP:

```json
{
  "mcpServers": {
    "muninn": {
      "httpUrl": "http://127.0.0.1:42069/mcp",
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
    "muninn": { "url": "http://127.0.0.1:42069/mcp?toolset=core" }
  }
}
```

## VS Code (GitHub Copilot agent mode)

`.vscode/mcp.json` in a workspace, or **MCP: Add Server** from the command palette:

```json
{
  "servers": {
    "muninn": { "type": "http", "url": "http://127.0.0.1:42069/mcp?toolset=core" }
  }
}
```

## Local models: LM Studio, Open WebUI, Ollama front ends

- **LM Studio** (0.3.17+): `~/.lmstudio/mcp.json` uses Cursor's format:
  `{"mcpServers": {"muninn": {"url": "http://127.0.0.1:42069/mcp?toolset=core"}}}`.
  Small models choose tools better from the `core` profile.
- **Open WebUI**: Admin Settings → Integrations → External Tool Servers → Add
  Connection, set Type to **MCP (Streamable HTTP)** and the URL to
  `http://127.0.0.1:42069/mcp?toolset=core` (from Docker use
  `http://host.docker.internal:42069/mcp?toolset=core`). Choose Bearer auth only if
  the server has a token.
- Clients without MCP can call the REST API directly; the OpenAPI schema is at
  `http://127.0.0.1:42069/openapi.json`.

---

## Remote access: ChatGPT, claude.ai, phones

ChatGPT and claude.ai connect from the cloud, so they need an HTTPS URL they can
reach; `127.0.0.1` will not work. ChatGPT connectors support OAuth or no
authentication, not a static bearer token, so **do not put Muninn on the public
internet with auth disabled**.

Recommended for ChatGPT: OpenAI's
[Secure MCP Tunnel](https://github.com/openai/tunnel-client). It runs an
outbound-only daemon next to Muninn, so nothing on your machine is exposed to
the internet, and it serves ChatGPT, Codex and the Responses API. Point it at
`http://127.0.0.1:42069/mcp?toolset=chatgpt` to give ChatGPT read-only
`search` and `fetch`.

In ChatGPT:

1. Without Developer Mode (deep research, company knowledge), ChatGPT only
   calls `search` and `fetch`; the `chatgpt` profile provides exactly those.
2. With Developer Mode (Settings → Apps & Connectors → Advanced), ChatGPT can
   use any tool. Use `?toolset=core` to let it add and update memories; ChatGPT
   asks for confirmation before write actions.

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
| `search_memory` finds nothing that `search` finds | `search_memory` scopes to the current git project first, then falls back to global memories; project-scoped memories from other projects are never returned |
