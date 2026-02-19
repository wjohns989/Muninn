# Muninn Development Handoff

> **Updated**: 2026-02-19
> **Branch**: `feature/v3.12.0-operational-hardening`
> **Version**: v3.12.0 (Phase 15 in progress)
> **Status**: Phase 14 MERGED (PR #43). Phase 15 branch open, PR #44 in progress.

---

## Current State

### What's Working
- **694 tests pass** (100% pass rate)
- **Server**: FastAPI on `http://localhost:42069`, auth token via `MUNINN_AUTH_TOKEN`
- **MCP**: Registered as "muninn" (tools: `mcp__muninn__*`) in Claude Code user config with auth token baked in
- **Claude Desktop**: Already correctly registered as "muninn"
- **Phase 14 (v3.11.0)**: Project-scoped memory — **MERGED** (PR #43, 2026-02-19)
- **Phase 15 (v3.12.0)**: Operational hardening branch open, PR #44 in progress

### Server Quick Start

```bash
# Start server (Windows — run from repo root)
$env:MUNINN_AUTH_TOKEN = (Get-Content .muninn_token)
python server.py

# Or from bash
MUNINN_AUTH_TOKEN=$(cat .muninn_token) python server.py
```

**Auth token** is stored in `.muninn_token` (gitignored). Also set permanently via `setx`:
```
Token: ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ
```

### Data Directory
```
C:\Users\wjohn\AppData\Local\AntigravityLabs\muninn\
├── metadata.db          # SQLite — 74 memories
├── qdrant_v8/           # Vector store — 73 vectors
└── kuzu_v12/            # Graph store — 0 nodes (graph feature not yet active)
```

---

## Recent Changes (This Session — 2026-02-18)

### Bug Fixes

#### 1. `sqlite_metadata.py` — Schema Migration Ordering Bug (CRITICAL)
**File**: `muninn/store/sqlite_metadata.py`
**Symptom**: `sqlite3.OperationalError: no such column: scope` on server startup against pre-v3.11.0 databases
**Root cause**: `idx_memories_scope` was in `CREATE_INDEXES` which runs BEFORE `_ensure_column_exists` adds the `scope` column for existing databases
**Fix**: Removed `idx_memories_scope` from `CREATE_INDEXES`; added it explicitly immediately after `_ensure_column_exists(conn, "memories", "scope", ...)` in `_initialize()`

#### 2. `handlers.py` — Hardcoded Version in MCP `initialize` Response
**File**: `muninn/mcp/handlers.py`
**Symptom**: `serverInfo.version` returned `"3.1.0"` to MCP clients instead of `"3.11.0"`
**Fix**: Import `__version__` from `muninn.version` and use it dynamically

#### 3. `lifecycle.py` — Stale Variable Naming
**File**: `muninn/mcp/lifecycle.py`
**Symptom**: Module-level var named `GLOBAL_MEMORY_DIR` (legacy from when project was "global-memory")
**Fix**: Renamed to `MUNINN_DIR` in all 3 occurrences

### Infrastructure Fixes

#### 4. MCP Registration — "global-memory" → "muninn"
**Problem**: Claude Code CLI had the MCP registered as "global-memory" in `~/.claude.json`
**Root cause**: Previous `claude mcp add global-memory ...` command
**Fix**: `claude mcp remove global-memory` then `claude mcp add -s user muninn -e MUNINN_AUTH_TOKEN=<token> -- python mcp_wrapper.py`
**Result**: Tools now appear as `mcp__muninn__*` instead of `mcp__global-memory__*`

#### 5. Auth Token Architecture Fix
**Problem**: `server.py` and `mcp_wrapper.py` each generate a RANDOM auth token when `MUNINN_AUTH_TOKEN` is not set → 401 on every tool call
**Fix**: Generated permanent token, set via `setx MUNINN_AUTH_TOKEN <token>` (persists across sessions) AND baked into Claude Code MCP registration with `-e MUNINN_AUTH_TOKEN=<token>`
**Note**: `.gitignore` updated to exclude `.muninn_token`

---

## Architecture Notes

### Auth Flow
```
Claude Code → spawns mcp_wrapper.py (with MUNINN_AUTH_TOKEN from MCP env config)
           → mcp_wrapper.py uses token in Authorization: Bearer header
           → server.py validates token from MUNINN_AUTH_TOKEN env var
```

Both processes MUST share the same token. The MCP registration `-e` flag is the reliable cross-session mechanism.

### Token Mismatch Risk (IMPORTANT)
If `MUNINN_AUTH_TOKEN` is not set in the MCP server registration env AND not in the system environment, each process generates a different random token → all tool calls fail silently. If this happens:
1. Check `MUNINN_AUTH_TOKEN` is in the claude MCP config: `claude mcp get muninn`
2. Or set env var and restart: `claude mcp add -s user muninn -e MUNINN_AUTH_TOKEN=<token> ...`

### MCP Server Registration (User-Scope)
```json
{
  "muninn": {
    "type": "stdio",
    "command": "C:/Users/wjohn/miniconda3/python.exe",
    "args": ["C:/Users/wjohn/muninn_mcp/mcp_wrapper.py"],
    "env": {
      "MUNINN_AUTH_TOKEN": "ij0w9VmdPH5dxnE7vG-lZCXPPWhX9uU7HpBJODg0zoQ"
    }
  }
}
```

---

## Phase 14 (v3.11.0) Summary

**Feature**: Project-scoped memory with strict isolation
**PR**: #43 (open for review)

### Core Design
```
MemoryRecord.scope: Literal["project", "global"] = "project"
  scope="project"  → visible only within its project; NEVER returned in cross-project fallback
  scope="global"   → always visible regardless of current project
```

### Key Files Changed (Phase 14)
- `muninn/core/types.py` — `MemoryRecord.scope` field
- `muninn/store/sqlite_metadata.py` — DB column, migration, SQL filters
- `muninn/mcp/handlers.py` — `add_memory` tool exposes `scope` parameter; fallback restricted to `scope="global"`
- `muninn/store/vector_store.py` — scope in Qdrant payload for pre-filter
- `tests/test_v3_11_0_project_scope.py` — 43 tests, 5 correctness invariants proven

### Correctness Invariants
1. `scope="project"` memories in project A never appear in project B queries
2. `scope="global"` memories appear in all contexts including fallback
3. Fallback search ONLY returns `scope="global"` — no project-scoped memory leaks
4. Pre-v3.11.0 rows default to `scope="project"` (backward compat)
5. Migration against existing DB is idempotent

### Feature Flag
- `MUNINN_PROJECT_SCOPE_STRICT=1` — disables fallback retry entirely (zero cross-project leakage)

---

## Open Items / Next Steps

### Immediate
- [ ] **Restart Claude Code** to activate new MCP registration (with auth token) — MCP tools will work as `mcp__muninn__*` in next session
- [ ] **PR #44 implementation** — Phase 15 operational hardening items (see SOTA_PLUS_PLAN.md Phase 15)

### Phase 15 Priorities (v3.12.0)
- [ ] **Auth propagation fix** in `lifecycle.py:start_server()` — pass `MUNINN_AUTH_TOKEN` to spawned server
- [ ] **Graph chains activation** — smoke test KuzuDB memory chains end-to-end
- [ ] **OTel activation validation** — verify GenAI spans with correct attributes
- [ ] **LongMemEval adapter** — external benchmark grounding for SOTA+ claims
- [ ] Evaluate: should `.muninn_token` be rotated periodically? Add rotation docs.
- [ ] Consider: start_server() in lifecycle.py should pass `MUNINN_AUTH_TOKEN` to spawned server process (currently not done → requires token to be in system env)

### Known Architectural Concern
`muninn/mcp/lifecycle.py:start_server()` spawns `server.py` without propagating `MUNINN_AUTH_TOKEN`. If the system env var is not set and the MCP registration env var IS set (for the wrapper), then auto-starting the backend server fails silently with auth mismatch. The server needs to be started manually or via the startup mechanism that has the token available. See: `lifecycle.py` lines 91–103.

---

## Test Suite
```bash
# Run full suite
pytest tests/ -x -q

# Run Phase 14 tests only
pytest tests/test_v3_11_0_project_scope.py -v
```

**Expected**: 694 pass, 0 fail

---

## Uncommitted Changes (as of 2026-02-18)

| File | Change |
|------|--------|
| `muninn/store/sqlite_metadata.py` | Fixed schema migration ordering bug (scope index) |
| `muninn/mcp/handlers.py` | Dynamic `serverInfo.version` from `muninn.version.__version__` |
| `muninn/mcp/lifecycle.py` | Renamed `GLOBAL_MEMORY_DIR` → `MUNINN_DIR` |
| `.gitignore` | Added `.muninn_token` exclusion |
| `HANDOFF.md` | Created this file |
