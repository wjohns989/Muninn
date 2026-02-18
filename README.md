# Muninn

Local-first persistent memory infrastructure for coding agents and MCP-compatible tools.

Muninn provides deterministic, explainable memory retrieval with robust transport behavior and production-grade operational controls. It is designed for long-running development workflows where continuity, auditability, and measurable quality matter.

## 🚦 Status
**Current Version:** v3.9.0 (Phase 12 Complete)
**Stability:** Production Beta

### 🆕 New in v3.9.0
- **Distributed Entity Scoping:** Composite `user_id/namespace/name` entity IDs for multi-tenant isolation.
- **Unified Security:** Centralized `muninn.core.security` module with FastAPI and MCP parity.
- **Multi-Namespace Integrity:** Consolidation daemon enforces user and namespace boundaries.
- **Federation Scoping:** All federation endpoints enforce user-level access control.

## 🚀 Features
- **Local-First:** Zero cloud dependency.

## Runtime Modes

- **Huginn mode**: browser-first standalone UX for direct ingestion/search/admin workflows.
- **Muninn mode**: MCP wrapper surface for active assistant/IDE sessions.
- Both modes use the same underlying Muninn memory engine and data.

## Why Muninn

- Local-first data residency by default
- Cross-session memory continuity
- Explainable retrieval traces with per-signal attribution
- Multi-signal retrieval with configurable fusion and reranking
- Memory lifecycle operations (decay, merge, promotion, replay)
- Deterministic ingestion with provenance-aware metadata
- MCP bridge plus REST and Python SDK surfaces

## Core Capabilities

1. Retrieval engine:
- Dense similarity
- Lexical retrieval
- Graph traversal
- Temporal relevance
- Goal relevance and chain expansion (feature-gated)

2. Memory integrity:
- Conflict detection (feature-gated)
- Near-duplicate suppression (feature-gated)
- Adaptive weighting from feedback signals (feature-gated)

3. Operations:
- Background consolidation daemon
- Runtime profile policy controls and audit events
- Editable user profile/global context for skills, paths, environment, and hardware hints
- Transport hardening for framed/line JSON-RPC and timeout-window guardrails
- Browser control center for ingestion/search/admin flows

## Quick Start

```bash
git clone https://github.com/wjohns989/Muninn.git
cd Muninn
pip install -e .
```

Run the backend service:

```bash
python server.py
```

Run the standalone browser-first launcher (Huginn mode):

```bash
python muninn_standalone.py
```

Run the MCP wrapper (Muninn MCP mode for active assistants):

```bash
python mcp_wrapper.py
```

Build a standalone executable package (PyInstaller):

```bash
python scripts/build_standalone.py --name HuginnControlCenter --windowed
```

## Minimal MCP Client Configuration

Use any MCP-compatible client by launching the wrapper as a stdio server:

```json
{
  "mcpServers": {
    "muninn": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_wrapper.py"]
    }
  }
}
```

## Python SDK

```python
from muninn import Memory

client = Memory(base_url="http://127.0.0.1:42069")
client.add(content="Use deterministic gates for release checks", metadata={"project": "muninn"})
results = client.search("release checks", limit=5)
print(results)
```

## REST Surface

Primary endpoints:

- `GET /health`
- `POST /add`
- `POST /search`
- `GET /get_all`
- `PUT /update`
- `DELETE /delete/{memory_id}`
- `POST /ingest`
- `POST /ingest/legacy/discover`
- `POST /ingest/legacy/import`
- `GET /profiles/model`
- `POST /profiles/model`
- `GET /profiles/model/events`
- `GET /profile/user/get`
- `POST /profile/user/set`

## Evaluation and SOTA+ Readiness

Muninn includes an evaluation toolchain to enforce measurable quality before promotion:

- retrieval metrics (`nDCG@k`, `Recall@k`, `MRR`)
- latency budgets (p50/p95)
- policy gates with significance-aware checks
- artifact integrity and reproducibility verification
- hygiene gates for branch/PR/test discipline

Reference plans:

- `SOTA_PLUS_PLAN.md`
- `docs/MUNINN_COMPREHENSIVE_ROADMAP.md`
- `docs/plans/`

## Agent Continuation

If a coding session is interrupted and another agent needs to resume implementation, use:

- `docs/AGENT_CONTINUATION_RUNBOOK.md`

## Data and Security Posture

- Default local data directory: `~/.muninn/data`
- No mandatory cloud dependency
- Optional observability is opt-in and privacy-bounded
- Transport and retry logic designed to fail fast with explicit error semantics

## Licensing

- Code license: Apache License 2.0 (`LICENSE`)
- Third-party dependency licenses remain with their respective owners
- No trademark endorsement is claimed by this project
- Third-party names are used only for factual interoperability references

## Documentation Index

- `docs/AGENT_CONTINUATION_RUNBOOK.md`
- `docs/ARCHITECTURE.md`
- `docs/PLAN_GAP_EVALUATION.md`
- `docs/WEB_RESEARCH_VIBECODER_SOTA.md`
- `docs/PYTHON_SDK.md`
- `docs/INGESTION_PIPELINE.md`
- `docs/OTEL_GENAI_OBSERVABILITY.md`