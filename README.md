# Muninn: Persistent Memory for AI Agents

```text
███╗   ███╗██╗   ██╗███╗   ██╗██╗███╗   ██╗███╗   ██╗
████╗ ████║██║   ██║████╗  ██║██║████╗  ██║████╗  ██║
██╔████╔██║██║   ██║██╔██╗ ██║██║██╔██╗ ██║██╔██╗ ██║
██║╚██╔╝██║██║   ██║██║╚██╗██║██║██║╚██╗██║██║╚██╗██║
██║ ╚═╝ ██║╚██████╔╝██║ ╚████║██║██║ ╚████║██║ ╚████║
╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝
     ᛗ The Persistent Memory MCP ᚱ
```

Muninn is a **local-first, assistant-agnostic memory layer** for AI agents. It decouples memory from any single assistant (Claude, Gemini, Codex, and others) and provides persistent, searchable memory via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

> **v3.0 — Muninn Native Engine**: Fully replaces the Mem0 dependency with a purpose-built memory framework featuring neuroscience-inspired consolidation, multi-factor importance scoring, and LLM-free extraction.

## Key Features

- **Assistant Agnostic**: Works with Claude Code, Gemini CLI, Codex CLI, Claude Desktop, and any MCP-compatible client
- **100% Local**: Qdrant vectors + Kuzu graph + SQLite metadata. Zero cloud calls, zero data leaves your machine
- **Hybrid Search**: Vector similarity + BM25 keyword search + graph traversal with Reciprocal Rank Fusion and cross-encoder reranking
- **Memory Lifecycle**: Neuroscience-inspired 4-tier memory hierarchy (Working → Episodic → Semantic → Procedural) with automatic consolidation
- **Importance Scoring**: Multi-factor importance combining recency decay, access frequency, graph centrality, novelty, and provenance
- **LLM-Free Extraction**: Rule-based entity/relation extraction (Tier 1) with optional xLAM chain-of-extraction (Tier 2) and Ollama fallback (Tier 3)
- **Background Consolidation**: Daemon that decays, merges, promotes, and replays memories on a configurable schedule
- **Silent Operation**: Runs invisibly in the background with optional Windows system tray control

## Architecture

```
muninn_mcp/
├── muninn/                    # Core framework package
│   ├── core/                  # Memory engine, types, config
│   │   ├── memory.py          # Main MuninnMemory class
│   │   ├── types.py           # Pydantic models & enums
│   │   └── config.py          # Centralized configuration
│   ├── store/                 # Storage backends
│   │   ├── vector_store.py    # Qdrant HNSW vectors
│   │   ├── graph_store.py     # Kuzu knowledge graph
│   │   └── sqlite_metadata.py # SQLite memory records
│   ├── extraction/            # Entity & relation extraction
│   │   ├── rules.py           # Rule-based (zero-latency)
│   │   └── pipeline.py        # Tiered extraction orchestrator
│   ├── retrieval/             # Multi-signal search
│   │   ├── hybrid.py          # RRF fusion retriever
│   │   ├── bm25.py            # In-memory BM25 index
│   │   └── reranker.py        # Jina cross-encoder reranker
│   ├── scoring/               # Importance scoring
│   │   └── importance.py      # Multi-factor importance formula
│   └── consolidation/         # Background memory lifecycle
│       ├── daemon.py          # Consolidation loop
│       ├── merge.py           # Near-duplicate merging
│       └── promote.py         # Memory type promotion
├── server.py                  # FastAPI backend
├── mcp_wrapper.py             # MCP stdio bridge (auto-starts server)
├── tray_app.py                # Windows system tray application
├── dashboard.html             # Web dashboard for memory visualization
├── ingest_history.py          # History ingestion from multiple assistants
├── pyproject.toml             # Package metadata
└── requirements.txt           # Dependencies
```

**Data Directory**: `~/.muninn/data/` (platform-independent, user-level)
**Server**: `http://localhost:42069` (FastAPI)
**Embeddings**: Ollama `nomic-embed-text` (768-dim, local)

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with `nomic-embed-text` model pulled
- (Optional) xLAM function-calling model for advanced entity extraction

### Install

```bash
git clone https://github.com/yourusername/muninn_mcp.git
cd muninn_mcp
pip install -r requirements.txt
```

### Pull Embedding Model

```bash
ollama pull nomic-embed-text
```

## Configuration

### Claude Code

```bash
claude mcp add muninn -s user -- python /path/to/muninn_mcp/mcp_wrapper.py
```

### Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "muninn": {
      "command": "python",
      "args": ["/path/to/muninn_mcp/mcp_wrapper.py"]
    }
  }
}
```

### Gemini CLI (`.gemini/settings.json`)

```json
{
  "mcpServers": {
    "muninn": {
      "command": "python",
      "args": ["/path/to/muninn_mcp/mcp_wrapper.py"]
    }
  }
}
```

### Codex CLI (`.codex/config.toml`)

```toml
[mcp_servers.muninn]
command = "python"
args = ["/path/to/muninn_mcp/mcp_wrapper.py"]
```

> **Note:** Replace `/path/to/muninn_mcp/` with the actual path to your installation. On Windows, use the full path to your Python executable (e.g., `C:\\Python313\\python.exe`).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MUNINN_DATA_DIR` | `~/.muninn/data` | Base data directory |
| `MUNINN_HOST` | `127.0.0.1` | Server bind address |
| `MUNINN_PORT` | `42069` | Server port |
| `MUNINN_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `MUNINN_EMBEDDING_DIMS` | `768` | Embedding dimensions |
| `MUNINN_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `MUNINN_RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `MUNINN_CONSOLIDATION_ENABLED` | `true` | Enable background consolidation |
| `MUNINN_SERVER_URL` | `http://localhost:42069` | Server URL (for mcp_wrapper) |

## MCP Tools

| Tool | Description |
|------|-------------|
| `add_memory` | Store a new memory with optional metadata |
| `search_memory` | Hybrid search with reranking |
| `get_all_memories` | Retrieve all stored memories |
| `update_memory` | Update an existing memory by ID |
| `delete_memory` | Delete a specific memory by ID |
| `delete_all_memories` | Delete all memories (requires confirmation) |

## Memory Type Hierarchy

Inspired by Complementary Learning Systems (CLS) theory from neuroscience:

```
Working Memory (ephemeral, session-scoped, 24h TTL)
    │
    │  ── promotion via importance threshold ──>
    │
Episodic Memory (specific events, conversations, decisions)
    │
    │  ── consolidation via pattern extraction ──>
    │
Semantic Memory (distilled facts, preferences, knowledge)
    │
    │  ── formalization via repeated access ──>
    │
Procedural Memory (workflows, tool usage patterns, habits)
```

## Dashboard

Visit `http://localhost:42069` when the server is running to visualize your memory graph and browse stored memories.

## History Ingestion

Import existing conversation history from your assistants:

```bash
# Dry run to see what would be ingested
python ingest_history.py --dry-run --agent all

# Ingest from a specific assistant
python ingest_history.py --agent claude
python ingest_history.py --agent codex
python ingest_history.py --agent antigravity
```

## Testing

```bash
python -m pytest tests/ -v
```

## Documentation

- [Architecture Deep Dive](docs/ARCHITECTURE.md) — Full design document with research references
- [Citations](CITATIONS.md) — Academic sources and cross-domain inspiration

## License

Apache License 2.0
