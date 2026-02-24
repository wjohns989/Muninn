# Muninn Memory Architecture: Post-Mem0 Native Framework

> Strategic design document synthesizing SOTA memory research, cross-domain novel methods,
> and xLAM skill enhancement findings into Muninn's next-generation architecture.

## Executive Summary

Muninn replaces the Mem0 dependency with a purpose-built, local-first memory framework that
borrows from neuroscience (complementary learning systems, sleep consolidation), information
theory (minimum description length), biological immune systems (clonal selection, danger theory),
and distributed systems (CRDTs, vector clocks). The result is a memory system that no existing
MCP memory server offers: **LLM-free extraction, importance-based consolidation, cross-assistant
sharing, and neuroscience-inspired memory dynamics** — all running 100% locally.

### Why Replace Mem0

| Limitation | Impact | Muninn Solution |
|---|---|---|
| LLM dependency for extraction | Slow, expensive, unreliable | xLAM chain-of-extraction + rule fallback |
| Cloud-first, API-gated features | Privacy risk, lock-in | 100% local with zero cloud calls |
| Bundled monolithic architecture | Can't swap components | Composable engine design |
| No importance scoring | All memories equal weight | Multi-factor importance formula |
| No consolidation/decay | Unbounded growth | Neuroscience-inspired lifecycle |
| Single-assistant design | No cross-agent sharing | Assistant-agnostic neutral storage |

### Competitive Landscape Position

```
                    LLM Required
                        ^
                        |
         Mem0 ●         |      ● Letta
                        |
         Zep ●          |      ● A-MEM
                        |
    ────────────────────+────────────────────> Feature Rich
                        |
     KuzuMem ●          |
                        |
    Memoripy ●          |      ★ MUNINN (target)
                        |
                    LLM Optional
```

Muninn targets the **bottom-right quadrant**: maximum features with minimal LLM dependency.

---

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP Interface Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ add_memory│  │ search   │  │ get_all  │  │ update   │     │
│  │          │  │ _memory  │  │ _memories│  │ _memory  │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │              │              │              │           │
├───────┴──────────────┴──────────────┴──────────────┴──────────┤
│                    Engine Layer                                │
│                                                               │
│  ┌─────────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │  Extraction      │  │  Retrieval     │  │ Consolidation  │  │
│  │  Engine          │  │  Engine        │  │ Engine         │  │
│  │                  │  │                │  │                │  │
│  │ • xLAM Chain-of- │  │ • Multi-vector │  │ • Importance   │  │
│  │   Extraction     │  │   (ColBERT)    │  │   scoring      │  │
│  │ • Rule-based     │  │ • Graph trav.  │  │ • Merge/prune  │  │
│  │   fallback       │  │ • Temporal     │  │ • Decay curves │  │
│  │ • Entity resolv. │  │ • BM25 keyword │  │ • Replay       │  │
│  │ • Relation trpl  │  │ • Reranking    │  │   (background) │  │
│  └────────┬─────────┘  └───────┬───────┘  └───────┬────────┘  │
│           │                    │                   │           │
├───────────┴────────────────────┴───────────────────┴──────────┤
│                  Unified Memory Store                          │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Kuzu Graph   │  │  Qdrant       │  │  SQLite          │    │
│  │  Database     │  │  Vectors      │  │  Metadata        │    │
│  │               │  │               │  │                  │    │
│  │ • Entities    │  │ • Embeddings  │  │ • Memory records │    │
│  │ • Relations   │  │ • ColBERT     │  │ • Importance     │    │
│  │ • Temporal    │  │   multi-vec   │  │ • Access history │    │
│  │   edges       │  │ • HNSW index  │  │ • Provenance     │    │
│  │ • Provenance  │  │               │  │ • Timestamps     │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
│                                                               │
│  Data Directory: ~/.muninn/data/                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Memory Type Hierarchy

Inspired by Complementary Learning Systems (CLS) theory from neuroscience:

```
Working Memory (ephemeral, session-scoped)
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

### Memory Record Schema (SQLite)

```sql
CREATE TABLE memories (
    id              TEXT PRIMARY KEY,      -- UUID
    content         TEXT NOT NULL,         -- Raw memory text (or transcription)
    memory_type     TEXT DEFAULT 'episodic', -- working|episodic|semantic|procedural
    media_type      TEXT DEFAULT 'text',   -- text|image|audio|video|sensor

    -- Importance Scoring (Multi-Factor)
    importance      REAL DEFAULT 0.5,      -- Composite score [0, 1]
    recency_score   REAL DEFAULT 1.0,      -- Exponential decay
    access_count    INTEGER DEFAULT 0,     -- Frequency tracking
    novelty_score   REAL DEFAULT 0.5,      -- Surprise vs existing knowledge

    -- Temporal (Bi-Temporal from Zep/Graphiti research)
    created_at      REAL NOT NULL,         -- Event timestamp
    ingested_at     REAL NOT NULL,         -- System timestamp
    last_accessed   REAL,                  -- For decay calculation
    expires_at      REAL,                  -- TTL for working memory

    -- Provenance (Cross-Assistant Tracking)
    source_agent    TEXT DEFAULT 'unknown', -- claude|gemini|codex|user
    project         TEXT DEFAULT 'global',  -- Git project name
    branch          TEXT,                   -- Git branch
    namespace       TEXT DEFAULT 'global',  -- Isolation boundary

    -- Embedding Reference
    vector_id       TEXT,                  -- Qdrant point ID
    embedding_model TEXT DEFAULT 'nomic-embed-text',

    -- Consolidation State
    consolidated    BOOLEAN DEFAULT FALSE, -- Has been merged/promoted
    parent_id       TEXT,                  -- Merged-from reference
    consolidation_gen INTEGER DEFAULT 0    -- Number of consolidation passes
);

CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_project ON memories(project);
CREATE INDEX idx_memories_namespace ON memories(namespace);
CREATE INDEX idx_memories_created ON memories(created_at DESC);
```

### Importance Scoring Formula

Adapted from neuroscience synaptic tagging and immune system danger signals:

```python
def calculate_importance(memory, existing_memories, context):
    """
    Multi-factor importance: combine recency, frequency, graph centrality,
    novelty, and provenance into a single [0,1] score.

    Inspired by:
    - Synaptic Tagging & Capture (STC) — neuroscience
    - Danger Theory — immune systems
    - PageRank — graph centrality
    """
    # 1. Recency Decay (exponential, half-life = 7 days)
    age_days = (now() - memory.created_at) / 86400
    recency = math.exp(-0.693 * age_days / 7.0)

    # 2. Access Frequency (log-scaled)
    frequency = math.log1p(memory.access_count) / math.log1p(100)

    # 3. Graph Centrality (degree centrality from Kuzu)
    centrality = get_entity_centrality(memory.id) if memory.has_entities else 0.0

    # 4. Novelty (1 - max_similarity to existing semantic memories)
    if existing_memories:
        max_sim = max(cosine_similarity(memory.embedding, m.embedding)
                      for m in existing_memories[:50])
        novelty = 1.0 - max_sim
    else:
        novelty = 1.0

    # 5. Provenance Weight (user-explicit > assistant-inferred)
    provenance = {
        'user_explicit': 1.0,
        'assistant_confirmed': 0.8,
        'auto_extracted': 0.5,
        'ingested': 0.3
    }.get(memory.provenance, 0.5)

    # Weighted combination
    importance = (
        0.25 * recency +
        0.15 * frequency +
        0.20 * centrality +
        0.25 * novelty +
        0.15 * provenance
    )

    return min(1.0, max(0.0, importance))
```

---

## Engine Specifications

### 1. Extraction Engine

Replaces Mem0's LLM-dependent extraction with a **tiered pipeline**:

```
Input Text
    │
    ├── Tier 1: Rule-Based (0ms latency, always available)
    │   ├── spaCy NER (if installed) or regex patterns
    │   ├── Date/time extraction (dateutil)
    │   ├── URL/email/path extraction
    │   └── Key-value pattern matching
    │
    ├── Tier 2: xLAM Chain-of-Extraction (~200ms, local GPU)
    │   ├── Step 1: get_entities(text) → entities[]
    │   ├── Step 2: search_relations(text, entities) → triples[]
    │   ├── Step 3: rank_results(entities, triples) → scored[]
    │   ├── Step 4: parse_dates(text) → temporal_context[]
    │   ├── Step 5: find_references(text, entities) → refs[]
    │   └── Step 6: summarize_text(text) → memory_summary
    │
    └── Tier 3: Ollama LLM (fallback, ~2s, optional)
        └── Full LLM extraction for complex/ambiguous text
```

#### xLAM Tool Schemas (PA-Tool Aligned)

Following the PA-Tool principle — rename tools to align with model pre-training patterns:

```python
XLAM_TOOLS = [
    {
        "name": "get_entities",
        "description": "Extract named entities from text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text"},
                "entity_type": {
                    "type": "string",
                    "enum": ["person", "org", "tech", "concept", "location",
                             "project", "file", "preference"]
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "search_relations",
        "description": "Find relationships between entities in text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "subject": {"type": "string"},
                "predicate": {
                    "type": "string",
                    "enum": ["uses", "prefers", "created", "depends_on",
                             "located_in", "works_with", "knows", "part_of"]
                },
                "object": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "rank_results",
        "description": "Score importance of extracted information",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "score": {"type": "integer", "description": "1-10 importance"},
                "reason": {"type": "string"}
            },
            "required": ["item", "score"]
        }
    },
    {
        "name": "parse_dates",
        "description": "Extract temporal information from text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "temporal_type": {
                    "type": "string",
                    "enum": ["deadline", "created", "updated", "scheduled",
                             "recurring", "relative"]
                }
            },
            "required": ["text"]
        }
    }
]
```

### 2. Retrieval Engine

Multi-signal retrieval combining 4 strategies:

```
Query
  │
  ├── Vector Search (Qdrant HNSW, cosine similarity)
  │   └── Top-K candidates by embedding distance
  │
  ├── Graph Traversal (Kuzu Cypher)
  │   └── Entity-linked memories via relationship paths
  │
  ├── Temporal Filtering (SQLite)
  │   └── Bi-temporal range queries (event_time, ingestion_time)
  │
  └── BM25 Keyword (in-memory inverted index)
      └── Exact term matching for precision
  │
  ▼
  Fusion (Reciprocal Rank Fusion)
  │
  ▼
  Reranking (Jina Tiny cross-encoder)
  │
  ▼
  Importance-Weighted Results
```

#### Retrieval Scoring

```python
def hybrid_retrieve(query, limit=10, filters=None):
    """
    Reciprocal Rank Fusion across 4 retrieval signals.

    Inspired by:
    - ColBERT late interaction for fine-grained matching
    - Reciprocal Rank Fusion for multi-signal combination
    - Kalman filtering for adaptive signal weighting
    """
    k = 60  # RRF constant

    # Parallel retrieval (all independent)
    vector_results = qdrant_search(query, limit=limit*3)
    graph_results = kuzu_entity_search(query, limit=limit*2)
    temporal_results = sqlite_temporal_search(query, filters)
    bm25_results = bm25_keyword_search(query, limit=limit*2)

    # Reciprocal Rank Fusion
    scores = defaultdict(float)
    for rank, result in enumerate(vector_results):
        scores[result.id] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(graph_results):
        scores[result.id] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(temporal_results):
        scores[result.id] += 0.5 / (k + rank + 1)  # Lower weight
    for rank, result in enumerate(bm25_results):
        scores[result.id] += 0.8 / (k + rank + 1)

    # Apply importance weighting
    for mem_id, rrf_score in scores.items():
        importance = get_importance(mem_id)
        scores[mem_id] = rrf_score * (0.7 + 0.3 * importance)

    # Sort and rerank top candidates
    candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit*2]

    if reranker:
        reranked = reranker.rerank(query, [get_content(c[0]) for c in candidates])
        return reranked[:limit]

    return candidates[:limit]
```

### 3. Consolidation Engine

Background daemon inspired by neuroscience sleep consolidation and immune system memory maturation:

```
┌─────────────────────────────────────────────────────┐
│              Consolidation Cycle (every 6h)          │
│                                                      │
│  Phase 1: DECAY                                      │
│  ├── Calculate importance for all memories           │
│  ├── Soft-delete memories below threshold (< 0.1)    │
│  └── Demote working → expired if TTL exceeded        │
│                                                      │
│  Phase 2: MERGE                                      │
│  ├── Find near-duplicate episodic memories           │
│  │   (cosine_sim > 0.92)                            │
│  ├── Merge content, combine metadata                 │
│  ├── Preserve highest importance score               │
│  └── Link merged → parent via parent_id              │
│                                                      │
│  Phase 3: PROMOTE                                    │
│  ├── Episodic memories accessed 5+ times             │
│  │   → Promote to semantic memory                    │
│  ├── Semantic memories with stable patterns           │
│  │   → Promote to procedural memory                  │
│  └── Extract and store distilled knowledge            │
│                                                      │
│  Phase 4: REPLAY (Hippocampal Replay)                │
│  ├── Re-embed high-importance memories               │
│  │   with latest model                               │
│  ├── Re-index graph relationships                    │
│  └── Update centrality scores                        │
│                                                      │
│  Phase 5: STATISTICS                                 │
│  ├── Update memory count metrics                     │
│  ├── Log consolidation report                        │
│  └── Write checkpoint to SQLite                      │
└─────────────────────────────────────────────────────┘
```

#### Consolidation Algorithm

```python
async def consolidation_cycle():
    """
    Neuroscience-inspired memory consolidation.

    References:
    - Complementary Learning Systems (McClelland et al., 1995)
    - Sleep replay consolidation (Walker & Stickgold, 2006)
    - Synaptic homeostasis hypothesis (Tononi & Cirelli, 2003)
    """
    logger.info("Starting consolidation cycle...")

    # Phase 1: Decay
    all_memories = get_all_memories_with_scores()
    for mem in all_memories:
        new_importance = calculate_importance(mem)
        update_importance(mem.id, new_importance)

        if new_importance < DECAY_THRESHOLD:  # 0.1
            soft_delete(mem.id)
            logger.debug(f"Decayed memory {mem.id[:8]} (importance={new_importance:.3f})")

    # Phase 2: Merge near-duplicates
    embeddings = get_all_embeddings()
    similarity_matrix = cosine_similarity_batch(embeddings)
    merge_pairs = find_pairs_above_threshold(similarity_matrix, threshold=0.92)

    for id_a, id_b in merge_pairs:
        merged = merge_memories(id_a, id_b)
        logger.debug(f"Merged {id_a[:8]} + {id_b[:8]} → {merged.id[:8]}")

    # Phase 3: Promote based on access patterns
    frequent_episodic = get_memories(
        memory_type='episodic',
        min_access_count=5,
        consolidated=False
    )
    for mem in frequent_episodic:
        promoted = promote_to_semantic(mem)
        logger.debug(f"Promoted {mem.id[:8]} to semantic")

    # Phase 4: Replay (re-embed high-importance memories)
    high_importance = get_memories(importance_min=0.8, limit=50)
    for mem in high_importance:
        new_embedding = embed(mem.content)
        update_embedding(mem.id, new_embedding)

    logger.info(f"Consolidation complete: {len(merge_pairs)} merges, "
                f"{len(frequent_episodic)} promotions, "
                f"{len(high_importance)} replays")
```

---

## Novel Cross-Domain Techniques

### From Neuroscience

| Technique | Inspiration | Application |
|---|---|---|
| Complementary Learning Systems | Hippocampus (fast) + Neocortex (slow) | Working memory (fast capture) → Semantic memory (slow consolidation) |
| Sleep Replay | Memory reactivation during sleep | Background re-embedding and re-indexing cycle |
| Synaptic Tagging | Tag important synapses for consolidation | Importance scoring with multi-factor formula |
| Sparse Distributed Coding | Brain uses sparse activation patterns | Sparse embedding representations for efficiency |
| Predictive Coding | Brain predicts and learns from errors | Novelty = prediction error → higher importance |

### From Immune Systems

| Technique | Inspiration | Application |
|---|---|---|
| Clonal Selection | B-cells amplify useful antibodies | Frequently accessed memories get replicated/promoted |
| Danger Theory | Immune response to actual threats | Provenance-weighted importance (user explicit > auto) |
| T-cell / B-cell Tiers | Fast response + deep memory | Working memory (fast) + Semantic memory (deep) |
| Affinity Maturation | Antibodies improve over generations | Consolidation generations improve memory quality |

### From Information Theory

| Technique | Inspiration | Application |
|---|---|---|
| Minimum Description Length | Compress knowledge optimally | Merge near-duplicates to minimize storage |
| Rate-Distortion | Trade accuracy for compression | Configurable consolidation aggressiveness |
| Kolmogorov Complexity | Algorithmic information content | Novelty scoring based on compressibility |

### From Distributed Systems

| Technique | Inspiration | Application |
|---|---|---|
| CRDTs | Conflict-free replicated data types | Future: multi-device memory sync without conflicts |
| Vector Clocks | Causal ordering of events | Bi-temporal timestamps for event causality |
| Merkle DAGs | Content-addressed storage | Deduplication via content hashing |

---

## Implementation Phases

### Phase 1: Core Foundation (Current Sprint)

**Goal**: Replace Mem0 with native Muninn memory store, maintain full MCP compatibility.

```
Priority: CRITICAL
Timeline: 2-3 weeks
Dependencies: None (builds on existing Qdrant + Kuzu)

Deliverables:
├── muninn/store/sqlite_metadata.py    # Memory record CRUD
├── muninn/store/vector_store.py       # Qdrant wrapper (keep existing)
├── muninn/store/graph_store.py        # Kuzu wrapper (keep existing)
├── muninn/extraction/rules.py         # Rule-based entity extraction
├── muninn/extraction/pipeline.py      # Extraction orchestrator
├── muninn/retrieval/hybrid.py         # Multi-signal retrieval + RRF
├── muninn/retrieval/reranker.py       # Jina Tiny reranker wrapper
├── muninn/core/memory.py              # Main Memory class (replaces Mem0)
├── muninn/core/types.py               # Pydantic models
└── server.py                          # Updated to use muninn.core.memory
```

**Migration Path**:
1. Create `muninn/` package alongside existing `server.py`
2. Implement SQLite metadata store with schema above
3. Wrap existing Qdrant operations in `VectorStore` class
4. Wrap existing Kuzu operations in `GraphStore` class
5. Build `Memory` class that composes all stores
6. Update `server.py` endpoints to use new `Memory` class
7. Remove `mem0` dependency from requirements
8. Run migration script to transfer existing Mem0 data → Muninn native format

### Phase 2: Intelligence Layer (Next Sprint)

**Goal**: Add xLAM chain-of-extraction and importance scoring.

```
Priority: HIGH
Timeline: 2-3 weeks
Dependencies: Phase 1 complete

Deliverables:
├── muninn/extraction/xlam.py          # xLAM tool-calling pipeline
├── muninn/extraction/schemas.py       # PA-Tool aligned tool schemas
├── muninn/extraction/resolver.py      # Entity resolution + dedup
├── muninn/scoring/importance.py       # Multi-factor importance
├── muninn/scoring/novelty.py          # Novelty detection
└── muninn/scoring/decay.py            # Temporal decay curves
```

### Phase 3: Consolidation Engine (Following Sprint)

**Goal**: Background memory lifecycle management.

```
Priority: MEDIUM-HIGH
Timeline: 2 weeks
Dependencies: Phase 2 complete

Deliverables:
├── muninn/consolidation/daemon.py     # Background consolidation loop
├── muninn/consolidation/merge.py      # Near-duplicate merging
├── muninn/consolidation/promote.py    # Memory type promotion
├── muninn/consolidation/decay.py      # Soft deletion / archival
├── muninn/consolidation/replay.py     # Re-embedding high-value memories
└── muninn/consolidation/stats.py      # Health metrics + reporting
```

### Phase 4: Differentiation Features

**Goal**: Features no other MCP memory server offers.

```
Priority: MEDIUM
Timeline: Ongoing
Dependencies: Phase 3 complete

Deliverables:
├── muninn/advanced/colbert.py         # Multi-vector ColBERT embeddings
├── muninn/advanced/temporal_kg.py     # Bi-temporal knowledge graph edges
├── muninn/advanced/predictive.py      # Predictive pre-fetch cache  
├── muninn/advanced/cross_agent.py     # Cross-assistant Hive Mind synchronization
└── muninn/dashboard/                  # Enhanced web dashboard      
```

---

## Multimodal Hive Mind (Phase 20)

Muninn extends the unified embedding space to support cross-assistant shared multimodal memory.

### Multimodal Ingestion
Images and audio are automatically processed during ingestion:
- **Audio**: Transcribed via `AudioAdapter` using Whisper-compatible endpoints.
- **Vision**: Described via `VisionAdapter` using VLMs (Ollama LLaVA).
- **Sensor**: Native support for structured sensor data strings.

### Hive Mind Federation
A push-based Merkle-DAG inspired sync protocol reconciles states between disparate runtimes:
- **Low-Latency Sync**: `sync_on_add` triggers immediate broadcast to peer instances.
- **Delta Bundles**: Efficient reconciliation via lightweight sync manifests.
- **User Scoping**: Multi-tenant isolation preserved across federated instances.

---

## Package Structure (Target)

```
muninn_mcp/
├── muninn/                        # Core framework package
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory.py              # Main Memory class
│   │   ├── types.py               # Pydantic models & enums
│   │   └── config.py              # Configuration management
│   ├── store/
│   │   ├── __init__.py
│   │   ├── sqlite_metadata.py     # SQLite memory records
│   │   ├── vector_store.py        # Qdrant operations
│   │   └── graph_store.py         # Kuzu operations
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Extraction orchestrator
│   │   ├── rules.py               # Rule-based extraction
│   │   ├── xlam.py                # xLAM chain-of-extraction
│   │   ├── schemas.py             # PA-Tool aligned schemas
│   │   ├── vision_adapter.py      # VLM image description
│   │   ├── audio_adapter.py       # STT audio transcription
│   │   └── resolver.py            # Entity resolution
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py              # Multi-signal retrieval + RRF
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   └── bm25.py                # In-memory BM25 index
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── importance.py          # Multi-factor importance
│   │   ├── novelty.py             # Novelty detection
│   │   └── decay.py               # Temporal decay curves
│   └── consolidation/
│       ├── __init__.py
│       ├── daemon.py              # Background consolidation loop
│       ├── merge.py               # Near-duplicate merging
│       ├── promote.py             # Memory type promotion
│       └── replay.py              # Re-embedding cycle
├── server.py                      # FastAPI server (uses muninn/)
├── mcp_wrapper.py                 # MCP stdio bridge
├── tray_app.py                    # Windows system tray
├── dashboard.html                 # Web dashboard
├── ingest_history.py              # History import tool
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Dependencies
├── LICENSE                        # Apache-2.0
├── README.md                      # Documentation
└── docs/
    ├── ARCHITECTURE.md            # This document
    └── MIGRATION.md               # Mem0 → Muninn migration guide
```

---

## Key Differentiators vs Competition

| Feature | Mem0 | Zep | Letta | A-MEM | **Muninn** |
|---|---|---|---|---|---|
| LLM-free extraction | No | No | No | No | **Yes** (xLAM + rules) |
| Importance scoring | No | Partial | No | Partial | **Yes** (multi-factor) |
| Memory consolidation | No | No | No | No | **Yes** (neuro-inspired) |
| Memory type hierarchy | No | No | Partial | Yes | **Yes** (4-tier CLS) |
| Cross-assistant | No | No | No | No | **Yes** (agent-agnostic) |
| 100% local | Partial | No | Partial | Yes | **Yes** (zero cloud) |
| Graph-enhanced | Partial | Yes | No | Partial | **Yes** (Kuzu native) |
| Bi-temporal | No | Yes | No | No | **Yes** (event + system) |
| Background consolidation | No | No | No | No | **Yes** (daemon) |
| Novelty detection | No | No | No | Partial | **Yes** (info theory) |

---

## Dependencies (Post-Mem0)

### Required
- `fastapi` + `uvicorn` — HTTP server
- `qdrant-client` — Vector storage
- `kuzu` — Graph database
- `fastembed` — Embeddings + reranking
- `pydantic` — Data validation

### Optional
- `llama-cpp-python` — xLAM local inference (Tier 2 extraction)
- `spacy` — Enhanced NER (Tier 1 extraction)
- `pystray` + `Pillow` — System tray (Windows)
- `pywin32` — Windows service wrapper

### Removed
- ~~`mem0ai`~~ — Replaced by `muninn.core.memory`

---

## Research Sources

### Memory Architecture
- McClelland et al. (1995) — Complementary Learning Systems
- Walker & Stickgold (2006) — Sleep-Dependent Memory Consolidation
- Tononi & Cirelli (2003) — Synaptic Homeostasis Hypothesis
- Khattab & Zaharia (2020) — ColBERT: Efficient Passage Retrieval

### Scoring & Retrieval
- Frey & Durstewitz (1997) — Synaptic Tagging & Capture
- Croft et al. (2009) — Reciprocal Rank Fusion
- Robertson & Zaragoza (2009) — BM25 Probabilistic Retrieval

### Cross-Domain Adaptations
- Matzinger (2002) — Danger Theory (Immunology)
- de Castro & Timmis (2002) — Artificial Immune Systems
- Shapiro et al. (2011) — CRDTs (Distributed Systems)
- Rissanen (1978) — Minimum Description Length (Information Theory)

### xLAM & Extraction
- Zhang et al. (2024) — xLAM: Tool-Use Language Models
- Liu et al. (2024) — PA-Tool: Schema Alignment for Small LMs
- Salesforce Research (2024) — APIGen-MT Training