# Muninn SOTA+ Implementation Plan

> **Version**: v3.1.0 → v3.3.0 Roadmap
> **Status**: Research Complete, Ready for Implementation
> **Estimated Effort**: 22–32 developer-days across 3 phases
> **License Constraint**: Apache-2.0 — all dependencies verified compatible
> **Backward Compatibility**: 100% — all enhancements are additive & optional

---

## Executive Summary

This plan advances Muninn from v3.0 (the most technically complete local-first MCP memory server) to v3.3.0 (definitively SOTA+ in the MCP memory category) by addressing 4 identified gaps and implementing 5 advancement features across 3 phased releases.

**Gaps Addressed:**
1. LLM extraction less nuanced than cloud competitors
2. No Python SDK for programmatic use
3. Windows-centric deployment (excludes 70%+ of developers)
4. No multi-source ingestion (files, conversations, APIs)

**Advancements Implemented:**
5. Explainable recall traces (UNIQUE — no competitor has this)
6. Adaptive retrieval weights (entropy-based dynamic fusion)
7. NLI-based conflict detection (UNIQUE — no competitor has this)
8. Memory chains with temporal/causal linking
9. Semantic deduplication at ingestion + consolidation

**After all features, Muninn will be the ONLY memory system that combines:**
- Local-first architecture (no cloud dependency)
- 4-signal hybrid retrieval with adaptive weights
- Explainable recall traces with per-signal attribution
- NLI-based conflict detection for memory integrity
- Structured extraction via Instructor (matches Mem0 quality)
- Memory chains with temporal/causal linking
- Semantic deduplication at ingestion + consolidation
- Cross-platform (Windows/Linux/macOS/Docker)
- Multi-source ingestion (files, conversations, APIs)
- Python SDK + REST API + MCP protocol
- Background consolidation daemon (neuroscience-inspired)
- 4-tier memory hierarchy with promotion
- Bi-temporal provenance tracking

---

## Phase 1: Foundation (v3.1.0)

> **Priority**: 🔴 HIGHEST
> **Risk**: LOW
> **Effort**: 6–9 days
> **Theme**: Unblock users, improve quality, create unique differentiator

### 1A. Platform Abstraction

**Problem**: Muninn is Windows-centric. pywin32/pystray optional deps, some hardcoded paths, subprocess flags are Windows-only. This excludes Linux, macOS, and Docker users (70%+ of developer market).

**Solution**: Abstract all platform-specific code behind `muninn/platform.py` using `platformdirs` library.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/platform.py` | Cross-platform path resolution, process flags, detection utilities |
| `Dockerfile` | Container deployment image |
| `docker-compose.yml` | Full stack: Muninn + Ollama |
| `tests/test_platform.py` | Platform detection and path tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/config.py` | Use `platform.get_data_dir()` for default paths |
| `mcp_wrapper.py` | Use `platform.get_process_creation_flags()` instead of Windows-specific flags |
| `pyproject.toml` | Add `platformdirs` dependency (MIT license) |

**Technical Design:**

```python
# muninn/platform.py
import os, sys
from pathlib import Path

try:
    from platformdirs import user_data_dir, user_config_dir, user_log_dir
except ImportError:
    def user_data_dir(appname, appauthor=None):
        if sys.platform == "win32":
            return str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / appname)
        elif sys.platform == "darwin":
            return str(Path.home() / "Library" / "Application Support" / appname)
        else:
            return str(Path(os.environ.get("XDG_DATA_HOME",
                           Path.home() / ".local" / "share")) / appname)

def get_data_dir() -> Path:
    return Path(os.environ.get("MUNINN_DATA_DIR", user_data_dir("muninn")))

def get_process_creation_flags() -> int:
    if sys.platform == "win32":
        import subprocess
        return subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    return 0

def is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("MUNINN_DOCKER") == "1"
```

**Docker Support:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[all]"
ENV MUNINN_DATA_DIR=/data MUNINN_HOST=0.0.0.0
EXPOSE 42069
VOLUME /data
CMD ["python", "-m", "muninn.server"]
```

**Dependency**: `platformdirs` — MIT License ✅

---

### 1B. Extraction Enhancement (Instructor Integration)

**Problem**: Current xLAM extraction uses raw prompt → JSON parse, which is fragile and unvalidated. Ollama extraction is unimplemented. No structured output guarantees.

**Solution**: Replace fragile JSON parsing with the `instructor` library, which uses Pydantic models to guarantee structured output with automatic retry. Works with ANY OpenAI-compatible endpoint (xLAM, Ollama, vLLM, LM Studio, cloud).

**New Files:**
| File | Purpose |
|---|---|
| `muninn/extraction/models.py` | Pydantic schemas for structured extraction |
| `muninn/extraction/instructor_extractor.py` | Instructor-based extractor |
| `tests/test_instructor_extractor.py` | Extraction quality tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/extraction/pipeline.py` | Integrate Instructor as Tier 2, improve routing |
| `muninn/core/config.py` | Add `extraction.provider` config (instructor/xlam/ollama) |
| `pyproject.toml` | Add `instructor` dependency (MIT license) |

**Technical Design:**

```python
# muninn/extraction/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name, properly capitalized")
    entity_type: str = Field(description="One of: person|org|tech|concept|project|file|preference|location")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

class ExtractedRelation(BaseModel):
    subject: str = Field(description="Source entity name")
    predicate: str = Field(description="Relationship: uses|prefers|created|depends_on|knows|part_of|works_with")
    object: str = Field(description="Target entity name")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    temporal_context: Optional[str] = Field(default=None)

class ExtractedMemoryFacts(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list, description="Atomic factual statements")
    summary: Optional[str] = Field(default=None, description="One-sentence summary")
    temporal_context: Optional[str] = Field(default=None)
```

```python
# muninn/extraction/instructor_extractor.py
import instructor
from openai import OpenAI
from .models import ExtractedMemoryFacts

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine. Given text from a conversation
or document, extract structured facts including entities (people, technologies, organizations,
concepts, projects, preferences), relationships between entities, key factual statements,
and a one-sentence summary. Be precise and factual. Only extract what is explicitly stated."""

class InstructorExtractor:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        self.client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=api_key),
            mode=instructor.Mode.JSON
        )
        self.model = model

    def extract(self, text: str) -> ExtractedMemoryFacts:
        return self.client.chat.completions.create(
            model=self.model,
            response_model=ExtractedMemoryFacts,
            max_retries=2,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract structured facts:\n\n{text}"}
            ],
        )
```

**Routing Logic (Enhanced Pipeline):**
```
Tier 1: Rules (always, ~0ms) — guaranteed baseline
Tier 2: Instructor (if endpoint configured, ~200ms–2s) — structured, validated
Merge: Union entities/relations, deduplicate by lowercase name
```

**Key Insight**: Instructor eliminates the need for SEPARATE xLAM and Ollama extractors. One extractor handles ALL backends because they all speak the OpenAI API format. This simplifies the codebase while dramatically improving quality.

**Dependency**: `instructor` — MIT License ✅

---

### 1C. Explainable Recall Traces

**Problem**: search() returns opaque combined scores. Users/agents have no idea WHY a particular memory was recalled, making debugging and trust impossible.

**Solution**: Track per-signal contributions through RRF fusion and return detailed `RecallTrace` objects explaining each retrieval decision.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/core/recall_trace.py` | RecallTrace, SignalContribution dataclasses |
| `tests/test_recall_trace.py` | Trace generation and explanation tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/types.py` | Import and re-export RecallTrace types |
| `muninn/retrieval/hybrid.py` | Track per-signal scores, generate traces |
| `server.py` | Include trace in search response when requested |
| `mcp_wrapper.py` | Format trace in search results |

**Technical Design:**

```python
# muninn/core/recall_trace.py
from pydantic import BaseModel
from typing import List, Optional

class SignalContribution(BaseModel):
    signal: str                  # "vector"|"bm25"|"graph"|"temporal"
    raw_score: float             # Original score from that signal
    rank: int                    # Rank position in that signal's results
    rrf_contribution: float      # Actual RRF score contribution
    weight: float                # Weight applied to this signal
    explanation: str             # Human-readable explanation

class RecallTrace(BaseModel):
    memory_id: str
    final_score: float
    signals: List[SignalContribution]
    rerank_score: Optional[float] = None
    importance_boost: float = 0.0
    dominant_signal: str         # Which signal contributed most
    explanation: str             # Full human-readable explanation
```

**Implementation in HybridRetriever:**

```python
def _rrf_fusion_with_traces(self, signal_results):
    """Modified RRF that tracks attribution per signal per document"""
    traces = {}
    for signal_name, results in signal_results.items():
        weight = self.weights[signal_name]
        for rank, (memory_id, raw_score) in enumerate(results):
            if memory_id not in traces:
                traces[memory_id] = RecallTrace(memory_id=memory_id, signals=[], ...)
            rrf_contrib = weight / (RRF_K + rank + 1)
            traces[memory_id].signals.append(SignalContribution(
                signal=signal_name, raw_score=raw_score, rank=rank,
                rrf_contribution=rrf_contrib, weight=weight,
                explanation=self._explain_signal(signal_name, raw_score, rank)
            ))
    for trace in traces.values():
        trace.final_score = sum(s.rrf_contribution for s in trace.signals)
        trace.dominant_signal = max(trace.signals, key=lambda s: s.rrf_contribution).signal
        trace.explanation = self._generate_explanation(trace)
    return traces
```

**Example Output:**
```json
{
  "memory": {"id": "abc123", "content": "User prefers Python for backend..."},
  "score": 0.847,
  "trace": {
    "dominant_signal": "vector",
    "explanation": "Recalled primarily due to semantic similarity (0.91) to query, with keyword match on 'Python' (BM25 rank #1) and graph connection via 'Python' entity",
    "signals": [
      {"signal": "vector", "raw_score": 0.91, "rank": 0, "rrf_contribution": 0.016, "explanation": "High semantic similarity (0.91) to query"},
      {"signal": "bm25", "raw_score": 4.2, "rank": 0, "rrf_contribution": 0.013, "explanation": "Keyword match: 'Python' (BM25 score 4.2)"},
      {"signal": "graph", "raw_score": 1.0, "rank": 2, "rrf_contribution": 0.016, "explanation": "Connected via entity 'Python' (2-hop graph traversal)"},
      {"signal": "temporal", "raw_score": 0.72, "rank": 5, "rrf_contribution": 0.008, "explanation": "Moderately recent (3 days old, importance 0.72)"}
    ]
  }
}
```

**Competitive Advantage**: NO competitor (Mem0, Graphiti, Memento, MemoryGraph) provides per-signal retrieval explanations. This is a unique differentiator.

**Dependencies**: None (uses existing Pydantic, existing retrieval infrastructure)

---

### 1D. Feature Flags System

**Purpose**: Centralize all feature toggles to prevent configuration sprawl.

**New File:**
| File | Purpose |
|---|---|
| `muninn/core/feature_flags.py` | Centralized feature flag management |

```python
from dataclasses import dataclass, field

@dataclass
class FeatureFlags:
    # Phase 1 (default ON — low cost, high value)
    explainable_recall: bool = True
    instructor_extraction: bool = True  # When endpoint available

    # Phase 2 (default OFF — higher cost, opt-in)
    conflict_detection: bool = False
    semantic_dedup: bool = False
    adaptive_weights: bool = False

    # Phase 3 (default OFF — require additional deps)
    memory_chains: bool = False
    multi_source_ingestion: bool = False

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        import os
        return cls(
            explainable_recall=os.environ.get("MUNINN_EXPLAIN_RECALL", "1") == "1",
            conflict_detection=os.environ.get("MUNINN_CONFLICT_DETECTION", "0") == "1",
            # ... etc
        )
```

---

## Phase 2: Intelligence (v3.2.0)

> **Priority**: 🟡 HIGH
> **Risk**: MEDIUM
> **Effort**: 7–10 days
> **Theme**: Memory integrity, retrieval quality, intelligent adaptation

### 2A. NLI-Based Conflict Detection

**Problem**: Contradictory memories coexist silently. "User prefers Python" and "User switched to Rust" both retrieved, confusing agents.

**Solution**: Use Natural Language Inference (NLI) cross-encoder model to detect contradiction between new and existing memories during add().

**New Files:**
| File | Purpose |
|---|---|
| `muninn/conflict/__init__.py` | Conflict detection package |
| `muninn/conflict/detector.py` | NLI-based contradiction detection |
| `muninn/conflict/resolver.py` | Resolution strategies (supersede, merge, flag) |
| `tests/test_conflict_detector.py` | Detection accuracy tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | Add conflict check in `add()` method |
| `muninn/core/types.py` | Add `ConflictResult`, `ConflictResolution` |
| `muninn/core/config.py` | Conflict detection config options |
| `server.py` | Return conflicts in add response |

**Technical Design:**

```python
# muninn/conflict/detector.py
class ConflictDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        # DeBERTa-v3-small: 44MB, Apache-2.0 license, 91.65% accuracy on SNLI
        # Outputs: [contradiction_score, entailment_score, neutral_score]
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def detect_conflicts(self, new_content: str,
                         existing_memories: List[MemoryRecord],
                         threshold: float = 0.7) -> List[ConflictResult]:
        conflicts = []
        for existing in existing_memories:
            features = self.tokenizer(new_content, existing.content,
                                      padding=True, truncation=True, return_tensors="pt")
            scores = self.model(**features).logits.softmax(dim=1)[0]
            # scores = [contradiction, entailment, neutral]
            if scores[0].item() > threshold:
                conflicts.append(ConflictResult(
                    new_content=new_content,
                    existing_memory=existing,
                    contradiction_score=scores[0].item(),
                    suggested_resolution=self._suggest_resolution(existing, scores)
                ))
        return conflicts
```

**Resolution Strategies:**
| Strategy | When Applied | Action |
|---|---|---|
| `SUPERSEDE` | New memory is clearly more recent, same topic | Archive old, store new |
| `MERGE` | Both partially true, complementary | Combine into unified memory |
| `FLAG_FOR_REVIEW` | High contradiction, both seem valid | Return to user/agent for decision |
| `KEEP_EXISTING` | New memory low confidence, existing high | Discard new |

**Performance Mitigation:**
- Only runs when `feature_flags.conflict_detection == True`
- Only checks against memories with similarity > 0.8 (pre-filtered by vector search)
- DeBERTa-v3-small is fast (~10ms per pair on CPU)
- Model loaded once, cached in memory

**Model**: `cross-encoder/nli-deberta-v3-small` — Apache-2.0 License ✅
**Dependency**: `transformers` + `torch` (only when conflict detection enabled) — Apache-2.0 ✅

---

### 2B. Semantic Deduplication

**Problem**: Same fact expressed differently creates redundant memories. "I use VS Code" and "My editor is Visual Studio Code" both stored separately.

**Solution**: Embedding-based near-duplicate detection at ingestion time using existing vector infrastructure.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/dedup/__init__.py` | Deduplication package |
| `muninn/dedup/semantic_dedup.py` | Embedding-based near-duplicate detection |
| `tests/test_semantic_dedup.py` | Dedup accuracy and false-positive tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | Add dedup check in `add()` before storage |
| `muninn/consolidation/daemon.py` | Enhanced merge with dedup |

**Technical Design:**

```python
# muninn/dedup/semantic_dedup.py
class SemanticDedup:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold  # Higher than consolidation merge (0.92)

    def check_duplicate(self, embedding, vector_store, content, metadata_store):
        matches = vector_store.search(embedding, limit=3, score_threshold=self.threshold)
        for memory_id, score in matches:
            existing = metadata_store.get(memory_id)
            if existing and score >= self.threshold:
                if self._content_overlap(content, existing.content) > 0.8:
                    return DedupResult(
                        is_duplicate=True,
                        existing_memory_id=memory_id,
                        similarity=score,
                        strategy=DedupStrategy.UPDATE_EXISTING
                    )
        return None
```

**Strategies:**
| Strategy | Action | When |
|---|---|---|
| `UPDATE_EXISTING` | Merge new info into existing memory | High similarity, new has additional detail |
| `SKIP` | Discard new memory entirely | Near-identical content |
| `LINK` | Store both, mark as related | Similar but distinct perspectives |

**Dependencies**: None — uses existing fastembed embeddings and Qdrant search ✅

---

### 2C. Adaptive Retrieval Weights

**Problem**: Fixed signal weights (`vector: 1.0, graph: 1.0, bm25: 0.8, temporal: 0.5`) are suboptimal for different query types. Short keyword queries benefit from BM25; temporal queries need temporal boost.

**Solution**: Entropy-based dynamic weighting that adapts per-query based on signal confidence and query characteristics.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/retrieval/weight_adapter.py` | Entropy-based + feedback-based weight adaptation |
| `tests/test_weight_adapter.py` | Weight adaptation correctness tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/retrieval/hybrid.py` | Use `WeightAdapter` instead of fixed `SIGNAL_WEIGHTS` |
| `muninn/store/sqlite_metadata.py` | Add `retrieval_feedback` table |

**Technical Design:**

```python
# muninn/retrieval/weight_adapter.py
import math

class WeightAdapter:
    DEFAULT_WEIGHTS = {"vector": 1.0, "graph": 1.0, "bm25": 0.8, "temporal": 0.5}

    def compute_weights(self, query: str, signal_results: dict) -> dict:
        weights = dict(self.DEFAULT_WEIGHTS)

        # 1. Query-based adaptation
        tokens = query.lower().split()
        if len(tokens) <= 3:
            weights["bm25"] *= 1.3           # Short queries → keyword match
        if any(t in tokens for t in ["recent", "latest", "today", "yesterday", "new"]):
            weights["temporal"] *= 2.0        # Temporal queries → recency
        if any(t in tokens for t in ["related", "connected", "about", "who"]):
            weights["graph"] *= 1.5           # Relational queries → graph

        # 2. Entropy-based confidence per signal
        for signal, results in signal_results.items():
            if results and len(results) > 1:
                scores = [r[1] for r in results if r[1] > 0]
                entropy = self._normalized_entropy(scores)
                confidence = 1.0 - entropy
                weights[signal] *= (0.5 + confidence)  # Scale [0.5, 1.5]

        return weights

    def _normalized_entropy(self, scores: list) -> float:
        if not scores or all(s == 0 for s in scores):
            return 1.0
        total = sum(scores)
        probs = [s / total for s in scores if s > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
```

**Academic Basis**: Inspired by "Entropy-Based Dynamic Hybrid Retrieval for Adaptive Query Weighting in RAG Pipelines" (ICML 2025 Workshop) and "Multi-Field Adaptive Retrieval" (ICLR 2025 Spotlight).

**Dependencies**: None — pure Python math ✅

---

## Phase 3: Ecosystem (v3.3.0)

> **Priority**: 🟢 MEDIUM
> **Risk**: MEDIUM
> **Effort**: 9–13 days
> **Theme**: Advanced capabilities, broader adoption, growth features

### 3A. Memory Chains

**Problem**: Memories are isolated nodes. Sequential events ("Started project X" → "Encountered bug" → "Fixed bug" → "Completed project") have no causal/temporal linking.

**Solution**: Detect temporal and causal chain relationships between memories, store as directed graph edges, enable chain retrieval.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/chains/__init__.py` | Chain detection package |
| `muninn/chains/detector.py` | Temporal/causal chain link detection |
| `muninn/chains/retriever.py` | Chain traversal and reconstruction |
| `tests/test_chain_detector.py` | Chain detection accuracy tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/store/graph_store.py` | New `PRECEDES`/`CAUSES` edge types, chain traversal queries |
| `muninn/retrieval/hybrid.py` | Chain expansion in graph search results |
| `muninn/core/types.py` | `ChainLink`, `MemoryChain` types |

**Chain Detection Algorithm:**
```
Score = temporal_proximity + entity_overlap + causal_markers + project_match
- temporal_proximity: 1.0 - (hours_apart / 24.0), capped at [0, 1]
- entity_overlap: shared_entities * 0.3
- causal_markers: +0.5 if causal language detected ("because", "therefore", "led to")
- project_match: +0.2 if same project context
- Threshold: score >= 0.6 → create chain link
```

**Graph Schema Extensions:**
```cypher
CREATE REL TABLE PRECEDES(FROM Memory TO Memory, confidence DOUBLE, shared_entities STRING[])
CREATE REL TABLE CAUSES(FROM Memory TO Memory, confidence DOUBLE, evidence STRING)
```

**Dependencies**: None — uses existing Kuzu graph store ✅

---

### 3B. Multi-Source Ingestion

**Problem**: Muninn only accepts memory via `add()` API call with string content. Users can't import existing knowledge bases, conversation histories, or documents.

**Solution**: Pluggable ingestion pipeline: Source → Parse → Chunk → Extract → Store.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/ingestion/__init__.py` | Ingestion package |
| `muninn/ingestion/pipeline.py` | Main ingestion orchestrator |
| `muninn/ingestion/chunker.py` | Semantic chunking with overlap |
| `muninn/ingestion/parsers/text.py` | .txt and .md parsing |
| `muninn/ingestion/parsers/pdf.py` | .pdf parsing (pypdf) |
| `muninn/ingestion/parsers/docx.py` | .docx parsing (python-docx) |
| `muninn/ingestion/parsers/html.py` | .html parsing (beautifulsoup4) |
| `muninn/ingestion/parsers/json_parser.py` | .json conversation import |
| `muninn/ingestion/parsers/csv_parser.py` | .csv data import |
| `muninn/ingestion/sources/chatgpt.py` | ChatGPT export format adapter |
| `muninn/ingestion/sources/obsidian.py` | Obsidian vault adapter |
| `tests/test_ingestion_pipeline.py` | Pipeline integration tests |
| `tests/test_chunker.py` | Chunking quality tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/core/memory.py` | New `ingest()` method |
| `server.py` | New `/ingest` endpoint |
| `mcp_wrapper.py` | New `ingest_file` MCP tool |
| `pyproject.toml` | New `[ingestion]` extras: pypdf, python-docx, beautifulsoup4 |

**Supported Formats:**
| Format | Library | License |
|---|---|---|
| .txt, .md | Built-in | — |
| .pdf | pypdf | BSD-3 ✅ |
| .docx | python-docx | MIT ✅ |
| .html | beautifulsoup4 | MIT ✅ |
| .json | Built-in | — |
| .csv | Built-in | — |

**⚠️ CRITICAL**: pymupdf/PyMuPDF is AGPL-3.0 — INCOMPATIBLE with Apache-2.0. Use `pypdf` (BSD) instead.

**Chunking Strategy:**
```
1. Parse → typed elements (paragraph, heading, code_block, table, list)
2. Keep code blocks and tables as single chunks (preserve context)
3. Merge small paragraphs up to max_tokens (512 default)
4. Split large paragraphs at sentence boundaries
5. Add overlap from previous chunk (50 tokens default) for context
```

---

### 3C. Python SDK

**Problem**: Muninn can only be used via REST API (server.py) or MCP protocol (mcp_wrapper.py). No programmatic Python API for direct integration.

**Solution**: Sync + Async Python SDK wrapping MuninnMemory directly.

**New Files:**
| File | Purpose |
|---|---|
| `muninn/sdk/__init__.py` | Public SDK API |
| `muninn/sdk/client.py` | `MuninnClient` (sync) |
| `muninn/sdk/async_client.py` | `AsyncMuninnClient` (async) |
| `muninn/sdk/models.py` | SDK-specific Pydantic models |
| `tests/test_sdk_client.py` | Sync client tests |
| `tests/test_sdk_async.py` | Async client tests |

**Modified Files:**
| File | Change |
|---|---|
| `muninn/__init__.py` | Export `Memory = MuninnClient` for convenience |
| `pyproject.toml` | New `[sdk]` extras group |

**API Design (Mem0-Compatible):**

```python
from muninn import Memory

# Simple usage
m = Memory()
m.add("User prefers Python for backend", user_id="alice")
results = m.search("programming language preference", user_id="alice")
m.close()

# Context manager
with Memory(data_dir="~/.myapp/memory") as m:
    m.add("Important fact", metadata={"project": "myapp"})
    results = m.search("fact", explain=True)  # With recall traces

# Async
from muninn import AsyncMemory
async with AsyncMemory() as m:
    await m.add("Async fact")
    results = await m.search("fact")
```

**Dependencies**: None — wraps existing MuninnMemory engine ✅

---

## Dependency Summary

### New Dependencies by Phase

| Phase | Dependency | License | Required/Optional | Size |
|---|---|---|---|---|
| 1 | `platformdirs` | MIT | Required | ~50KB |
| 1 | `instructor` | MIT | Optional (extraction) | ~200KB |
| 1 | `openai` | Apache-2.0 | Optional (via instructor) | ~500KB |
| 2 | `transformers` | Apache-2.0 | Optional (conflict detection) | ~8MB |
| 2 | `torch` | BSD-3 | Optional (conflict detection) | ~200MB |
| 3 | `pypdf` | BSD-3 | Optional (ingestion) | ~2MB |
| 3 | `python-docx` | MIT | Optional (ingestion) | ~1MB |
| 3 | `beautifulsoup4` | MIT | Optional (ingestion) | ~500KB |

**pyproject.toml Extras Groups:**
```toml
[project.optional-dependencies]
extraction = ["instructor>=1.0", "openai>=1.0"]
conflict = ["transformers>=4.30", "torch>=2.0"]
ingestion = ["pypdf>=4.0", "python-docx>=1.0", "beautifulsoup4>=4.12"]
sdk = []  # No additional deps, uses core
all = ["muninn-mcp[extraction,conflict,ingestion]"]
```

---

## Risk Matrix

| # | Feature | Risk | Impact if Fails | Mitigation |
|---|---|---|---|---|
| 1A | Platform Abstraction | LOW | Path errors on new platforms | Comprehensive platform tests, CI matrix |
| 1B | Extraction Enhancement | LOW-MED | Extraction quality regression | Rules tier always works, Instructor is additive |
| 1C | Explainable Recall | LOW | Slightly slower search response | Trace construction is O(n) lightweight |
| 2A | Conflict Detection | MEDIUM | Slower add() operations | Optional flag, async option, small model |
| 2B | Semantic Dedup | LOW-MED | False positive dedup (loses memory) | High threshold (0.95), secondary text check |
| 2C | Adaptive Weights | MEDIUM | Retrieval quality regression | Falls back to fixed weights, A/B testable |
| 3A | Memory Chains | LOW-MED | False chain links | Confidence threshold, only causal markers |
| 3B | Multi-Source Ingestion | MEDIUM | Parser failures on edge-case docs | Per-parser error handling, skip-and-continue |
| 3C | Python SDK | LOW | Event loop conflicts | Detect async context, clear error messages |

---

## ROI Prioritization

| Rank | Feature | Effort | Value | Unique? |
|---|---|---|---|---|
| 1 | Explainable Recall Traces | 2 days | 🔴 HIGHEST | ✅ UNIQUE |
| 2 | Platform Abstraction | 2-3 days | 🔴 HIGH | Growing expectation |
| 3 | Extraction Enhancement | 2-3 days | 🔴 HIGH | Parity + quality |
| 4 | Conflict Detection | 3-4 days | 🟡 HIGH | ✅ UNIQUE |
| 5 | Semantic Dedup | 2-3 days | 🟡 MEDIUM-HIGH | Common need |
| 6 | Python SDK | 3-4 days | 🟡 MEDIUM-HIGH | Distribution req. |
| 7 | Adaptive Weights | 2-3 days | 🟢 MEDIUM | Research-backed |
| 8 | Memory Chains | 2-3 days | 🟢 MEDIUM | Advanced graph |
| 9 | Multi-Source Ingestion | 4-5 days | 🟢 MEDIUM | Growth feature |

---

## Validation Checklist

- [ ] ALL existing APIs remain unchanged (backward compatible)
- [ ] ALL new features are optional and disabled by default where appropriate
- [ ] ALL new dependencies are Apache-2.0/MIT/BSD compatible
- [ ] NO code copied from competitors (100% original implementation)
- [ ] NO AGPL dependencies (pymupdf explicitly avoided)
- [ ] Every new module has corresponding unit tests
- [ ] Feature flags centralize all toggles
- [ ] Docker deployment works with stdio MCP and REST modes
- [ ] Cross-platform CI validates Windows + Linux + macOS
- [ ] Performance impact measured for each hot-path addition (add, search)

---

## Implementation Timeline

```
Phase 1 (v3.1.0): Weeks 1-2
├── 1A. Platform Abstraction (Days 1-3)
├── 1B. Extraction Enhancement (Days 3-5)
├── 1C. Explainable Recall (Days 5-7)
├── 1D. Feature Flags (Day 7)
└── Testing + Validation (Days 8-9)

Phase 2 (v3.2.0): Weeks 3-4
├── 2A. Conflict Detection (Days 1-4)
├── 2B. Semantic Dedup (Days 4-6)
├── 2C. Adaptive Weights (Days 6-8)
└── Testing + Validation (Days 9-10)

Phase 3 (v3.3.0): Weeks 5-7
├── 3A. Memory Chains (Days 1-3)
├── 3B. Multi-Source Ingestion (Days 3-8)
├── 3C. Python SDK (Days 8-11)
└── Testing + Validation (Days 12-13)
```

---

## Appendix: Competitive Positioning After SOTA+

| Feature | Muninn v3.3 | Mem0 | Graphiti/Zep | Memento | MemoryGraph |
|---|---|---|---|---|---|
| Local-first | ✅ | ❌ Cloud | ❌ Neo4j | ✅ | ✅ |
| Explainable recall | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Conflict detection | ✅ UNIQUE | Partial | ❌ | ❌ | ❌ |
| Hybrid retrieval (4-signal) | ✅ | Vector only | Graph + Vector | Vector | Graph |
| Adaptive weights | ✅ | ❌ | ❌ | ❌ | ❌ |
| Memory chains | ✅ | ❌ | ✅ Temporal KG | ❌ | ❌ |
| Structured extraction | ✅ Instructor | ✅ LLM | ✅ LLM | ❌ | ❌ |
| Multi-source ingestion | ✅ | ❌ | ✅ Episodes | ❌ | ❌ |
| Python SDK | ✅ | ✅ | ✅ | ❌ | ❌ |
| MCP protocol | ✅ | ❌ | ❌ | ✅ | ✅ |
| Cross-platform | ✅ | ✅ | ✅ | ✅ | ✅ |
| Background consolidation | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Memory hierarchy (4-tier) | ✅ UNIQUE | ❌ | ❌ | ❌ | ❌ |
| Bi-temporal provenance | ✅ | ❌ | ✅ | ❌ | ❌ |
| Semantic dedup | ✅ | ❌ | ❌ | ❌ | ❌ |
