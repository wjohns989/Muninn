"""
Muninn Core Types
-----------------
Pydantic models and enums for the Muninn memory framework.

v3.1.0: Added RecallTrace support in SearchResult for explainable recall.
"""

import uuid
import time
from enum import Enum
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from muninn.core.recall_trace import RecallTrace


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class Provenance(str, Enum):
    USER_EXPLICIT = "user_explicit"
    ASSISTANT_CONFIRMED = "assistant_confirmed"
    AUTO_EXTRACTED = "auto_extracted"
    INGESTED = "ingested"


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC

    # Importance scoring
    importance: float = 0.5
    recency_score: float = 1.0
    access_count: int = 0
    novelty_score: float = 0.5

    # Temporal (bi-temporal)
    created_at: float = Field(default_factory=time.time)
    ingested_at: float = Field(default_factory=time.time)
    last_accessed: Optional[float] = None
    expires_at: Optional[float] = None

    # Provenance
    source_agent: str = "unknown"
    project: str = "global"
    branch: Optional[str] = None
    namespace: str = "global"
    provenance: Provenance = Provenance.AUTO_EXTRACTED

    # Project isolation scope (v3.11.0)
    # "project" — visible only within its project; NEVER returned in cross-project fallback search
    # "global"  — always visible regardless of current project (user prefs, universal rules)
    scope: Literal["project", "global"] = "project"

    # Multimodal support (v3.20.0)
    media_type: MediaType = MediaType.TEXT

    # Embedding reference
    vector_id: Optional[str] = None
    embedding_model: str = "nomic-embed-text"

    # Consolidation state
    consolidated: bool = False
    parent_id: Optional[str] = None
    consolidation_gen: int = 0

    # Metadata (flexible key-value)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    memory: MemoryRecord
    score: float = 0.0
    source: str = "vector"  # vector|graph|bm25|temporal|hybrid|hybrid+rerank
    trace: Optional[RecallTrace] = Field(
        default=None,
        description="RecallTrace explaining why this memory was retrieved (v3.1.0). "
        "Set when explain=True in search request and explainable_recall flag is ON.",
    )


class Entity(BaseModel):
    name: str
    entity_type: str  # person|org|tech|concept|location|project|file|preference
    source_memory_id: Optional[str] = None


class Relation(BaseModel):
    subject: str
    predicate: str  # uses|prefers|created|depends_on|located_in|works_with|knows|part_of
    object: str
    source_memory_id: Optional[str] = None
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    summary: Optional[str] = None
    temporal_context: Optional[str] = None


# --- API Request/Response Models ---

class AddMemoryRequest(BaseModel):
    content: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    user_id: Optional[str] = "global_user"
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: Optional[bool] = None
    namespace: Optional[str] = "global"
    scope: Literal["project", "global"] = "project"
    media_type: MediaType = MediaType.TEXT


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "global_user"
    agent_id: Optional[str] = None
    limit: int = 10
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    namespaces: Optional[List[str]] = None
    media_type: Optional[MediaType] = None
    explain: bool = Field(
        default=False,
        description="When True, include RecallTrace explaining retrieval signals (v3.1.0).",
    )


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    data: str


class HealthResponse(BaseModel):
    status: str
    memory_count: int = 0
    graph_nodes: int = 0
    reranker: str = "inactive"
    backend: str = "muninn-native"