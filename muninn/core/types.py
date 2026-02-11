"""
Muninn Core Types
-----------------
Pydantic models and enums for the Muninn memory framework.
"""

import uuid
import time
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


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
    source: str = "vector"  # vector|graph|bm25|temporal


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


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "global_user"
    agent_id: Optional[str] = None
    limit: int = 10
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    namespaces: Optional[List[str]] = None


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    data: str


class HealthResponse(BaseModel):
    status: str
    memory_count: int = 0
    graph_nodes: int = 0
    reranker: str = "inactive"
    backend: str = "muninn-native"
