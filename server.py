#!/usr/bin/env python3
"""
Muninn Memory Server — Native Local Backend
=============================================

Architecture (Muninn-native, Mem0-free):
- Memory Engine: muninn.core.MuninnMemory
- Embeddings: FastEmbed or Ollama nomic-embed-text (768 dims)
- Vector Store: Qdrant local (on_disk, cosine)
- Graph Store: Kuzu embedded (entity/relation knowledge graph)
- Metadata: SQLite (WAL mode, importance scoring, bi-temporal)
- Extraction: 3-tier pipeline (rules → xLAM → Ollama)
- Retrieval: Hybrid (vector + graph + BM25 + temporal) with RRF
- Reranking: Jina Tiny cross-encoder (fastembed)
- Consolidation: Background daemon (decay/merge/promote/replay)

VRAM usage is profile/model dependent.
Use MUNINN_VRAM_BUDGET_GB and extraction profile env vars to keep
runtime helper paths lightweight during active development.

Usage:
    python server.py              # Start server on localhost:42069
    python server.py --port 8000  # Custom port
"""

import os
import sys
import argparse
import logging
import time
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import secrets

from muninn.core.memory import MuninnMemory
from muninn.core.config import MuninnConfig, SUPPORTED_MODEL_PROFILES
from muninn.core.security import SecurityContext, verify_token as core_verify_token, initialize_security
from muninn.version import __version__
from muninn.ingestion.pipeline import (
    MAX_CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_SIZE_CHARS,
    MAX_INGEST_FILE_SIZE_BYTES,
)
from muninn.core.types import (
    AddMemoryRequest,
    SearchMemoryRequest,
    UpdateMemoryRequest,
    MemoryType,
    Provenance,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("muninn_server.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Muninn")

# --- Global State ---
memory: Optional[MuninnMemory] = None
GLOBAL_AUTH_TOKEN: Optional[str] = None


# --- Pydantic Models (API compatibility) ---
from pydantic import BaseModel
from pydantic import Field


class DeleteMemoryRequest(BaseModel):
    memory_id: str


class DeleteAllRequest(BaseModel):
    user_id: Optional[str] = None
    agent_id: Optional[str] = None


class DeleteBatchRequest(BaseModel):
    memory_ids: List[str]


class HandoverRequest(BaseModel):
    source_agent_id: str
    target_agent_id: str
    memory_id: str
    reason: Optional[str] = "Handover for context alignment"


class SynthesisRequest(BaseModel):
    namespaces: List[str]
    target_namespace: str = "global"
    query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SetProjectGoalRequest(BaseModel):
    user_id: str = "global_user"
    namespace: str = "global"
    project: str
    goal_statement: str
    constraints: List[str] = Field(default_factory=list)


class SetUserProfileRequest(BaseModel):
    user_id: str = "global_user"
    profile: Dict[str, Any] = Field(default_factory=dict)
    merge: bool = True
    source: str = "api"


class ExportHandoffRequest(BaseModel):
    user_id: str = "global_user"
    namespace: str = "global"
    project: str
    limit: int = 25


class ImportHandoffRequest(BaseModel):
    bundle: Dict[str, Any]
    user_id: str = "global_user"
    namespace: str = "global"
    project: str
    source: str = "handoff_import"


class RetrievalFeedbackRequest(BaseModel):
    query: str
    memory_id: str
    outcome: float
    rank: Optional[int] = Field(default=None, ge=1)
    sampling_prob: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    user_id: str = "global_user"
    namespace: str = "global"
    project: str = "global"
    signals: Dict[str, float] = Field(default_factory=dict)
    source: str = "manual"


class IngestSourcesRequest(BaseModel):
    sources: List[str]
    user_id: str = "global_user"
    namespace: str = "global"
    project: str = "global"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recursive: bool = False
    chronological_order: str = Field(default="none", pattern="^(none|oldest_first|newest_first)$")
    max_file_size_bytes: Optional[int] = Field(
        default=None,
        gt=0,
        le=MAX_INGEST_FILE_SIZE_BYTES,
    )
    chunk_size_chars: Optional[int] = Field(
        default=None,
        gt=0,
        le=MAX_CHUNK_SIZE_CHARS,
    )
    chunk_overlap_chars: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_CHUNK_OVERLAP_CHARS,
    )
    min_chunk_chars: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_CHUNK_SIZE_CHARS,
    )


class DiscoverLegacySourcesRequest(BaseModel):
    roots: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    include_unsupported: bool = False
    max_results_per_provider: int = Field(default=100, ge=1, le=5000)


class IngestLegacySourcesRequest(BaseModel):
    selected_source_ids: List[str] = Field(default_factory=list)
    selected_paths: List[str] = Field(default_factory=list)
    roots: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    include_unsupported: bool = False
    max_results_per_provider: int = Field(default=100, ge=1, le=5000)
    user_id: str = "global_user"
    namespace: str = "global"
    project: str = "global"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recursive: bool = False
    chronological_order: str = Field(default="none", pattern="^(none|oldest_first|newest_first)$")
    max_file_size_bytes: Optional[int] = Field(
        default=None,
        gt=0,
        le=MAX_INGEST_FILE_SIZE_BYTES,
    )
    chunk_size_chars: Optional[int] = Field(
        default=None,
        gt=0,
        le=MAX_CHUNK_SIZE_CHARS,
    )
    chunk_overlap_chars: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_CHUNK_OVERLAP_CHARS,
    )
    min_chunk_chars: Optional[int] = Field(
        default=None,
        ge=1,
        le=MAX_CHUNK_SIZE_CHARS,
    )


MODEL_PROFILE_PATTERN = f"^({'|'.join(SUPPORTED_MODEL_PROFILES)})$"


class SetModelProfilesRequest(BaseModel):
    model_profile: Optional[str] = Field(default=None, pattern=MODEL_PROFILE_PATTERN)
    runtime_model_profile: Optional[str] = Field(default=None, pattern=MODEL_PROFILE_PATTERN)
    ingestion_model_profile: Optional[str] = Field(default=None, pattern=MODEL_PROFILE_PATTERN)
    legacy_ingestion_model_profile: Optional[str] = Field(default=None, pattern=MODEL_PROFILE_PATTERN)
    source: str = "api"


# --- Security ---
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """FastAPI dependency for token verification."""
    token = credentials.credentials if credentials else None
    if not core_verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory

    logger.info("Muninn Server starting...")

    try:
        # Initialize configuration from environment
        config = MuninnConfig.from_env()

        # Auth Token Initialization (Phase 10 / v3.7.0)
        initialize_security(config.server.auth_token)

        # Initialize memory engine
        memory = MuninnMemory(config)
        await memory.initialize()

        # Phase 5C.3: Removed GLOBAL_LOCK in favor of granular MuninnMemory._write_lock
        # GLOBAL_LOCK = asyncio.Lock()
        # logger.info("Global concurrency lock initialized")

        yield
    finally:
        logger.info("Shutting down Muninn Server...")
        if memory:
            await memory.shutdown()
        logger.info("Muninn Server stopped.")


app = FastAPI(
    title="Muninn Memory Server",
    description="Local-first persistent memory for AI agents — Muninn native engine",
    version=__version__,
    lifespan=lifespan,
)

# --- CORS ---
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow local dashboard via file:// or other ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")


def _load_dashboard_html() -> str:
    try:
        return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to load dashboard HTML: %s", exc)
        return (
            "<html><body><h1>Huginn UI unavailable</h1>"
            "<p>dashboard.html could not be loaded.</p></body></html>"
        )


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    """Serve the browser UI for memory operations."""
    return HTMLResponse(content=_load_dashboard_html())

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if memory is None:
        return {"status": "initializing", "backend": "muninn-native"}

    try:
        health = await memory.health()
        return health
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "error", "error": str(e), "backend": "muninn-native"}


@app.post("/add", dependencies=[Depends(verify_token)])
async def add_memory_endpoint(req: AddMemoryRequest):
    """Add a memory to the store with entity extraction and auto-indexing."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock; concurrency managed by memory engine
        # if GLOBAL_LOCK is None:
        #     raise HTTPException(status_code=503, detail="Server not initialized")

        # async with GLOBAL_LOCK:
        # Support both messages list and simple content string
        if req.messages:
            # Convert messages to a single content string
            content = " ".join(
                msg.get("content", "") for msg in req.messages
                if msg.get("role") in ("user", "assistant")
            )
        elif req.content:
            content = req.content
        else:
            raise HTTPException(
                status_code=400, detail="Either 'messages' or 'content' required"
            )

        if not content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        # Determine provenance
        provenance = Provenance.AUTO_EXTRACTED
        if req.metadata and req.metadata.get("provenance"):
            try:
                provenance = Provenance(req.metadata["provenance"])
            except ValueError:
                pass

        # Chunking for long content
        if len(content) > 1000:
            logger.info("Content > 1000 chars, applying semantic splitting...")
            import re
            chunks = re.split(r"\n\n|(?<!\s\w)\.\s", content)
            merged_chunks = []
            current = ""
            for c in chunks:
                if len(current) + len(c) < 800:
                    current += " " + c
                else:
                    if current.strip():
                        merged_chunks.append(current.strip())
                    current = c
            if current.strip():
                merged_chunks.append(current.strip())

            # Parallelize chunk additions to overlap extractions/embeddings
            tasks = [
                memory.add(
                    content=chunk,
                    user_id=req.user_id or "global_user",
                    agent_id=req.agent_id,
                    metadata=req.metadata,
                    namespace=req.namespace or "global",
                    provenance=provenance,
                    scope=req.scope,
                )
                for chunk in merged_chunks
            ]
            results = await asyncio.gather(*tasks)
            return {"success": True, "data": {"chunks_added": len(results), "results": results}}

        result = await memory.add(
            content=content,
            user_id=req.user_id or "global_user",
            agent_id=req.agent_id,
            metadata=req.metadata,
            namespace=req.namespace or "global",
            provenance=provenance,
            scope=req.scope,
        )

        logger.info("Added memory for user %s", req.user_id)
        return {"success": True, "data": result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error adding memory: %s", e)
        err_msg = str(e).lower()
        if "validation" in err_msg or "argument" in err_msg:
            raise HTTPException(status_code=400, detail=str(e))
        elif "lock" in err_msg or "access" in err_msg:
            raise HTTPException(status_code=503, detail="Database busy, please retry")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", dependencies=[Depends(verify_token)])
async def search_memory_endpoint(req: SearchMemoryRequest):
    """Search for relevant memories using hybrid multi-signal retrieval."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        results = await memory.search(
            query=req.query,
            user_id=req.user_id or "global_user",
            agent_id=req.agent_id,
            limit=req.limit,
            rerank=req.rerank,
            filters=req.filters,
            namespaces=req.namespaces,
            explain=req.explain,
        )

        return {"success": True, "data": results}
    except Exception as e:
        logger.error("Error searching memories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goal/set", dependencies=[Depends(verify_token)])
async def set_project_goal_endpoint(req: SetProjectGoalRequest):
    """Set/update active project goal used for drift checks and retrieval prior."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.set_project_goal(
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project,
            goal_statement=req.goal_statement,
            constraints=req.constraints,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error setting project goal: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profile/user/set", dependencies=[Depends(verify_token)])
async def set_user_profile_endpoint(req: SetUserProfileRequest):
    """Set/update editable user profile and global context."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.set_user_profile(
            user_id=req.user_id or "global_user",
            profile=req.profile,
            merge=req.merge,
            source=req.source,
        )
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error setting user profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while setting the user profile.",
        )


@app.get("/profile/user/get", dependencies=[Depends(verify_token)])
async def get_user_profile_endpoint(
    user_id: str = "global_user",
):
    """Fetch editable user profile and global context."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        result = await memory.get_user_profile(
            user_id=user_id or "global_user",
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error getting user profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while getting the user profile.",
        )


@app.get("/goal/get", dependencies=[Depends(verify_token)])
async def get_project_goal_endpoint(
    user_id: str = "global_user",
    namespace: str = "global",
    project: str = "global",
):
    """Fetch active project goal for a scope."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        result = await memory.get_project_goal(
            user_id=user_id or "global_user",
            namespace=namespace or "global",
            project=project or "global",
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error getting project goal: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiles/model", dependencies=[Depends(verify_token)])
async def get_model_profiles_endpoint():
    """Return active runtime extraction profile policy."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        result = await memory.get_model_profiles()
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting model profiles: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/profiles/model", dependencies=[Depends(verify_token)])
async def set_model_profiles_endpoint(req: SetModelProfilesRequest):
    """Update runtime extraction profile policy without server restart."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    payload = req.model_dump(exclude_none=True)
    if set(payload.keys()) <= {"source"}:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one profile field to update.",
        )

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.set_model_profiles(**payload)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error setting model profiles: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiles/model/events", dependencies=[Depends(verify_token)])
async def get_model_profile_events_endpoint(limit: int = 25):
    """Return recent runtime profile policy mutation events."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    try:
        result = await memory.get_model_profile_events(limit=limit)
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting model profile events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/handoff/export", dependencies=[Depends(verify_token)])
async def export_handoff_endpoint(req: ExportHandoffRequest):
    """Export portable handoff bundle for cross-assistant continuity."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        result = await memory.export_handoff(
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project,
            limit=req.limit,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error exporting handoff: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/handoff/import", dependencies=[Depends(verify_token)])
async def import_handoff_endpoint(req: ImportHandoffRequest):
    """Import portable handoff bundle (idempotent via event ledger)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.import_handoff(
            bundle=req.bundle,
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project,
            source=req.source,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error importing handoff: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/retrieval", dependencies=[Depends(verify_token)])
async def retrieval_feedback_endpoint(req: RetrievalFeedbackRequest):
    """Record retrieval feedback for adaptive weighting calibration."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.record_retrieval_feedback(
            query=req.query,
            memory_id=req.memory_id,
            outcome=req.outcome,
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project or "global",
            rank=req.rank,
            sampling_prob=req.sampling_prob,
            signals=req.signals,
            source=req.source,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error recording retrieval feedback: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", dependencies=[Depends(verify_token)])
async def ingest_sources_endpoint(req: IngestSourcesRequest):
    """Ingest multiple local sources with fail-open behavior per source/chunk."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.ingest_sources(
            sources=req.sources,
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project or "global",
            metadata=req.metadata,
            recursive=req.recursive,
            chronological_order=req.chronological_order,
            max_file_size_bytes=req.max_file_size_bytes,
            chunk_size_chars=req.chunk_size_chars,
            chunk_overlap_chars=req.chunk_overlap_chars,
            min_chunk_chars=req.min_chunk_chars,
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error ingesting sources: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/legacy/discover")
async def discover_legacy_sources_endpoint(req: DiscoverLegacySourcesRequest):
    """Discover local legacy assistant/MCP memory artifacts available for import."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        result = await memory.discover_legacy_sources(
            roots=req.roots,
            providers=req.providers,
            include_unsupported=req.include_unsupported,
            max_results_per_provider=req.max_results_per_provider,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error discovering legacy sources: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/legacy/import")
async def ingest_legacy_sources_endpoint(req: IngestLegacySourcesRequest):
    """Ingest user-selected legacy assistant/MCP sources with contextual metadata."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.ingest_legacy_sources(
            selected_source_ids=req.selected_source_ids,
            selected_paths=req.selected_paths,
            roots=req.roots,
            providers=req.providers,
            include_unsupported=req.include_unsupported,
            max_results_per_provider=req.max_results_per_provider,
            user_id=req.user_id or "global_user",
            namespace=req.namespace or "global",
            project=req.project or "global",
            metadata=req.metadata,
            recursive=req.recursive,
            chronological_order=req.chronological_order,
            max_file_size_bytes=req.max_file_size_bytes,
            chunk_size_chars=req.chunk_size_chars,
            chunk_overlap_chars=req.chunk_overlap_chars,
            min_chunk_chars=req.min_chunk_chars,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error importing legacy sources: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_all", dependencies=[Depends(verify_token)])
async def get_all_memories_endpoint(
    user_id: Optional[str] = "global_user",
    agent_id: Optional[str] = None,
    limit: int = 10,
    namespace: Optional[str] = None,
):
    """Get all memories for a user."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        memories = await memory.get_all(
            user_id=user_id or "global_user",
            agent_id=agent_id,
            limit=limit,
        )
        return {"success": True, "data": memories}
    except Exception as e:
        logger.error("Error getting memories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update", dependencies=[Depends(verify_token)])
async def update_memory_endpoint(req: UpdateMemoryRequest):
    """Update a specific memory."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.update(req.memory_id, req.data)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error updating memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete/{memory_id}", dependencies=[Depends(verify_token)])
async def delete_memory_endpoint(memory_id: str):
    """Delete a specific memory."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.delete(memory_id)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error deleting memory: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_all", dependencies=[Depends(verify_token)])
async def delete_all_endpoint(req: DeleteAllRequest):
    """Delete all memories for a user."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        result = await memory.delete_all(user_id=req.user_id or "global_user")
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error deleting all memories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_batch", dependencies=[Depends(verify_token)])
async def delete_batch_endpoint(req: DeleteBatchRequest):
    """Delete a batch of memories."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Phase 5C.3: Removed global lock
        results = []
        for mem_id in req.memory_ids:
            try:
                # Delete logic is thread-safe via core lock
                await memory.delete(mem_id)
                results.append({"id": mem_id, "status": "deleted"})
            except Exception as e:
                logger.warning("Failed to delete %s: %s", mem_id, e)
                results.append({"id": mem_id, "status": "failed", "error": str(e)})
        return {"success": True, "data": results}
    except Exception as e:
        logger.error("Error batch deleting memories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph", dependencies=[Depends(verify_token)])
async def get_graph_endpoint(user_id: Optional[str] = "global_user"):
    """Get the memory graph (entities and relationships)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        entities = memory._graph.get_all_entities()
        return {
            "success": True,
            "data": {
                "entities": entities,
                "entity_count": len(entities),
            },
        }
    except Exception as e:
        logger.error("Error getting graph: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/handover", dependencies=[Depends(verify_token)])
async def context_handover_endpoint(req: HandoverRequest):
    """Context handover between agents."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Retrieve the source memory
        records = memory._metadata.get_by_ids([req.memory_id])
        if not records:
            raise HTTPException(status_code=404, detail="Memory not found")

        record = records[0]

        # Create a copy in the target agent's namespace
        metadata = record.metadata.copy()
        metadata["handed_over_from"] = req.source_agent_id
        metadata["handover_reason"] = req.reason

        result = await memory.add(
            content=record.content,
            agent_id=req.target_agent_id,
            metadata=metadata,
            namespace=req.target_agent_id,
        )

        logger.info("Handover: %s -> %s", req.source_agent_id, req.target_agent_id)
        return {"success": True, "transferred_id": result.get("id")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Handover failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/federated/search", dependencies=[Depends(verify_token)])
async def federated_search_endpoint(req: SearchMemoryRequest):
    """Federated search across namespaces."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        results = await memory.search(
            query=req.query,
            user_id=req.user_id or "global_user",
            limit=req.limit,
            rerank=req.rerank,
            filters=req.filters,
            namespaces=req.namespaces,
        )
        return {"success": True, "data": results}
    except Exception as e:
        logger.error("Federated search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consolidation/run", dependencies=[Depends(verify_token)])
async def trigger_consolidation():
    """Manually trigger a consolidation cycle."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        if memory._consolidation:
            result = await memory._consolidation.run_cycle()
            return {"success": True, "data": result}
        return {"success": False, "error": "Consolidation daemon not available"}
    except Exception as e:
        logger.error("Consolidation trigger failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/consolidation/status", dependencies=[Depends(verify_token)])
async def consolidation_status():
    """Get consolidation daemon status."""
    if memory is None:
        return {"status": "not_initialized"}

    if memory._consolidation:
        return {"success": True, "data": memory._consolidation.status}
    return {"success": False, "data": {"running": False}}


# --- Phase 6 Endpoints ---

@app.get("/knowledge/temporal", dependencies=[Depends(verify_token)])
async def get_temporal_knowledge_endpoint(
    timestamp: Optional[float] = None,
    limit: int = 50,
    user_id: str = "global_user",
):
    """Query the Temporal Knowledge Graph (scoped to user)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        data = await memory.get_temporal_knowledge(timestamp=timestamp, limit=limit, user_id=user_id)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error("Temporal query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/federation/manifest", dependencies=[Depends(verify_token)])
async def create_federation_manifest_endpoint(
    project: str = "global",
    user_id: str = "global_user",
):
    """Generate a federation manifest for sync (scoped to user)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        fed = await memory.get_federation_manager()
        manifest = await fed.generate_manifest(project=project, user_id=user_id)
        return {"success": True, "data": manifest}
    except Exception as e:
        logger.error("Manifest generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/federation/delta", dependencies=[Depends(verify_token)])
async def calculate_federation_delta_endpoint(local: Dict[str, Any], remote: Dict[str, Any]):
    """Calculate delta between two manifests."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    
    try:
        fed = await memory.get_federation_manager()
        delta = await fed.calculate_delta(local_manifest=local, remote_manifest=remote)
        return {"success": True, "data": delta}
    except Exception as e:
        logger.error("Delta calculation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/federation/bundle", dependencies=[Depends(verify_token)])
async def create_federation_bundle_endpoint(
    memory_ids: List[str],
    user_id: str = "global_user",
):
    """Create a sync bundle for requested memories (scoped to user)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        fed = await memory.get_federation_manager()
        bundle = await fed.create_sync_bundle(memory_ids=memory_ids, user_id=user_id)
        return {"success": True, "data": bundle}
    except Exception as e:
        logger.error("Bundle creation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/federation/apply", dependencies=[Depends(verify_token)])
async def apply_federation_bundle_endpoint(
    bundle: Dict[str, Any],
    user_id: str = "global_user",
):
    """Apply a sync bundle (scoped to user)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        fed = await memory.get_federation_manager()
        applied = await fed.apply_sync_bundle(bundle=bundle, user_id=user_id)
        return {"success": True, "data": {"applied": applied}}
    except Exception as e:
        logger.error("Bundle apply failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main ---

def main():
    config = MuninnConfig.from_env()

    parser = argparse.ArgumentParser(description="Muninn Memory Server")
    parser.add_argument("--host", default=config.server.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.server.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    logger.info("Starting Muninn Memory Server on %s:%d", args.host, args.port)

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=config.server.log_level,
    )


if __name__ == "__main__":
    main()