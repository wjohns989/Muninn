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
import portalocker

from muninn.core.memory import MuninnMemory
from muninn.core.config import MuninnConfig, SUPPORTED_MODEL_PROFILES
from muninn.core.security import SecurityContext, verify_token as core_verify_token, initialize_security, get_token, is_security_enabled
from muninn.version import __version__
from muninn.ingestion.pipeline import (
    MAX_CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_SIZE_CHARS,
    MAX_INGEST_FILE_SIZE_BYTES,
)
from muninn.ingestion.periodic import (
    PeriodicIngestionScheduler,
    PeriodicIngestionSettings,
)
from muninn.ingestion.legacy_scheduler import LegacyDiscoveryScheduler
from muninn.core.types import (
    AddMemoryRequest,
    SearchMemoryRequest,
    UpdateMemoryRequest,
    MemoryType,
    MediaType,
    Provenance,
)
from muninn.retrieval.synthesis import synthesize_hunt_results
from muninn.mimir.api import init_mimir, mimir_router
from muninn.mimir.relay import MimirRelay
from muninn.mimir.store import MimirStore

# Configure detailed logging to file with a robust, absolute path
server_log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'muninn_server.log'))
file_handler = logging.FileHandler(server_log_path, mode='a')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("Muninn")

# --- Global State ---
memory: Optional[MuninnMemory] = None
GLOBAL_AUTH_TOKEN: Optional[str] = None
_mimir_store: Optional[MimirStore] = None
_mimir_relay: Optional[MimirRelay] = None
_periodic_ingestion: Optional[PeriodicIngestionScheduler] = None
_legacy_discovery: Optional[LegacyDiscoveryScheduler] = None
_SERVER_INSTANCE_LOCK_HANDLE: Optional[portalocker.Lock] = None
_SERVER_INSTANCE_LOCK_PATH: Optional[Path] = None


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


class ReasoningRequest(BaseModel):
    query: str
    context: Optional[str] = None
    user_id: str = "global_user"
    limit: int = 10


class DistillationRequest(BaseModel):
    force: bool = True


class CorrectionRequest(BaseModel):
    memory_id: str
    correction: str


class ForagingRequest(BaseModel):
    query: str
    ambiguity_threshold: float = 0.7
    user_id: str = "global_user"
    limit: int = 10


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
    max_results_per_provider: int = Field(default=50000, ge=1, le=50000)
    use_cache: bool = True


class IngestLegacySourcesRequest(BaseModel):
    selected_source_ids: List[str] = Field(default_factory=list)
    selected_paths: List[str] = Field(default_factory=list)
    roots: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)
    include_unsupported: bool = False
    max_results_per_provider: int = Field(default=50000, ge=1, le=50000)
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


class HuntMemoryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    limit: int = 10
    depth: int = 2
    namespaces: Optional[List[str]] = None
    media_type: Optional[MediaType] = None
    synthesize: bool = False  # v3.18.0: optional LLM narration of discovery path


# --- Security ---
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    """FastAPI dependency for token verification."""
    if not is_security_enabled():
        return credentials
    token = credentials.credentials if credentials else None
    if not core_verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


def _server_instance_lock_timeout_seconds() -> float:
    raw = os.environ.get("MUNINN_SERVER_INSTANCE_LOCK_TIMEOUT_SEC", "0.25").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "Invalid MUNINN_SERVER_INSTANCE_LOCK_TIMEOUT_SEC='%s'; using default 0.25s",
            raw,
        )
        return 0.25


def _acquire_server_instance_lock(config: MuninnConfig) -> None:
    """
    Acquire an exclusive process-wide server lease for the configured data dir.

    This prevents duplicate `server.py` instances from racing into Qdrant local
    mode, which is single-process per storage path.
    """
    global _SERVER_INSTANCE_LOCK_HANDLE, _SERVER_INSTANCE_LOCK_PATH

    data_dir = Path(config.data_dir)
    lock_path = data_dir / ".muninn_server.instance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_handle = portalocker.Lock(
        str(lock_path),
        mode="a",
        timeout=_server_instance_lock_timeout_seconds(),
        flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        fail_when_locked=True,
    )

    try:
        lock_handle.acquire()
    except portalocker.exceptions.LockException as exc:
        raise RuntimeError(
            "Muninn server instance lock is already held for data directory "
            f"'{data_dir}'. Reuse the existing server or stop it before starting "
            "another instance."
        ) from exc

    _SERVER_INSTANCE_LOCK_HANDLE = lock_handle
    _SERVER_INSTANCE_LOCK_PATH = lock_path
    logger.info("Acquired server instance lock: %s", lock_path)


def _release_server_instance_lock() -> None:
    """Release the process-wide server lease if held."""
    global _SERVER_INSTANCE_LOCK_HANDLE, _SERVER_INSTANCE_LOCK_PATH
    lock_handle = _SERVER_INSTANCE_LOCK_HANDLE
    lock_path = _SERVER_INSTANCE_LOCK_PATH
    _SERVER_INSTANCE_LOCK_HANDLE = None
    _SERVER_INSTANCE_LOCK_PATH = None
    if lock_handle is None:
        return

    try:
        lock_handle.release()
    except Exception:
        pass
    try:
        lock_handle.close()
    except Exception:
        pass
    if lock_path is not None:
        logger.info("Released server instance lock: %s", lock_path)


# --- Application Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, _mimir_store, _mimir_relay, _periodic_ingestion, _legacy_discovery

    logger.info("Muninn Server starting...")

    try:
        # Initialize configuration from environment
        config = MuninnConfig.from_env()

        # Auth Token Initialization (Phase 10 / v3.7.0)
        initialize_security(config.server.auth_token)

        # Single-owner guard for local embedded stores (especially Qdrant local).
        _acquire_server_instance_lock(config)

        # Initialize memory engine
        memory = MuninnMemory(config)
        await memory.initialize()

        # Phase 5C.3: Removed GLOBAL_LOCK in favor of granular MuninnMemory._write_lock
        # GLOBAL_LOCK = asyncio.Lock()
        # logger.info("Global concurrency lock initialized")

        # Initialise Mimir IRP/1 interop relay.  MimirStore shares the Muninn
        # metadata SQLite connection so all Mimir tables co-locate in the same
        # WAL-mode database file — no extra file, no separate connection pool.
        # The connection is owned by SQLiteMetadataStore; memory.shutdown()
        # closes it, so MimirStore has no separate teardown responsibility.
        _mimir_store = MimirStore(memory._metadata._get_conn())
        _mimir_relay = MimirRelay(
            mimir_store=_mimir_store,
            metadata_store=memory._metadata
        )
        init_mimir(_mimir_relay, _mimir_store)
        logger.info("Mimir relay initialised (db=%s)", memory._metadata.db_path)

        periodic_settings = PeriodicIngestionSettings.from_env()
        _periodic_ingestion = PeriodicIngestionScheduler(
            memory=memory,
            settings=periodic_settings,
        )
        if periodic_settings.enabled_on_startup:
            started = await _periodic_ingestion.start()
            if started:
                logger.info(
                    "Periodic ingestion enabled on startup (interval=%.1fs)",
                    periodic_settings.interval_seconds,
                )
            else:
                logger.warning(
                    "Periodic ingestion requested on startup but scheduler did not start"
                )

        if config.legacy_discovery.enabled:
            _legacy_discovery = LegacyDiscoveryScheduler(
                memory=memory,
                interval_seconds=config.legacy_discovery.interval_hours * 3600.0,
            )
            await _legacy_discovery.start()
            logger.info(
                "Legacy discovery scheduler enabled (interval=%.1fh)",
                config.legacy_discovery.interval_hours,
            )

        yield
    finally:
        logger.info("Shutting down Muninn Server...")
        if _legacy_discovery:
            await _legacy_discovery.stop()
            _legacy_discovery = None
        if _periodic_ingestion:
            await _periodic_ingestion.stop()
            _periodic_ingestion = None
        if memory:
            await memory.shutdown()
        _release_server_instance_lock()
        logger.info("Muninn Server stopped.")


app = FastAPI(
    title="Muninn Memory Server",
    description="Local-first persistent memory for AI agents — Muninn native engine",
    version=__version__,
    lifespan=lifespan,
)

# Register Mimir IRP/1 relay router (/mimir/* endpoints).
# The router carries its own /mimir prefix and _verify_token dependency;
# no additional prefix or auth wrapper is needed here.
app.include_router(mimir_router)

# --- CORS ---
# Security design: Muninn runs on localhost only and all data-mutating endpoints
# require a Bearer token (Depends(verify_token)).  The wildcard origin is needed
# so the static dashboard (opened as file:// or from a different local port) can
# reach the API.  Per the Fetch spec, wildcard origins CANNOT be combined with
# allow_credentials=True, so session cookies are not usable — this is intentional.
# Authentication is Bearer-token only (Authorization header), which browsers do
# NOT send automatically; no cross-site request forgery is possible.
# allow_methods is restricted to the verbs actually used by the server.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
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
    content = _load_dashboard_html()
    
    # Automate Auth Token handling for the local dashboard (v3.18.2)
    # We inject the current token so the user doesn't have to enter it manually.
    try:
        active_token = get_token()
        no_auth = not is_security_enabled()
        
        # Inject active token using robust placeholder
        content = content.replace("{{MUNINN_TOKEN}}", active_token)
        
        # Inject security status for absolute bypass in UI
        if no_auth:
            content = content.replace(
                'let SECURITY_ENABLED = true;',
                'let SECURITY_ENABLED = false;'
            )
            
    except Exception as e:
        logger.warning("Failed to inject auth token into dashboard: %s", e)
        
    return HTMLResponse(content=content)

@app.get("/dashboard.css")
async def dashboard_css():
    """Serve the dashboard CSS file."""
    css_path = Path(__file__).with_name("dashboard.css")
    try:
        return HTMLResponse(content=css_path.read_text(encoding="utf-8"), media_type="text/css")
    except Exception as exc:
        logger.error("Failed to load dashboard CSS: %s", exc)
        return HTMLResponse(content="", status_code=404)

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
                    media_type=req.media_type,
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
            media_type=req.media_type,
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
            media_type=req.media_type,
            explain=req.explain,
        )

        return {"success": True, "data": results}
    except Exception as e:
        logger.error("Error searching memories: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hunt", dependencies=[Depends(verify_token)])
async def hunt_memory_endpoint(req: HuntMemoryRequest):
    """Perform agentic multi-hop retrieval to discover hidden context.     

    When ``synthesize=True`` and ANTHROPIC_API_KEY is configured, the response
    includes a ``synthesis`` field: a brief LLM-generated narrative explaining
    what was found and why.  Gracefully returns ``synthesis: ""`` on any failure.
    """
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        results = await memory.hunt(
            query=req.query,
            user_id=req.user_id or "global_user",
            agent_id=req.agent_id,
            limit=req.limit,
            depth=req.depth,
            namespaces=req.namespaces,
            media_type=req.media_type,
        )

        synthesis = ""
        if req.synthesize and results:
            synthesis = await synthesize_hunt_results(req.query, results)

        return {"success": True, "data": results, "synthesis": synthesis}
    except Exception as e:
        logger.error("Error hunting memories: %s", e)
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


@app.get("/profiles/model/alerts", dependencies=[Depends(verify_token)])
async def get_model_profile_alerts_endpoint(
    window_seconds: Optional[float] = None,
    churn_threshold: Optional[int] = None,
    source_churn_threshold: Optional[int] = None,
    distinct_sources_threshold: Optional[int] = None,
):
    """Evaluate profile-policy mutation churn against alert thresholds."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    try:
        result = await memory.get_model_profile_alerts(
            window_seconds=window_seconds,
            churn_threshold=churn_threshold,
            source_churn_threshold=source_churn_threshold,
            distinct_sources_threshold=distinct_sources_threshold,
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error evaluating model profile alerts: %s", e)
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


@app.post("/ingest/legacy/discover", dependencies=[Depends(verify_token)])
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
            use_cache=req.use_cache,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error discovering legacy sources: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/legacy/catalog", dependencies=[Depends(verify_token)])
async def legacy_catalog_endpoint(
    limit: int = 1000,
    offset: int = 0,
    providers: Optional[str] = None,
):
    """Retrieve the cached catalog of discovered legacy sources."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        provider_list = [p.strip() for p in providers.split(",")] if providers else None
        result = await memory.discover_legacy_sources(
            use_cache=True,
            providers=provider_list,
            max_results_per_provider=limit,
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Error fetching legacy catalog: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/legacy/import", dependencies=[Depends(verify_token)])
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


@app.post("/ingest/legacy/import-all", dependencies=[Depends(verify_token)])
async def ingest_all_legacy_sources_endpoint():
    """Discover and import ALL legacy sources in one shot."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Step 1: Discover everything
        result = await memory.discover_legacy_sources(
            use_cache=False,
            max_results_per_provider=50000,
        )

        sources = result.get("sources", [])
        if not sources:
            return {"success": True, "data": {"imported": 0, "total_discovered": 0, "message": "No sources found"}}

        # Step 2: Extract all source IDs
        all_ids = [s["source_id"] for s in sources if s.get("parser_supported", False)]
        logger.info("Bulk import: %d parser-supported sources out of %d total", len(all_ids), len(sources))

        if not all_ids:
            return {"success": True, "data": {"imported": 0, "total_discovered": len(sources), "message": "No parser-supported sources"}}

        # Step 3: Import in batches to avoid overwhelming the pipeline
        batch_size = 50
        total_imported = 0
        errors = []

        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i:i + batch_size]
            try:
                batch_result = await memory.ingest_legacy_sources(
                    selected_source_ids=batch,
                    max_results_per_provider=50000,
                )
                count = batch_result.get("count", 0) if isinstance(batch_result, dict) else 0
                total_imported += count
                logger.info("Bulk import batch %d/%d: imported %d nodes",
                           (i // batch_size) + 1,
                           (len(all_ids) + batch_size - 1) // batch_size,
                           count)
            except Exception as batch_err:
                logger.warning("Bulk import batch %d failed: %s", (i // batch_size) + 1, batch_err)
                errors.append(str(batch_err))

        return {
            "success": True,
            "data": {
                "imported": total_imported,
                "total_discovered": len(sources),
                "total_supported": len(all_ids),
                "batches": (len(all_ids) + batch_size - 1) // batch_size,
                "errors": errors[:10] if errors else [],
            }
        }
    except Exception as e:
        logger.error("Error in bulk legacy import: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/legacy/status", dependencies=[Depends(verify_token)])
async def legacy_discovery_status_endpoint():
    """Get runtime status for the background legacy scan scheduler."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    if _legacy_discovery is None:
        return {
            "success": True,
            "data": {
                "running": False,
                "last_run_at": None,
                "last_sync_result": None,
                "cache_stats": {},
            },
        }

    return {"success": True, "data": _legacy_discovery.status}


@app.post("/ingest/legacy/run", dependencies=[Depends(verify_token)])
async def legacy_discovery_run_endpoint():
    """Manually trigger a background legacy discovery scan."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    if _legacy_discovery is None:
        raise HTTPException(status_code=503, detail="Legacy discovery scheduler not initialized")

    result = await _legacy_discovery.trigger_once()
    return {"success": True, "data": result}


@app.get("/ingest/periodic/status", dependencies=[Depends(verify_token)])
async def periodic_ingestion_status_endpoint():
    """Get runtime status for the periodic ingestion scheduler."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    if _periodic_ingestion is None:
        return {
            "success": True,
            "data": {
                "configured": {"enabled_on_startup": False, "sources": []},
                "runtime": {"running": False, "inflight": False},
                "last_result": None,
            },
        }

    return {"success": True, "data": _periodic_ingestion.status}


@app.post("/ingest/periodic/run", dependencies=[Depends(verify_token)])
async def periodic_ingestion_run_endpoint():
    """Manually trigger one periodic-ingestion cycle."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    if _periodic_ingestion is None:
        raise HTTPException(status_code=503, detail="Periodic ingestion scheduler not initialized")

    result = await _periodic_ingestion.trigger_once(reason="manual")
    return {"success": bool(result.get("success")), "data": result}


@app.post("/ingest/periodic/start", dependencies=[Depends(verify_token)])
async def periodic_ingestion_start_endpoint():
    """Start periodic ingestion loop without restarting server."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    if _periodic_ingestion is None:
        raise HTTPException(status_code=503, detail="Periodic ingestion scheduler not initialized")

    started = await _periodic_ingestion.start()
    return {"success": True, "data": {"started": started, "status": _periodic_ingestion.status}}


@app.post("/ingest/periodic/stop", dependencies=[Depends(verify_token)])
async def periodic_ingestion_stop_endpoint():
    """Stop periodic ingestion loop without restarting server."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    if _periodic_ingestion is None:
        raise HTTPException(status_code=503, detail="Periodic ingestion scheduler not initialized")

    stopped = await _periodic_ingestion.stop()
    return {"success": True, "data": {"stopped": stopped, "status": _periodic_ingestion.status}}


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


@app.post("/reasoning/detect-gaps", dependencies=[Depends(verify_token)])  
async def detect_gaps_endpoint(req: ReasoningRequest):
    """Analyze query and context for missing information (Omission Filtering)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        # Import dynamically to avoid circular deps at module level if any
        from muninn.reasoning.omission import OmissionDetector
        detector = OmissionDetector(memory)
        
        result = await detector.detect_gaps(
            query=req.query,
            context=req.context,
            user_id=req.user_id,
            limit=req.limit
        )
        return {"success": True, "data": result.model_dump()}
    except Exception as e:
        logger.error("Gap detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/distill", dependencies=[Depends(verify_token)])
async def trigger_distillation_endpoint(req: DistillationRequest):
    """Trigger Knowledge Distillation."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        from muninn.optimization.distillation import DistillationDaemon
        # Use existing daemon if available, or create transient one
        # Ideally, MuninnMemory should manage the daemon instance.
        # Assuming MuninnMemory doesn't have it yet (Phase 25), we create one.
        # But wait, daemon is stateful. We should attach it to memory in initialize().
        # For now, we'll instantiate ad-hoc for the trigger or check if memory has it.
        
        daemon = getattr(memory, "_distillation_daemon", None)
        if not daemon:
            daemon = DistillationDaemon(memory)
            # Attach for reuse
            setattr(memory, "_distillation_daemon", daemon)
            
        result = await daemon.run_cycle()
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Distillation trigger failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/correct", dependencies=[Depends(verify_token)])
async def correct_memory_endpoint(req: CorrectionRequest):
    """Execute Memory Surgery."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        from muninn.optimization.surgeon import MemorySurgeon
        surgeon = MemorySurgeon(memory)
        success = await surgeon.correct_memory(req.memory_id, req.correction)
        if not success:
             raise HTTPException(status_code=404, detail="Memory not found or correction failed")
        return {"success": True, "data": {"memory_id": req.memory_id, "status": "corrected"}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Memory correction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/forage", dependencies=[Depends(verify_token)])
async def forage_knowledge_endpoint(req: ForagingRequest):
    """Execute Epistemic Foraging (Active Inference)."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        from muninn.optimization.foraging import ForagingEngine
        # Initial search to get baseline results for entropy analysis
        initial_results = await memory.search(
            query=req.query,
            user_id=req.user_id,
            limit=req.limit
        )
        
        engine = ForagingEngine(memory)
        result = await engine.forage(
            initial_query=req.query,
            initial_results=initial_results,
            ambiguity_threshold=req.ambiguity_threshold
        )
        return {"success": True, "data": result}
    except Exception as e:
        logger.error("Foraging failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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

    import sys
    import os

    try:
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=config.server.log_level,
        )
    except OSError as e:
        if e.errno in (98, 10048):
            print(f"\n\033[91m{'='*60}\033[0m")
            print("\033[91mCRITICAL ERROR: PORT ALREADY IN USE\033[0m")
            print(f"\033[91m{'='*60}\033[0m")
            print(f"Muninn failed to start because port {args.port} is already bound.")
            print("This usually means another instance of Muninn is currently running in the background.")
            print("")
            error_msg = f"Failed to start server on port {args.port}. Port is likely in use."
            logger.error(error_msg)
            print(f"\n[ERROR] Port {args.port} is already in use.")
            print(f"Please check the server log at: {server_log_path}")
            print("\nTo forcefully kill the existing process using this port on Windows, use:")
            print(f"  netstat -ano | findstr :{args.port}")
            print("  taskkill /PID <PID> /F")
            print(f"To forcefully kill the existing process using this port on Linux/macOS, use:")
            print(f"  lsof -i :{args.port}")
            print("  kill -9 <PID>")
            print(f"\033[91m{'='*60}\033[0m\n")
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()