"""
Mimir Interop Relay — FastAPI Router
======================================
HTTP API surface for the Mimir relay module, exposing:

  POST   /mimir/relay              — execute a relay request (IRP/1)
  GET    /mimir/providers          — score all configured providers
  GET    /mimir/runs               — paginated relay run history
  GET    /mimir/runs/{run_id}      — retrieve a specific run record
  GET    /mimir/runs/{run_id}/audit — audit trail for a run
  GET    /mimir/settings           — fetch interop settings for a user
  POST   /mimir/settings           — update interop settings for a user
  GET    /mimir/connections        — list provider connections for a user
  POST   /mimir/audit/purge        — delete audit events beyond retention window

All endpoints require Bearer token authentication (MUNINN_API_KEY env var).
If MUNINN_API_KEY is unset the server accepts all requests (dev/test mode).

Synchronous MimirStore operations are dispatched via asyncio.to_thread()
so they never block the FastAPI event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .models import (
    InteropSettings,
    IRPEnvelope,
    IRPHop,
    IRPMode,
    IRPPolicy,
    MimirRelayRequest,
    ProviderName,
    RelayResult,
    RunStatus,
)
from .relay import MimirRelay
from .store import MimirStore
from muninn.core.security import verify_api_token

logger = logging.getLogger("Muninn.Mimir.api")

# ---------------------------------------------------------------------------
# Module-level singletons — injected by server.py during lifespan startup
# ---------------------------------------------------------------------------

_relay: Optional[MimirRelay] = None
_store: Optional[MimirStore] = None


def init_mimir(relay: MimirRelay, store: MimirStore) -> None:
    """
    Bind the Mimir API module to its relay and store singletons.

    Called once during FastAPI application lifespan startup, before any
    request is served.  ``server.py`` is responsible for constructing
    ``MimirRelay`` and ``MimirStore`` and calling this function.
    """
    global _relay, _store
    _relay = relay
    _store = store
    logger.info(
        "Mimir API module initialised (relay=%s, store=%s)",
        type(relay).__name__,
        type(store).__name__,
    )


# ---------------------------------------------------------------------------
# Guards — raise 503 when singletons are not yet initialised
# ---------------------------------------------------------------------------


def _get_relay() -> MimirRelay:
    """Return the relay singleton or raise HTTP 503."""
    if _relay is None:
        raise HTTPException(
            status_code=503,
            detail="Mimir relay is not initialised. Check server lifespan configuration.",
        )
    return _relay


def _get_store() -> MimirStore:
    """Return the store singleton or raise HTTP 503."""
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Mimir store is not initialised. Check server lifespan configuration.",
        )
    return _store


# ---------------------------------------------------------------------------
# Authentication — defined locally to avoid circular import with server.py
# ---------------------------------------------------------------------------

_security = HTTPBearer(auto_error=False)


async def _verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> None:
    """
    Validate the Bearer token against the global Muninn auth token.

    Raises HTTP 401 on failure.
    """
    token = credentials.credentials if credentials else None
    if not verify_api_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _ok(data: Any) -> Dict[str, Any]:
    """Wrap payload in the standard Muninn success envelope."""
    return {"success": True, "data": data}


def _fail(message: str, status_code: int = 500) -> HTTPException:
    """
    Build an HTTPException whose detail follows the Muninn error envelope.

    Raise the returned exception — it is not raised here to keep mypy happy.
    """
    return HTTPException(
        status_code=status_code,
        detail={"success": False, "error": message},
    )


# ---------------------------------------------------------------------------
# Pydantic request bodies for endpoints that don't reuse Mimir models
# ---------------------------------------------------------------------------


class AuditPurgeRequest(BaseModel):
    """Request body for ``POST /mimir/audit/purge``."""

    retention_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description=(
            "Delete audit events older than this many days. "
            "Must be in the range [1, 3650]."
        ),
    )


# ---------------------------------------------------------------------------
# Router definition
# ---------------------------------------------------------------------------

mimir_router = APIRouter(
    prefix="/mimir",
    tags=["mimir"],
    dependencies=[Depends(_verify_token)],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_probe_envelope() -> IRPEnvelope:
    """
    Construct a minimal ``IRPEnvelope`` used as a diagnostic probe for the
    ``GET /mimir/providers`` endpoint.

    The envelope carries no real instruction — it exists only to allow the
    ``MemoryAwareRouter`` to compute availability and composite scores for
    each configured provider.
    """
    return IRPEnvelope.model_validate(
        {
            "id": "MIMIR_PROVIDER_PROBE",
            "from": "muninn",
            "to": "auto",
            "mode": IRPMode.ADVISORY.value,
            "hop": IRPHop().model_dump(),
            "policy": IRPPolicy().model_dump(),
            "request": {"instruction": "[provider availability probe]"},
            "context": {},
            "ts": time.time(),
        }
    )


def _serialise_run_status(status: Optional[str]) -> Optional[str]:
    """
    Validate and normalise a ``status`` query parameter against
    ``RunStatus`` enum values.

    Returns the validated string or raises HTTP 422.
    """
    if status is None:
        return None
    valid = {s.value for s in RunStatus}
    if status not in valid:
        raise _fail(
            "Invalid status value. ",
            status_code=422,
        )
    return status


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@mimir_router.post(
    "/relay",
    summary="Execute an IRP/1 relay request",
    response_description="RelayResult with run metadata and provider output",
)
async def relay_request(request: MimirRelayRequest) -> Dict[str, Any]:
    """
    Dispatch an instruction to an AI provider via the IRP/1 relay protocol.

    **Modes**

    | mode | description |
    |------|-------------|
    | ``A`` | Advisory — single provider, advisory response |
    | ``B`` | Structured — single provider, structured JSON output |
    | ``C`` | Reconcile — fan-out to all viable providers; reconcile results |

    **Target selection**

    Set ``target`` to ``"auto"`` (default) for automatic routing based on
    capability, availability, cost, safety, and historical success scores.
    Or specify a provider explicitly: ``"claude_code"``, ``"codex_cli"``,
    ``"gemini_cli"``.

    The response always contains a ``RelayResult`` — errors are surfaced as
    ``status="FAILED"`` / ``status="POLICY_BLOCKED"`` within the result
    rather than as HTTP error codes, ensuring callers always receive
    structured metadata.
    """
    relay = _get_relay()
    try:
        result: RelayResult = await relay.run(request)
        return _ok(result.model_dump(mode="json"))
    except Exception as exc:
        logger.exception("Unexpected error in POST /mimir/relay: %s", exc)
        raise _fail("Internal relay error.", status_code=500)


@mimir_router.get(
    "/providers",
    summary="Score all configured providers",
    response_description="Mapping of provider name to RoutingScore breakdown",
)
async def list_providers(
    user_id: str = Query(
        default="global_user",
        description=(
            "User context for history-based scoring. "
            "Affects the history_score component via recent run outcomes."
        ),
    ),
) -> Dict[str, Any]:
    """
    Return routing scores for every configured provider.

    Scores are computed concurrently against live provider availability
    and recent run history from the Muninn memory store.  Use this
    endpoint for health dashboards, routing diagnostics, and capacity
    planning.

    Score components (all in ``[0.0, 1.0]``):

    | field | weight | description |
    |-------|--------|-------------|
    | ``capability_score`` | 0.35 | how well the provider handles this mode |
    | ``availability_score`` | 0.25 | is the provider reachable right now? |
    | ``cost_score`` | 0.15 | estimated cost (lower cost → higher score) |
    | ``safety_score`` | 0.15 | policy compliance posture |
    | ``history_score`` | 0.10 | recent success rate from Muninn memory |
    | ``composite`` | — | weighted sum of the above components |
    """
    relay = _get_relay()
    try:
        envelope = _build_probe_envelope()
        router = relay._router  # type: ignore[attr-defined]
        scores = await router.score_all(envelope)

        payload: Dict[str, Any] = {}
        for provider, score in scores.items():
            payload[provider.value] = {
                "provider": score.provider.value,
                "capability_score": score.capability_score,
                "availability_score": score.availability_score,
                "cost_score": score.cost_score,
                "safety_score": score.safety_score,
                "history_score": score.history_score,
                "composite": score.composite,
            }

        return _ok(payload)

    except Exception as exc:
        logger.exception("Error computing provider scores: %s", exc)
        raise _fail("Provider scoring failed.", status_code=500)


@mimir_router.get(
    "/runs",
    summary="List relay run history",
    response_description="Paginated list of run records",
)
async def list_runs(
    user_id: str = Query(
        default="global_user",
        description="Filter runs to this user.",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of run records to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip (for pagination).",
    ),
    provider: Optional[str] = Query(
        default=None,
        description="Optional provider filter (e.g. claude_code, codex_cli, gemini_cli).",
    ),
    status: Optional[str] = Query(
        default=None,
        description=(
            "Filter by run status. One of: "
            "PENDING, RUNNING, SUCCESS, FAILED, CANCELLED, POLICY_BLOCKED."
        ),
    ),
) -> Dict[str, Any]:
    """
    Return a paginated list of relay run records for the specified user.

    Records are returned in reverse-chronological order (newest first).
    Use ``offset`` and ``limit`` for pagination.  Filter by ``status``
    to narrow results (e.g. ``status=FAILED`` for error analysis).
    """
    store = _get_store()

    try:
        normalised_status = _serialise_run_status(status)
    except HTTPException:
        raise

    try:
        runs: List[Dict[str, Any]] = await asyncio.to_thread(
            store.list_runs,
            user_id=user_id,
            provider=provider,
            limit=limit,
            offset=offset,
            status=normalised_status,
        )
        return _ok(
            {
                "user_id": user_id,
                "limit": limit,
                "offset": offset,
                "provider_filter": provider,
                "status_filter": normalised_status,
                "count": len(runs),
                "runs": runs,
            }
        )
    except Exception as exc:
        logger.exception("Error listing runs for user=%s: %s", user_id, exc)
        raise _fail("Failed to list runs.", status_code=500)


@mimir_router.get(
    "/runs/{run_id}",
    summary="Get a specific run record",
    response_description="Full run record including provider, latency, and token counts",
)
async def get_run(run_id: str) -> Dict[str, Any]:
    """
    Retrieve the complete run record for a single relay invocation.

    Returns all metadata including mode, provider selection, latency,
    token usage, redaction count, error details (if any), and the
    prompt hash used for audit correlation.
    """
    store = _get_store()
    try:
        record: Optional[Dict[str, Any]] = await asyncio.to_thread(
            store.get_run, run_id
        )
        if record is None:
            raise _fail("Run not found.", status_code=404)
        return _ok(record)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching run_id=%s: %s", run_id, exc)
        raise _fail("Failed to fetch run.", status_code=500)


@mimir_router.get(
    "/runs/{run_id}/audit",
    summary="Get audit trail for a run",
    response_description="Ordered list of audit events emitted during the run",
)
async def get_run_audit(
    run_id: str,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of audit events to return.",
    ),
) -> Dict[str, Any]:
    """
    Return the ordered audit event trail for a specific relay run.

    Events are returned in ascending timestamp order and cover the full
    relay lifecycle: ``relay_start`` → routing decisions → provider calls
    → reconciliation (Mode C) → ``relay_complete`` / ``relay_failed``.

    Policy violations (``policy_blocked``, ``hop_limit_exceeded``) and
    secret redactions (``redaction_applied``) are also recorded here.
    """
    store = _get_store()
    try:
        events: List[Dict[str, Any]] = await asyncio.to_thread(
            store.get_audit_events, run_id, limit
        )
        return _ok(
            {
                "run_id": run_id,
                "count": len(events),
                "events": events,
            }
        )
    except Exception as exc:
        logger.exception(
            "Error fetching audit trail for run_id=%s: %s", run_id, exc
        )
        raise _fail("Failed to fetch audit trail.", status_code=500)


@mimir_router.get(
    "/settings",
    summary="Get interop settings for a user",
    response_description="InteropSettings object with current IRP/1 policy",
)
async def get_settings(
    user_id: str = Query(
        default="global_user",
        description="User whose settings to retrieve.",
    ),
) -> Dict[str, Any]:
    """
    Retrieve the IRP/1 policy settings for the specified user.

    If the user has no custom settings persisted, the system defaults are
    returned (interop enabled, all providers allowed, balanced redaction,
    2-hop limit).
    """
    store = _get_store()
    try:
        settings: InteropSettings = await asyncio.to_thread(
            store.get_settings, user_id
        )
        return _ok(settings.model_dump(mode="json"))
    except Exception as exc:
        logger.exception(
            "Error fetching settings for user=%s: %s", user_id, exc
        )
        raise _fail("Failed to fetch settings.", status_code=500)


@mimir_router.post(
    "/settings",
    summary="Update interop settings for a user",
    response_description="Confirmation with the updated user_id",
)
async def update_settings(settings: InteropSettings) -> Dict[str, Any]:
    """
    Persist updated IRP/1 policy settings for a user.

    The complete ``InteropSettings`` object must be supplied — partial
    updates are not supported.  Changes take effect on the next relay
    invocation for the specified ``user_id``.

    **Key fields**

    | field | description |
    |-------|-------------|
    | ``enabled`` | globally enable / disable the relay for this user |
    | ``allowed_targets`` | whitelist of provider names |
    | ``policy_tools`` | ``"allowed"`` or ``"forbidden"`` |
    | ``hop_max`` | maximum relay hop depth (1–4) |
    | ``redaction`` | secret redaction level |
    | ``audit_retention_days`` | how long audit events are retained |
    """
    store = _get_store()
    try:
        await asyncio.to_thread(store.update_settings, settings)
        return _ok({"updated": True, "user_id": settings.user_id})
    except Exception as exc:
        logger.exception(
            "Error updating settings for user=%s: %s", settings.user_id, exc
        )
        raise _fail("Failed to update settings.", status_code=500)


@mimir_router.get(
    "/connections",
    summary="List provider connections for a user",
    response_description="List of connection records (credentials omitted)",
)
async def list_connections(
    user_id: str = Query(
        default="global_user",
        description="User whose provider connections to list.",
    ),
) -> Dict[str, Any]:
    """
    Return the registered provider connection records for a user.

    Each record contains provider metadata (name, last-seen timestamp,
    connection status) but **never** secrets or credentials.  Use this
    endpoint to verify which providers are configured and active.
    """
    store = _get_store()
    try:
        connections: List[Dict[str, Any]] = await asyncio.to_thread(
            store.list_connections, user_id
        )
        return _ok(
            {
                "user_id": user_id,
                "count": len(connections),
                "connections": connections,
            }
        )
    except Exception as exc:
        logger.exception(
            "Error listing connections for user=%s: %s", user_id, exc
        )
        raise _fail("Failed to list connections.", status_code=500)


@mimir_router.post(
    "/audit/purge",
    summary="Purge old audit events",
    response_description="Number of audit rows deleted",
)
async def purge_audit(body: AuditPurgeRequest) -> Dict[str, Any]:
    """
    Delete audit events that are older than ``retention_days`` days.

    This operation is **irreversible**.  It is intended for scheduled
    maintenance jobs (e.g. a nightly cron) to keep the audit table
    within manageable bounds.  The default ``retention_days=90`` matches
    the ``InteropSettings`` default.

    Returns the number of rows deleted.
    """
    store = _get_store()
    try:
        deleted: int = await asyncio.to_thread(
            store.purge_old_audit_events, body.retention_days
        )
        logger.info(
            "Audit purge complete: deleted=%d retention_days=%d",
            deleted,
            body.retention_days,
        )
        return _ok(
            {
                "deleted": deleted,
                "retention_days": body.retention_days,
            }
        )
    except Exception as exc:
        logger.exception("Error during audit purge: %s", exc)
        raise _fail("Audit purge failed.", status_code=500)