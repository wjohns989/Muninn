"""
Mimir Interop Relay — MimirRelay Orchestrator
=============================================
High-level orchestrator for the Interoperability Relay Protocol v1 (IRP/1).

Responsibilities
----------------
  1. Parse and validate incoming MimirRelayRequest
  2. Load per-user InteropSettings and merge with request overrides
  3. Enforce IRP/1 policy (hop limits, prompt size, allowed targets)
  4. Apply secret redaction to prompt and output
  5. Route to the optimal provider via MemoryAwareRouter
  6. Execute the relay call via the appropriate adapter
  7. For Mode C (RECONCILE): fan-out to all viable providers, then reconcile
  8. Persist run records and emit audit events to MimirStore
  9. Return a structured RelayResult to the caller

Exception flow
--------------
  run()
    └─ _execute()          ← catches PolicyError → _policy_blocked()
         |                    catches RoutingError  → _fail(ROUTING_FAILED)
         └─ _pipeline()    ← PolicyError and RoutingError propagate upward
              └─ _run_mode_ab() / _run_mode_c()

All unexpected exceptions in run() are caught and returned as INTERNAL_ERROR.
_emit_audit() and _upsert_run() are fire-and-forget: they swallow errors so
storage failures never kill a relay call.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional

from .adapters import get_adapter
from .models import (
    AuditEvent,
    AuditEventType,
    IRPEnvelope,
    IRPHop,
    IRPMode,
    IRPRequest,
    InteropSettings,
    MimirRelayRequest,
    ProviderName,
    ProviderResult,
    RelayResult,
    RunRecord,
    RunStatus,
)
from .policy import PolicyEngine, PolicyError
from .reconcile import Reconciler
from .routing import MemoryAwareRouter, RoutingError

logger = logging.getLogger("Muninn.Mimir.relay")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_id() -> str:
    """Generate a 26-character uppercase hex identifier (no external deps)."""
    return uuid.uuid4().hex.upper()[:26]


def _elapsed_ms(start: float) -> int:
    """Return elapsed time in milliseconds since *start* (monotonic clock)."""
    return int((time.monotonic() - start) * 1000)


# ---------------------------------------------------------------------------
# MimirRelay
# ---------------------------------------------------------------------------


class MimirRelay:
    """
    High-level orchestrator for IRP/1 relay calls.

    Parameters
    ----------
    store : optional MimirStore
        If provided, run records and audit events are persisted, and
        InteropSettings are loaded per-user. Without a store, defaults
        are used and nothing is written to disk.
    """

    def __init__(self, store=None) -> None:
        self._store = store
        self._router = MemoryAwareRouter(mimir_store=store)
        self._reconciler = Reconciler(mimir_store=store)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: MimirRelayRequest) -> RelayResult:
        """
        Execute an IRP/1 relay call and return a RelayResult.

        This method is the sole public entry point.  All policy errors,
        routing failures, and unexpected exceptions are caught here and
        converted to a RelayResult with an appropriate error status.

        Parameters
        ----------
        request : MimirRelayRequest
            Validated relay request from the MCP tool layer.

        Returns
        -------
        RelayResult
            Always returned — never raises.
        """
        start_ts = time.monotonic()
        run_id = _new_id()
        irp_id = _new_id()
        mode = IRPMode(request.mode)

        run_record = RunRecord(
            run_id=run_id,
            irp_id=irp_id,
            user_id=request.user_id,
            mode=mode,
            status=RunStatus.PENDING,
        )
        self._upsert_run(run_record)

        try:
            return await self._execute(
                run_id=run_id,
                irp_id=irp_id,
                request=request,
                mode=mode,
                run_record=run_record,
                start_ts=start_ts,
            )
        except Exception as exc:
            logger.exception(
                "relay run=%s unexpected internal error: %s", run_id, exc
            )
            return self._fail(
                run_id=run_id,
                irp_id=irp_id,
                mode=mode,
                error_code="INTERNAL_ERROR",
                error_message=str(exc),
                run_record=run_record,
                start_ts=start_ts,
            )

    # ------------------------------------------------------------------
    # Two-level exception boundary
    # ------------------------------------------------------------------

    async def _execute(
        self,
        *,
        run_id: str,
        irp_id: str,
        request: MimirRelayRequest,
        mode: IRPMode,
        run_record: RunRecord,
        start_ts: float,
    ) -> RelayResult:
        """
        Second-level exception handler.

        Catches PolicyError → _policy_blocked()
        Catches RoutingError → _fail(ROUTING_FAILED)
        All other exceptions propagate to run() for INTERNAL_ERROR handling.
        """
        try:
            return await self._pipeline(
                run_id=run_id,
                irp_id=irp_id,
                request=request,
                mode=mode,
                run_record=run_record,
                start_ts=start_ts,
            )
        except PolicyError as exc:
            logger.info(
                "relay run=%s policy blocked: code=%s msg=%s",
                run_id, exc.code, exc.message,
            )
            return self._policy_blocked(
                run_id=run_id,
                irp_id=irp_id,
                mode=mode,
                error_code=exc.code,
                error_message=exc.message,
                run_record=run_record,
                start_ts=start_ts,
            )
        except RoutingError as exc:
            logger.warning(
                "relay run=%s routing failed: %s", run_id, exc.message
            )
            return self._fail(
                run_id=run_id,
                irp_id=irp_id,
                mode=mode,
                error_code="ROUTING_FAILED",
                error_message=exc.message,
                run_record=run_record,
                start_ts=start_ts,
            )

    async def _pipeline(
        self,
        *,
        run_id: str,
        irp_id: str,
        request: MimirRelayRequest,
        mode: IRPMode,
        run_record: RunRecord,
        start_ts: float,
    ) -> RelayResult:
        """
        Core relay pipeline — PolicyError and RoutingError propagate upward.

        Steps
        -----
        1.  Load InteropSettings for user
        2.  Validate interop enabled + allowed target
        3.  Build merged IRPPolicy
        4.  Construct IRP/1 envelope
        5.  Structural validations (hop limit, prompt size)
        6.  Secret redaction on prompt
        7.  Mark run RUNNING + emit RELAY_START
        8.  Dispatch to Mode A/B or Mode C handler
        """
        # 1. Load settings
        settings = self._load_settings(request.user_id)

        # 2. Global policy guards (raise PolicyError on violation)
        PolicyEngine.validate_interop_enabled(settings.enabled)
        PolicyEngine.validate_allowed_target(request.target, settings.allowed_targets)

        # 3. Derive allowed-provider string list for routing
        allowed_targets: Optional[List[str]] = (
            settings.allowed_targets if settings.allowed_targets else None
        )

        # 4. Build merged IRP policy (defaults from settings, overrides from request)
        merged_policy = PolicyEngine.build_policy(
            defaults={"tools": settings.policy_tools},
            overrides=request.policy,
        )

        # 5. Construct IRP/1 envelope
        hop = IRPHop(max=settings.hop_max)
        irp_request = IRPRequest(instruction=request.instruction)
        envelope = IRPEnvelope(
            id=irp_id,
            from_agent=request.from_agent,
            to=request.target,
            mode=mode,
            hop=hop,
            policy=merged_policy,
            context=request.context,
            request=irp_request,
        )

        # 6. Structural validations (raise PolicyError on violation)
        PolicyEngine.validate_hop_limit(envelope)
        PolicyEngine.validate_prompt_size(envelope)

        # 7. Secret redaction on prompt (returns new envelope + count)
        envelope, prompt_redact_count = PolicyEngine.redact_prompt(envelope)
        if prompt_redact_count:
            self._emit_audit(
                run_id=run_id,
                event_type=AuditEventType.REDACTION_APPLIED,
                provider="*",
                status=RunStatus.RUNNING,
                payload={"count": prompt_redact_count, "direction": "prompt"},
            )

        # 8. Record prompt hash for deduplication / audit
        run_record.prompt_hash = PolicyEngine.hash_prompt(
            envelope.request.instruction
        )

        # 9. Mark run as RUNNING
        run_record.status = RunStatus.RUNNING
        self._upsert_run(run_record)

        # 10. Emit RELAY_START
        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.RELAY_START,
            provider=request.target,
            status=RunStatus.RUNNING,
            payload={
                "mode": mode.value,
                "from_agent": request.from_agent,
                "user_id": request.user_id,
                "irp_id": irp_id,
            },
        )

        # 11. Dispatch
        if mode == IRPMode.RECONCILE:
            return await self._run_mode_c(
                envelope=envelope,
                request=request,
                run_id=run_id,
                irp_id=irp_id,
                run_record=run_record,
                prompt_redact_count=prompt_redact_count,
                start_ts=start_ts,
                allowed_targets=allowed_targets,
            )
        else:
            return await self._run_mode_ab(
                envelope=envelope,
                request=request,
                run_id=run_id,
                irp_id=irp_id,
                run_record=run_record,
                prompt_redact_count=prompt_redact_count,
                start_ts=start_ts,
                allowed_targets=allowed_targets,
            )

    # ------------------------------------------------------------------
    # Mode A / B (single-provider)
    # ------------------------------------------------------------------

    async def _run_mode_ab(
        self,
        *,
        envelope: IRPEnvelope,
        request: MimirRelayRequest,
        run_id: str,
        irp_id: str,
        run_record: RunRecord,
        prompt_redact_count: int,
        start_ts: float,
        allowed_targets: Optional[List[str]],
    ) -> RelayResult:
        """
        Execute an Advisory (A) or Structured (B) relay to a single provider.

        Raises
        ------
        RoutingError
            If no provider can be selected.
        PolicyError
            If a loop is detected in the hop path, or tool usage violates policy.
        """
        # Route to optimal provider
        provider = await self._router.route(
            envelope, allowed_providers=allowed_targets
        )
        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.ROUTING_DECISION,
            provider=provider.value,
            status=RunStatus.RUNNING,
            payload={"selected": provider.value, "mode": envelope.mode.value},
        )

        # Loop detection
        PolicyEngine.validate_hop_path(envelope, provider.value)

        # Execute provider call
        adapter = get_adapter(provider)
        result: ProviderResult = await adapter.call(envelope)

        # Provider-level error
        if result.error:
            self._emit_audit(
                run_id=run_id,
                event_type=AuditEventType.PROVIDER_UNAVAILABLE,
                provider=provider.value,
                status=RunStatus.FAILED,
                payload={"error": result.error},
            )
            return self._fail(
                run_id=run_id,
                irp_id=irp_id,
                mode=envelope.mode,
                error_code="PROVIDER_ERROR",
                error_message=result.error,
                run_record=run_record,
                start_ts=start_ts,
                provider=provider,
            )

        # No-tools policy check
        if envelope.policy.tools == "forbidden":
            PolicyEngine.check_no_tools_result(result)

        # Redact provider output
        output_text, out_redact_count = PolicyEngine.redact_output(
            result.raw_output, envelope.policy.redaction
        )
        if out_redact_count:
            self._emit_audit(
                run_id=run_id,
                event_type=AuditEventType.REDACTION_APPLIED,
                provider=provider.value,
                status=RunStatus.RUNNING,
                payload={"count": out_redact_count, "direction": "output"},
            )

        # Output size check
        PolicyEngine.validate_output_size(output_text, envelope.policy)

        total_redact = prompt_redact_count + out_redact_count
        latency = _elapsed_ms(start_ts)

        # Persist successful run
        run_record.status = RunStatus.SUCCESS
        run_record.selected_provider = provider
        run_record.latency_ms = latency
        run_record.input_tokens = result.input_tokens
        run_record.output_tokens = result.output_tokens
        run_record.redaction_count = total_redact
        run_record.completed_at = time.time()
        self._upsert_run(run_record)

        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.RELAY_COMPLETE,
            provider=provider.value,
            status=RunStatus.SUCCESS,
            payload={
                "latency_ms": latency,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "redaction_count": total_redact,
            },
        )

        return RelayResult(
            run_id=run_id,
            irp_id=irp_id,
            mode=envelope.mode,
            provider=provider,
            status=RunStatus.SUCCESS,
            output=output_text,
            latency_ms=latency,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            redaction_count=total_redact,
            hop_count=envelope.hop.count,
            trace=envelope.trace,
        )

    # ------------------------------------------------------------------
    # Mode C (multi-provider reconciliation)
    # ------------------------------------------------------------------

    async def _run_mode_c(
        self,
        *,
        envelope: IRPEnvelope,
        request: MimirRelayRequest,
        run_id: str,
        irp_id: str,
        run_record: RunRecord,
        prompt_redact_count: int,
        start_ts: float,
        allowed_targets: Optional[List[str]],
    ) -> RelayResult:
        """
        Execute a Mode C (RECONCILE) relay: fan-out to all viable providers,
        redact outputs, then reconcile into a consensus/synthesis.

        Raises
        ------
        RoutingError
            If no providers are available for fan-out.
        """
        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.RECONCILIATION_START,
            provider="*",
            status=RunStatus.RUNNING,
            payload={"mode": "C", "irp_id": irp_id},
        )

        # Score all providers to determine viable fan-out candidates
        all_scores = await self._router.score_all(envelope)
        viable_providers: List[ProviderName] = [
            p
            for p, s in all_scores.items()
            if s.availability_score > 0.0
            and (allowed_targets is None or p.value in allowed_targets)
        ]

        if not viable_providers:
            raise RoutingError(
                "No providers are available for Mode C reconciliation. "
                f"Checked: {[p.value for p in all_scores]}"
            )

        logger.info(
            "relay run=%s mode=C fan-out to providers=%s",
            run_id,
            [p.value for p in viable_providers],
        )

        # Fan-out: call all viable providers concurrently
        async def _call_one(
            prov: ProviderName,
        ) -> tuple[ProviderName, ProviderResult]:
            try:
                adp = get_adapter(prov)
                res = await adp.call(envelope)
                return prov, res
            except Exception as exc:
                logger.warning(
                    "relay run=%s mode=C provider=%s call failed: %s",
                    run_id, prov.value, exc,
                )
                return prov, ProviderResult(
                    provider=prov,
                    raw_output="",
                    error=str(exc),
                    available=False,
                )

        tasks = [asyncio.create_task(_call_one(p)) for p in viable_providers]
        raw_pairs: List[tuple[ProviderName, ProviderResult]] = await asyncio.gather(
            *tasks
        )

        # Process results: redact outputs, accumulate token/redaction totals
        provider_results: Dict[ProviderName, ProviderResult] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_out_redact = 0

        for prov, result in raw_pairs:
            if result.error:
                # Pass error results through unchanged for reconciler visibility
                provider_results[prov] = result
                self._emit_audit(
                    run_id=run_id,
                    event_type=AuditEventType.PROVIDER_UNAVAILABLE,
                    provider=prov.value,
                    status=RunStatus.RUNNING,
                    payload={"error": result.error},
                )
                continue

            out_text, out_redact_count = PolicyEngine.redact_output(
                result.raw_output, envelope.policy.redaction
            )
            total_out_redact += out_redact_count
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            provider_results[prov] = result.model_copy(
                update={
                    "raw_output": out_text,
                    "redaction_count": out_redact_count,
                }
            )

        if total_out_redact:
            self._emit_audit(
                run_id=run_id,
                event_type=AuditEventType.REDACTION_APPLIED,
                provider="*",
                status=RunStatus.RUNNING,
                payload={"count": total_out_redact, "direction": "output"},
            )

        total_redact = prompt_redact_count + total_out_redact

        # Reconcile across all provider outputs
        reconciliation = await self._reconciler.reconcile(envelope, provider_results)

        # Determine final status from reconciliation outcome
        if reconciliation.escalated:
            status = RunStatus.FAILED
            error_code: Optional[str] = "RECONCILIATION_ESCALATED"
            error_message: Optional[str] = reconciliation.escalation_reason
        else:
            status = RunStatus.SUCCESS
            error_code = None
            error_message = None

        latency = _elapsed_ms(start_ts)

        # Persist run record
        run_record.status = status
        run_record.latency_ms = latency
        run_record.input_tokens = total_input_tokens
        run_record.output_tokens = total_output_tokens
        run_record.redaction_count = total_redact
        run_record.completed_at = time.time()
        if error_code:
            run_record.error_code = error_code
            run_record.error_message = error_message
        self._upsert_run(run_record)

        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.RECONCILIATION_COMPLETE,
            provider="*",
            status=status,
            payload={
                "escalated": reconciliation.escalated,
                "consensus_count": len(reconciliation.consensus_claims),
                "conflicting_count": len(reconciliation.conflicting_claims),
                "latency_ms": latency,
                "providers_used": [p.value for p in viable_providers],
            },
        )

        return RelayResult(
            run_id=run_id,
            irp_id=irp_id,
            mode=envelope.mode,
            status=status,
            output=reconciliation.synthesis,
            reconciliation=reconciliation,
            latency_ms=latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            redaction_count=total_redact,
            hop_count=envelope.hop.count,
            error_code=error_code,
            error_message=error_message,
            trace=envelope.trace,
        )

    # ------------------------------------------------------------------
    # Terminal state helpers
    # ------------------------------------------------------------------

    def _fail(
        self,
        *,
        run_id: str,
        irp_id: str,
        mode: IRPMode,
        error_code: str,
        error_message: str,
        run_record: RunRecord,
        start_ts: float,
        provider: Optional[ProviderName] = None,
    ) -> RelayResult:
        """Persist a FAILED run record and return a failure RelayResult."""
        latency = _elapsed_ms(start_ts)
        run_record.status = RunStatus.FAILED
        run_record.latency_ms = latency
        run_record.error_code = error_code
        run_record.error_message = error_message
        run_record.completed_at = time.time()
        if provider is not None:
            run_record.selected_provider = provider
        self._upsert_run(run_record)

        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.RELAY_FAILED,
            provider=provider.value if provider else "unknown",
            status=RunStatus.FAILED,
            payload={"error_code": error_code, "error_message": error_message},
        )

        return RelayResult(
            run_id=run_id,
            irp_id=irp_id,
            mode=mode,
            provider=provider,
            status=RunStatus.FAILED,
            latency_ms=latency,
            error_code=error_code,
            error_message=error_message,
        )

    def _policy_blocked(
        self,
        *,
        run_id: str,
        irp_id: str,
        mode: IRPMode,
        error_code: str,
        error_message: str,
        run_record: RunRecord,
        start_ts: float,
        provider: Optional[ProviderName] = None,
    ) -> RelayResult:
        """Persist a POLICY_BLOCKED run record and return a blocked RelayResult."""
        latency = _elapsed_ms(start_ts)
        run_record.status = RunStatus.POLICY_BLOCKED
        run_record.latency_ms = latency
        run_record.error_code = error_code
        run_record.error_message = error_message
        run_record.completed_at = time.time()
        self._upsert_run(run_record)

        self._emit_audit(
            run_id=run_id,
            event_type=AuditEventType.POLICY_BLOCKED,
            provider=provider.value if provider else "policy",
            status=RunStatus.POLICY_BLOCKED,
            payload={"error_code": error_code, "error_message": error_message},
        )

        return RelayResult(
            run_id=run_id,
            irp_id=irp_id,
            mode=mode,
            provider=provider,
            status=RunStatus.POLICY_BLOCKED,
            latency_ms=latency,
            error_code=error_code,
            error_message=error_message,
        )

    # ------------------------------------------------------------------
    # Storage helpers (fire-and-forget)
    # ------------------------------------------------------------------

    def _load_settings(self, user_id: str) -> InteropSettings:
        """
        Load per-user InteropSettings from the store.

        Falls back to default InteropSettings if the store is unavailable
        or the user has no saved settings.
        """
        if self._store is None:
            return InteropSettings(user_id=user_id)
        try:
            settings = self._store.get_settings(user_id=user_id)
            if settings is not None:
                return settings
        except Exception as exc:
            logger.warning(
                "Failed to load settings for user=%s, using defaults: %s",
                user_id, exc,
            )
        return InteropSettings(user_id=user_id)

    def _upsert_run(self, run_record: RunRecord) -> None:
        """
        Persist (create or update) a RunRecord.  Errors are suppressed
        so storage failures never kill a relay call.
        """
        if self._store is None:
            return
        try:
            self._store.upsert_run(run_record)
        except Exception as exc:
            logger.debug(
                "Failed to upsert run=%s: %s", run_record.run_id, exc
            )

    def _emit_audit(
        self,
        *,
        run_id: str,
        event_type: AuditEventType,
        provider: str,
        status: RunStatus,
        payload: Optional[dict] = None,
    ) -> None:
        """
        Insert an AuditEvent into the store.  Errors are suppressed so
        audit failures never block the relay pipeline.
        """
        if self._store is None:
            return
        try:
            event = AuditEvent(
                id=_new_id(),
                run_id=run_id,
                event_type=event_type,
                provider=provider,
                status=status,
                payload=payload,
            )
            self._store.insert_audit_event(event)
        except Exception as exc:
            logger.debug(
                "Failed to emit audit event type=%s run=%s: %s",
                event_type.value, run_id, exc,
            )
