"""One-attempt batch transport and restart-safe local publication.

Submission requires a caller's separately scoped temporary/nontraining consent.
Recovery does not: it retrieves only a known provider ID and never sends source
text again. A lost submission response remains visibly blocked, not retried.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.historical_batch import (
    MODEL,
    TERMINAL,
    BatchError,
    BatchOutbox,
    billed_cost,
    payload,
    terminal_results,
    validate_item,
)
from muninn.history.remote_accounting import Admission, reserve

_API = "https://openrouter.ai/api/v1/batches"
_MAX_RESPONSE = 4 * 1024 * 1024  # Transport bound, never a transcript-size cutoff.


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BatchError("batch_response_invalid")
        result[key] = value
    return result


def _decode(raw):
    if len(raw) > _MAX_RESPONSE:
        raise BatchError("batch_response_bound")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise BatchError("batch_response_invalid") from exc
    if not isinstance(value, dict):
        raise BatchError("batch_response_invalid")
    return value


async def transport(method, provider_id=None, body=None):
    """Fixed origin, no redirects/proxies/retries; bound before JSON parsing."""
    from muninn.history import llm_settings
    from muninn.history.historical_batch import _provider_id
    if (method not in {"GET", "POST"} or method == "GET" and not _provider_id(provider_id)
            or method == "POST" and provider_id is not None):
        raise BatchError("batch_transport_invalid")
    key = llm_settings.api_key()
    if not key:
        raise BatchError("batch_key_missing")
    url = _API if method == "POST" else f"{_API}/{provider_id}"
    # Serialization preserves the required routing-field-before-requests order.
    content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode() if body is not None else None
    async with asyncio.timeout(60):
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5),
                                     trust_env=False, follow_redirects=False) as client:
            async with client.stream(method, url, content=content,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}) as reply:
                # HTTP rejection is not proof that a submitted batch is free.
                if reply.status_code not in ({202} if method == "POST" else {200}):
                    raise BatchError("batch_transport_rejected")
                chunks, size = [], 0
                async for chunk in reply.aiter_bytes():
                    size += len(chunk)
                    if size > _MAX_RESPONSE:
                        raise BatchError("batch_response_bound")
                    chunks.append(chunk)
    return _decode(b"".join(chunks))


class HistoricalBatchWorker:
    """Shared service's serial inference consumer; no independent service/task.

    authorize_submit(generation) must re-read separately revocable retention
    consent and managed remote consent. mark_unknown is the linearization point
    authorizing ONE attempt; revocation prevents subsequent attempts, not one
    already committed. Publication may resume after consent/budget revocation.
    """
    def __init__(self, journal, *, authorize_submit=lambda generation: False,
                 send=transport, provider_status=None, clock=time.monotonic):
        self.journal = journal
        self.authorize_submit = authorize_submit
        self.send = send
        self.provider_status = provider_status
        self.clock = clock
        self.next_poll = 0.0
        self.status = {"state": "idle"}
        self.next_step = 0.0

    def _admission(self, ident):
        with self.journal._connect() as db:
            _head, owner = self.journal._historical_batch_head(db)
            if owner is None or owner["id"] != ident or owner["phase"] != "sent":
                raise BatchError("batch_admission_unresolved")
            return Admission(self.journal.policy_root, owner["admission_id"], owner["generation"])

    async def step(self):
        if self.clock() < self.next_step:
            return True
        # Failed local publication, unavailable GET, and unresolved evidence
        # retry at a bounded cadence. This never authorizes another POST.
        self.next_step = self.clock() + 60
        owner = await asyncio.to_thread(self.journal.historical_batch_owner)
        if owner is None or owner["phase"] == "passed":
            self.status = {"state": "idle" if owner is None else "passed"}
            self.next_step = 0.0
            return False
        self.status = {"state": "working", "items": owner["items"]}
        outbox = await asyncio.to_thread(BatchOutbox, self.journal.archive)
        record = await asyncio.to_thread(outbox.read, owner["id"])
        if record["state"] == "submission_unknown":
            self.status["state"] = "submission_unknown"
            return True  # Block advance, without hot polling or blind retry.
        if record["state"] == "prepared":
            if not self.authorize_submit(owner["generation"]):
                self.status["state"] = "consent_required"
                return True
            from muninn.history.auto_routing import openrouter_key_status
            status = await asyncio.to_thread(self.provider_status or openrouter_key_status,
                                            policy_root=self.journal.policy_root)
            admitted = await asyncio.to_thread(reserve, self.journal.policy_root,
                owner["generation"], status, batch_owner=owner["id"])
            try:
                # Reauthenticate the ACTUAL stored prompt, not just a safe flag.
                from muninn.history.historical_batch import prepare_items
                source = await asyncio.to_thread(CitedAnalysisSource, self.journal.archive)
                checked = await asyncio.to_thread(prepare_items, source,
                    [(i["job_id"], i["window"]) for i in record["items"]])
                if any(a["body"] != b["body"] for a, b in zip(checked, record["items"])):
                    raise BatchError("batch_input_binding_invalid")
                body = payload(record["items"])
                if not self.authorize_submit(owner["generation"]):
                    raise BatchError("batch_consent_revoked")
                # Shield/drain all durable fences; cancelling a to_thread does
                # not cancel its transaction. No POST until all three commit.
                async def fence():
                    await asyncio.to_thread(admitted.mark_unknown)
                    await asyncio.to_thread(outbox.begin_submission, owner["id"], record["revision"])
                    await asyncio.to_thread(self.journal.mark_historical_batch_dispatched,
                                            owner["id"], admitted.identifier)
                task = asyncio.create_task(fence())
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await asyncio.gather(task, return_exceptions=True)
                    raise
                reply = await self.send("POST", body=body)
                await asyncio.to_thread(outbox.save_submission, owner["id"], record["revision"] + 1, reply)
                self.next_poll = self.clock() + 60
                self.status["state"] = "submitted"
                return True
            finally:
                # This only releases a reservation that NEVER crossed the
                # uncertainty fence; unknown or settled holds remain intact.
                await asyncio.to_thread(admitted.release_reserved)
        if record["state"] == "submitted":
            if self.clock() < self.next_poll:
                self.status["state"] = "awaiting_provider"
                return True
            self.next_poll = self.clock() + 60
            reply = await self.send("GET", provider_id=record["provider_id"])
            if (reply.get("id") != record["provider_id"] or reply.get("model") != MODEL
                    or reply.get("endpoint") != "/v1/chat/completions"
                    or reply.get("status") not in {"validating", "in_progress", "finalizing", *TERMINAL}):
                raise BatchError("batch_poll_identity_invalid")
            if reply["status"] not in TERMINAL:
                self.status["state"] = "awaiting_provider"
                return True
            await asyncio.to_thread(outbox.save_terminal, owner["id"], record["revision"], reply)
            record = await asyncio.to_thread(outbox.read, owner["id"])
        if record["state"] not in {"terminal_saved", "cleaned"}:
            raise BatchError("batch_state_conflict")
        paid = await asyncio.to_thread(self._admission, owner["id"])
        billed_cost(record["terminal"])  # Missing/BYOK cost stays unknown.
        if not await asyncio.to_thread(paid.settle_response, record["terminal"]):
            raise BatchError("batch_cost_unresolved")
        rows = terminal_results(record["items"], record["terminal"])
        source = await asyncio.to_thread(CitedAnalysisSource, self.journal.archive)
        invalid = 0
        for item in record["items"]:
            try:
                outcome = await asyncio.to_thread(validate_item, source, item, rows[item["custom_id"]])
            except ValueError:
                invalid += 1
                continue  # Retain failed evidence; successful siblings need no rerun.
            job = await asyncio.to_thread(self.journal.claim_historical_batch_result,
                                         owner["id"], item["job_id"])
            if job is None:
                continue  # Already published or an unexpired publication lease.
            stage = {**outcome["extraction"], "admission_id": paid.identifier}
            await self._publish(source, job, stage)
        passed = not invalid and await asyncio.to_thread(self.journal.finish_historical_batch, owner["id"])
        self.status = {"state": "passed" if passed else "checkpoint_unresolved", "items": owner["items"],
                       "invalid_items": invalid}
        if passed:
            self.next_step = 0.0
        return True

    async def _publish(self, source, job, stage):
        async def heartbeat():
            while True:
                await asyncio.sleep(15)
                if not await asyncio.to_thread(self.journal.heartbeat_analysis, job.job_id, job.lease_token):
                    return
        task = asyncio.create_task(heartbeat())
        try:
            if job.extraction is not None and job.extraction != stage:
                raise BatchError("batch_publication_binding_invalid")
            if job.state != "publishing":
                if not await asyncio.to_thread(self.journal.stage_analysis, job.job_id, job.lease_token, stage):
                    return
                if not await asyncio.to_thread(self.journal.begin_publication, job.job_id, job.lease_token):
                    return
            refs = await asyncio.to_thread(source.record_proposals, stage["window"], stage["proposals"],
                                           model_identity=stage["model_identity"])
            await asyncio.to_thread(self.journal.acknowledge_publication, job.job_id, job.lease_token, refs)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
