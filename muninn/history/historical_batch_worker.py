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
    MODEL_IDENTITIES,
    TERMINAL,
    BatchError,
    BatchOutbox,
    billed_cost,
    payload,
    prepare_items,
    resolved_extractions,
)
from muninn.history.remote_accounting import Admission, reserve, settled_response, unknown_response

_API = "https://openrouter.ai/api/v1/batches"
_MAX_RESPONSE = 16 * 1024 * 1024  # Enough for bounded bulk results; not a source-size cutoff.


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
    from muninn.history.historical_batch import MAX_REQUEST_BYTES, _provider_id, _wire_json
    if (method not in {"GET", "POST"} or method == "GET" and not _provider_id(provider_id)
            or method == "POST" and provider_id is not None):
        raise BatchError("batch_transport_invalid")
    key = llm_settings.api_key()
    if not key:
        raise BatchError("batch_key_missing")
    url = _API if method == "POST" else f"{_API}/{provider_id}"
    # Serialization preserves the required routing-field-before-requests order.
    content = _wire_json(body) if body is not None else None
    if content is not None and len(content) > MAX_REQUEST_BYTES:
        raise BatchError("batch_request_bound")
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
                 authorize_transaction=None, send=transport, provider_status=None, clock=time.monotonic,
                 repair_id=None):
        self.journal = journal
        self.authorize_submit = authorize_submit
        self.authorize_transaction = authorize_transaction
        self.send = send
        self.provider_status = provider_status
        self.clock = clock
        self.next_poll = 0.0
        self.status = {"state": "idle"}
        self.next_step = 0.0
        self.repair_id = repair_id
        self.repair_worker = None

    def _private_owner(self):
        with self.journal._connect() as db:
            return self.journal._historical_batch_head(db)[1]

    def _settle_retained_repairs(self, outbox, parent):
        """Restart may find a sealed terminal BEFORE its separate bill commit."""
        for child in outbox.repair_records(parent):
            if child["state"] not in {"terminal_saved", "cleaned"}:
                continue
            ident, generation = child.get("repair_admission"), child["consent_generation"]
            if not ident or not (unknown_response(self.journal.policy_root, ident, generation,
                    batch_owner=child["id"]) or settled_response(self.journal.policy_root, ident,
                    generation, batch_owner=child["id"])):
                raise BatchError("batch_cost_unresolved")
            billed_cost(child["terminal"])
            if not Admission(self.journal.policy_root, ident, generation).settle_response(child["terminal"]):
                raise BatchError("batch_cost_unresolved")

    def _admission(self, ident):
        if self.repair_id is not None:
            owner = self.journal.historical_batch_owner()
            outbox = BatchOutbox(self.journal.archive)
            records = outbox.repair_records(outbox.read(owner["id"]))
            child = next((r for r in records if r["id"] == ident), None)
            if owner["phase"] != "sent" or child is None or not child.get("repair_admission"):
                raise BatchError("batch_admission_unresolved")
            return Admission(self.journal.policy_root, child["repair_admission"], owner["generation"])
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
        parent_id = owner["id"]
        if self.repair_id is not None:
            children = await asyncio.to_thread(outbox.repair_records, record)
            record = next((r for r in children if r["id"] == self.repair_id), None)
            if record is None or children[-1]["id"] != self.repair_id:
                raise BatchError("batch_repair_binding_invalid")
            owner = {**owner, "id": self.repair_id, "items": len(record["items"])}
        if record["state"] == "submission_unknown":
            candidate = record.get("recovery_candidate")
            if candidate is None:
                self.status["state"] = "submission_unknown"
                return True  # Block advance, without hot polling or blind retry.
            reply = await self.send("GET", provider_id=candidate)
            if reply.get("id") != candidate:
                raise BatchError("batch_poll_identity_invalid")
            if reply.get("status") != "completed":
                self.status["state"] = "awaiting_provider_identity"
                return True  # Candidate metadata alone never establishes ownership.
            await asyncio.to_thread(outbox.recover_submission, owner["id"], record["revision"], reply)
            record = await asyncio.to_thread(outbox.read, owner["id"])
            await asyncio.to_thread(outbox.save_terminal, owner["id"], record["revision"], reply)
            record = await asyncio.to_thread(outbox.read, owner["id"])
        if record["state"] == "prepared":
            if not self.authorize_submit(owner["generation"]):
                self.status["state"] = "consent_required"
                return True
            if self.repair_id is not None:
                # Reprove failed-only selection before any new provider attempt.
                original = await asyncio.to_thread(outbox.read, parent_id)
                parent_owner = await asyncio.to_thread(self._private_owner)
                source = await asyncio.to_thread(CitedAnalysisSource, self.journal.archive)
                _valid, unresolved = await asyncio.to_thread(resolved_extractions, outbox, original,
                    source, self.journal.policy_root, parent_owner["admission_id"])
                if any(i["job_id"] not in unresolved for i in record["items"]):
                    raise BatchError("batch_repair_repeated_success")
                with self.journal._connect() as db:
                    for item in record["items"]:
                        row = db.execute("SELECT state,sealed_extraction,sealed_receipt,publication_started "
                                         "FROM history_analysis_jobs WHERE job_id=?", (item["job_id"],)).fetchone()
                        if (row is None or row["state"] not in {"pending", "retry", "outcome_unknown"}
                                or any(row[k] for k in ("sealed_extraction", "sealed_receipt", "publication_started"))):
                            raise BatchError("batch_repair_publication_pending")
                from muninn.history.batch_activation import bind_consent, read_batch_policy
                await asyncio.to_thread(bind_consent, self.journal, outbox, record["id"],
                                        read_batch_policy(self.journal.policy_root))
                if record.get("repair_admission") is not None:
                    # A persisted uncertainty/reservation must never be replaced.
                    self.status = {"state": "repair_admission_unresolved", "items": owner["items"]}
                    return True
            from muninn.history.auto_routing import openrouter_key_status
            status = await asyncio.to_thread(self.provider_status or openrouter_key_status,
                                            policy_root=self.journal.policy_root)
            admitted = await asyncio.to_thread(reserve, self.journal.policy_root,
                owner["generation"], status, batch_owner=owner["id"])
            try:
                # Reauthenticate the ACTUAL stored prompt, not just a safe flag.
                source = await asyncio.to_thread(CitedAnalysisSource, self.journal.archive)
                checked = await asyncio.to_thread(prepare_items, source,
                    [(i["job_id"], i["window"]) for i in record["items"]])
                if any(a["body"] != b["body"] for a, b in zip(checked, record["items"])):
                    raise BatchError("batch_input_binding_invalid")
                body = payload(record["items"])
                if not self.authorize_submit(owner["generation"]):
                    raise BatchError("batch_consent_revoked")
                if self.repair_id is not None:
                    await asyncio.to_thread(outbox.bind_repair_admission, owner["id"], record["revision"],
                                            admitted.identifier)
                    record = await asyncio.to_thread(outbox.read, owner["id"])
                # Shield/drain all durable fences; cancelling a to_thread does
                # not cancel its transaction. No POST until all three commit.
                async def fence():
                    import hashlib

                    from muninn.history.historical_batch import _json
                    digest = hashlib.sha256(_json(record["items"])).hexdigest()
                    guard = (None if self.authorize_transaction is None else
                             lambda db: self.authorize_transaction(db, owner["id"], digest))
                    await asyncio.to_thread(admitted.mark_unknown, policy_guard=guard)
                    await asyncio.to_thread(outbox.begin_submission, owner["id"], record["revision"])
                    if self.repair_id is None:
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
            if (reply.get("id") != record["provider_id"] or reply.get("model") not in MODEL_IDENTITIES
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
        parent = await asyncio.to_thread(outbox.read, parent_id)
        await asyncio.to_thread(self._settle_retained_repairs, outbox, parent)
        source = await asyncio.to_thread(CitedAnalysisSource, self.journal.archive)
        parent_owner = await asyncio.to_thread(self._private_owner)
        stages, unresolved = await asyncio.to_thread(resolved_extractions, outbox, parent, source,
            self.journal.policy_root, parent_owner["admission_id"])
        for job_id, stage in stages.items():
            job = await asyncio.to_thread(self.journal.claim_historical_batch_result,
                                         parent_id, job_id)
            if job is None:
                continue  # Already published or an unexpired publication lease.
            await self._publish(source, job, stage)
        invalid = len(unresolved)
        passed = not invalid and await asyncio.to_thread(self.journal.finish_historical_batch, parent_id)
        self.status = {"state": "passed" if passed else "checkpoint_unresolved", "items": owner["items"],
                       "invalid_items": invalid}
        if passed:
            self.next_step = 0.0
        elif invalid and self.repair_id is None:
            children = await asyncio.to_thread(outbox.repair_records, parent)
            if not children or children[-1]["state"] in {"terminal_saved", "cleaned"}:
                if len(children) >= 2:
                    self.status["state"] = "repair_limit_reached"
                    return True
                if not self.authorize_submit(owner["generation"]):
                    self.status["state"] = "consent_required"
                    return True
                from muninn.history.batch_activation import read_batch_policy
                policy = await asyncio.to_thread(read_batch_policy, self.journal.policy_root)
                if not policy["enabled"] or not policy["remaining_batches"]:
                    return True
                items = await asyncio.to_thread(prepare_items, source,
                    [(i["job_id"], i["window"]) for i in parent["items"] if i["job_id"] in unresolved])
                child_id = await asyncio.to_thread(outbox.prepare_repair, parent_id, items)
            else:
                child_id = children[-1]["id"]
            if self.repair_worker is None or self.repair_worker.repair_id != child_id:
                self.repair_worker = HistoricalBatchWorker(self.journal, authorize_submit=self.authorize_submit,
                    authorize_transaction=self.authorize_transaction, send=self.send,
                    provider_status=self.provider_status, clock=self.clock, repair_id=child_id)
            await self.repair_worker.step()
            updated = await asyncio.to_thread(outbox.read, parent_id)
            self.status = {**self.repair_worker.status, "repair_only": True,
                           "parent_items": len(parent["items"]), "repair_round": len(updated["repairs"])}
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
