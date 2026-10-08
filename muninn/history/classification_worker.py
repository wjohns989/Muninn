"""One purpose-bound ZDR turn in the existing serial inference consumer."""
import asyncio
import json
import threading
import time
from decimal import Decimal

import httpx

from muninn.history import llm_settings
from muninn.history.auto_routing import openrouter_key_status, remote_policy_snapshot
from muninn.history.insights import Provider
from muninn.history.memory_classification import (
    BUCKETS, PROMPT, REASONS, ClassificationError, prepare_classification, revalidate_classification,
)
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.remote_accounting import AdmissionError, reserve
from muninn.history.secure_analysis import _request_safe

_HEARTBEAT_SECONDS = 15
_PRICE = {"prompt": 0.25, "completion": 1, "request": 0}
_MODEL = "openai/gpt-6-luna-pro"


def _model_context_bound():
    """Public metadata only; no credential or private input sent to the catalog."""
    try:
        with httpx.Client(timeout=15, trust_env=False, follow_redirects=False) as client:
            with client.stream("GET", "https://openrouter.ai/api/v1/models", params={"q": _MODEL}) as reply:
                reply.raise_for_status()
                chunks, size = [], 0
                for chunk in reply.iter_bytes():
                    size += len(chunk)
                    if size > 1024 * 1024:
                        raise AdmissionError("classification_catalog_unavailable")
                    chunks.append(chunk)
        rows = json.loads(b"".join(chunks))["data"]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError
        matches = [row for row in rows if row.get("id") == _MODEL]
        context = matches[0]["context_length"] if len(matches) == 1 else None
        if type(context) is not int or not 4096 <= context <= 8_000_000:
            raise ValueError
        return context
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise AdmissionError("classification_catalog_unavailable") from exc


def _cost_ceiling(body, context):
    """Full-context bound, not chars/token estimation or an expected bill.

    Count a full context for BOTH input and output, deliberately overestimating
    even reasoning providers that treat the requested completion cap differently.
    No tools, web search, fallback, premium tier or extra fixed fees are allowed.
    """
    if (type(context) is not int or not 4096 <= context <= 8_000_000
            or set(body) != {"model", "models", "messages", "provider", "reasoning", "usage",
                              "response_format", "max_completion_tokens"}
            or body.get("model") != _MODEL or body.get("models") != [_MODEL]
            or body.get("provider") != {"zdr": True, "data_collection": "deny",
                                       "require_parameters": True, "max_price": _PRICE}
            or type(body.get("max_completion_tokens")) is not int or body["max_completion_tokens"] != 4096
            or body.get("reasoning") != {"effort": "low", "exclude": True}
            or body.get("usage") != {"include": True} or not _request_safe(body)):
        raise AdmissionError("classification_price_contract_invalid")
    return Decimal(context) * (Decimal(str(_PRICE["prompt"])) + Decimal(str(_PRICE["completion"]))) / 1_000_000


def _dispatch_period_open():
    # Transport's whole lifetime is bounded by 90s. Avoid crossing the UTC
    # accounting period, including month rollover; this is not a queue failure.
    return time.time() % 86400 < 86400 - 120


def request_body(prepared):
    slots = [row["id"] for row in prepared.payload()["candidates"]]
    fields = {"id": {"type": "string", "enum": slots},
        "bucket": {"type": "string", "enum": sorted(BUCKETS)},
        "disposition": {"type": "string", "enum": ["accepted", "needs_user", "conflict"]},
        "evidence_refs": {"type": "array", "maxItems": 24, "items": {"type": "string"}},
        "reason": {"type": "string", "enum": sorted(REASONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}}
    # An unmeasured fallback is not Luna semantic-quality proof.
    provider = Provider("openrouter", llm_settings.OPENROUTER_API, [llm_settings.DEFAULT_MODEL])
    body = provider.request_body([{"role": "system", "content": PROMPT},
        {"role": "user", "content": prepared.payload_json}])
    body["max_completion_tokens"] = 4096
    body["provider"]["max_price"] = dict(_PRICE)
    body["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "memory_placement_v1", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": ["items"],
            "properties": {"items": {"type": "array", "minItems": len(slots), "maxItems": len(slots),
                "items": {"type": "object", "additionalProperties": False,
                    "required": list(fields), "properties": fields}}}}}}
    if not _request_safe(body):
        raise ClassificationError("classification_withheld")
    return body


async def transport(body, *, before_post):
    """Single bounded POST, fixed origin, no redirects/proxy or automatic retry."""
    key = llm_settings.api_key()
    if not key:
        raise AdmissionError("classification_key_missing")
    async with asyncio.timeout(90):
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5), trust_env=False,
                                     follow_redirects=False) as client:
            if not await before_post():
                return None  # Explicit proven-unsent refusal, not a timeout guess.
            async with client.stream("POST", llm_settings.OPENROUTER_API + "/chat/completions",
                    json=body, headers={"Authorization": f"Bearer {key}"}) as response:
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 128 * 1024:
                        raise ClassificationError("classification_response_bound")
                    chunks.append(chunk)
                return response.status_code, json.loads(b"".join(chunks), parse_float=Decimal)


async def _drain_thread(function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


async def process_classification(journal, *, enabled, recovery_only=False,
                                 send=transport, key_status=None, note_dispatch=lambda: None):
    """Recovery needs no consent/model; new work requires every current gate.

    Test injection supplies synthetic replies, never a hidden alternate provider.
    The caller is the existing serial consumer, not a second background worker.
    """
    if not recovery_only:
        if not enabled() or not _dispatch_period_open() or not await asyncio.to_thread(journal.classification_ready):
            return False
        checkpoint = await asyncio.to_thread(journal.historical_batch_owner)
        if checkpoint is None or checkpoint["phase"] != "passed":
            return False  # General review cannot bypass exact paid recovery.
        await _drain_thread(journal.discover_classifications, limit=8)
    job = await _drain_thread(journal.claim_classification, include_pending=not recovery_only)
    if job is None:
        return False
    if job["state"] == "staged":
        try:
            await _drain_thread(journal.publish_classification, job["job_id"])
        except ClassificationError:
            await _drain_thread(journal.stop_classification, job["job_id"], None, reason="input_changed")
        return True
    cancelled = threading.Event()

    async def heartbeat():
        while not cancelled.is_set():
            await asyncio.sleep(_HEARTBEAT_SECONDS)
            if not await _drain_thread(journal.heartbeat_classification, job["job_id"], job["lease"]):
                cancelled.set()

    async def run():
        pulse = asyncio.create_task(heartbeat())
        admission, started = None, False
        stop_reason = "missing_evidence"
        try:
            prepared = await asyncio.to_thread(prepare_classification, MemoryLedger(journal.archive, read_only=True), job["refs"])
            await asyncio.to_thread(journal.prepare_classification_job, job["job_id"], job["lease"], prepared)
            body = request_body(prepared)
            context_bound = await asyncio.to_thread(_model_context_bound)
            ceiling = _cost_ceiling(body, context_bound)
            policy = remote_policy_snapshot(journal.policy_root)

            def gate():
                current = remote_policy_snapshot(journal.policy_root)
                owner = journal.historical_batch_owner()
                return (not cancelled.is_set() and enabled() and _dispatch_period_open() and current.enabled
                    and current.generation == policy.generation and journal.classification_ready()
                    and owner is not None and owner["phase"] == "passed" and owner["id"] == checkpoint["id"])

            if not gate():
                return False
            status = (await asyncio.to_thread(openrouter_key_status, policy_root=journal.policy_root)
                      if key_status is None else key_status())
            # reserve/mark_unknown are synchronous durable boundaries: no
            # cancelled to_thread writer can race a second reservation.
            admission = reserve(journal.policy_root, policy.generation, status,
                classification_job=job["job_id"], classification_input=prepared.input_sha256,
                classification_once=True, cost_ceiling_usd=ceiling)
            if not gate():
                return False
            async def before_post():
                nonlocal started, stop_reason
                # Client setup has completed. Pin fresh source/review evidence
                # after every preceding wait, not just the initial preparation.
                await asyncio.to_thread(revalidate_classification,
                    MemoryLedger(journal.archive, read_only=True), prepared)
                if not gate() or not _request_safe(body):
                    return False
                admission.mark_unknown(policy_guard=lambda _db: gate())
                await _drain_thread(journal.mark_classification_dispatch, job["job_id"], job["lease"],
                                        admission.identifier, policy.generation)
                # The durable journal marker itself awaited a writer.
                fresh_status = (await asyncio.to_thread(openrouter_key_status, policy_root=journal.policy_root)
                                if key_status is None else key_status())
                await asyncio.to_thread(revalidate_classification,
                    MemoryLedger(journal.archive, read_only=True), prepared)
                if not gate():
                    return False
                admission.check_headroom(fresh_status, cost_ceiling_usd=_cost_ceiling(body, context_bound))
                stop_reason = "reply_invalid"
                note_dispatch()
                started = True  # Transport must admit HTTP immediately next.
                return True

            reply = await send(body, before_post=before_post)
            if reply is None:
                return False
            code, data = reply
            if not admission.settle_response(data):
                raise AdmissionError("remote_cost_unresolved")
            if code != 200:
                raise ClassificationError("classification_provider_rejected")
            content = data["choices"][0]["message"]["content"]
            model = data["model"]
            await asyncio.to_thread(journal.stage_classification, job["job_id"], job["lease"], content, model=model)
            stop_reason = "input_changed"
            await asyncio.to_thread(journal.publish_classification, job["job_id"])
        except AdmissionError:
            if started:
                await asyncio.to_thread(journal.stop_classification, job["job_id"], job["lease"], reason="reply_invalid")
            else:
                return False
        except (httpx.HTTPError, TimeoutError):
            if not started:
                return False
            await asyncio.to_thread(journal.stop_classification, job["job_id"], job["lease"], reason="reply_invalid")
        except (ClassificationError, ValueError, KeyError, TypeError):
            if not started and admission is not None:
                admission.release_unsent()
            await asyncio.to_thread(journal.stop_classification, job["job_id"],
                None if stop_reason == "input_changed" else job["lease"], reason=stop_reason)
        finally:
            # Never infer free transport from a timeout. Only pre-send exits
            # can release; expiry also recognizes this exact unsent proof.
            try:
                if not started:
                    if admission is not None:
                        admission.release_unsent()
                    with journal._connect() as db:
                        current = journal._classification_job(db.execute(
                            "SELECT * FROM memory_classification_jobs WHERE job_id=?", (job["job_id"],)).fetchone())
                    if current["state"] == "running":
                        await asyncio.to_thread(journal.defer_unsent_classification, job["job_id"], job["lease"])
            finally:
                cancelled.set()
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)

    task = asyncio.create_task(run())
    try:
        outcome = await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        # Drain bounded transport + durable writes before shutdown returns.
        await asyncio.gather(task, return_exceptions=True)
        raise
    return outcome is not False
