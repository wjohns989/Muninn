"""One fixed synthetic ZDR streaming probe; no queue or batch replacement.

The bounded receipt is encrypted by DiagnosticStore. Missing billing always
leaves the admission unknown. SSE keepalives are not counted as model output.
"""
from __future__ import annotations

import asyncio
import codecs
import json
import time

import httpx

from muninn.history.batch_diagnostic import DiagnosticStore
from muninn.history.historical_batch import BatchError, MODEL, MODEL_IDENTITIES, _wire_json
from muninn.history.remote_accounting import Admission, AdmissionError, _MAX, _db, _micros

MAX_RECEIPT_BYTES = 16 * 1024


def request_body():
    # Luna's existing private-history route uses Azure ZDR, not the OpenAI
    # temporary-retention batch host. Never carry the batch pin into ZDR.
    return {"model": MODEL, "provider": {"only": ["azure"], "allow_fallbacks": False,
        "zdr": True, "data_collection": "deny"}, "messages": [{"role": "user",
        "content": "Reply with exactly these five numbers separated by spaces: 1 2 3 4 5"}],
        "max_tokens": 128, "stream": True, "stream_options": {"include_usage": True}}


class StreamReceipt:
    def __init__(self):
        self.started = time.monotonic()
        self.value = {"http_status": None, "header_seconds": None, "first_content_seconds": None,
            "elapsed_seconds": None, "content_chunks": [], "events": [], "comments": 0,
            "done": False, "error": None, "usage": None, "generation_id": None,
            "transport_generation_id": None,
            "identity_valid": True}
        self.size, self.pending = 0, []
        self.seen_model = False
        self.model = None
        self.decoder, self.buffer, self.wire_bytes = codecs.getincrementaldecoder("utf-8")(), "", 0

    def feed(self, chunk):
        self.wire_bytes += len(chunk)
        if self.wire_bytes > MAX_RECEIPT_BYTES:
            raise BatchError("stream_receipt_bound")
        self.buffer += self.decoder.decode(chunk)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.line(line.removesuffix("\r"))

    def line(self, line):
        self.size += len(line.encode("utf-8")) + 1
        if self.size > MAX_RECEIPT_BYTES:
            raise BatchError("stream_receipt_bound")
        if line.startswith(":"):
            self.value["comments"] += 1
        elif line.startswith("data:"):
            self.pending.append(line[5:].lstrip(" "))
        elif not line and self.pending:
            self.event("\n".join(self.pending))
            self.pending.clear()

    def event(self, data):
        if data == "[DONE]":
            self.value["done"] = True
            return
        from muninn.history.historical_batch_worker import _pairs
        event = json.loads(data, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(event, dict):
            raise BatchError("stream_event_invalid")
        self.value["events"].append(event)
        if "error" in event:
            self.value["error"] = "provider_stream_error"
        model = event.get("model")
        if model is not None:
            self.seen_model = True
            self.value["identity_valid"] &= model in MODEL_IDENTITIES
            if self.model is None:
                self.model = model
            else:
                self.value["identity_valid"] &= model == self.model
        ident = event.get("id")
        if ident is not None:
            if not isinstance(ident, str) or not ident or len(ident) > 256:
                self.value["identity_valid"] = False
            elif self.value["generation_id"] is None:
                self.value["generation_id"] = ident
            elif ident != self.value["generation_id"]:
                self.value["identity_valid"] = False
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                if not isinstance(content, str):
                    raise BatchError("stream_content_invalid")
                if self.value["first_content_seconds"] is None:
                    self.value["first_content_seconds"] = round(time.monotonic() - self.started, 3)
                self.value["content_chunks"].append(content)
        if event.get("usage") is not None:
            if self.value["usage"] is not None and self.value["usage"] != event["usage"]:
                self.value["identity_valid"] = False
            self.value["usage"] = event["usage"]

    def finish(self):
        self.value["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        self.value["identity_valid"] &= self.seen_model and self.value["generation_id"] is not None
        return self.value


async def transport(body):
    if body != request_body():
        raise BatchError("diagnostic_fixed_input_required")
    from muninn.history import llm_settings
    key = llm_settings.api_key()
    if not key:
        raise BatchError("stream_key_missing")
    receipt = StreamReceipt()
    try:
        async with asyncio.timeout(90):
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5),
                    trust_env=False, follow_redirects=False) as client:
                async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                        content=_wire_json(body), headers={"Authorization": f"Bearer {key}",
                            "Content-Type": "application/json", "Accept": "text/event-stream"}) as reply:
                    receipt.value.update(http_status=reply.status_code,
                        header_seconds=round(time.monotonic() - receipt.started, 3),
                        transport_generation_id=reply.headers.get("X-Generation-Id"))
                    if reply.status_code != 200 or not reply.headers.get("content-type", "").startswith("text/event-stream"):
                        receipt.value["error"] = "stream_http_rejected"
                        raw = bytearray()
                        async for chunk in reply.aiter_bytes():
                            if len(raw) + len(chunk) > MAX_RECEIPT_BYTES:
                                raise BatchError("stream_receipt_bound")
                            raw.extend(chunk)
                        # Retained encrypted only; never echo arbitrary error
                        # bodies or exceptions into operator/agent output.
                        receipt.value["http_error_body"] = raw.decode("utf-8", errors="replace")
                    else:
                        async for chunk in reply.aiter_bytes():
                            receipt.feed(chunk)
                            if receipt.value["done"]:
                                break
    except (Exception, asyncio.CancelledError) as exc:
        # Retain already-received synthetic chunks, never raw exception/key text.
        receipt.value["error"] = ("stream_cancelled" if isinstance(exc, asyncio.CancelledError) else
            "stream_timeout" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else "stream_transport_or_parse_error")
    return receipt.finish()


class StreamingStore(DiagnosticStore):
    def retained_status(self, ident):
        """Read one authenticated snapshot; operator debits are not provider bills."""
        with _db(self.root) as (db, _):
            record = self._read(db, ident)
            if record.get("kind") != "streaming":
                raise BatchError("stream_kind_invalid")
            state, cost, resolution = db.execute(
                "SELECT state,cost_micro,resolution FROM remote_admissions WHERE id=?", (ident,)
            ).fetchone()
            response_settled = state == "settled" and resolution == "response"
            if state == "settled":
                if (type(cost) is not int or not 0 <= cost <= _MAX
                        or resolution not in {"response", "operator"}):
                    raise BatchError("stream_accounting_invalid")
                if response_settled:
                    response = record.get("response") or {}
                    usage = response.get("usage")
                    try:
                        if (response.get("identity_valid") is not True or not isinstance(usage, dict)
                                or usage.get("is_byok") is not False or _micros(usage.get("cost")) != cost):
                            raise BatchError("stream_accounting_mismatch")
                    except AdmissionError as exc:
                        raise BatchError("stream_accounting_mismatch") from exc
            elif not (cost is None and (
                    state in {"reserved", "unknown"} and resolution is None
                    or state == "released" and resolution == "unsent")):
                raise BatchError("stream_accounting_invalid")
            result = self.stream_summary(record, settled=response_settled)
            result.update(admission_settled=state == "settled", admission_resolution=resolution,
                          provider_billing="confirmed" if response_settled else "unknown")
            if state == "settled" and resolution == "operator":
                result["operator_reconciled_cost_usd"] = cost / 1_000_000
            return result

    async def submit_stream(self, ident, send=transport, *, provider_status):
        record = await self.begin_submission(ident, provider_status=provider_status, kind="streaming")
        response = await send(record["body"])
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            current = self._read(db, ident)
            if current["state"] != "submission_unknown":
                raise BatchError("diagnostic_state_conflict")
            current.update(response=response, state="stream_receipt_saved")
            self._save(db, current)
        return self.reconcile(ident)

    def reconcile(self, ident):
        record = self.read(ident)
        if record.get("kind") != "streaming" or record["state"] != "stream_receipt_saved":
            raise BatchError("stream_receipt_not_saved")
        response = record["response"]
        # SSE IDs must agree within this fixed authenticated response; the
        # separate HTTP generation header is retained only for correlation.
        usage = response.get("usage")
        settled = (response.get("identity_valid") is True and isinstance(usage, dict)
            and usage.get("is_byok") is False) and Admission(
            self.root, ident, record["generation"]).settle_response(response)
        return self.stream_summary(record, settled=settled)

    @staticmethod
    def stream_summary(record, *, settled=None):
        response = record.get("response") or {}
        output = "".join(response.get("content_chunks", []))
        valid = response.get("identity_valid") is True and response.get("done") is True and not response.get("error")
        result = {"diagnostic_id": record["id"], "kind": "streaming", "local_state": record["state"],
            "http_status": response.get("http_status"), "header_seconds": response.get("header_seconds"),
            "first_content_seconds": response.get("first_content_seconds"),
            "elapsed_seconds": response.get("elapsed_seconds"), "content_chunks": len(response.get("content_chunks", [])),
            "keepalives": response.get("comments", 0), "done": response.get("done", False),
            "validated_output": bool(valid and output.split() == ["1", "2", "3", "4", "5"]),
            "error": response.get("error"), "billing_settled": settled,
            "backlog_publications": 0, "private_data_sent": False}
        if settled:
            result["actual_cost_usd"] = response["usage"]["cost"]
        return result
