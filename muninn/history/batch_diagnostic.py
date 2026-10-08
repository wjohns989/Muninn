"""Explicit one-shot synthetic batch probe, never a backlog checkpoint bypass.

Only the operator CLI calls submit. Ordinary admission still blocks on ALL
unresolved rows. An encrypted probe and its separate admission live atomically
in the existing portable accounting database. Unknown submissions never retry.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from decimal import Decimal

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.historical_batch import (
    BatchError, MODEL, MODEL_IDENTITIES, PROVIDER, TERMINAL, _provider_id, _wire_json, billed_cost,
)
from muninn.history.remote_accounting import (
    Admission, AdmissionError, _check_id, _db, _policy, _periods, _reported_usage, _require_headroom,
)

CEILING = Decimal("0.01")


def request_body():
    from muninn.history.secure_analysis import _CITED_SCHEMA, _cited_prompt
    text = "The synthetic project requires keeping source citations."
    window = {"text": text, "role": "user", "provider": "codex",
        "event_at": 1791331200, "time_basis": "provider_record", "project_basis": "provider_record",
        "citation_ranges": [{"start": 0, "length": len(text)}]}
    return {"endpoint": "/v1/chat/completions", "model": MODEL,
        "provider": {"only": [PROVIDER]}, "completion_window": "24h", "requests": [
            {"custom_id": uuid.uuid4().hex, "body": {"messages": [
                {"role": "user", "content": 'Return exactly {"ok":true} as JSON.'}],
                "max_tokens": 512, "response_format": {"type": "json_object"}}},
            {"custom_id": uuid.uuid4().hex, "body": {"messages": _cited_prompt(window),
                "max_tokens": 512, "response_format": {"type": "json_schema", "json_schema": {
                    "name": "secure_excerpt_analysis", "strict": True, "schema": _CITED_SCHEMA}}}}]}


def verify_price(body, catalog):
    # Use the highest published OpenAI price (including long-context overrides),
    # without assuming a batch discount. UTF-8 bytes overestimate input tokens.
    prices = []
    for endpoint in catalog.get("data", {}).get("endpoints", []):
        if endpoint.get("provider_name") == "OpenAI":
            pricing = endpoint.get("pricing", {})
            prices.extend([pricing, *pricing.get("overrides", [])])
    if not prices:
        raise BatchError("diagnostic_price_unknown")
    try:
        prompt = max(Decimal(p["prompt"]) for p in prices)
        completion = max(Decimal(p["completion"]) for p in prices)
        estimate = Decimal(len(_wire_json(body)) + 1024) * prompt + Decimal(1024) * completion
        if not all(p.is_finite() and p >= 0 for p in (prompt, completion, estimate)) or estimate > CEILING:
            raise ValueError
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        raise BatchError("diagnostic_price_bound") from exc
    return estimate


class DiagnosticStore:
    def __init__(self, archive, root):
        self.archive, self.root = archive, root
        self.key = hmac.new(archive._key, b"muninn-batch-diagnostic-v1", hashlib.sha256).digest()
        self.aad = b"muninn-batch-diagnostic-v1\0" + archive.vault_id.encode("ascii")

    def _seal(self, record):
        nonce = os.urandom(12)
        return nonce + AESGCM(self.key).encrypt(nonce, _wire_json(record),
                                               self.aad + record["id"].encode("ascii"))

    def _read(self, db, ident):
        _check_id(ident)
        row = db.execute("SELECT sealed FROM batch_diagnostics WHERE id=?", (ident,)).fetchone()
        try:
            sealed = row[0]
            record = json.loads(AESGCM(self.key).decrypt(sealed[:12], sealed[12:],
                                                       self.aad + ident.encode("ascii")))
            if record["id"] != ident:
                raise ValueError
            return record
        except Exception as exc:
            raise BatchError("diagnostic_evidence_invalid") from exc

    def read(self, ident):
        with _db(self.root) as (db, _):
            return self._read(db, ident)

    def _save(self, db, record):
        db.execute("INSERT INTO batch_diagnostics VALUES(?,?) ON CONFLICT(id) DO UPDATE SET sealed=excluded.sealed",
                   (record["id"], self._seal(record)))

    def prepare(self, parent, generation, retention_generation, provider_status, body):
        """Private preimage first, then additive schema plus one atomic reserve.

        Caller already authenticated a known submitted parent from BatchOutbox.
        The admission's parent ownership/generation is rechecked under the lock.
        No released/unknown diagnostic ever grants a second attempt for a parent.
        """
        _check_id(parent)
        expected = request_body()
        try:
            ids = [request["custom_id"] for request in body["requests"]]
            for ident in ids:
                _check_id(ident)
            if len(set(ids)) != 2:
                raise ValueError
            for request, ident in zip(expected["requests"], ids):
                request["custom_id"] = ident
            if body != expected or list(body) != list(expected):
                raise ValueError
        except (ValueError, KeyError, TypeError) as exc:
            raise BatchError("diagnostic_fixed_input_required") from exc
        reported = _reported_usage(provider_status)
        now = time.time()
        day, month = _periods(now)
        with _db(self.root, initialize=True, generation=generation) as (db, _):
            caps = _policy(db, generation)
            owners = db.execute("SELECT batch_owner,generation,state FROM remote_admissions "
                                "WHERE state IN ('reserved','unknown')").fetchall()
            if owners != [(parent, generation, "unknown")]:
                raise AdmissionError("diagnostic_parent_not_sole_unknown")
            if db.execute("SELECT enabled,generation FROM batch_policy WHERE id=1").fetchone() != (1, retention_generation):
                raise AdmissionError("remote_consent_revoked")
            _require_headroom(db, caps, reported, day, month, 10000)
            from muninn.history.batch_activation import _backup_batch_policy, _database
            columns = {r[1] for r in db.execute("PRAGMA table_info(remote_admissions)")}
            if "diagnostic_parent" not in columns:
                _backup_batch_policy(self.root, _database(self.root))
                db.execute("ALTER TABLE remote_admissions ADD COLUMN diagnostic_parent TEXT")
                db.execute("CREATE TABLE batch_diagnostics(id TEXT PRIMARY KEY,sealed BLOB NOT NULL)")
                db.execute("DROP INDEX one_remote_admission")
                db.execute("CREATE UNIQUE INDEX one_remote_admission ON remote_admissions((1)) "
                    "WHERE state IN ('reserved','unknown') AND diagnostic_parent IS NULL")
                db.execute("CREATE UNIQUE INDEX one_diagnostic_admission ON remote_admissions((1)) "
                    "WHERE state IN ('reserved','unknown') AND diagnostic_parent IS NOT NULL")
                db.execute("CREATE UNIQUE INDEX one_diagnostic_per_parent ON remote_admissions(diagnostic_parent) "
                    "WHERE diagnostic_parent IS NOT NULL")
            if db.execute("SELECT 1 FROM remote_admissions WHERE diagnostic_parent=?", (parent,)).fetchone():
                raise AdmissionError("diagnostic_already_attempted")
            from muninn.history.portable_accounting import _validate
            _validate(db)
            ident, owner = uuid.uuid4().hex, uuid.uuid4().hex
            db.execute("INSERT INTO remote_admissions(id,generation,state,started,start_day,start_month,"
                "batch_owner,diagnostic_parent) VALUES(?,?,'reserved',?,?,?,?,?)",
                (ident, generation, now, day, month, owner, parent))
            record = {"id": ident, "parent": parent, "owner": owner, "generation": generation,
                "retention_generation": retention_generation, "state": "prepared", "body": body,
                "provider_id": None, "response": None, "created_at": now, "ceiling_usd": str(CEILING)}
            self._save(db, record)
        return ident

    async def submit(self, ident, send, *, provider_status):
        # Admission + encrypted uncertainty phase must commit together before HTTP.
        reported = _reported_usage(await asyncio.to_thread(provider_status))
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            record = self._read(db, ident)
            if record["state"] != "prepared":
                raise BatchError("diagnostic_never_resubmit")
            caps = _policy(db, record["generation"])
            day, month = _periods(time.time())
            _require_headroom(db, caps, reported, day, month, 10000)
            if db.execute("SELECT enabled,generation FROM batch_policy WHERE id=1").fetchone() != (1, record["retention_generation"]):
                raise AdmissionError("remote_consent_revoked")
            if db.execute("SELECT batch_owner,generation,state FROM remote_admissions "
                    "WHERE diagnostic_parent IS NULL AND state IN ('reserved','unknown')").fetchall() != [
                    (record["parent"], record["generation"], "unknown")]:
                raise AdmissionError("diagnostic_parent_not_sole_unknown")
            if db.execute("UPDATE remote_admissions SET state='unknown' WHERE id=? AND state='reserved'",
                          (ident,)).rowcount != 1:
                raise AdmissionError("remote_accounting_conflict")
            record["state"] = "submission_unknown"
            self._save(db, record)
        response = await send("POST", body=record["body"])
        # Preserve every bounded received receipt BEFORE identity validation.
        # A received ID with mismatched metadata is an untrusted GET candidate,
        # not permission to resubmit or settle a potentially foreign batch bill.
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            current = self._read(db, ident)
            if current["state"] != "submission_unknown":
                raise BatchError("diagnostic_state_conflict")
            current["response"] = response
            if isinstance(response, dict) and _provider_id(response.get("id")):
                current["recovery_candidate"] = response["id"]
            self._save(db, current)
        self._identity(record, response, submitted=True)
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            current = self._read(db, ident)
            if current["state"] != "submission_unknown":
                raise BatchError("diagnostic_state_conflict")
            current.update(state="submitted", provider_id=response["id"], response=response)
            self._save(db, current)
        return self.summary(current)

    @staticmethod
    def _identity(record, response, *, submitted=False):
        if (not isinstance(response, dict) or not _provider_id(response.get("id"))
                or not submitted and response["id"] != record["provider_id"]
                or response.get("model") not in MODEL_IDENTITIES
                or response.get("endpoint") != "/v1/chat/completions"
                or response.get("completion_window") != "24h"
                or response.get("status") not in {"validating", "in_progress", "finalizing", *TERMINAL}
                or type((response.get("request_counts") or {}).get("total")) is not int
                or (response.get("request_counts") or {}).get("total") != 2):
            raise BatchError("diagnostic_provider_identity_invalid")

    async def poll(self, ident, send):
        record = self.read(ident)
        if record["state"] not in {"submitted", "terminal_saved"}:
            raise BatchError("diagnostic_provider_identity_unknown")
        response = record["response"] if record["state"] == "terminal_saved" else await send(
            "GET", provider_id=record["provider_id"])
        self._identity(record, response)
        with _db(self.root) as (db, _):
            db.rollback()
            db.execute("BEGIN IMMEDIATE")
            current = self._read(db, ident)
            current.update(response=response, state="terminal_saved" if response["status"] in TERMINAL else "submitted")
            self._save(db, current)
        summary = self.summary(current)
        if response["status"] in TERMINAL:
            validated = validate_results(current)
            cost = billed_cost(response)
            Admission(self.root, ident, current["generation"]).settle_response(response)
            summary["actual_aggregate_cost_usd"] = float(cost)
            summary["validated_requests"] = validated
        return summary

    @staticmethod
    def summary(record):
        response = record.get("response") or {}
        counts = response.get("request_counts") or {}
        count = lambda name: counts.get(name) if type(counts.get(name)) is int and 0 <= counts[name] <= 2 else None
        provider_state = response.get("status")
        if provider_state not in {"validating", "in_progress", "finalizing", *TERMINAL}:
            provider_state = "unknown"
        return {"diagnostic_id": record["id"],
            "local_state": record["state"], "provider_state": provider_state,
            "requests": 2, "completed": count("completed"), "failed": count("failed"),
            "age_seconds": round(time.time() - record["created_at"], 1),
            "backlog_publications": 0, "private_data_sent": False}


def validate_results(record):
    response = record["response"]
    rows = response.get("results")
    ids = [r["custom_id"] for r in record["body"]["requests"]]
    if response["status"] != "completed" or not isinstance(rows, list) or len(rows) != 2:
        raise BatchError("diagnostic_results_incomplete")
    matched = {}
    for row in rows:
        ident = row.get("custom_id") if isinstance(row, dict) else None
        if ident not in ids or ident in matched:
            raise BatchError("diagnostic_result_binding_invalid")
        matched[ident] = row
    valid = 0
    from jsonschema import validate
    from muninn.history.secure_analysis import _CITED_SCHEMA
    for index, ident in enumerate(ids):
        row = matched[ident]
        reply = row.get("response") or {}
        body = reply.get("body") or {}
        if row.get("error") is not None or reply.get("status_code") != 200 or body.get("model") not in MODEL_IDENTITIES:
            continue
        try:
            choice = body["choices"][0]
            if choice["finish_reason"] != "stop":
                continue
            value = json.loads(choice["message"]["content"])
            if index == 0:
                if value != {"ok": True} or type(value.get("ok")) is not bool:
                    continue
            else:
                validate(value, _CITED_SCHEMA)
                text = "The synthetic project requires keeping source citations."
                for proposal in value["proposals"]:
                    start, quote = proposal["start"], proposal["quote"]
                    if type(start) is not int or text[start:start + len(quote)] != quote:
                        raise ValueError
            valid += 1
        except Exception:
            continue
    return valid
