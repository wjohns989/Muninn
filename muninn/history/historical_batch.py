"""Encrypted batch recovery boundary; deliberately no HTTP or queue activation.

This is not a substitute for capture ownership, consent or budget escrow. The
caller must establish those before dispatch. No transport is exposed here until
that integration exists. Records use the portable archive key, never a bearer.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from decimal import Decimal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file, verify_private

MODEL = "openai/gpt-6-luna-pro"
PROVIDER = "openai"
MAX_ITEMS = 24  # A bounded batch, not a whole-source or historical size limit.
TERMINAL = {"completed", "failed", "expired", "cancelled"}
_STATES = {"prepared", "submission_unknown", "submitted", "terminal_saved", "cleaned"}
_MARKER = b"muninn-historical-batch-outbox-v1\n"


class BatchError(ValueError):
    """Fixed categories only: never echo a provider body or transcript."""


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise BatchError("batch_json_invalid") from exc


def _opaque(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def _provider_id(value):
    return isinstance(value, str) and re.fullmatch(r"batch_[A-Za-z0-9_-]{1,160}", value) is not None


def prepare_items(source, bindings):
    """Authenticate and screen each actual source before constructing a prompt.

    Binding tuples are (opaque job ID, immutable cited descriptor). Acquisition
    of exclusive capture ownership is a separate required integration step.
    """
    from muninn.history.secure_analysis import _CITED_SCHEMA, _cited_prompt, _request_safe
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= MAX_ITEMS:
        raise BatchError("batch_item_count_invalid")
    items, seen = [], set()
    for job_id, descriptor in bindings:
        if not _opaque(job_id) or job_id in seen:
            raise BatchError("batch_binding_invalid")
        seen.add(job_id)
        window = source.remote_input(descriptor)
        if window is None:
            raise BatchError("source_not_remote_safe")
        body = {"messages": _cited_prompt(window), "max_completion_tokens": 2048,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "secure_excerpt_analysis", "strict": True, "schema": _CITED_SCHEMA}}}
        if not _request_safe(body):
            raise BatchError("source_not_remote_safe")
        items.append({"custom_id": uuid.uuid4().hex, "job_id": job_id,
                      "window": descriptor, "body": body})
    return json.loads(_json(items))  # Detach caller-owned mutable descriptors.


def payload(items):
    """Required routing fields precede requests for OpenRouter's stream parser."""
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise BatchError("batch_item_count_invalid")
    custom_ids, jobs = set(), set()
    for item in items:
        if (not isinstance(item, dict) or set(item) != {"custom_id", "job_id", "window", "body"}
                or not _opaque(item["custom_id"]) or not _opaque(item["job_id"])
                or item["custom_id"] in custom_ids or item["job_id"] in jobs
                or not isinstance(item["body"], dict)):
            raise BatchError("batch_binding_invalid")
        CitedAnalysisSource.validate_descriptor(item["window"])
        custom_ids.add(item["custom_id"])
        jobs.add(item["job_id"])
    value = {"endpoint": "/v1/chat/completions", "model": MODEL,
             "provider": {"only": [PROVIDER]}, "completion_window": "24h",
             "requests": [{"custom_id": i["custom_id"], "body": i["body"]} for i in items]}
    return json.loads(_json(value))


def terminal_results(items, response):
    """Transport-only reconciliation; HTTP 200 is NOT accepted cited output.

    Count checks describe provider requests, not memory success. Every matched
    item still requires validate_item(), budget settlement and fenced staging.
    """
    expected = {r["custom_id"] for r in payload(items)["requests"]}
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise BatchError("batch_not_completed")
    rows = response.get("results")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise BatchError("batch_results_incomplete")
    matched = {}
    succeeded = 0
    for row in rows:
        if not isinstance(row, dict):
            raise BatchError("batch_result_invalid")
        ident = row.get("custom_id")
        if not _opaque(ident) or ident not in expected or ident in matched:
            raise BatchError("batch_result_binding_invalid")
        reply, error = row.get("response"), row.get("error")
        if (reply is None) == (error is None):
            raise BatchError("batch_result_invalid")
        if reply is not None:
            if not isinstance(reply, dict) or type(reply.get("status_code")) is not int:
                raise BatchError("batch_result_invalid")
            if reply["status_code"] == 200:
                body = reply.get("body")
                if not isinstance(body, dict) or body.get("model") != MODEL:
                    raise BatchError("batch_result_model_mismatch")
                succeeded += 1
        elif not isinstance(error, dict):
            raise BatchError("batch_result_invalid")
        matched[ident] = row
    counts = response.get("request_counts")
    expected_counts = {"total": len(expected), "completed": succeeded,
                       "failed": len(expected) - succeeded}
    if (not isinstance(counts, dict) or any(type(counts.get(k)) is not int
            or counts[k] != v for k, v in expected_counts.items())):
        raise BatchError("batch_result_counts_invalid")
    return matched


def validate_item(source, item, row):
    """Validate one stored reply against its exact authenticated cited window.

    Returns an unstaged extraction, NOT publication authority or a source ACK.
    No item-level cost is invented from an aggregate batch bill.
    """
    from muninn.history.secure_analysis import _cited_outcome
    if not isinstance(row, dict) or row.get("custom_id") != item["custom_id"]:
        raise BatchError("batch_result_binding_invalid")
    reply = row.get("response")
    if (row.get("error") is not None or not isinstance(reply, dict)
            or type(reply.get("status_code")) is not int or reply["status_code"] != 200):
        raise BatchError("batch_item_failed")
    body = reply.get("body")
    if not isinstance(body, dict) or body.get("model") != MODEL:
        raise BatchError("batch_result_model_mismatch")
    choices = body.get("choices")
    if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), dict)
            or not isinstance(choices[0]["message"].get("content"), str)):
        raise BatchError("batch_item_output_invalid")
    return _cited_outcome(choices[0]["message"]["content"], source,
                          item["window"], "openrouter", MODEL)


def billed_cost(response):
    """Missing/nonfinite/BYOK cost is unresolved, never silently zero."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict) or usage.get("is_byok") is not False:
        raise BatchError("batch_cost_unresolved")
    cost = usage.get("cost")
    if type(cost) not in (int, float):
        raise BatchError("batch_cost_unresolved")
    amount = Decimal(str(cost))
    if not amount.is_finite() or amount < 0:
        raise BatchError("batch_cost_unresolved")
    return amount


class BatchOutbox:
    """FULL-synchronous encrypted CAS records, portable with the archive key.

    An uncertainty marker is irreversible here. No timeout clears it, and there
    is intentionally no approximate workspace-list recovery or retry operation.
    Restoring this store requires its marker AND database, alongside the archive.
    """

    def __init__(self, archive):
        self.archive = archive
        self.path = archive.root.absolute() / "historical-batches.db"
        self.marker = archive.root.absolute() / "historical-batches-managed"
        self._key = hmac.new(archive._key, b"muninn-historical-batch-v1", hashlib.sha256).digest()
        self._aad = b"muninn-historical-batch-v1\0" + archive.vault_id.encode("ascii")
        verify_private(archive.root)
        exists = self.path.exists() or self.path.is_symlink()
        managed = self.marker.exists() or self.marker.is_symlink()
        if exists != managed:
            raise VaultIntegrityError("Batch outbox recovery pair is incomplete")
        if not managed:
            create_private_file(self.marker)
            with self.marker.open("wb") as stream:
                stream.write(_MARKER)
                stream.flush()
                os.fsync(stream.fileno())
            create_private_file(self.path)
            with self._db(check_schema=False) as db:
                db.execute("CREATE TABLE batches(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, "
                           "state TEXT NOT NULL, sealed BLOB NOT NULL)")
                db.execute("CREATE TABLE sentinel(version INTEGER NOT NULL)")
                db.execute("INSERT INTO sentinel VALUES(1)")
        with self._db():
            pass

    @contextmanager
    def _db(self, *, check_schema=True):
        for path in (self.archive.root, self.marker, self.path):
            verify_private(path)
        if self.marker.read_bytes() != _MARKER:
            raise VaultIntegrityError("Batch outbox marker is invalid")
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True, timeout=1)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            if check_schema:
                if db.execute("SELECT version FROM sentinel").fetchall() != [(1,)]:
                    raise VaultIntegrityError("Batch outbox schema is unavailable")
                db.execute("SELECT id,revision,state,sealed FROM batches LIMIT 0")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            db.rollback()
            raise VaultIntegrityError("Batch outbox storage is unavailable") from exc
        finally:
            db.close()

    def _seal(self, record):
        nonce = os.urandom(12)
        aad = self._aad + record["id"].encode("ascii")
        return nonce + AESGCM(self._key).encrypt(nonce, _json(record), aad)

    def _read(self, row):
        if row is None:
            raise BatchError("batch_reference_missing")
        ident, revision, state, sealed = row
        if not _opaque(ident) or type(revision) is not int or revision < 0 or state not in _STATES:
            raise VaultIntegrityError("Batch outbox identity is invalid")
        try:
            record = json.loads(AESGCM(self._key).decrypt(
                sealed[:12], sealed[12:], self._aad + ident.encode("ascii")))
        except (InvalidTag, ValueError, UnicodeError, TypeError) as exc:
            raise VaultIntegrityError("Batch outbox authentication failed") from exc
        if (not isinstance(record, dict) or record.get("id") != ident
                or record.get("revision") != revision or record.get("state") != state):
            raise VaultIntegrityError("Batch outbox state authentication failed")
        return record

    def prepare(self, items, *, consent_generation):
        payload(items)
        if type(consent_generation) is not int or consent_generation < 1:
            raise BatchError("batch_consent_invalid")
        ident = uuid.uuid4().hex
        record = {"id": ident, "revision": 0, "state": "prepared",
                  "retention": "temporary_nontraining", "consent_generation": consent_generation,
                  "items": json.loads(_json(items)), "provider_id": None,
                  "terminal": None, "deletion": None}
        with self._db() as db:
            db.execute("INSERT INTO batches VALUES(?,?,?,?)",
                       (ident, 0, "prepared", self._seal(record)))
        return ident

    def read(self, ident):
        with self._db() as db:
            return self._read(db.execute("SELECT * FROM batches WHERE id=?", (ident,)).fetchone())

    def _transition(self, ident, expected_revision, before, after, change):
        if type(expected_revision) is not int or expected_revision < 0:
            raise BatchError("batch_revision_invalid")
        with self._db() as db:
            record = self._read(db.execute("SELECT * FROM batches WHERE id=?", (ident,)).fetchone())
            if record["revision"] != expected_revision or record["state"] != before:
                raise BatchError("batch_state_conflict")
            change(record)
            record.update(state=after, revision=expected_revision + 1)
            db.execute("UPDATE batches SET revision=?,state=?,sealed=? WHERE id=? AND revision=?",
                       (record["revision"], after, self._seal(record), ident, expected_revision))
        return record["revision"]

    def begin_submission(self, ident, expected_revision):
        # This must commit before HTTP. A second caller cannot pass the CAS.
        return self._transition(ident, expected_revision, "prepared", "submission_unknown", lambda r: None)

    def save_submission(self, ident, expected_revision, response):
        def change(record):
            if (not isinstance(response, dict) or not _provider_id(response.get("id"))
                    or response.get("model") != MODEL
                    or response.get("endpoint") != "/v1/chat/completions"
                    or response.get("completion_window") != "24h"
                    or response.get("status") not in {"validating", "in_progress", "finalizing", *TERMINAL}
                    or type((response.get("request_counts") or {}).get("total")) is not int
                    or response["request_counts"]["total"] != len(record["items"])):
                raise BatchError("batch_submission_identity_invalid")
            record["provider_id"] = response["id"]
        return self._transition(ident, expected_revision, "submission_unknown", "submitted", change)

    def save_terminal(self, ident, expected_revision, response):
        def change(record):
            if (not isinstance(response, dict) or response.get("id") != record["provider_id"]
                    or response.get("model") != MODEL or response.get("status") not in TERMINAL
                    or response.get("endpoint") != "/v1/chat/completions"):
                raise BatchError("batch_terminal_identity_invalid")
            # Save full evidence even if item mapping, billing or schema fails.
            record["terminal"] = json.loads(_json(response))
        revision = self._transition(ident, expected_revision, "submitted", "terminal_saved", change)
        self.read(ident)  # Verify durable local ciphertext before any deletion.
        return revision

    def save_cleanup(self, ident, expected_revision, response):
        def change(record):
            if (record["terminal"] is None or not isinstance(response, dict)
                    or response.get("id") != record["provider_id"]
                    or not isinstance(response.get("deletion"), dict)
                    or response["deletion"].get("openrouter") != "deleted"):
                raise BatchError("batch_deletion_unverified")
            record["deletion"] = json.loads(_json(response))
        return self._transition(ident, expected_revision, "terminal_saved", "cleaned", change)
