"""Encrypted, idempotent capture outbox; queueing is not inference coverage.

Uses the capture journal's existing transaction/encryption/portable backup
boundary. No provider dispatch, raw text, path or service lifecycle belongs here.
"""
import hashlib
import hmac
import json
import re
import time

from muninn.history.credential_crypto import VaultIntegrityError

_FIELDS = {"vault_id", "blob", "sha256", "version", "commit_generation", "provider", "kind"}
_CONTROL_ID = "0" * 32
_PROVIDERS = {"codex", "claude_code", "gemini_cli"}


class CaptureEnrichmentMixin:
    def _init_enrichment(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS capture_enrichment_control ("
                   "id INTEGER PRIMARY KEY CHECK(id=1), sealed_config BLOB NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS capture_enrichment_sources ("
                   "work_id TEXT PRIMARY KEY, sealed_receipt BLOB NOT NULL, created_at REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS capture_enrichment_progress ("
                   "id INTEGER PRIMARY KEY CHECK(id=1), sealed_cursor BLOB NOT NULL)")

    @staticmethod
    def _enrichment_limit(limit):
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid capture enrichment batch limit")

    def _enrichment_baseline(self, db):
        row = db.execute("SELECT sealed_config FROM capture_enrichment_control WHERE id=1").fetchone()
        if row is None:
            return None
        value = self._open_search(row[0], _CONTROL_ID, "capture-enrichment-config-v1")
        if (not isinstance(value, dict) or set(value) != {"format", "starting_generation"}
                or type(value["format"]) is not int or value["format"] != 1
                or type(value["starting_generation"]) is not int or value["starting_generation"] < 0):
            raise VaultIntegrityError("Capture enrichment baseline is invalid")
        return value["starting_generation"]

    def configure_enrichment(self, starting_generation):
        """Called under the archive lock when enabling; never reset the watermark."""
        if type(starting_generation) is not int or starting_generation < 0:
            raise ValueError("Invalid capture enrichment baseline")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            baseline = self._enrichment_baseline(db)
            if baseline is not None:
                return baseline
            sealed = self._seal_search({"format": 1, "starting_generation": starting_generation},
                                       _CONTROL_ID, "capture-enrichment-config-v1")
            db.execute("INSERT INTO capture_enrichment_control VALUES(1,?)", (sealed,))
            cursor = {"format": 1, "after_generation": starting_generation,
                      "through_generation": starting_generation, "source_index": 0, "version_index": 0}
            db.execute("INSERT INTO capture_enrichment_progress VALUES(1,?)", (
                self._seal_search(cursor, _CONTROL_ID, "capture-enrichment-progress-v1"),))
        return starting_generation

    def _validate_enrichment_receipt(self, value):
        if (not isinstance(value, dict) or set(value) != _FIELDS
                or value["vault_id"] != self.archive.vault_id
                or not isinstance(value["blob"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["blob"])
                or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
                or type(value["version"]) is not int or value["version"] < 0
                or (value["commit_generation"] is not None
                    and (type(value["commit_generation"]) is not int or value["commit_generation"] < 1))
                or not isinstance(value["provider"], str) or value["provider"] not in _PROVIDERS
                or value["kind"] != "transcript"):
            raise VaultIntegrityError("Capture enrichment receipt is invalid")

    def _enrichment_id(self, receipt):
        self._validate_enrichment_receipt(receipt)
        return hmac.new(self._key, b"capture-enrichment-source-v1\0" + json.dumps(
            receipt, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()

    def _store_enrichment_receipt(self, db, receipt, baseline):
        ident = self._enrichment_id(receipt)
        if receipt["commit_generation"] is None or receipt["commit_generation"] <= baseline:
            return "before_watermark"
        if db.execute("SELECT 1 FROM capture_enrichment_sources WHERE work_id=?", (ident,)).fetchone():
            return "existing"
        sealed = self._seal_search(receipt, ident, "capture-enrichment-receipt-v1")
        db.execute("INSERT INTO capture_enrichment_sources(work_id,sealed_receipt,created_at,sealed_planning_state) "
                   "VALUES(?,?,?,?)", (ident, sealed, time.time(), self._new_capture_planning_state(ident)))
        self._adjust_capture_schedule(db, pending=1, planning=1)
        return "queued"

    def enqueue_enrichment_receipt(self, receipt):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            baseline = self._enrichment_baseline(db)
            if baseline is None:
                return "disabled"
            self._capture_schedule(db)
            return self._store_enrichment_receipt(db, receipt, baseline)

    def _enrichment_progress(self, db, baseline):
        row = db.execute("SELECT sealed_cursor FROM capture_enrichment_progress WHERE id=1").fetchone()
        if row is None:
            raise VaultIntegrityError("Capture enrichment progress is missing")
        value = self._open_search(row[0], _CONTROL_ID, "capture-enrichment-progress-v1")
        fields = {"format", "after_generation", "through_generation", "source_index", "version_index"}
        if (not isinstance(value, dict) or set(value) != fields
                or any(type(item) is not int for item in value.values()) or value["format"] != 1
                or not baseline <= value["after_generation"] <= value["through_generation"]
                or value["source_index"] < 0 or value["version_index"] < 0
                or (value["after_generation"] == value["through_generation"]
                    and (value["source_index"] or value["version_index"]))):
            raise VaultIntegrityError("Capture enrichment progress is invalid")
        return row[0], value

    @staticmethod
    def _validate_cursor_position(cursor, manifest):
        keys = list(manifest["files"])
        source = cursor["source_index"]
        version = cursor["version_index"]
        if (source > len(keys) or (source == len(keys) and version)
                or (source < len(keys) and version > len(manifest["files"][keys[source]]))):
            raise VaultIntegrityError("Capture enrichment cursor exceeds snapshot")
        return keys

    def reconcile_enrichment(self, *, limit=128):
        """Examine at most limit entries, not just limit new inserts.

        The existing manifest format still requires O(catalog size) metadata
        loading. No raw source is read; this is not a bulk-scaling claim.
        """
        self._enrichment_limit(limit)
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            if baseline is None:
                return 0
            previous_seal, cursor = self._enrichment_progress(db, baseline)
        if cursor["after_generation"] == cursor["through_generation"]:
            manifest = self.archive._load_manifest()
            if manifest["generation"] < cursor["after_generation"]:
                raise VaultIntegrityError("Capture enrichment progress exceeds archive")
            if manifest["generation"] == cursor["after_generation"]:
                return 0
            cursor["through_generation"] = manifest["generation"]
        else:
            manifest = self.archive._load_manifest(generation=cursor["through_generation"])
        keys = self._validate_cursor_position(cursor, manifest)
        pending = []
        examined = 0
        while cursor["source_index"] < len(keys) and examined < limit:
            entries = manifest["files"][keys[cursor["source_index"]]]
            examined += 1
            if cursor["version_index"] == len(entries):
                cursor["source_index"] += 1
                cursor["version_index"] = 0
                continue
            version = cursor["version_index"]
            entry = entries[version]
            cursor["version_index"] += 1
            if cursor["version_index"] == len(entries):
                cursor["source_index"] += 1
                cursor["version_index"] = 0
            generation = entry.get("commit_generation")
            if (type(generation) is int
                    and cursor["after_generation"] < generation <= cursor["through_generation"]
                    and entry["provider"] in _PROVIDERS and entry["kind"] == "transcript"):
                pending.append(self.archive._snapshot_receipt(entry, version))
        if cursor["source_index"] == len(keys):
            cursor.update(after_generation=cursor["through_generation"], source_index=0, version_index=0)
        queued = 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._enrichment_baseline(db) != baseline:
                raise VaultIntegrityError("Capture enrichment baseline changed")
            # A concurrent reconciler may have advanced the checkpoint. Never
            # overwrite its progress; its receipts committed in the same txn.
            if self._enrichment_progress(db, baseline)[0] != previous_seal:
                return 0
            self._capture_schedule(db)
            for receipt in pending:
                queued += self._store_enrichment_receipt(db, receipt, baseline) == "queued"
            db.execute("UPDATE capture_enrichment_progress SET sealed_cursor=? WHERE id=1", (
                self._seal_search(cursor, _CONTROL_ID, "capture-enrichment-progress-v1"),))
        return queued

    def _read_enrichment_receipt(self, row, baseline):
        ident, sealed = row
        if not isinstance(ident, str) or not re.fullmatch(r"[0-9a-f]{64}", ident):
            raise VaultIntegrityError("Capture enrichment identity is invalid")
        value = self._open_search(sealed, ident, "capture-enrichment-receipt-v1")
        if (self._enrichment_id(value) != ident or baseline is None
                or value["commit_generation"] is None or value["commit_generation"] <= baseline):
            raise VaultIntegrityError("Capture enrichment identity is invalid")
        return value

    def pending_enrichment(self, *, limit=128):
        self._enrichment_limit(limit)
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            self._capture_schedule(db)
            result = []
            for row in db.execute("SELECT * FROM capture_enrichment_sources WHERE resolved=0 "
                                  "ORDER BY created_at,work_id LIMIT ?", (limit,)):
                self._capture_plan_state(row)
                result.append(self._read_enrichment_receipt(row[:2], baseline))
            return result

    def enrichment_status(self):
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            count = self._capture_schedule(db)["pending"]
        return {"configured": baseline is not None, "pending_sources": count}

    def _verify_enrichment(self, db):
        baseline = self._enrichment_baseline(db)
        if baseline is None:
            if (db.execute("SELECT 1 FROM capture_enrichment_sources LIMIT 1").fetchone()
                    or db.execute("SELECT 1 FROM capture_enrichment_progress LIMIT 1").fetchone()):
                raise VaultIntegrityError("Capture enrichment baseline is missing")
            return
        if baseline is not None and baseline > self.archive._load_manifest()["generation"]:
            raise VaultIntegrityError("Capture enrichment baseline exceeds archive generation")
        _seal, cursor = self._enrichment_progress(db, baseline)
        if cursor["through_generation"] > self.archive._load_manifest()["generation"]:
            raise VaultIntegrityError("Capture enrichment progress exceeds archive generation")
        if cursor["after_generation"] != cursor["through_generation"]:
            self._validate_cursor_position(cursor, self.archive._load_manifest(
                generation=cursor["through_generation"]))
        committed = {self._enrichment_id(receipt): receipt for receipt in
            self.archive.iter_committed_receipts(after_generation=baseline if baseline is not None else 0)
            if receipt["provider"] in _PROVIDERS and receipt["kind"] == "transcript"}
        for row in db.execute("SELECT work_id,sealed_receipt FROM capture_enrichment_sources"):
            value = self._read_enrichment_receipt(row, baseline)
            if committed.get(row[0]) != value:
                raise VaultIntegrityError("Capture enrichment commit is unavailable")
