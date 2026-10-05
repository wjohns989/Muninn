"""Explicit, bounded latest-snapshot enrollment; never inference permission."""
import hashlib
import re

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import verify_private, VaultPermissionError

_ID = "0" * 32
_PURPOSE = "capture-historical-cursor-v1"
_GRANT = "capture-historical-grant-v1"
_FIELDS = {"format", "selection", "generation", "manifest_sha", "source_index",
           "total_sources", "queued", "existing", "excluded", "complete"}


class HistoricalEnrollmentMixin:
    def _init_historical_enrollment(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS capture_historical_enrollment "
                   "(id INTEGER PRIMARY KEY CHECK(id=1), sealed_cursor BLOB NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS capture_historical_receipts "
                   "(work_id TEXT PRIMARY KEY, sealed_grant BLOB NOT NULL)")

    def _historical_progress(self, db):
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE name IN "
                  "('capture_historical_enrollment','capture_historical_receipts')")}
        if not tables:  # Read-only preview of an older journal; never migrate it.
            return None, None
        if len(tables) != 2:
            raise VaultIntegrityError("Historical enrollment schema is incomplete")
        row = db.execute("SELECT sealed_cursor FROM capture_historical_enrollment WHERE id=1").fetchone()
        if row is None:
            if db.execute("SELECT 1 FROM capture_historical_receipts LIMIT 1").fetchone():
                raise VaultIntegrityError("Historical enrollment cursor is missing")
            return None, None
        value = self._open_search(row[0], _ID, _PURPOSE)
        numbers = ("generation", "source_index", "total_sources", "queued", "existing", "excluded")
        if (not isinstance(value, dict) or set(value) != _FIELDS
                or type(value["format"]) is not int or value["format"] != 1
                or value["selection"] != "latest-v1"
                or any(type(value[k]) is not int or not 0 <= value[k] < 2**63 for k in numbers)
                or value["generation"] < 1
                or not isinstance(value["manifest_sha"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha"])
                or type(value["complete"]) is not bool
                or value["source_index"] != value["queued"] + value["existing"] + value["excluded"]
                or value["source_index"] > value["total_sources"]
                or value["complete"] != (value["source_index"] == value["total_sources"])):
            raise VaultIntegrityError("Historical enrollment cursor is invalid")
        return row[0], value

    def _historical_manifest(self, cursor):
        return self._verify_historical_pin(cursor, catalog=True)

    def _verify_historical_pin(self, cursor, *, catalog=False):
        path = self.archive.root / f"manifest-{cursor['generation']:012d}.enc"
        try:
            verify_private(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, VaultPermissionError) as exc:
            raise VaultIntegrityError("Historical enrollment manifest is unavailable") from exc
        if digest != cursor["manifest_sha"]:
            raise VaultIntegrityError("Historical enrollment manifest identity differs")
        identity = (cursor["generation"], digest, cursor["total_sources"])
        if not catalog and getattr(self, "_historical_verified_identity", None) == identity:
            return  # Cache only proof identity, never paths or source text.
        manifest = self.archive._load_manifest(generation=cursor["generation"])
        # Bind the decrypted catalog to exactly the bytes just checked.
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise VaultIntegrityError("Historical enrollment manifest changed during verification")
        if len(manifest["files"]) != cursor["total_sources"]:
            raise VaultIntegrityError("Historical enrollment source count differs")
        self._historical_verified_identity = identity
        return manifest if catalog else None

    def historical_enrollment_status(self):
        with self._connect() as db:
            _seal, cursor = self._historical_progress(db)
        return cursor

    def _historical_grant(self, db, ident):
        _seal, cursor = self._historical_progress(db)
        if cursor is None:
            raise VaultIntegrityError("Historical receipt has no enrollment authority")
        self._verify_historical_pin(cursor)  # Fresh byte hash even on a proof-identity cache hit.
        row = db.execute("SELECT sealed_grant FROM capture_historical_receipts WHERE work_id=?", (ident,)).fetchone()
        if row is None:
            raise VaultIntegrityError("Historical receipt grant is missing")
        grant = self._open_search(row[0], ident, _GRANT)
        expected = {"format": 1, "generation": cursor["generation"], "manifest_sha": cursor["manifest_sha"]}
        if (not isinstance(grant, dict) or grant != expected
                or type(grant["format"]) is not int or type(grant["generation"]) is not int):
            raise VaultIntegrityError("Historical receipt grant differs from enrollment")
        return grant

    def _store_historical_grant(self, db, ident, cursor):
        grant = {"format": 1, "generation": cursor["generation"], "manifest_sha": cursor["manifest_sha"]}
        db.execute("INSERT INTO capture_historical_receipts VALUES(?,?)", (
            ident, self._seal_search(grant, ident, _GRANT)))

    def _historical_selection(self, cursor):
        if cursor is None:
            manifest = self.archive._load_manifest()
            generation = manifest["generation"]
            path = self.archive.root / f"manifest-{generation:012d}.enc"
            cursor = {"format": 1, "selection": "latest-v1", "generation": generation,
                      "manifest_sha": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "source_index": 0, "total_sources": len(manifest["files"]),
                      "queued": 0, "existing": 0, "excluded": 0,
                      "complete": not manifest["files"]}
        manifest = self._historical_manifest(cursor)
        return cursor, manifest

    def preview_historical_latest(self, *, limit=128):
        """Counts only. The caller may provide a query-only database connection."""
        self._enrichment_limit(limit)
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            _seal, cursor = self._historical_progress(db)
            if baseline is None:
                raise VaultIntegrityError("Historical enrollment requires configured capture")
            cursor, manifest = self._historical_selection(cursor)
            counts = {"would_queue": 0, "would_existing": 0, "would_exclude": 0}
            entries = list(manifest["files"].values())[cursor["source_index"]:cursor["source_index"] + limit]
            for versions in entries:
                if not versions or versions[-1]["provider"] not in {"codex", "claude_code", "gemini_cli"} or versions[-1]["kind"] != "transcript":
                    counts["would_exclude"] += 1
                    continue
                receipt = self.archive._snapshot_receipt(versions[-1], len(versions) - 1)
                ident = self._enrichment_id(receipt)
                row = db.execute("SELECT work_id,sealed_receipt FROM capture_enrichment_sources WHERE work_id=?", (ident,)).fetchone()
                if row is not None and self._read_enrichment_receipt(row, baseline, db=db) != receipt:
                    raise VaultIntegrityError("Existing enrichment receipt differs")
                counts["would_existing" if row is not None else "would_queue"] += 1
        return {"stage": "preview", "batch_sources": len(entries), "enrollment": cursor, **counts}

    def enroll_historical_latest(self, *, limit=128):
        """Explicit owner operation. Queue one pinned latest snapshot per source."""
        self._enrichment_limit(limit)
        with self._connect() as db:
            baseline = self._enrichment_baseline(db)
            if baseline is None:
                raise VaultIntegrityError("Historical enrollment requires configured capture")
            previous, cursor = self._historical_progress(db)
        cursor, manifest = self._historical_selection(cursor)
        if baseline > manifest["generation"]:
            raise VaultIntegrityError("Capture enrichment baseline exceeds archive generation")
        keys = list(manifest["files"])
        end = min(cursor["source_index"] + limit, len(keys))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._enrichment_baseline(db) != baseline:
                raise VaultIntegrityError("Capture enrichment baseline changed")
            current_seal, current = self._historical_progress(db)
            if current_seal != previous:
                return current  # A concurrent batch already committed its cursor.
            self._capture_schedule(db)
            for key in keys[cursor["source_index"]:end]:
                entries = manifest["files"][key]
                if not entries or entries[-1]["provider"] not in {"codex", "claude_code", "gemini_cli"} or entries[-1]["kind"] != "transcript":
                    cursor["excluded"] += 1
                    continue
                receipt = self.archive._snapshot_receipt(entries[-1], len(entries) - 1)
                outcome = self._store_enrichment_receipt(db, receipt, baseline, historical=cursor)
                cursor["queued" if outcome == "queued" else "existing"] += 1
            cursor.update(source_index=end, complete=end == len(keys))
            db.execute("INSERT INTO capture_historical_enrollment VALUES(1,?) "
                       "ON CONFLICT(id) DO UPDATE SET sealed_cursor=excluded.sealed_cursor", (
                self._seal_search(cursor, _ID, _PURPOSE),))
        return cursor

    def _verify_historical_enrollment(self, db, baseline):
        _seal, cursor = self._historical_progress(db)
        if cursor is None:
            return {}
        if baseline is None:
            raise VaultIntegrityError("Historical enrollment has no live baseline")
        manifest = self._historical_manifest(cursor)
        selected = {}
        for entries in list(manifest["files"].values())[:cursor["source_index"]]:
            if entries and entries[-1]["provider"] in {"codex", "claude_code", "gemini_cli"} and entries[-1]["kind"] == "transcript":
                receipt = self.archive._snapshot_receipt(entries[-1], len(entries) - 1)
                selected[self._enrichment_id(receipt)] = receipt
        if len(selected) != cursor["queued"] + cursor["existing"]:
            raise VaultIntegrityError("Historical enrollment selection differs from cursor")
        for ident in selected:
            row = db.execute("SELECT work_id,sealed_receipt FROM capture_enrichment_sources WHERE work_id=?", (ident,)).fetchone()
            if row is None or self._read_enrichment_receipt(row, baseline, db=db) != selected[ident]:
                raise VaultIntegrityError("Historical enrollment receipt is unavailable")
        for row in db.execute("SELECT work_id FROM capture_historical_receipts"):
            if row[0] not in selected:
                raise VaultIntegrityError("Historical receipt grant is orphaned")
            self._historical_grant(db, row[0])
        return selected
