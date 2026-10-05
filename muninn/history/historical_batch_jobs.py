"""Encrypted, persistent batch ownership; no HTTP, activation or deletion.

Batch members do not hold expiring worker leases while awaiting the provider.
Only local result publication takes a fresh short lease. The authenticated head
is the exclusion authority, not a plaintext phase or membership hint.
"""
import hashlib
import os
import time

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.historical_batch import MAX_ITEMS, BatchError, BatchOutbox, _opaque

_CONTROL = "0" * 32
_HEAD = "historical-batch-head-v1"
_OWNER = "historical-batch-owner-v1"


class HistoricalBatchJobsMixin:
    def _init_historical_batch_jobs(self, db):
        existed = db.execute("SELECT 1 FROM sqlite_master WHERE name='historical_batch_control'").fetchone()
        db.execute("CREATE TABLE IF NOT EXISTS historical_batch_control "
                   "(id INTEGER PRIMARY KEY CHECK(id=1),sealed_head BLOB NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS historical_batch_owners "
                   "(owner_id TEXT PRIMARY KEY,sealed_owner BLOB NOT NULL)")
        if not db.execute("SELECT 1 FROM historical_batch_control WHERE id=1").fetchone():
            if existed or db.execute("SELECT 1 FROM historical_batch_owners LIMIT 1").fetchone():
                raise VaultIntegrityError("Historical batch head is missing")
            db.execute("INSERT INTO historical_batch_control VALUES(1,?)", (
                self._seal_search({"format": 1, "id": None, "sha256": None, "sequence": 0}, _CONTROL, _HEAD),))
        self._historical_batch_head(db)

    def _historical_batch_record(self, db, ident, digest):
        row = db.execute("SELECT * FROM historical_batch_owners WHERE owner_id=?", (ident,)).fetchone()
        if (row is None or not _opaque(ident) or not isinstance(digest, str)
                or hashlib.sha256(row["sealed_owner"]).hexdigest() != digest):
            raise VaultIntegrityError("Historical batch ownership is incomplete")
        value = self._open_search(row["sealed_owner"], ident, _OWNER)
        if (not isinstance(value, dict) or set(value) != {
                "format", "id", "phase", "generation", "admission_id", "members", "previous",
                "input_sha256", "sequence"}
                or type(value["format"]) is not int or value["format"] != 1
                or value["id"] != ident or value["phase"] not in {"owned", "sent", "passed"}
                or type(value["generation"]) is not int or value["generation"] < 1
                or type(value["sequence"]) is not int or value["sequence"] < 1
                or not isinstance(value["input_sha256"], str) or len(value["input_sha256"]) != 64
                or (value["admission_id"] is not None and not _opaque(value["admission_id"]))
                or (value["phase"] == "owned") != (value["admission_id"] is None)
                or not isinstance(value["members"], list) or not 1 <= len(value["members"]) <= MAX_ITEMS):
            raise VaultIntegrityError("Historical batch ownership is invalid")
        previous = value["previous"]
        if previous is not None and (not isinstance(previous, dict)
                or set(previous) != {"id", "sha256"} or not _opaque(previous["id"])
                or not isinstance(previous["sha256"], str) or len(previous["sha256"]) != 64):
            raise VaultIntegrityError("Historical batch predecessor is invalid")
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        seen = set()
        for item in value["members"]:
            if (not isinstance(item, dict) or set(item) != {"job_id", "target_sha256", "window"}
                    or not _opaque(item["job_id"]) or item["job_id"] in seen
                    or not isinstance(item["target_sha256"], str) or len(item["target_sha256"]) != 64):
                raise VaultIntegrityError("Historical batch member is invalid")
            try:
                CitedAnalysisSource.validate_descriptor(item["window"])
            except ValueError as exc:
                raise VaultIntegrityError("Historical batch window is invalid") from exc
            seen.add(item["job_id"])
        return value

    def _historical_batch_head(self, db):
        row = db.execute("SELECT sealed_head FROM historical_batch_control WHERE id=1").fetchone()
        if row is None:
            raise VaultIntegrityError("Historical batch head is missing")
        head = self._open_search(row[0], _CONTROL, _HEAD)
        if (not isinstance(head, dict) or set(head) != {"format", "id", "sha256", "sequence"}
                or type(head["format"]) is not int or head["format"] != 1
                or type(head["sequence"]) is not int or head["sequence"] < 0
                or head["sequence"] != db.execute("SELECT COUNT(*) FROM historical_batch_owners").fetchone()[0]):
            raise VaultIntegrityError("Historical batch head is invalid")
        if head["id"] is None:
            if head["sha256"] is not None or db.execute("SELECT 1 FROM historical_batch_owners LIMIT 1").fetchone():
                raise VaultIntegrityError("Historical batch head is incomplete")
            return head, None
        owner = self._historical_batch_record(db, head["id"], head["sha256"])
        if owner["sequence"] != head["sequence"]:
            raise VaultIntegrityError("Historical batch head sequence differs")
        return head, owner

    def _historical_batch_member_row(self, db, owner, member):
        row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (member["job_id"],)).fetchone()
        if (row is None or row["lane"] != 1 or row["remote_policy_generation"] != owner["generation"]
                or hashlib.sha256(row["sealed_target"]).hexdigest() != member["target_sha256"]
                or self._read_analysis_window(row) != member["window"]
                or row["remote_dispatched"] != int(owner["phase"] != "owned")):
            raise VaultIntegrityError("Historical batch member binding differs")
        self._validated_analysis_target(row, db)
        return row

    def _historical_batch_blocked_jobs(self, db):
        _head, owner = self._historical_batch_head(db)
        if owner is None or owner["phase"] == "passed":
            return set()
        for member in owner["members"]:
            self._historical_batch_member_row(db, owner, member)
        return {member["job_id"] for member in owner["members"]}

    def _save_historical_batch_owner(self, db, owner, *, insert=False):
        sealed = self._seal_search(owner, owner["id"], _OWNER)
        if insert:
            db.execute("INSERT INTO historical_batch_owners VALUES(?,?)", (owner["id"], sealed))
        else:
            db.execute("UPDATE historical_batch_owners SET sealed_owner=? WHERE owner_id=?", (sealed, owner["id"]))
        db.execute("UPDATE historical_batch_control SET sealed_head=? WHERE id=1", (
            self._seal_search({"format": 1, "id": owner["id"],
                               "sequence": owner["sequence"],
                               "sha256": hashlib.sha256(sealed).hexdigest()}, _CONTROL, _HEAD),))

    def historical_batch_owner(self):
        """Nonsecret status only; private descriptors stay inside sealed stores."""
        with self._connect() as db:
            _head, owner = self._historical_batch_head(db)
            self._historical_batch_blocked_jobs(db)
        return None if owner is None else {"id": owner["id"], "phase": owner["phase"],
            "generation": owner["generation"], "items": len(owner["members"])}

    def reserve_historical_batch(self, ident):
        """Bind an already encrypted prepared outbox; does not authorize egress."""
        outbox = BatchOutbox(self.archive).read(ident)
        if outbox["state"] != "prepared":
            raise BatchError("batch_state_conflict")
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        from muninn.history.historical_batch import prepare_items
        expected = prepare_items(CitedAnalysisSource(self.archive), [
            (item["job_id"], item["window"]) for item in outbox["items"]])
        if any(a["body"] != b["body"] for a, b in zip(expected, outbox["items"])):
            raise BatchError("batch_input_binding_invalid")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            head, previous = self._historical_batch_head(db)
            if previous is not None and previous["phase"] != "passed":
                raise BatchError("batch_checkpoint_unresolved")
            members = []
            for item in outbox["items"]:
                row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (item["job_id"],)).fetchone()
                if (row is None or row["lane"] != 1 or row["state"] not in {"pending", "retry"}
                        or row["due_at"] > time.time() or row["remote_dispatched"] != 0
                        or row["remote_policy_generation"] != outbox["consent_generation"]
                        or row["publication_started"] != 0 or row["cancel_requested"] != 0
                        or any(row[k] is not None for k in ("lease_token", "lease_until", "sealed_extraction",
                                                            "sealed_receipt", "sealed_reuse", "sealed_result"))):
                    raise BatchError("batch_member_ineligible")
                target = self._validated_analysis_target(row, db)
                window = item["window"]
                self._assert_capture_window(row, window)
                if any(window[k] != target[k] for k in ("blob", "sha256", "version")):
                    raise BatchError("batch_input_binding_invalid")
                old = self._read_analysis_window(row)
                if old is not None and old != window:
                    raise BatchError("batch_input_binding_invalid")
                if old is None:
                    db.execute("UPDATE history_analysis_jobs SET sealed_window=? WHERE job_id=?", (
                        self._seal_search(window, item["job_id"], self._window_purpose(row, db)), item["job_id"]))
                members.append({"job_id": item["job_id"], "target_sha256": hashlib.sha256(
                    row["sealed_target"]).hexdigest(), "window": window})
            if db.execute("SELECT 1 FROM historical_batch_owners WHERE owner_id=?", (ident,)).fetchone():
                raise BatchError("batch_state_conflict")
            self._save_historical_batch_owner(db, {"format": 1, "id": ident, "phase": "owned",
                "sequence": head["sequence"] + 1,
                "generation": outbox["consent_generation"], "admission_id": None, "members": members,
                "input_sha256": hashlib.sha256(self._stage_json(outbox["items"])).hexdigest(),
                "previous": None if previous is None else {"id": head["id"], "sha256": head["sha256"]}}, insert=True)

    def mark_historical_batch_dispatched(self, ident, admission_id):
        """Durable pre-POST fence; unsent/unknown ownership is never timed out."""
        if not _opaque(admission_id):
            raise BatchError("batch_admission_invalid")
        from muninn.history.remote_accounting import unknown_response
        outbox = BatchOutbox(self.archive).read(ident)
        if outbox["state"] != "submission_unknown":
            raise BatchError("batch_state_conflict")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _head, owner = self._historical_batch_head(db)
            if owner is None or owner["id"] != ident or owner["phase"] != "owned":
                raise BatchError("batch_state_conflict")
            if not unknown_response(self.policy_root, admission_id, owner["generation"], batch_owner=ident):
                raise BatchError("batch_admission_unresolved")
            if hashlib.sha256(self._stage_json(outbox["items"])).hexdigest() != owner["input_sha256"]:
                raise BatchError("batch_input_binding_invalid")
            for member in owner["members"]:
                row = self._historical_batch_member_row(db, owner, member)
                if row["state"] not in {"pending", "retry"} or row["cancel_requested"]:
                    raise BatchError("batch_member_ineligible")
                db.execute("UPDATE history_analysis_jobs SET remote_dispatched=1,updated_at=? WHERE job_id=?",
                           (time.time(), member["job_id"]))
            self._save_historical_batch_owner(db, {**owner, "phase": "sent", "admission_id": admission_id})

    def claim_historical_batch_result(self, ident, job_id):
        """Claim only local publication of a paid, settled, exact batch member."""
        from muninn.history.remote_accounting import settled_response
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._recover_analysis(db, time.time())
            _head, owner = self._historical_batch_head(db)
            if owner is None or owner["id"] != ident or owner["phase"] != "sent":
                return None
            member = next((i for i in owner["members"] if i["job_id"] == job_id), None)
            if member is None:
                return None
            row = self._historical_batch_member_row(db, owner, member)
            if (row["state"] not in {"pending", "retry", "outcome_unknown", "publication_pending"}
                    or row["cancel_requested"] or not settled_response(
                        self.policy_root, owner["admission_id"], owner["generation"], batch_owner=ident)):
                return None
            token, now = os.urandom(16).hex(), time.time()
            db.execute("UPDATE history_analysis_jobs SET state=CASE WHEN publication_started=1 "
                       "THEN 'publishing' ELSE 'running' END,attempt=attempt+1,lease_token=?,lease_until=?,"
                       "updated_at=? WHERE job_id=?", (token, now + 120.0, now, job_id))
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._analysis_row(row)

    def _historical_batch_expected(self, outbox):
        from muninn.history.historical_batch import billed_cost, terminal_results, validate_item
        if outbox["state"] not in {"terminal_saved", "cleaned"}:
            raise VaultIntegrityError("Historical batch terminal reply is missing")
        matched = terminal_results(outbox["items"], outbox["terminal"])
        billed_cost(outbox["terminal"])
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        source = CitedAnalysisSource(self.archive)
        return {item["job_id"]: validate_item(source, item, matched[item["custom_id"]])["extraction"]
                for item in outbox["items"]}

    def finish_historical_batch(self, ident):
        outbox = BatchOutbox(self.archive).read(ident)
        if outbox["state"] not in {"terminal_saved", "cleaned"}:
            return False
        expected = self._historical_batch_expected(outbox)
        # Expensive cross-store reference validation precedes the writer lock.
        self.verify_publications(job_ids=[item["job_id"] for item in outbox["items"]])
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _head, owner = self._historical_batch_head(db)
            if owner is None or owner["id"] != ident:
                return False
            if owner["phase"] == "passed":
                return True
            if owner["phase"] != "sent":
                return False
            if hashlib.sha256(self._stage_json(outbox["items"])).hexdigest() != owner["input_sha256"]:
                raise BatchError("batch_input_binding_invalid")
            for member in owner["members"]:
                row = self._historical_batch_member_row(db, owner, member)
                if row["state"] != "succeeded" or self._read_publication_receipt(row) is None:
                    return False
                if self._read_extraction(row) != {**expected[member["job_id"]], "admission_id": owner["admission_id"]}:
                    raise VaultIntegrityError("Historical batch publication differs from its reply")
            self._save_historical_batch_owner(db, {**owner, "phase": "passed"})
            return True

    def _verify_historical_batch_jobs(self, db):
        head, owner = self._historical_batch_head(db)
        seen = set()
        if owner is None:
            return 0
        if not all((self.archive.root / name).is_file() for name in (
                "historical-batches.db", "historical-batches-managed")):
            raise VaultIntegrityError("Historical batch outbox is missing")
        outbox = BatchOutbox(self.archive)
        while owner is not None:
            if owner["id"] in seen:
                raise VaultIntegrityError("Historical batch ownership chain is invalid")
            seen.add(owner["id"])
            stored = outbox.read(owner["id"])
            if (stored["consent_generation"] != owner["generation"]
                    or hashlib.sha256(self._stage_json(stored["items"])).hexdigest() != owner["input_sha256"]
                    or [(i["job_id"], i["window"]) for i in stored["items"]] != [
                        (i["job_id"], i["window"]) for i in owner["members"]]):
                raise VaultIntegrityError("Historical batch outbox binding differs")
            expected = self._historical_batch_expected(stored) if owner["phase"] == "passed" else None
            for member in owner["members"]:
                row = self._historical_batch_member_row(db, owner, member)
                stage = self._read_extraction(row)
                if stage is not None and stage.get("admission_id") != owner["admission_id"]:
                    raise VaultIntegrityError("Historical batch admission binding differs")
                if owner["phase"] == "passed" and (row["state"] != "succeeded"
                        or self._read_publication_receipt(row) is None):
                    raise VaultIntegrityError("Historical batch checkpoint lacks publication")
                if expected is not None and stage != {**expected[member["job_id"]],
                                                       "admission_id": owner["admission_id"]}:
                    raise VaultIntegrityError("Historical batch restored publication differs from its reply")
            previous = owner["previous"]
            sequence = owner["sequence"]
            owner = None if previous is None else self._historical_batch_record(db, previous["id"], previous["sha256"])
            if (owner is None and sequence != 1 or owner is not None
                    and (owner["phase"] != "passed" or owner["sequence"] != sequence - 1)):
                raise VaultIntegrityError("Historical batch predecessor is unresolved")
        if len(seen) != db.execute("SELECT COUNT(*) FROM historical_batch_owners").fetchone()[0]:
            raise VaultIntegrityError("Historical batch ownership chain is incomplete")
        return len(seen)
