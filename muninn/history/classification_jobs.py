"""Durable classification discovery/stages in the existing capture journal.

No inference, credentials, new service, or implicit admission authority here.
Late publication ACKs are discovered by identity, not a lossy row-number cursor.
"""
import hashlib
import hmac
import math
import time
import uuid

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.classification_enrollment import ClassificationEnrollmentMixin, related_cohorts
from muninn.history.memory_classification import (
    VERSION, PreparedClassification, _prepared, validate_classification,
)
from muninn.history.memory_ledger import MemoryLedger, _json

PURPOSE = "memory-classification-job-v1"
STATES = {"pending", "running", "staged", "published", "needs_user", "outcome_unknown", "failed"}


class CaptureClassificationMixin(ClassificationEnrollmentMixin):
    def _init_classifications(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS memory_classification_jobs("
            "job_id TEXT PRIMARY KEY, sealed BLOB NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS memory_classification_acks("
            "ack_id TEXT PRIMARY KEY, sealed BLOB NOT NULL)")
        self._init_classification_members(db)

    def _classification_job(self, row):
        if (row is None or not isinstance(row["job_id"], str) or len(row["job_id"]) != 32
                or any(c not in "0123456789abcdef" for c in row["job_id"])):
            raise VaultIntegrityError("Classification job identity is invalid")
        value = self._open_search(row["sealed"], row["job_id"], PURPOSE)
        try:
            if (not isinstance(value, dict) or set(value) != {"refs", "state", "lease", "lease_until",
                    "attempt", "prepared", "admission", "generation", "stage", "reason"}
                    or value["state"] not in STATES or value["state"] != row["state"]
                    or not isinstance(value["refs"], list) or not 1 <= len(value["refs"]) <= 12
                    or any(not MemoryLedger._hex(ref) for ref in value["refs"])
                    or len(set(value["refs"])) != len(value["refs"])
                    or type(value["attempt"]) is not int or value["attempt"] < 0
                    or type(value["generation"]) is not int or value["generation"] < -1
                    or value["reason"] not in {"", "missing_evidence", "input_changed", "reply_invalid", "dispatch_unknown"}):
                raise ValueError
            if value["prepared"] is not None:
                payload, bindings = _prepared(PreparedClassification(**value["prepared"]))
                if [binding["id"] for binding in bindings[:len(payload["candidates"])]] != value["refs"]:
                    raise ValueError
            if value["admission"] is not None:
                if (value["state"] == "pending" or value["prepared"] is None or value["generation"] < 1
                        or not isinstance(value["admission"], str) or len(value["admission"]) != 32
                        or any(c not in "0123456789abcdef" for c in value["admission"])):
                    raise ValueError
                from muninn.history.remote_accounting import classification_admission_state
                if classification_admission_state(self.policy_root, value["admission"], value["generation"],
                        row["job_id"], value["prepared"]["input_sha256"]) is None:
                    raise ValueError
            elif value["generation"] != -1 or value["state"] in {"staged", "published", "outcome_unknown"}:
                raise ValueError
            if value["stage"] is not None:
                stage = value["stage"]
                if (value["prepared"] is None or set(stage) != {"raw", "model_identity", "receipt"}
                        or not MemoryLedger._hex(stage["model_identity"])
                        or stage["receipt"]["admission_id"] != value["admission"]
                        or stage["receipt"]["policy_generation"] != value["generation"]):
                    raise ValueError
                validate_classification(PreparedClassification(**value["prepared"]), stage["raw"])
                identity = hashlib.sha256(_json({"purpose": VERSION,
                    "input": value["prepared"]["input_sha256"], "model": stage["receipt"]["model"]})).hexdigest()
                stage_id = hmac.new(self._key, b"classification-stage-v1\0" + _json({"job": row["job_id"],
                    "input": value["prepared"]["input_sha256"], "raw": stage["raw"], "model": identity,
                    "admission": value["admission"]}), hashlib.sha256).hexdigest()
                if identity != stage["model_identity"] or stage_id != stage["receipt"]["stage_id"]:
                    raise ValueError
                from muninn.history.remote_accounting import settled_response
                if not settled_response(self.policy_root, value["admission"], value["generation"], require_unowned=True,
                        classification_job=row["job_id"], classification_input=value["prepared"]["input_sha256"]):
                    raise ValueError
            if (value["state"] in {"staged", "published"} and value["stage"] is None
                    or value["stage"] is not None and value["state"] not in {"staged", "published", "needs_user"}):
                raise ValueError
            if value["state"] == "running":
                if (not isinstance(value["lease"], str) or len(value["lease"]) != 32
                        or any(c not in "0123456789abcdef" for c in value["lease"])
                        or type(value["lease_until"]) not in (int, float)
                        or not math.isfinite(value["lease_until"]) or not 0 < value["lease_until"]):
                    raise ValueError
            elif value["lease"] is not None or value["lease_until"] is not None:
                raise ValueError
            return value
        except (ValueError, TypeError, KeyError) as exc:
            raise VaultIntegrityError("Classification journal authentication failed") from exc

    def _save_classification(self, db, job_id, value):
        db.execute("UPDATE memory_classification_jobs SET sealed=?,state=? WHERE job_id=?",
            (self._seal_search(value, job_id, PURPOSE), value["state"], job_id))

    def discover_classifications(self, *, limit=16):
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid classification discovery bound")
        with self._connect() as db:
            rows = list(db.execute("SELECT a.* FROM history_analysis_jobs a WHERE a.sealed_receipt IS NOT NULL "
                "AND NOT EXISTS(SELECT 1 FROM memory_classification_acks c WHERE c.ack_id=a.job_id) "
                "ORDER BY a.created_at,a.job_id LIMIT ?", (limit,)))
        if not rows:
            return {"acks": 0, "jobs": 0}
        # Exact stage/ACK/source/ledger proofs, outside the journal writer.
        self.verify_publications(job_ids=[row["job_id"] for row in rows])
        ledger = MemoryLedger(self.archive, read_only=True)
        with ledger._connect() as db:
            db.execute("BEGIN")
            _report, candidates, states, placements, _revisions, humans, _stages = ledger._snapshot(db)
        queued = 0
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            jobs = {row["job_id"]: self._classification_job(row)
                    for row in db.execute("SELECT * FROM memory_classification_jobs")}
            members = self._classification_members(db, candidates, jobs)
            for old in rows:
                current = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (old["job_id"],)).fetchone()
                ack = self._read_publication_receipt(current)
                if ack != self._read_publication_receipt(old) or current["state"] != "succeeded":
                    raise VaultIntegrityError("Classification source ACK changed")
                refs = [ref for ref in ack["refs"] if candidates[ref]["credential_risk"] is False
                    and states[ref] == "provisional" and ref not in humans
                    and (ref not in placements or placements[ref]["status"] == "stale")]
                db.execute("INSERT OR IGNORE INTO memory_classification_acks VALUES(?,?)",
                    (old["job_id"], self._seal_search(ack, old["job_id"], "classification-source-ack-v1")))
                available = [ref for ref in refs if self._classification_member_id(ref) not in members]
                for cohort in related_cohorts([(ref, candidates[ref]["project_ref"]) for ref in available]):
                    job_id = self._classification_member_id(cohort[0])
                    value = {"refs": cohort, "state": "pending", "lease": None, "lease_until": None,
                        "attempt": 0, "prepared": None, "admission": None, "generation": -1, "stage": None, "reason": ""}
                    db.execute("INSERT INTO memory_classification_jobs VALUES(?,?,?,?)",
                        (job_id, self._seal_search(value, job_id, PURPOSE), "pending", old["created_at"]))
                    self._add_classification_members(db, job_id, cohort, old["job_id"], candidates)
                    members.update({self._classification_member_id(ref): (job_id, ref) for ref in cohort})
                    queued += 1
        return {"acks": len(rows), "jobs": queued}

    def claim_classification(self, *, now=None, include_pending=True):
        if type(include_pending) is not bool:
            raise ValueError("Invalid classification claim gate")
        now = time.time() if now is None else now
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM memory_classification_jobs WHERE state='running'").fetchall():
                value = self._classification_job(row)
                if value["lease_until"] <= now:
                    unsent = value["admission"] is None
                    if not unsent:
                        from muninn.history.remote_accounting import classification_admission_state
                        unsent = classification_admission_state(self.policy_root, value["admission"], value["generation"],
                            row["job_id"], value["prepared"]["input_sha256"]) == "released"
                    if unsent:
                        value.update(admission=None, generation=-1)
                    value.update(state="pending" if unsent else "outcome_unknown",
                                 lease=None, lease_until=None, reason="" if unsent else "dispatch_unknown")
                    self._save_classification(db, row["job_id"], value)
            row = db.execute("SELECT * FROM memory_classification_jobs WHERE state IN ('staged',?) "
                             "ORDER BY CASE state WHEN 'staged' THEN 0 ELSE 1 END,created_at,job_id LIMIT 1",
                             ("pending" if include_pending else "staged",)).fetchone()
            if row is None:
                return None
            value = self._classification_job(row)
            if value["state"] == "pending":
                value.update(state="running", lease=uuid.uuid4().hex, lease_until=now + 120, attempt=value["attempt"] + 1)
                self._save_classification(db, row["job_id"], value)
            return {"job_id": row["job_id"], **value}

    def prepare_classification_job(self, job_id, lease, prepared):
        from dataclasses import asdict
        _prepared(prepared)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if (value["state"] != "running" or value["lease"] != lease or value["lease_until"] <= time.time()
                    or value["admission"] is not None
                    or value["prepared"] is not None and value["prepared"] != asdict(prepared)
                    or [binding["id"] for binding in prepared.bindings()[:len(prepared.payload()["candidates"])]] != value["refs"]):
                raise ValueError("Classification preparation lease changed")
            value["prepared"] = asdict(prepared)
            self._save_classification(db, job_id, value)

    def classification_ready(self):
        owner = self.historical_batch_owner()
        if owner is not None and owner["phase"] != "passed":
            return False
        with self._connect() as db:
            return (not self._foreground_search_pending(db, time.time())
                    and db.execute("SELECT 1 FROM history_analysis_jobs WHERE lane=0 AND cancel_requested=0 "
                        "AND state IN ('pending','retry','running','publishing','publication_pending') "
                        "AND due_at<=? LIMIT 1", (time.time(),)).fetchone() is None)

    def heartbeat_classification(self, job_id, lease):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if value["state"] != "running" or value["lease"] != lease or value["lease_until"] <= time.time():
                return False
            value["lease_until"] = time.time() + 120
            self._save_classification(db, job_id, value)
        return True

    def defer_unsent_classification(self, job_id, lease):
        """Transport-proven unsent only; exact released bookkeeping survives a crash."""
        from muninn.history.remote_accounting import classification_admission_state
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if value["state"] != "running" or value["lease"] != lease:
                raise ValueError("Classification unsent lease changed")
            if value["admission"] is not None and classification_admission_state(self.policy_root,
                    value["admission"], value["generation"], job_id, value["prepared"]["input_sha256"]) != "released":
                raise ValueError("Classification dispatch is not proven unsent")
            value.update(state="pending", lease=None, lease_until=None, admission=None, generation=-1, reason="")
            self._save_classification(db, job_id, value)

    def mark_classification_dispatch(self, job_id, lease, admission, generation):
        if (not isinstance(admission, str) or len(admission) != 32 or any(c not in "0123456789abcdef" for c in admission)
                or type(generation) is not int or generation < 1):
            raise ValueError("Invalid classification admission")
        from muninn.history.remote_accounting import unowned_unknown_response
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for other in db.execute("SELECT * FROM memory_classification_jobs").fetchall():
                prior = self._classification_job(other)
                if prior["admission"] == admission:
                    raise ValueError("Classification admission already bound")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if (value["state"] != "running" or value["lease"] != lease or value["lease_until"] <= time.time()
                    or value["prepared"] is None or value["admission"] is not None):
                raise ValueError("Classification dispatch lease changed")
            if not unowned_unknown_response(self.policy_root, admission, generation,
                    classification_job=job_id, classification_input=value["prepared"]["input_sha256"]):
                raise ValueError("Classification admission is not a fresh unowned dispatch")
            value.update(admission=admission, generation=generation)
            self._save_classification(db, job_id, value)

    def stage_classification(self, job_id, lease, raw, *, model):
        from muninn.history.remote_accounting import settled_response
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if (value["state"] != "running" or value["lease"] != lease or value["admission"] is None
                    or value["lease_until"] <= time.time()):
                raise ValueError("Classification stage lease changed")
            plan = PreparedClassification(**value["prepared"])
            validate_classification(plan, raw)
            if not settled_response(self.policy_root, value["admission"], value["generation"], require_unowned=True,
                    classification_job=job_id, classification_input=plan.input_sha256):
                raise ValueError("Classification settlement missing")
            if not isinstance(model, str) or not 1 <= len(model) <= 128 or not MemoryLedger(self.archive, read_only=True)._screen({"model": model}):
                raise ValueError("Classification model invalid")
            identity = hashlib.sha256(_json({"purpose": VERSION, "input": plan.input_sha256, "model": model})).hexdigest()
            stage_id = hmac.new(self._key, b"classification-stage-v1\0" + _json({"job": job_id,
                "input": plan.input_sha256, "raw": raw, "model": identity, "admission": value["admission"]}), hashlib.sha256).hexdigest()
            value.update(state="staged", lease=None, lease_until=None, stage={"raw": raw, "model_identity": identity,
                "receipt": {"stage_id": stage_id, "admission_id": value["admission"], "policy_generation": value["generation"],
                    "provider": "openrouter", "model": model, "purpose": VERSION}})
            self._save_classification(db, job_id, value)
        return stage_id

    def publish_classification(self, job_id):
        with self._connect() as db:
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
        if value["state"] not in {"staged", "published"}:
            raise ValueError("Classification has no immutable stage")
        stage = value["stage"]
        refs = MemoryLedger(self.archive).commit_classification(PreparedClassification(**value["prepared"]),
            stage["raw"], model_identity=stage["model_identity"], receipt=stage["receipt"])
        if value["state"] == "published":
            return refs
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if current != value:
                raise ValueError("Classification publication changed")
            current["state"] = "published"
            self._save_classification(db, job_id, current)
        return refs

    def classification_status(self):
        with self._connect() as db:
            return {row[0]: row[1] for row in db.execute("SELECT state,COUNT(*) FROM memory_classification_jobs GROUP BY state")}

    def stop_classification(self, job_id, lease, *, reason):
        """A charged failure or stale stage is consultation, never fresh inference."""
        if reason not in {"missing_evidence", "input_changed", "reply_invalid"}:
            raise ValueError("Invalid classification stop reason")
        from muninn.history.remote_accounting import settled_response
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            value = self._classification_job(db.execute("SELECT * FROM memory_classification_jobs WHERE job_id=?", (job_id,)).fetchone())
            if value["state"] not in {"running", "staged"} or value["state"] == "running" and value["lease"] != lease:
                raise ValueError("Classification stop lease changed")
            settled = value["admission"] is not None and settled_response(
                self.policy_root, value["admission"], value["generation"], require_unowned=True,
                classification_job=job_id, classification_input=value["prepared"]["input_sha256"])
            from muninn.history.remote_accounting import classification_admission_state
            unsent = value["admission"] is None or classification_admission_state(self.policy_root,
                value["admission"], value["generation"], job_id, value["prepared"]["input_sha256"]) == "released"
            # Preserve the validated immutable stage for recovery/audit even
            # when evidence changed. Sent work without a bill remains unknown.
            value.update(state="needs_user" if unsent or settled else "outcome_unknown",
                         lease=None, lease_until=None, reason=reason)
            self._save_classification(db, job_id, value)

    def verify_classifications(self):
        """Cross-check encrypted work/ACKs and published placement receipts.

        Safe for a copied portable runtime after its accounting is restored.
        Does not enroll work, publish, or dispatch a provider request.
        """
        with self._connect() as db:
            rows = list(db.execute("SELECT * FROM memory_classification_jobs"))
            for row in db.execute("SELECT * FROM memory_classification_acks"):
                old = self._open_search(row["sealed"], row["ack_id"], "classification-source-ack-v1")
                current = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (row["ack_id"],)).fetchone()
                if current is None or self._read_publication_receipt(current) != old:
                    raise VaultIntegrityError("Classification enrollment ACK is missing")
        values = [(row["job_id"], self._classification_job(row)) for row in rows]
        if not (self.archive.root / "memory-ledger" / "ledger.sqlite3").is_file():
            if values:
                raise VaultIntegrityError("Classification candidate ledger is missing")
            return 0
        ledger = MemoryLedger(self.archive, read_only=True)
        with ledger._connect() as db:
            db.execute("BEGIN")
            _report, candidates, _states, _placements, _revisions, _humans, stages = ledger._snapshot(db)
        with self._connect() as db:
            self._classification_members(db, candidates, dict(values))
        admissions, claimed_stages = set(), set()
        for _job_id, value in values:
            if value["admission"] is not None:
                if value["admission"] in admissions:
                    raise VaultIntegrityError("Classification admission has multiple owners")
                admissions.add(value["admission"])
            if any(ref not in candidates for ref in value["refs"]):
                raise VaultIntegrityError("Classification candidate is missing")
            if value["stage"] is not None:
                claimed_stages.add(value["stage"]["receipt"]["stage_id"])
            if value["stage"] is not None:
                event = stages.get(value["stage"]["receipt"]["stage_id"])
                plan, stage = PreparedClassification(**value["prepared"]), value["stage"]
                expected = {"event": "classification", "version": VERSION, "input_sha256": plan.input_sha256,
                    "bindings": plan.bindings(), "items": validate_classification(plan, stage["raw"]),
                    "model_identity": stage["model_identity"], "receipt": stage["receipt"]}
                if event is not None and event != expected or value["state"] == "published" and event is None:
                    raise VaultIntegrityError("Classification durable publication is missing")
        if not set(stages) <= claimed_stages:
            raise VaultIntegrityError("Classification ledger stage has no journal owner")
        return len(values)
