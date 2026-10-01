"""Trusted outbox-to-window admission, distinct from term-targeted search jobs.

All identities, plan positions and completion counters are authenticated. Queue
insertion is not processing coverage; only the existing publication ACK advances
the completed counter. This module does not call providers or start workers.
"""
import hashlib
import hmac
import os
import re
import time

from muninn.history.credential_crypto import VaultIntegrityError

_ACTIVE = "('pending','running','retry','publishing','publication_pending')"
_TARGET_FIELDS = {"kind", "vault_id", "blob", "sha256", "version", "work_id",
                  "plan_attempt", "ordinal", "descriptor_sha256"}
_SCHEDULE_ID = "0" * 32


class CaptureWindowJobsMixin:
    def _init_capture_window_jobs(self, db):
        columns = {row[1] for row in db.execute("PRAGMA table_info(capture_enrichment_sources)")}
        if "sealed_plan" not in columns:
            db.execute("ALTER TABLE capture_enrichment_sources ADD COLUMN sealed_plan BLOB")
        for column, definition in (("planning_complete", "INTEGER NOT NULL DEFAULT 0"),
                                   ("resolved", "INTEGER NOT NULL DEFAULT 0"),
                                   ("last_planned_at", "REAL NOT NULL DEFAULT 0")):
            if column not in columns:
                db.execute(f"ALTER TABLE capture_enrichment_sources ADD COLUMN {column} {definition}")
        db.execute("CREATE INDEX IF NOT EXISTS capture_enrichment_due ON capture_enrichment_sources "
                   "(planning_complete,last_planned_at,created_at,work_id)")
        db.execute("CREATE INDEX IF NOT EXISTS capture_enrichment_pending ON capture_enrichment_sources "
                   "(resolved,created_at,work_id)")
        db.execute("CREATE TABLE IF NOT EXISTS capture_enrichment_windows ("
                   "work_id TEXT NOT NULL, ordinal INTEGER NOT NULL, job_id TEXT NOT NULL UNIQUE, "
                   "sealed_binding BLOB NOT NULL, PRIMARY KEY(work_id,ordinal))")
        existed = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                             "AND name='capture_enrichment_schedule'").fetchone()
        db.execute("CREATE TABLE IF NOT EXISTS capture_enrichment_schedule ("
                   "id INTEGER PRIMARY KEY CHECK(id=1), sealed_totals BLOB NOT NULL)")
        if db.execute("SELECT 1 FROM capture_enrichment_schedule WHERE id=1").fetchone() is None:
            if existed:
                raise VaultIntegrityError("Capture scheduling totals are missing")
            # One migration pass authenticates existing state, not mirror counts.
            for row in db.execute("SELECT * FROM capture_enrichment_sources"):
                self._read_enrichment_receipt(row[:2], self._enrichment_baseline(db))
                self._capture_plan_state(row)
            pending, planning = self._capture_schedule_counts(db)
            self._write_capture_schedule(db, {"format": 1, "pending": pending, "planning": planning})

    @staticmethod
    def _capture_schedule_counts(db):
        pending = db.execute("SELECT COUNT(*) FROM capture_enrichment_sources WHERE resolved=0").fetchone()[0]
        planning = db.execute("SELECT COUNT(*) FROM capture_enrichment_sources WHERE planning_complete=0").fetchone()[0]
        return pending, planning

    def _write_capture_schedule(self, db, value):
        db.execute("INSERT INTO capture_enrichment_schedule VALUES(1,?) "
                   "ON CONFLICT(id) DO UPDATE SET sealed_totals=excluded.sealed_totals", (
            self._seal_search(value, _SCHEDULE_ID, "capture-schedule-totals-v1"),))

    def _capture_schedule(self, db, *, check_mirrors=True):
        row = db.execute("SELECT sealed_totals FROM capture_enrichment_schedule WHERE id=1").fetchone()
        if row is None:
            raise VaultIntegrityError("Capture scheduling totals are missing")
        value = self._open_search(row[0], _SCHEDULE_ID, "capture-schedule-totals-v1")
        if (not isinstance(value, dict) or set(value) != {"format", "pending", "planning"}
                or any(type(item) is not int for item in value.values()) or value["format"] != 1
                or not 0 <= value["planning"] <= value["pending"]):
            raise VaultIntegrityError("Capture scheduling totals are invalid")
        if check_mirrors and self._capture_schedule_counts(db) != (value["pending"], value["planning"]):
            raise VaultIntegrityError("Capture scheduling hints differ from authenticated totals")
        return value

    def _adjust_capture_schedule(self, db, *, pending=0, planning=0):
        value = self._capture_schedule(db, check_mirrors=False)
        value["pending"] += pending
        value["planning"] += planning
        if not 0 <= value["planning"] <= value["pending"]:
            raise VaultIntegrityError("Capture scheduling totals exceed durable work")
        self._write_capture_schedule(db, value)

    def _capture_outbox_row(self, db, receipt):
        ident = self._enrichment_id(receipt)
        row = db.execute("SELECT * FROM capture_enrichment_sources "
                         "WHERE work_id=?", (ident,)).fetchone()
        baseline = self._enrichment_baseline(db)
        if row is None or self._read_enrichment_receipt(row[:2], baseline) != receipt:
            raise VaultIntegrityError("Capture window has no authenticated outbox source")
        return row

    def _capture_plan_state(self, row):
        if row["sealed_plan"] is None:
            if row["planning_complete"] != 0 or row["resolved"] != 0:
                raise VaultIntegrityError("Unplanned capture source has completion hints")
            return None
        value = self._open_search(row["sealed_plan"], row["work_id"], "capture-window-plan-v1")
        if (not isinstance(value, dict)
                or set(value) != {"format", "attempt", "count", "next_ordinal", "acknowledged"}
                or type(value["format"]) is not int or value["format"] != 1
                or not isinstance(value["attempt"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["attempt"])
                or any(type(value[k]) is not int for k in ("count", "next_ordinal", "acknowledged"))
                or not 0 <= value["acknowledged"] <= value["next_ordinal"] <= value["count"]):
            raise VaultIntegrityError("Capture window plan state is invalid")
        if (row["planning_complete"] != (value["next_ordinal"] == value["count"])
                or row["resolved"] != (value["acknowledged"] == value["count"])):
            raise VaultIntegrityError("Capture window scheduling hints differ from sealed state")
        return value

    def next_capture_plan(self):
        """Indexed round-robin hint; authenticate the selected source before use."""
        with self._connect() as db:
            self._capture_schedule(db)
            row = db.execute("SELECT * FROM capture_enrichment_sources WHERE planning_complete=0 "
                             "ORDER BY last_planned_at,created_at,work_id LIMIT 1").fetchone()
            if row is None:
                return None
            self._capture_plan_state(row)
            return self._read_enrichment_receipt(row[:2], self._enrichment_baseline(db))

    def _capture_plan_source(self, receipt):
        from muninn.history.cited_windows import CitedWindowPlanStore
        plans = CitedWindowPlanStore(self.archive)
        entry = plans.source.ledger._entries.get((receipt["blob"], receipt["version"]))
        if entry is None or self.archive._snapshot_receipt(entry, receipt["version"]) != receipt:
            raise VaultIntegrityError("Capture window source commit is unavailable")
        return plans, entry

    def queue_capture_windows(self, receipt, *, limit=4, should_cancel=lambda: False):
        """Internal trusted admission only; no API accepts lane or target options.

        Plan/source authentication and descriptor reopening happen outside the
        journal writer. Inside it, compare the sealed cursor and commit jobs,
        mapping and cursor advancement together. Saturation consumes no ordinal.
        """
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Invalid capture window batch limit")
        with self._connect() as db:
            before = self._capture_outbox_row(db, receipt)
            state = self._capture_plan_state(before)
        plans, entry = self._capture_plan_source(receipt)
        attempt = plans.build_snapshot(entry, receipt["version"], should_cancel=should_cancel)
        count = plans.count_pages(entry, receipt["version"], attempt)
        initial = {"format": 1, "attempt": attempt, "count": count,
                   "next_ordinal": 0, "acknowledged": 0}
        if state is None:
            state = initial
        elif state["attempt"] != attempt or state["count"] != count:
            raise VaultIntegrityError("Capture window plan became stale")
        with self._connect() as db:
            capacity = self._capture_window_capacity(db)
        descriptors = []
        for ordinal in range(state["next_ordinal"], min(count, state["next_ordinal"] + min(limit, capacity))):
            if should_cancel():
                from muninn.history.structured_projector import ProjectionCancelled
                raise ProjectionCancelled("Capture window scheduling cancelled")
            descriptor = plans.window_at(entry, receipt["version"], attempt, ordinal)
            descriptors.append((ordinal, hashlib.sha256(self._stage_json(descriptor)).hexdigest()))
        queued = 0
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._capture_schedule(db)
            row = self._capture_outbox_row(db, receipt)
            if row["sealed_plan"] != before["sealed_plan"]:
                return {"state": "concurrent_advance", "queued": 0}
            capacity = self._capture_window_capacity(db)
            for ordinal, digest in descriptors[:capacity]:
                target = {"kind": "capture_window", "vault_id": self.archive.vault_id,
                          "blob": receipt["blob"], "sha256": receipt["sha256"], "version": receipt["version"],
                          "work_id": row["work_id"], "plan_attempt": attempt,
                          "ordinal": ordinal, "descriptor_sha256": digest}
                dedup = hmac.new(self._key, b"capture-window-dedup-v1\0" + self._stage_json(target),
                                 hashlib.sha256).hexdigest()
                job_id = os.urandom(16).hex()
                db.execute("INSERT INTO history_analysis_jobs(job_id,vault_id,dedup_key,sealed_target,"
                           "state,created_at,updated_at,remote_policy_generation,lane) VALUES(?,?,?,?,?,?,?,?,1)",
                           (job_id, self.archive.vault_id, dedup,
                            self._seal_search(target, job_id, "analysis-target"), "pending", now, now, -1))
                db.execute("INSERT INTO capture_enrichment_windows VALUES(?,?,?,?)", (
                    row["work_id"], ordinal, job_id,
                    self._seal_search(target, job_id, "capture-window-binding-v1")))
                state["next_ordinal"] = ordinal + 1
                queued += 1
            db.execute("UPDATE capture_enrichment_sources SET sealed_plan=?,planning_complete=?,resolved=?,"
                       "last_planned_at=? WHERE work_id=?", (
                self._seal_search(state, row["work_id"], "capture-window-plan-v1"),
                state["next_ordinal"] == count, state["acknowledged"] == count,
                now, row["work_id"]))
            self._adjust_capture_schedule(db,
                planning=row["planning_complete"] - (state["next_ordinal"] == count),
                pending=row["resolved"] - (state["acknowledged"] == count))
        return {"state": "no_context" if not count else "planned" if state["next_ordinal"] == count
                else "queued" if queued else "queue_full", "queued": queued,
                "next_ordinal": state["next_ordinal"], "windows": count}

    @staticmethod
    def _capture_window_capacity(db):
        total, automatic = db.execute(f"SELECT COUNT(*),COALESCE(SUM(lane=1),0) "
                                      f"FROM history_analysis_jobs WHERE state IN {_ACTIVE}").fetchone()
        return max(0, min(32 - total, 24 - automatic))

    def _validated_analysis_target(self, row, db=None):
        target = self._open_search(row["sealed_target"], row["job_id"], "analysis-target")
        if row["lane"] == 0:
            checked = self._analysis_target(target, target.get("terms", []) if isinstance(target, dict) else [],
                                            self.archive.vault_id)
            if checked is None:
                raise VaultIntegrityError("Search analysis target format is invalid")
            return checked
        if (row["lane"] != 1 or not isinstance(target, dict) or set(target) != _TARGET_FIELDS
                or target["kind"] != "capture_window" or target["vault_id"] != self.archive.vault_id
                or row["vault_id"] != self.archive.vault_id
                or row["remote_policy_generation"] != -1 or row["remote_dispatched"] != 0
                or row["provider"] not in (None, "ollama")
                or any(not isinstance(target[k], str) or not re.fullmatch(r"[0-9a-f]{64}", target[k])
                       for k in ("work_id", "sha256", "descriptor_sha256"))
                or any(not isinstance(target[k], str) or not re.fullmatch(r"[0-9a-f]{32}", target[k])
                       for k in ("blob", "plan_attempt"))
                or type(target["version"]) is not int or target["version"] < 0
                or type(target["ordinal"]) is not int or target["ordinal"] < 0):
            raise VaultIntegrityError("Capture window lane or target is invalid")
        if db is None:
            with self._connect() as connection:
                self._validate_capture_authority(connection, row, target)
        else:
            self._validate_capture_authority(db, row, target)
        return target

    def _validate_capture_authority(self, db, job, target):
        mapping = db.execute("SELECT * FROM capture_enrichment_windows WHERE job_id=?", (job["job_id"],)).fetchone()
        row = db.execute("SELECT * FROM capture_enrichment_sources "
                         "WHERE work_id=?", (target["work_id"],)).fetchone()
        if (mapping is None or row is None or mapping["work_id"] != target["work_id"]
                or mapping["ordinal"] != target["ordinal"]
                or self._open_search(mapping["sealed_binding"], job["job_id"], "capture-window-binding-v1") != target):
            raise VaultIntegrityError("Capture window lacks trusted outbox binding")
        receipt = self._read_enrichment_receipt(row[:2], self._enrichment_baseline(db))
        state = self._capture_plan_state(row)
        if (any(target[k] != receipt[k] for k in ("vault_id", "blob", "sha256", "version"))
                or state is None or target["plan_attempt"] != state["attempt"]
                or not target["ordinal"] < state["next_ordinal"]):
            raise VaultIntegrityError("Capture window exceeds its authenticated plan")

    def _assert_capture_window(self, row, window):
        target = self._validated_analysis_target(row)
        if row["lane"] == 1 and hashlib.sha256(self._stage_json(window)).hexdigest() != target["descriptor_sha256"]:
            raise VaultIntegrityError("Capture window is not the admitted plan ordinal")

    def _ack_capture_window(self, db, row):
        target = self._validated_analysis_target(row, db)
        if row["lane"] != 1:
            return
        self._capture_schedule(db)
        source = db.execute("SELECT * FROM capture_enrichment_sources WHERE work_id=?", (target["work_id"],)).fetchone()
        state = self._capture_plan_state(source)
        state["acknowledged"] += 1
        if state["acknowledged"] > state["next_ordinal"]:
            raise VaultIntegrityError("Capture window completion exceeds scheduling")
        db.execute("UPDATE capture_enrichment_sources SET sealed_plan=?,resolved=? WHERE work_id=?", (
            self._seal_search(state, target["work_id"], "capture-window-plan-v1"),
            state["acknowledged"] == state["count"], target["work_id"]))
        self._adjust_capture_schedule(db,
            pending=source["resolved"] - (state["acknowledged"] == state["count"]))

    def capture_window_status(self, receipt):
        with self._connect() as db:
            row = self._capture_outbox_row(db, receipt)
            state = self._capture_plan_state(row)
            if state is None:
                return {"state": "pending"}
            counts = dict(db.execute("SELECT j.state,COUNT(*) FROM capture_enrichment_windows w "
                                     "JOIN history_analysis_jobs j ON w.job_id=j.job_id WHERE w.work_id=? GROUP BY j.state",
                                     (row["work_id"],)))
            if sum(counts.values()) != state["next_ordinal"] or counts.get("succeeded", 0) != state["acknowledged"]:
                raise VaultIntegrityError("Capture window coverage records are inconsistent")
        outcome = ("no_context" if not state["count"] else "completed" if state["acknowledged"] == state["count"]
                   else "failed" if any(counts.get(k) for k in ("failed", "cancelled", "outcome_unknown"))
                   else "deferred" if counts.get("retry") else "processing")
        return {"state": outcome, "windows": state["count"], "queued": state["next_ordinal"],
                "acknowledged": state["acknowledged"], "jobs": counts}

    def _verify_capture_window_jobs(self, db):
        self._capture_schedule(db)
        for row in db.execute("SELECT * FROM capture_enrichment_sources"):
            receipt = self._read_enrichment_receipt(row[:2], self._enrichment_baseline(db))
            state = self._capture_plan_state(row)
            mappings = db.execute("SELECT * FROM capture_enrichment_windows WHERE work_id=? ORDER BY ordinal",
                                  (row["work_id"],)).fetchall()
            if state is None:
                if mappings:
                    raise VaultIntegrityError("Capture window mapping has no sealed plan")
                continue
            plans, entry = self._capture_plan_source(receipt)
            if plans.count_pages(entry, receipt["version"], state["attempt"]) != state["count"]:
                raise VaultIntegrityError("Capture window count differs from sealed EOF")
            if len(mappings) != state["next_ordinal"]:
                raise VaultIntegrityError("Capture window scheduling is incomplete")
            acknowledged = 0
            for ordinal, mapping in enumerate(mappings):
                job = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (mapping["job_id"],)).fetchone()
                if job is None or mapping["ordinal"] != ordinal or job["lane"] != 1:
                    raise VaultIntegrityError("Capture window mapping identity is invalid")
                target = self._validated_analysis_target(job, db)
                descriptor = plans.window_at(entry, receipt["version"], state["attempt"], ordinal)
                if hashlib.sha256(self._stage_json(descriptor)).hexdigest() != target["descriptor_sha256"]:
                    raise VaultIntegrityError("Capture window plan membership is invalid")
                if job["state"] == "succeeded":
                    if self._read_publication_receipt(job) is None:
                        raise VaultIntegrityError("Capture window completion lacks publication ACK")
                    acknowledged += 1
            if acknowledged != state["acknowledged"]:
                raise VaultIntegrityError("Capture window completion counter is invalid")
        if db.execute("SELECT 1 FROM capture_enrichment_windows w LEFT JOIN capture_enrichment_sources s "
                      "ON w.work_id=s.work_id WHERE s.work_id IS NULL LIMIT 1").fetchone():
            raise VaultIntegrityError("Capture window mapping source is missing")
