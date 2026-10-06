"""Authenticated whitespace completion; never inference, reuse or publication."""
import hashlib
import hmac
import re
import time

from muninn.history.credential_crypto import VaultIntegrityError


class CaptureNoContextMixin:
    def _empty_plan_source(self, row, window):
        from muninn.history.cited_windows import CitedWindowPlanStore
        from muninn.history.secure_projection_store import ProjectionIntegrityError
        target = self._validated_analysis_target(row)
        with self._connect() as db:
            source = db.execute("SELECT * FROM capture_enrichment_sources WHERE work_id=?",
                                (target["work_id"],)).fetchone()
            receipt = self._read_enrichment_receipt(source[:2], self._enrichment_baseline(db), db=db)
            state = self._capture_plan_state(source)
        try:
            plans = CitedWindowPlanStore(self.archive, read_only=True)
            entry = plans.source.ledger._entries[(target["blob"], target["version"])]
            if (self.archive._snapshot_receipt(entry, target["version"]) != receipt
                    or plans.count_pages(entry, target["version"], target["plan_attempt"]) != state["count"]
                    or plans.window_at(entry, target["version"], target["plan_attempt"],
                                       target["ordinal"]) != window):
                raise VaultIntegrityError("Empty coverage plan binding changed")
            return plans.source
        except (ProjectionIntegrityError, KeyError) as exc:
            raise VaultIntegrityError("Empty coverage plan is unavailable") from exc

    def _no_context_purpose(self, row):
        return "capture-no-context-v1:" + hashlib.sha256(
            self._stage_json(self._validated_analysis_target(row))).hexdigest()

    @staticmethod
    def _empty_immutable(row):
        return (row["lane"] == 1 and not row["remote_dispatched"]
                and not row["publication_started"] and not row["cancel_requested"]
                and row["provider"] is None and row["model"] is None
                and row["result_expires_at"] is None
                and all(row[k] is None for k in (
                    "sealed_extraction", "extraction_id", "sealed_receipt", "sealed_reuse")))

    def _read_capture_no_context(self, row, *, source=None):
        if row["state"] != "no_context":
            return None
        if (not self._empty_immutable(row) or row["sealed_result"] is None
                or row["lease_token"] is not None or row["lease_until"] is not None):
            raise VaultIntegrityError("Empty coverage state is invalid")
        target = self._validated_analysis_target(row)
        window = self._read_analysis_window(row)
        proof = self._open_search(row["sealed_result"], row["job_id"], self._no_context_purpose(row))
        if (not isinstance(proof, dict) or set(proof) != {"format", "basis", "target", "window"}
                or type(proof["format"]) is not int or proof["format"] != 1
                or proof["basis"] != "authenticated_whitespace"
                or proof["target"] != target or window is None or proof["window"] != window):
            raise VaultIntegrityError("Empty coverage binding is invalid")
        reader = self._empty_plan_source(row, window)
        if reader.reopen(window)["text"].strip():
            raise VaultIntegrityError("Empty coverage has substantive source text")
        return proof

    def acknowledge_capture_no_context(self, job_id, *, lease_token=None,
                                       expected_attempt=None, expected_target_sha256=None):
        """Finish only original whitespace; atomically retain its proof and ACK.

        Running claims use their current lease. Legacy failed/insufficient_context
        rows need an exact attempt and encrypted-target preimage. No paid item,
        sent request, retained result or masked projection can enter this path.
        """
        if (not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None
                or lease_token is not None and (not isinstance(lease_token, str)
                    or re.fullmatch(r"[0-9a-f]{32}", lease_token) is None)
                or expected_attempt is not None and (type(expected_attempt) is not int or expected_attempt < 0)
                or expected_target_sha256 is not None and (not isinstance(expected_target_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected_target_sha256) is None)):
            raise ValueError("Invalid empty coverage request")
        before = self._publication_row(job_id)
        if before is None:
            return False
        if before["state"] == "no_context":
            self._read_capture_no_context(before)
            return False
        def eligible(row):
            return (self._empty_immutable(row) and row["sealed_result"] is None
                and (lease_token is not None and row["state"] == "running"
                     and row["lease_token"] == lease_token and row["lease_until"] is not None
                     and row["lease_until"] > time.time()
                     or lease_token is None and row["state"] == "failed"
                     and row["error_code"] == "insufficient_context"
                     and expected_attempt is not None and row["attempt"] == expected_attempt
                     and expected_target_sha256 is not None
                     and hmac.compare_digest(hashlib.sha256(row["sealed_target"]).hexdigest(),
                                             expected_target_sha256)
                     and row["lease_token"] is None and row["lease_until"] is None))
        if not eligible(before):
            return False
        target = self._validated_analysis_target(before)
        window = self._read_analysis_window(before)
        if window is None:
            return False
        if self._empty_plan_source(before, window).reopen(window)["text"].strip():
            return False
        proof = {"format": 1, "basis": "authenticated_whitespace", "target": target, "window": window}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or not eligible(row) or job_id in self._historical_batch_blocked_jobs(db):
                return False
            if any(row[k] != before[k] for k in ("attempt", "lane", "sealed_target", "sealed_window")):
                return False
            self._validated_analysis_target(row, db)
            sealed = self._seal_search(proof, job_id, self._no_context_purpose(row))
            self._ack_capture_window(db, row)
            db.execute("UPDATE history_analysis_jobs SET state='no_context',sealed_result=?,"
                       "error_code='',lease_token=NULL,lease_until=NULL,updated_at=? WHERE job_id=?",
                       (sealed, time.time(), job_id))
        return True

    def reconcile_capture_no_context(self, *, limit=8):
        """Bounded CPU-only repair of old empty failures, never model readmission."""
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid empty reconciliation bound")
        with self._connect() as db:
            rows = list(db.execute("SELECT job_id,attempt,sealed_target FROM history_analysis_jobs "
                "WHERE lane=1 AND state='failed' AND error_code='insufficient_context' "
                "AND remote_dispatched=0 AND publication_started=0 AND cancel_requested=0 "
                "ORDER BY created_at,job_id LIMIT ?", (limit,)))
        return sum(self.acknowledge_capture_no_context(row["job_id"], expected_attempt=row["attempt"],
            expected_target_sha256=hashlib.sha256(row["sealed_target"]).hexdigest()) for row in rows)
