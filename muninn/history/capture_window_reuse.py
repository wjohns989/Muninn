"""Private direct-parent coverage reuse; never a new model/publication result."""
import hashlib
import re
import time

from muninn.history.credential_crypto import VaultIntegrityError


class CaptureWindowReuseMixin:
    def _capture_reuse_purpose(self, row):
        target = self._validated_analysis_target(row)
        return "capture-reuse-v1:" + hashlib.sha256(self._stage_json(target)).hexdigest()

    def _capture_reuse_parent(self, row):
        target = self._validated_analysis_target(row)
        if row["lane"] != 1:
            return None
        from muninn.history.cited_windows import CitedWindowPlanStore
        plans = CitedWindowPlanStore(self.archive)
        entry = plans.source.ledger._entries.get((target["blob"], target["version"]))
        if entry is None:
            raise VaultIntegrityError("Reuse source is unavailable")
        parent_window = plans.preserved_parent_window(entry, target["version"],
                                                      target["plan_attempt"], target["ordinal"])
        if parent_window is None:
            return None
        parent_entry = plans.source.ledger._entries[(parent_window["blob"], parent_window["version"])]
        receipt = self.archive._snapshot_receipt(parent_entry, parent_window["version"])
        work_id = self._enrichment_id(receipt)
        with self._connect() as db:
            db.execute("BEGIN")
            parent = db.execute("SELECT j.* FROM capture_enrichment_windows w "
                                "JOIN history_analysis_jobs j ON j.job_id=w.job_id "
                                "WHERE w.work_id=? AND w.ordinal=?", (work_id, target["ordinal"])).fetchone()
            if parent is None:
                return None
            parent_target = self._validated_analysis_target(parent, db)
        return plans, parent_window, parent, parent_target

    def _capture_reuse_material(self, row):
        found = self._capture_reuse_parent(row)
        if found is None:
            return None
        plans, parent_window, parent, parent_target = found
        # Direct originals only. Reused parents, cloud calls, legacy result-only
        # completions and interrupted publication are not original analysis ACKs.
        if (parent["state"] != "succeeded" or parent["lane"] != 1
                or parent["sealed_reuse"] is not None or parent["remote_dispatched"]
                or parent["provider"] != "ollama"):
            return None
        stage = self._read_extraction(parent)
        ack = self._read_publication_receipt(parent)
        if stage is None or ack is None:
            return None
        if (stage["window"] != parent_window or self._read_analysis_window(parent) != parent_window
                or stage["result"]["provider"] != "ollama"
                or parent["model"] != stage["result"]["model"]):
            raise VaultIntegrityError("Reuse parent analysis binding changed")
        return plans, parent_window, parent, parent_target, stage, ack

    def acknowledge_capture_reuse(self, job_id, lease_token, *, model, weights_digest,
                                  request_options=None, identity_guard=None):
        """Trusted worker admission; caller must read fresh installed weights.

        No public API accepts a digest or reuse authority. Source/ledger proof
        happens without a journal writer; final admission rechecks both rows.
        This component does not call a provider, start a cadence, or infer reuse
        across chains. A miss remains ordinary unfinished work.
        """
        if (not isinstance(model, str) or not 1 <= len(model) <= 128
                or not isinstance(weights_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", weights_digest) is None):
            raise ValueError("Invalid local reuse identity")
        if identity_guard is not None and not callable(identity_guard):
            raise ValueError("Invalid reuse identity guard")
        before = self._publication_row(job_id)
        if (before is None or before["lane"] != 1 or before["state"] != "running"
                or before["lease_token"] != lease_token or before["lease_until"] <= time.time()
                or before["cancel_requested"] or before["publication_started"]
                or before["sealed_extraction"] is not None or before["sealed_reuse"] is not None):
            return False
        window = self._read_analysis_window(before)
        if window is None:
            return False
        found = self._capture_reuse_material(before)
        if found is None:
            return False
        plans, parent_window, parent, parent_target, stage, ack = found
        if stage["result"]["model"] != model:
            return False
        from muninn.history.secure_analysis import _cited_model_identity
        if _cited_model_identity(plans.source.reopen(window), "ollama", model,
                                 weights_digest, request_options=request_options) != stage["model_identity"]:
            return False
        self._verify_reuse_refs(parent_window, stage, ack)
        if identity_guard is not None and not identity_guard():
            return False  # Fresh installed-weight check, outside the writer.
        target = self._validated_analysis_target(before)
        proof = {"format": 1, "target": target, "window": window,
                 "parent_job": parent["job_id"], "parent_target": parent_target,
                 "parent_window": parent_window, "parent_extraction_id": parent["extraction_id"],
                 "model_identity": stage["model_identity"], "weights_digest": weights_digest,
                 "refs": ack["refs"]}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
            original = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?",
                                  (parent["job_id"],)).fetchone()
            if (row is None or row["state"] != "running" or row["lease_token"] != lease_token
                    or row["lease_until"] <= time.time() or row["cancel_requested"]
                    or row["publication_started"] or row["sealed_extraction"] is not None
                    or row["sealed_reuse"] is not None):
                return False
            # Legitimate heartbeats may change timing but not these bindings.
            if any(row[k] != before[k] for k in ("lane", "sealed_target", "sealed_window")):
                raise VaultIntegrityError("Reuse current binding changed")
            if (original is None or original["state"] != "succeeded"
                    or any(original[k] != parent[k] for k in (
                        "lane", "sealed_target", "sealed_window", "sealed_extraction", "extraction_id",
                        "sealed_receipt", "sealed_reuse", "provider", "model", "remote_dispatched"))):
                raise VaultIntegrityError("Reuse original ACK changed")
            self._validated_analysis_target(original, db)
            self._ack_capture_window(db, row)
            db.execute("UPDATE history_analysis_jobs SET state='reused',sealed_reuse=?,provider='ollama',model=?,"
                       "lease_token=NULL,lease_until=NULL,error_code='',updated_at=? WHERE job_id=?", (
                self._seal_search(proof, job_id, self._capture_reuse_purpose(row)), model, time.time(), job_id))
        return True

    def _verify_reuse_refs(self, window, stage, ack):
        from muninn.history.cited_analysis_source import CitedAnalysisSource
        source = CitedAnalysisSource(self.archive)
        expected = source.expected_refs(window, stage["proposals"], model_identity=stage["model_identity"])
        if expected != ack["refs"] or not source.ledger.verify_refs(ack["refs"]):
            raise VaultIntegrityError("Reuse has no matching durable original memories")

    def _capture_reuse_state(self, row):
        if row["state"] != "reused":
            if row["sealed_reuse"] is not None:
                raise VaultIntegrityError("Reuse receipt has inconsistent job state")
            return False
        if (row["sealed_reuse"] is None or row["lane"] != 1
                or row["sealed_extraction"] is not None or row["sealed_receipt"] is not None
                or row["publication_started"] or row["remote_dispatched"]):
            raise VaultIntegrityError("Reused coverage has no valid receipt")
        return True

    def _read_capture_reuse(self, row):
        if not self._capture_reuse_state(row):
            return None
        proof = self._open_search(row["sealed_reuse"], row["job_id"], self._capture_reuse_purpose(row))
        fields = {"format", "target", "window", "parent_job", "parent_target", "parent_window",
                  "parent_extraction_id", "model_identity", "weights_digest", "refs"}
        if (not isinstance(proof, dict) or set(proof) != fields
                or type(proof["format"]) is not int or proof["format"] != 1
                or not isinstance(proof["weights_digest"], str)
                or re.fullmatch(r"[0-9a-f]{64}", proof["weights_digest"]) is None
                or proof["target"] != self._validated_analysis_target(row)
                or proof["window"] != self._read_analysis_window(row)):
            raise VaultIntegrityError("Reuse receipt binding is invalid")
        found = self._capture_reuse_material(row)
        if found is None:
            raise VaultIntegrityError("Reuse original analysis is unavailable")
        plans, parent_window, parent, parent_target, stage, ack = found
        if (proof["parent_job"] != parent["job_id"] or proof["parent_target"] != parent_target
                or proof["parent_window"] != parent_window
                or proof["parent_extraction_id"] != parent["extraction_id"]
                or proof["model_identity"] != stage["model_identity"] or proof["refs"] != ack["refs"]
                or row["model"] != stage["result"]["model"] or row["provider"] != "ollama"):
            raise VaultIntegrityError("Reuse original publication binding is invalid")
        # Historical coverage records the admitted contract. Do not compare it
        # with today's installed model/options or dispatch a model during restore.
        self._verify_reuse_refs(parent_window, stage, ack)
        return proof
