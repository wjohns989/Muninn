"""Encrypted evidence placement replay, separate from immutable memory truth.

Only the journal publisher may supply a settled, authenticated stage receipt.
This lower-level ledger boundary does not dispatch, admit, or authorize models.
"""
from __future__ import annotations

import hashlib


class MemoryPlacementMixin:
    def _placement_bindings_match(self, bindings, candidates, states, revisions, humans):
        from muninn.history.memory_ledger import _json
        for binding in bindings:
            ref = binding["id"]
            candidate = candidates.get(ref)
            if (candidate is None or binding["expected_state"] != states[ref]
                    or binding["decision_seq"] != revisions[ref]
                    or binding["human_reviewed"] != (ref in humans)
                    or binding["candidate_sha256"] != hashlib.sha256(_json(candidate)).hexdigest()
                    or binding["citation_sha256"] != hashlib.sha256(_json(candidate["citation"])).hexdigest()
                    or binding["project_ref"] != candidate["project_ref"]):
                return False
        return True

    def _check_placement(self, payload, candidates, states, revisions, humans):
        from muninn.history.memory_classification import BUCKETS, REASONS, VERSION
        from muninn.history.memory_ledger import MemoryLedgerIntegrityError
        try:
            if (set(payload) != {"event", "version", "input_sha256", "bindings", "items", "model_identity", "receipt"}
                    or payload["event"] != "classification" or payload["version"] != VERSION
                    or not self._hex(payload["input_sha256"]) or not self._hex(payload["model_identity"])):
                raise ValueError
            bindings, items, receipt = payload["bindings"], payload["items"], payload["receipt"]
            if (not isinstance(bindings, list) or not 1 <= len(bindings) <= 24
                    or not isinstance(items, list) or not 1 <= len(items) <= 12
                    or len(bindings) < len(items) or not isinstance(receipt, dict)
                    or set(receipt) != {"stage_id", "admission_id", "policy_generation", "provider", "model", "purpose"}
                    or not self._hex(receipt["stage_id"])
                    or not isinstance(receipt["admission_id"], str) or len(receipt["admission_id"]) != 32
                    or any(c not in "0123456789abcdef" for c in receipt["admission_id"])
                    or type(receipt["policy_generation"]) is not int or receipt["policy_generation"] < 0
                    or receipt["provider"] != "openrouter" or receipt["purpose"] != VERSION
                    or not isinstance(receipt["model"], str) or not 1 <= len(receipt["model"]) <= 128
                    or not self._screen({"model": receipt["model"]})):
                raise ValueError
            refs = set()
            for index, binding in enumerate(bindings):
                if (not isinstance(binding, dict) or set(binding) != {"id", "model_slot", "candidate_sha256",
                        "citation_sha256", "decision_seq", "expected_state", "human_reviewed", "project_ref"}
                        or not self._hex(binding["id"]) or binding["id"] in refs
                        or binding["model_slot"] != f"m{index}"
                        or type(binding["decision_seq"]) is not int or binding["decision_seq"] < 1
                        or binding["expected_state"] not in {"provisional", "filed", "needs_user", "rejected"}
                        or type(binding["human_reviewed"]) is not bool
                        or any(not self._hex(binding[k]) for k in ("candidate_sha256", "citation_sha256", "project_ref"))):
                    raise ValueError
                refs.add(binding["id"])
            if not self._placement_bindings_match(bindings, candidates, states, revisions, humans):
                raise ValueError
            project = bindings[0]["project_ref"]
            for index, item in enumerate(items):
                ref = bindings[index]["id"]
                candidate = candidates[ref]
                if (not isinstance(item, dict) or set(item) != {"id", "bucket", "disposition", "evidence_refs",
                        "reason", "truth_status", "comparison_scope", "global_conflict_coverage"}
                        or item["id"] != ref or item["bucket"] not in BUCKETS
                        or item["disposition"] not in {"accepted", "needs_user", "conflict"}
                        or item["reason"] not in REASONS or states[ref] != "provisional" or ref in humans
                        or item["truth_status"] != candidate["truth_status"]
                        or item["comparison_scope"] != "selected_peers_only" or item["global_conflict_coverage"] is not False
                        or not isinstance(item["evidence_refs"], list) or len(item["evidence_refs"]) > 24
                        or len(set(item["evidence_refs"])) != len(item["evidence_refs"])
                        or any(e not in refs for e in item["evidence_refs"])):
                    raise ValueError
                evidence = item["evidence_refs"]
                if item["disposition"] == "accepted":
                    if (ref not in evidence or item["reason"] != "source_supported"
                            or item["bucket"] in {"conflict", "duplicate"}
                            or any(states[e] == "rejected" for e in evidence)):
                        raise ValueError
                elif item["disposition"] == "conflict":
                    if ref not in evidence or len(evidence) < 2 or item["reason"] != "conflicting_evidence":
                        raise ValueError
                elif item["reason"] == "source_supported":
                    raise ValueError
            for binding in bindings:
                candidate = candidates[binding["id"]]
                if (candidate.get("credential_risk") is not False
                        or candidate["screening"] not in {"complete_unit", "original_ranges"}
                        or candidate["project_ref"] != project or candidate["project_basis"] == "unknown"
                        or candidate["event_at"] is None or candidate["time_basis"] != "provider_record"):
                    raise ValueError
        except (ValueError, TypeError, KeyError) as exc:
            raise MemoryLedgerIntegrityError("Memory placement authentication failed") from exc

    @staticmethod
    def _invalidate_placements(placements, revisions, seq):
        # A human or another classification can change a peer which invalidates
        # a dependent placement; that invalidation is itself a new revision.
        changed = True
        while changed:
            changed = False
            for ref, placement in placements.items():
                if (placement["status"] == "current" and any(revisions[e] != revision
                        for e, revision in placement["dependencies"].items())):
                    placement["status"] = "stale"
                    revisions[ref] = seq
                    changed = True

    def _snapshot(self, db):
        from muninn.history.memory_ledger import MemoryLedgerIntegrityError
        report = {"events": 0, "candidates": 0, "decisions": 0}
        candidates, states, placements, revisions, humans, stages = {}, {}, {}, {}, set(), {}
        for seq, (ref, payload) in enumerate(self._walk(db), 1):
            report["events"] += 1
            kind = payload.get("event")
            if kind == "candidate" and ref not in candidates:
                self._check_candidate(payload)
                candidates[ref], states[ref], revisions[ref] = payload, payload["state"], seq
                report["candidates"] += 1
            elif kind in {"decision", "human_review"} and ref in candidates:
                states[ref] = self._apply_review_event(ref, candidates[ref], states[ref], payload)
                revisions[ref] = seq
                if kind == "human_review":
                    humans.add(ref)
                report["decisions"] += 1
                self._invalidate_placements(placements, revisions, seq)
            elif kind == "classification" and ref in candidates:
                self._check_placement(payload, candidates, states, revisions, humans)
                if ref != payload["items"][0]["id"] or payload["receipt"]["stage_id"] in stages:
                    raise MemoryLedgerIntegrityError("Duplicate memory placement stage")
                stages[payload["receipt"]["stage_id"]] = payload
                targets = {item["id"] for item in payload["items"]}
                for target in targets:
                    revisions[target] = seq
                self._invalidate_placements(placements, revisions, seq)
                # This event revises its targets. Non-target peer evidence must
                # retain the revision actually observed, even if invalidating
                # an older dependency changed that peer during this replay.
                dependencies = {binding["id"]: seq if binding["id"] in targets else binding["decision_seq"]
                                for binding in payload["bindings"]}
                for item in payload["items"]:
                    placements[item["id"]] = {"decision": item, "status": "current",
                        "dependencies": dict(dependencies), "input_sha256": payload["input_sha256"],
                        "model_identity": payload["model_identity"]}
                self._invalidate_placements(placements, revisions, seq)
                report["decisions"] += 1
            else:
                raise MemoryLedgerIntegrityError("Invalid memory event sequence")
        return report, candidates, states, placements, revisions, humans, stages

    @staticmethod
    def _public_placement(placement):
        if placement is None:
            return None
        decision = placement["decision"]
        return {"bucket": decision["bucket"],
            "status": decision["disposition"] if placement["status"] == "current" else "stale",
            "reason": decision["reason"] if placement["status"] == "current" else "evidence_revision_changed",
            "evidence_refs": list(decision["evidence_refs"]), "actor": "model",
            "comparison_scope": "selected_peers_only", "global_conflict_coverage": False}

    def _read_view(self, ident):
        if not self._hex(ident):
            raise ValueError("Invalid memory reference")
        with self._connect() as db:
            db.execute("BEGIN")
            _report, candidates, states, placements, *_rest = self._snapshot(db)
        return candidates.get(ident), states.get(ident), placements.get(ident)

    def commit_classification(self, prepared, raw, *, model_identity, receipt):
        """Publisher-only append after authenticated journal-stage settlement.

        This primitive is NOT a service endpoint or model authorization. The
        caller must verify the immutable stage and its admission before calling.
        Replay, source privacy, exact-stage recovery and revision CAS are local
        ledger properties enforced here even for an already validated caller.
        """
        from muninn.history.memory_classification import (
            ClassificationError, VERSION, _prepared, revalidate_classification, validate_classification)
        from muninn.history.memory_ledger import MemoryLedger, _json
        if self.read_only:
            raise PermissionError("Memory placement requires a writer")
        _payload, bindings = _prepared(prepared)
        items = validate_classification(prepared, raw)
        event = {"event": "classification", "version": VERSION, "input_sha256": prepared.input_sha256,
                 "bindings": bindings, "items": items, "model_identity": model_identity, "receipt": receipt}
        # Recover exact previous commit before any current revision check. A
        # subsequent human review must neither resurrect nor force redispatch.
        with self._connect() as db:
            db.execute("BEGIN")
            *_first, stages = self._snapshot(db)
        previous = stages.get(receipt.get("stage_id")) if isinstance(receipt, dict) else None
        if previous is not None:
            if _json(previous) != _json(event):
                raise ClassificationError("classification_stage_conflict")
            return [item["id"] for item in items]
        revalidate_classification(MemoryLedger(self.archive, read_only=True), prepared)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _report, candidates, states, _placements, revisions, humans, stages = self._snapshot(db)
            previous = stages.get(receipt.get("stage_id")) if isinstance(receipt, dict) else None
            if previous is not None:
                if _json(previous) != _json(event):
                    raise ClassificationError("classification_stage_conflict")
                return [item["id"] for item in items]
            if not self._placement_bindings_match(bindings, candidates, states, revisions, humans):
                raise ClassificationError("classification_input_changed")
            self._check_placement(event, candidates, states, revisions, humans)
            head = self._head(db)
            seq, ref = head["seq"] + 1, items[0]["id"]
            sealed = self._seal({"previous": head["digest"], "payload": event}, "event", seq, ref)
            db.execute("INSERT INTO events VALUES(?,?,?)", (seq, ref, sealed))
            self._set_head(db, seq, self._digest(seq, ref, sealed))
        return [item["id"] for item in items]
