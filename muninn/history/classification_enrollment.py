"""Authenticated candidate ownership for related classification cohorts."""
import hashlib
import hmac

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.memory_ledger import MemoryLedger

MEMBER_PURPOSE = "classification-member-v1"


def related_cohorts(rows):
    """Preserve ACK order, never mix projects or group unknown attribution."""
    groups = []
    for ref, project in rows:
        if not groups or project is None or groups[-1][0] != project or len(groups[-1][1]) == 12:
            groups.append((project, []))
        groups[-1][1].append(ref)
    return [refs for _project, refs in groups]


class ClassificationEnrollmentMixin:
    def _classification_member_id(self, ref):
        return hmac.new(self._key, b"classification-candidate-v1\0" + ref.encode(), hashlib.sha256).hexdigest()[:32]

    def _classification_ack(self, db, ack_id):
        stored = db.execute("SELECT sealed FROM memory_classification_acks WHERE ack_id=?", (ack_id,)).fetchone()
        original = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (ack_id,)).fetchone()
        if stored is None or original is None:
            raise VaultIntegrityError("Classification membership ACK is missing")
        ack = self._open_search(stored[0], ack_id, "classification-source-ack-v1")
        if ack != self._read_publication_receipt(original) or original["state"] != "succeeded":
            raise VaultIntegrityError("Classification membership ACK changed")
        return ack

    def _add_classification_members(self, db, owner, refs, ack_id, candidates):
        ack = self._classification_ack(db, ack_id)
        if not set(refs) <= set(ack["refs"]):
            raise VaultIntegrityError("Classification membership source differs")
        for ref in refs:
            member = self._classification_member_id(ref)
            value = {"ref": ref, "ack_id": ack_id, "project_ref": candidates[ref]["project_ref"]}
            db.execute("INSERT INTO memory_classification_members VALUES(?,?,?)",
                (member, owner, self._seal_search(value, member, MEMBER_PURPOSE)))

    def _classification_members(self, db, candidates, jobs):
        expected = {}
        for owner, job in jobs.items():
            projects = set()
            for ref in job["refs"]:
                member = self._classification_member_id(ref)
                if member in expected or ref not in candidates:
                    raise VaultIntegrityError("Classification membership has multiple owners")
                expected[member] = (owner, ref)
                projects.add(candidates[ref]["project_ref"])
            if len(projects) != 1 or None in projects and len(job["refs"]) != 1:
                raise VaultIntegrityError("Classification membership crosses projects")
        actual, source_acks = {}, {}
        ack_cache = {}
        for row in db.execute("SELECT * FROM memory_classification_members"):
            value = self._open_search(row["sealed"], row["member_id"], MEMBER_PURPOSE)
            if (not isinstance(value, dict) or set(value) != {"ref", "ack_id", "project_ref"}
                    or not MemoryLedger._hex(value["ref"])
                    or not isinstance(value["ack_id"], str) or len(value["ack_id"]) != 32
                    or any(c not in "0123456789abcdef" for c in value["ack_id"])
                    or expected.get(row["member_id"]) != (row["owner"], value["ref"])
                    or candidates[value["ref"]]["project_ref"] != value["project_ref"]):
                raise VaultIntegrityError("Classification membership binding differs")
            ack_id = value["ack_id"]
            if ack_id not in ack_cache:
                ack_cache[ack_id] = self._classification_ack(db, ack_id)
            if value["ref"] not in ack_cache[ack_id]["refs"]:
                raise VaultIntegrityError("Classification membership source differs")
            source_acks.setdefault(row["owner"], set()).add(ack_id)
            actual[row["member_id"]] = (row["owner"], value["ref"])
        if actual != expected or any(len(ids) != 1 for ids in source_acks.values()):
            raise VaultIntegrityError("Classification membership is incomplete")
        return actual

    def _init_classification_members(self, db):
        legacy = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_classification_members'").fetchone() is None
        db.execute("CREATE TABLE IF NOT EXISTS memory_classification_members("
            "member_id TEXT PRIMARY KEY, owner TEXT NOT NULL, sealed BLOB NOT NULL)")
        jobs = {row["job_id"]: self._classification_job(row)
                for row in db.execute("SELECT * FROM memory_classification_jobs")}
        if not jobs:
            if db.execute("SELECT 1 FROM memory_classification_members LIMIT 1").fetchone():
                raise VaultIntegrityError("Classification membership has no owner")
            return
        ledger = MemoryLedger(self.archive, read_only=True)
        with ledger._connect() as source_db:
            _report, candidates, *_rest = ledger._snapshot(source_db)
        if legacy:
            acks = {row["ack_id"]: self._classification_ack(db, row["ack_id"])
                    for row in db.execute("SELECT ack_id FROM memory_classification_acks")}
            for owner, job in jobs.items():
                ack_id = next((key for key, ack in acks.items() if set(job["refs"]) <= set(ack["refs"])), None)
                if ack_id is None:
                    raise VaultIntegrityError("Classification membership legacy ACK is missing")
                self._add_classification_members(db, owner, job["refs"], ack_id, candidates)
        self._classification_members(db, candidates, jobs)
