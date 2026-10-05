"""Owner-local retained-batch consent and screened historical selection.

No secret configuration, provider transport, or deletion lives here. Consent is
separate from ZDR consent and binds an immutable encrypted outbox input to its
own revocable generation. Selection orders already planned historical windows;
it does not claim globally chronological coverage of unplanned sources.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing

from muninn.history.auto_routing import remote_policy_snapshot
from muninn.history.historical_batch import MAX_ITEMS, BatchError, BatchOutbox, _json, prepare_items
from muninn.history.private_acl import verify_private
from muninn.history.remote_policy import PolicyError, _paths


def _database(root):
    directory, marker, database = _paths(root)
    for path in (directory, marker, database):
        verify_private(path)
    return database


def read_batch_policy(root):
    remote = remote_policy_snapshot(root)
    if remote.source != "managed":
        return {"enabled": False, "generation": 0, "remaining_batches": 0}
    database = _database(root)
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=2)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='batch_policy'").fetchone():
            return {"enabled": False, "generation": 0, "remaining_batches": 0}
        row = db.execute("SELECT enabled,generation,max_batches FROM batch_policy WHERE id=1").fetchone()
        if (row is None or row[0] not in (0, 1) or type(row[1]) is not int or row[1] < 1
                or type(row[2]) is not int or not 1 <= row[2] <= 10000):
            raise PolicyError("Batch policy is invalid")
        used = db.execute("SELECT COUNT(*) FROM batch_consent WHERE retention_generation=?", (row[1],)).fetchone()[0]
    return {"enabled": bool(row[0]), "generation": row[1],
            "remaining_batches": max(0, row[2] - used)}


def configure_batch(root, *, enabled, max_batches=1):
    """Explicit local operator action. Existing spending/ZDR policy is untouched."""
    if type(enabled) is not bool or type(max_batches) is not int or not 1 <= max_batches <= 10000:
        raise ValueError("Invalid batch configuration")
    remote = remote_policy_snapshot(root)
    if remote.source != "managed" or enabled and not remote.enabled:
        raise PolicyError("Managed remote consent is required")
    with closing(sqlite3.connect(_database(root), timeout=5)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS batch_policy (id INTEGER PRIMARY KEY CHECK(id=1),"
                   "enabled INTEGER NOT NULL,generation INTEGER NOT NULL,max_batches INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS batch_policy_audit (generation INTEGER PRIMARY KEY,"
                   "changed_at REAL NOT NULL,enabled INTEGER NOT NULL,max_batches INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS batch_consent (batch_id TEXT PRIMARY KEY,"
                   "retention_generation INTEGER NOT NULL,remote_generation INTEGER NOT NULL,"
                   "input_sha256 TEXT NOT NULL)")
        row = db.execute("SELECT generation FROM batch_policy WHERE id=1").fetchone()
        generation = (row[0] if row else 0) + 1
        db.execute("INSERT INTO batch_policy VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                   "enabled=excluded.enabled,generation=excluded.generation,max_batches=excluded.max_batches",
                   (int(enabled), generation, max_batches))
        db.execute("INSERT INTO batch_policy_audit VALUES(?,?,?,?)",
                   (generation, time.time(), int(enabled), max_batches))
        db.commit()
    return read_batch_policy(root)


def _binding(record):
    return hashlib.sha256(_json(record["items"])).hexdigest()


def bind_consent(journal, outbox, ident, policy):
    record = outbox.read(ident)
    with closing(sqlite3.connect(_database(journal.policy_root), timeout=5)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT enabled,generation,max_batches FROM batch_policy WHERE id=1").fetchone()
        count = db.execute("SELECT COUNT(*) FROM batch_consent WHERE retention_generation=?",
                           (policy["generation"],)).fetchone()[0]
        if not row or row[0] != 1 or row[1] != policy["generation"] or count >= row[2]:
            raise PolicyError("Batch consent changed")
        db.execute("INSERT INTO batch_consent VALUES(?,?,?,?)",
                   (ident, row[1], record["consent_generation"], _binding(record)))
        db.commit()


def authorize_batch(journal, generation):
    """Fresh checks immediately before the worker's durable pre-POST fence."""
    policy = read_batch_policy(journal.policy_root)
    remote = remote_policy_snapshot(journal.policy_root)
    owner = journal.historical_batch_owner()
    if not policy["enabled"] or not remote.enabled or remote.generation != generation or owner is None:
        return False
    record = BatchOutbox(journal.archive).read(owner["id"])
    with closing(sqlite3.connect(f"{_database(journal.policy_root).as_uri()}?mode=ro", uri=True, timeout=2)) as db:
        row = db.execute("SELECT retention_generation,remote_generation,input_sha256 "
                         "FROM batch_consent WHERE batch_id=?", (owner["id"],)).fetchone()
    return row == (policy["generation"], generation, _binding(record))


def authorize_transaction(db, ident, digest):
    """Checked inside the SAME BEGIN IMMEDIATE as the paid attempt fence.

    Policy revocation either commits first and denies this attempt, or follows
    the already committed one-attempt authorization. No check/POST TOCTOU gap.
    """
    policy = db.execute("SELECT enabled,generation FROM batch_policy WHERE id=1").fetchone()
    binding = db.execute("SELECT retention_generation,remote_generation,input_sha256 "
                         "FROM batch_consent WHERE batch_id=?", (ident,)).fetchone()
    remote = db.execute("SELECT enabled,generation FROM policy WHERE id=1").fetchone()
    return bool(policy and binding and remote and policy[0] == remote[0] == 1
                and binding == (policy[1], remote[1], digest))


def _park_refusal(journal, job_id, descriptor):
    """Leave proved-unsent private input visible, without consuming runnable slots."""
    with journal._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        if (row is None or row["state"] not in {"pending", "retry"} or row["remote_dispatched"]
                or row["lease_token"] is not None or row["lease_until"] is not None
                or row["publication_started"] or row["cancel_requested"]
                or any(row[k] is not None for k in ("sealed_extraction", "sealed_receipt", "sealed_reuse"))
                or job_id in journal._historical_batch_blocked_jobs(db)):
            return False
        target = journal._validated_analysis_target(row, db)
        if hashlib.sha256(journal._stage_json(descriptor)).hexdigest() != target["descriptor_sha256"]:
            raise PolicyError("Screening descriptor changed")
        db.execute("UPDATE history_analysis_jobs SET state='retry',error_code='source_not_remote_safe',"
                   "due_at=0,updated_at=? WHERE job_id=?", (time.time(), job_id))
        return True


def prepare_next_batch(journal):
    """One checkpoint at a time; completed, dispatched and private work excluded."""
    policy = read_batch_policy(journal.policy_root)
    remote = remote_policy_snapshot(journal.policy_root)
    owner = journal.historical_batch_owner()
    if (not policy["enabled"] or not policy["remaining_batches"] or not remote.enabled
            or owner is not None and owner["phase"] != "passed"):
        return None
    from muninn.history.cited_analysis_source import CitedAnalysisSource
    source = CitedAnalysisSource(journal.archive)
    candidates = []
    with journal._connect() as db:
        _seal, cursor = journal._historical_progress(db)
        if cursor is None:
            return None
        manifest = journal._historical_manifest(cursor)
        selected = {journal._enrichment_id(journal.archive._snapshot_receipt(entries[-1], len(entries) - 1))
                    for entries in list(manifest["files"].values())[:cursor["source_index"]]
                    if entries and entries[-1]["provider"] in {"codex", "claude_code", "gemini_cli"}
                    and entries[-1]["kind"] == "transcript"}
        rows = db.execute("SELECT j.* FROM history_analysis_jobs j JOIN capture_enrichment_windows w "
                          "ON w.job_id=j.job_id "
                          "WHERE j.lane=1 AND j.state IN ('pending','retry') AND j.remote_dispatched=0 "
                          "AND j.remote_policy_generation=? AND j.cancel_requested=0 "
                          "AND j.lease_token IS NULL AND j.lease_until IS NULL AND j.sealed_extraction IS NULL "
                          "AND j.sealed_receipt IS NULL AND j.sealed_reuse IS NULL AND j.sealed_result IS NULL "
                          "AND j.publication_started=0 AND j.error_code<>'source_not_remote_safe' "
                          "AND j.due_at<=? ORDER BY j.created_at,j.job_id",
                          (remote.generation, time.time())).fetchall()
        for row in rows:
            target = journal._validated_analysis_target(row, db)
            if target["work_id"] in selected:
                candidates.append((row["job_id"], target))
        from muninn.history.capture_window_jobs import _RECOVERABLE_LOCAL_FAILURES
        failed = []
        for row in db.execute("SELECT * FROM history_analysis_jobs WHERE lane=1 AND state='failed' "
                              "AND remote_dispatched=0 ORDER BY created_at,job_id"):
            if row["error_code"] in _RECOVERABLE_LOCAL_FAILURES:
                target = journal._validated_analysis_target(row, db)
                if target["work_id"] in selected:
                    failed.append((row["job_id"], target, row["attempt"],
                                   hashlib.sha256(row["sealed_target"]).hexdigest()))
    from muninn.history.cited_windows import CitedWindowPlanStore
    plans = CitedWindowPlanStore(journal.archive)
    checked = []
    prepared = {}
    retry_proofs = {job_id: (attempt, digest) for job_id, _target, attempt, digest in failed}
    for job_id, target in candidates + [(job_id, target) for job_id, target, _, _ in failed]:
        # Reconstruct from the authenticated plan, never from an untrusted hint.
        entry = plans.source.ledger._entries[(target["blob"], target["version"])]
        descriptor = plans.window_at(entry, target["version"], target["plan_attempt"], target["ordinal"])
        window = source.remote_input(descriptor)
        if window is None or not window["text"].strip():
            if job_id not in retry_proofs and window is None:
                _park_refusal(journal, job_id, descriptor)
            continue
        try:
            prepared[job_id] = prepare_items(source, [(job_id, descriptor)])[0]
        except BatchError as exc:
            if str(exc) != "source_not_remote_safe":
                raise
            if job_id not in retry_proofs:
                _park_refusal(journal, job_id, descriptor)
            continue  # Serialized request screening is stricter than span screening.
        checked.append(((window["event_at"] is None, window["event_at"] or 0,
                         target["ordinal"], target["work_id"], job_id), (job_id, descriptor)))
    if not checked:
        return None
    bindings = []
    for _, binding in sorted(checked):
        job_id, _descriptor = binding
        if job_id in retry_proofs:
            attempt, digest = retry_proofs[job_id]
            if journal.retry_capture_window(job_id, expected_attempt=attempt,
                    remote_policy_generation=remote.generation, expected_target_sha256=digest) != "queued":
                continue
        bindings.append(binding)
        if len(bindings) == MAX_ITEMS:
            break
    if not bindings:
        return None
    outbox = BatchOutbox(journal.archive)
    ident = outbox.prepare([prepared[job_id] for job_id, _ in bindings], consent_generation=remote.generation)
    bind_consent(journal, outbox, ident, policy)
    journal.reserve_historical_batch(ident)
    return ident
