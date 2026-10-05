"""Preview or recover a bounded set of unsent local failures; never run models."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muninn.history.capture_journal import CaptureJournal
from muninn.history.capture_window_jobs import _RECOVERABLE_LOCAL_FAILURES
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_directory, create_private_file, verify_private
from muninn.history.remote_policy import read_policy
from muninn.history.secure_archive import SecureHistoryArchive
from scripts.enroll_history_backlog import ReadOnlyJournal
from scripts.reload_shared_local import require_unlinked_path


def candidates(journal, limit):
    """Nonsecret index hints; the writer authenticates each selected binding."""
    codes = sorted(_RECOVERABLE_LOCAL_FAILURES)
    predicate = ("j.lane=1 AND j.state='failed' AND j.remote_dispatched=0 "
                 "AND j.cancel_requested=0 AND j.publication_started=0 "
                 "AND j.lease_token IS NULL AND j.lease_until IS NULL "
                 "AND j.sealed_extraction IS NULL AND j.extraction_id IS NULL "
                 "AND j.sealed_receipt IS NULL AND j.sealed_reuse IS NULL "
                 "AND j.sealed_result IS NULL AND s.resolved=0 "
                 "AND j.error_code IN (" + ",".join("?" for _ in codes) + ")")
    joined = (" FROM history_analysis_jobs j JOIN capture_enrichment_windows w ON w.job_id=j.job_id "
              "JOIN capture_enrichment_sources s ON s.work_id=w.work_id WHERE " + predicate)
    with journal._connect() as db:
        db.execute("BEGIN") if not db.in_transaction else None
        rows = db.execute("SELECT j.*" + joined +
                          " ORDER BY s.planning_complete DESC,j.created_at,w.ordinal LIMIT ?", (*codes, limit)).fetchall()
        counts = dict(db.execute("SELECT j.error_code,count(*)" + joined + " GROUP BY j.error_code", codes))
        capacity = journal._capture_window_capacity(db)
        for row in rows:
            journal._validated_analysis_target(row, db)
            journal._read_analysis_window(row)
    return rows, {"eligible_windows": sum(counts.values()), "by_error": counts,
                  "selected": len(rows), "runnable_capacity": capacity}


def policy_for(journal):
    # Do not infer consent from an environment variable or reveal a provider key.
    return read_policy(getattr(journal, "policy_root", journal.archive.root.parent),
                       lambda: (False, 1, 30, False))


def backup_journal(journal, destination):
    require_unlinked_path(destination)
    if destination.exists():
        raise ValueError("Recovery preimage already exists")
    create_private_directory(destination)
    target = destination / "capture-jobs.db"
    create_private_file(target)
    with sqlite3.connect(journal.path.as_uri() + "?mode=ro", uri=True) as source:
        with sqlite3.connect(target) as backup:
            source.backup(backup)
            if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise VaultIntegrityError("Recovery preimage is invalid")
    verify_private(target)


def confirm_preview(journal, rows):
    """Compare selected rows after the preimage; unrelated live capture is safe."""
    with journal._connect() as db:
        for row in rows:
            current = db.execute("SELECT * FROM history_analysis_jobs WHERE job_id=?",
                                 (row["job_id"],)).fetchone()
            if current is None or tuple(current) != tuple(row):
                raise ValueError("Recovery preview changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--backup-before", type=Path)
    parser.add_argument("--prompt-passphrase", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 32:
        parser.error("limit must be 1-32")
    if args.apply and (args.expected_generation is None or args.backup_before is None):
        parser.error("apply requires an expected consent generation and a fresh preimage destination")
    try:
        require_unlinked_path(args.archive_root)
        for name in ("header.json", "capture-jobs.db"):
            existing = args.archive_root / name
            require_unlinked_path(existing)
            if not existing.is_file():
                raise ValueError("Existing recovery store is missing")
            verify_private(existing)
        passphrase = getpass.getpass("History recovery passphrase (hidden): ") if args.prompt_passphrase else None
        archive = SecureHistoryArchive(args.archive_root.resolve(), passphrase)
        del passphrase
        journal = ReadOnlyJournal(archive)
        rows, preview = candidates(journal, args.limit)
        policy = policy_for(journal)
        print(json.dumps({"stage": "preview", **preview, "remote_enabled": policy.enabled,
                          "remote_generation": policy.generation}), flush=True)
        if not args.apply:
            return 0
        if not policy.enabled or policy.generation != args.expected_generation:
            raise ValueError("Recovery consent changed")
        if not rows or preview["runnable_capacity"] == 0:
            print(json.dumps({"stage": "recovery_committed", "outcomes": {
                "queue_full" if rows else "no_eligible_work": 1}, "model_calls": 0,
                "sources_marked_complete": 0}), flush=True)
            return 0
        backup_journal(journal, args.backup_before.absolute())
        print(json.dumps({"stage": "preimage_validated"}), flush=True)
        confirm_preview(journal, rows)
        writer = CaptureJournal(archive, recover=False)
        outcomes = {}
        for row in rows:
            current = policy_for(writer)
            if not current.enabled or current.generation != args.expected_generation:
                outcomes["consent_changed"] = 1
                break
            result = writer.retry_capture_window(row["job_id"], expected_attempt=row["attempt"],
                remote_policy_generation=args.expected_generation,
                expected_target_sha256=hashlib.sha256(row["sealed_target"]).hexdigest())
            outcomes[result] = outcomes.get(result, 0) + 1
            if result == "queue_full":
                break
        print(json.dumps({"stage": "recovery_committed", "outcomes": outcomes,
                          "model_calls": 0, "sources_marked_complete": 0}), flush=True)
        return 0
    except (VaultIntegrityError, OSError, sqlite3.Error, ValueError, RuntimeError):
        print(json.dumps({"stage": "error", "code": "capture_recovery_unavailable"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
