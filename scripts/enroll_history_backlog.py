"""Preview or explicitly enroll latest historical snapshots, without inference."""
import argparse
from contextlib import contextmanager
import getpass
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muninn.history.capture_journal import CaptureJournal
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import verify_private
from muninn.history.secure_archive import SecureHistoryArchive


class ReadOnlyJournal(CaptureJournal):
    def __init__(self, archive):
        self.archive = archive
        # Match CaptureJournal's default policy boundary without its writer
        # initialization. Queue-capacity inspection must not guess or cache
        # batch consent, nor create/migrate the live journal.
        self.policy_root = archive.root.parent
        self.path = (archive.root / "capture-jobs.db").absolute()
        self._key = hmac.new(archive._key, b"muninn-capture-journal-key-v1", hashlib.sha256).digest()

    @contextmanager
    def _connect(self, **kwargs):
        verify_private(self.path)
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            yield db
        finally:
            db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Explicitly commit enrollment; never call a model")
    parser.add_argument("--all-versions", action="store_true",
                        help="Backfill every pinned version after latest enrollment is complete")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--prompt-passphrase", action="store_true", help="Prompt locally instead of using Windows unlock")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 128 or not 1 <= args.max_batches <= 10000:
        parser.error("limit must be 1-128; max-batches must be 1-10000")
    try:
        passphrase = getpass.getpass("History recovery passphrase (hidden): ") if args.prompt_passphrase else None
        archive = SecureHistoryArchive(args.archive_root.resolve(), passphrase)
        del passphrase
        if not args.apply:
            reader = ReadOnlyJournal(archive)
            preview = reader.preview_historical_versions if args.all_versions else reader.preview_historical_latest
            print(json.dumps(preview(limit=args.limit)))
            return 0
        journal = CaptureJournal(archive, recover=False)
        enroll = journal.enroll_historical_versions if args.all_versions else journal.enroll_historical_latest
        for _ in range(args.max_batches):
            state = enroll(limit=args.limit)
            print(json.dumps({"stage": "enrollment_batch", **state}), flush=True)
            if state["complete"]:
                break
        with journal._connect() as db:
            db.execute("BEGIN")
            journal._verify_enrichment(db)
        print(json.dumps({"stage": "enrollment_verified", "complete": state["complete"],
                          "processed_by_models": False}), flush=True)
        return 0
    except (VaultIntegrityError, OSError, sqlite3.Error, ValueError):
        print(json.dumps({"stage": "error", "code": "historical_enrollment_unavailable"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
