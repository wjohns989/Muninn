"""Explicit one-file restart recovery compaction. No batches or GC."""
import argparse
import getpass
import json
import time
from pathlib import Path
from muninn.history.private_acl import create_private_file, verify_private

from muninn.history.recovery_pool import RecoveryPool, RELATIVE, MARKER, _SNAPSHOT, unlinked, private_file
from muninn.history.secure_archive import SecureHistoryArchive


def eligible_snapshots(archive, keep_full=4, excluded=()):
    if keep_full < 4:
        raise ValueError("At least four full restart preimages must remain")
    parent = unlinked(archive.root / "operator-preimages")
    snapshots = sorted(unlinked(p) for p in parent.iterdir()
                       if _SNAPSHOT.fullmatch(p.name) and p.is_dir() and (p / RELATIVE).exists())
    for snapshot in snapshots:
        # Fail closed before selecting any retirement if a retained slot is a
        # linked/shared file, a non-file, or no longer privately accessible.
        private_file(snapshot / RELATIVE)
    # Count actual full databases, not newer directories already represented by
    # pooled manifests. Clock-skewed names must not consume the full-copy floor.
    return [p for p in snapshots[:-keep_full] if p.name not in excluded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compact", "restore", "backfill"))
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--keep-full", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--retire", action="store_true")
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--exclude-snapshot", action="append", default=[])
    args = parser.parse_args()
    def emit(value):
        line = json.dumps({"timestamp": time.time(), **value})
        print(line, flush=True)
        if args.log_path:
            path = unlinked(args.log_path)
            if not path.exists():
                create_private_file(path)
            verify_private(path)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    if args.action == "restore":
        if not args.destination or bool(args.snapshot) == bool(args.snapshot_id):
            parser.error("restore requires snapshot OR snapshot-id and a new destination")
        pool = RecoveryPool(args.pool_root, passphrase=getpass.getpass("History recovery passphrase (hidden): "))
        emit(pool.restore_id(args.snapshot_id, args.destination) if args.snapshot_id else
             pool.restore(args.snapshot, args.destination))
        return
    if not args.archive_root or args.keep_full < 4 or args.limit < 1:
        parser.error("compact requires archive-root, keep-full >= 4, limit >= 1")
    if any(not _SNAPSHOT.fullmatch(item) for item in args.exclude_snapshot):
        parser.error("excluded snapshot identity is invalid")
    archive = SecureHistoryArchive(unlinked(args.archive_root))
    if args.action == "backfill":
        pool = RecoveryPool(args.pool_root, archive=archive)
        emit({"stage": "backfill_complete", **pool.backfill_manifests(archive)})
        return
    candidates = eligible_snapshots(archive, args.keep_full, args.exclude_snapshot)
    if args.snapshot:
        candidates = [p for p in candidates if p == unlinked(args.snapshot)]
        if not candidates:
            parser.error("selected snapshot is absent, packed, or protected by keep-full")
    pool = RecoveryPool(args.pool_root, archive=archive)
    total = {"processed": 0, "retired_bytes": 0, "new_chunk_bytes": 0}
    for snapshot in candidates[:args.limit]:
        start = time.monotonic()
        emit({"stage": "packing", "snapshot": snapshot.name})
        result = pool.pack(archive, snapshot, retire=args.retire)
        emit({"stage": "verified", "elapsed_seconds": round(time.monotonic()-start, 1), **result})
        total["processed"] += 1
        total["retired_bytes"] += result["retired_bytes"]
        total["new_chunk_bytes"] += result["new_chunk_bytes"]
    emit({"stage": "complete", **total,
          "remaining_eligible_copies": max(0, len(candidates)-total["processed"])})


if __name__ == "__main__":
    main()
