"""Read-only, metadata-only coverage of credential-to-archive provenance.

No candidate, source hash, path, or credential value is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muninn.history.credential_provenance import archive_source_index
from muninn.history.credential_store import CredentialStore
from muninn.history.secure_archive import SecureHistoryArchive


def coverage(vault_root: Path, archive_root: Path) -> dict[str, int]:
    index = archive_source_index(SecureHistoryArchive(archive_root))
    store = CredentialStore(vault_root)
    report = {"transcript_rows": 0, "transcript_sources": 0,
              "matched_rows": 0, "missing_rows": 0, "nonunique_rows": 0,
              "project_rows": 0, "transcript_source_paths": 0,
              "review_groups": 0, "cross_path_groups": 0}
    source_paths = set()
    group_paths = {}
    with store._connect(readonly=True) as db:
        rows = db.execute("SELECT source_hash,origin,COUNT(*) AS n FROM ambiguity_queue "
                          "GROUP BY source_hash,origin")
        for row in rows:
            count = int(row["n"])
            if row["origin"] != "transcript":
                report["project_rows"] += count
                continue
            report["transcript_rows"] += count
            report["transcript_sources"] += 1
            if row["source_hash"] not in index:
                report["missing_rows"] += count
            elif index[row["source_hash"]] is None:
                report["nonunique_rows"] += count
            else:
                report["matched_rows"] += count
                source_paths.add(index[row["source_hash"]].source_path)
        for row in db.execute("SELECT id,group_digest,source_hash FROM ambiguity_queue "
                              "WHERE origin='transcript'"):
            evidence = index.get(row["source_hash"])
            group = (row["group_digest"] if row["group_digest"].startswith("v2:")
                     else row["id"])
            group_paths.setdefault(group, set()).add(
                evidence.source_path if evidence is not None else None)
    report["transcript_source_paths"] = len(source_paths)
    report["review_groups"] = len(group_paths)
    report["cross_path_groups"] = sum(len(paths) > 1 for paths in group_paths.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(coverage(args.root, args.archive_root), sort_keys=True))
        return 0
    except Exception as exc:
        # Errors can carry private paths; do not include exception text.
        print(json.dumps({"state": "unavailable", "error_category": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
