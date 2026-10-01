"""Check a real pending transcript source join without unlocking vault values.

Builds only encrypted derived sidecars. Reports counts, never values/paths.
This checks occurrence provenance, not agreement with the encrypted vault
candidate or correctness of a classifier's credential determination.
"""

import argparse
import json
from pathlib import Path

from muninn.history.credential_review_source import CredentialReviewSource
from muninn.history.credential_store import CredentialStore


def check(vault_root: Path, archive_root: Path, max_bytes: int) -> dict:
    review = CredentialReviewSource(archive_root)
    rows = CredentialStore(vault_root).list_ambiguities(limit=100)
    seen = set()
    for row in rows:
        source = review.index.get(row["source_hash"])
        if source is None or row["source_hash"] in seen or row["origin"] != "transcript":
            continue
        seen.add(row["source_hash"])
        entry = review.manifest["files"][source.source_path][source.version]
        if entry["size"] > max_bytes:
            continue
        prepared = review.prepare(row)
        if prepared is None:
            continue
        _source, _entry, units_attempt, contexts_attempt = prepared
        units = {part.unit.physical_line: part.unit for part in review.units.fragments(
            entry, source.version, units_attempt) if part.final and part.unit.physical_line is not None}
        count, joined, timed, projects, matching_metadata = 0, 0, 0, 0, 0
        for item in review.contexts.contexts(entry, source.version, contexts_attempt):
            count += 1
            matching_metadata += int((item.name, item.reason) == (row["name"], row["reason"]))
            unit = units.get(item.source_line)
            if unit is not None:
                joined += 1
                timed += int(unit.event_at is not None)
                projects += int(unit.cwd is not None)
        if matching_metadata:
            return {"state": "verified", "provider": source.provider, "source_bytes": entry["size"],
                    "ambiguity_occurrences": count, "line_joined": joined,
                    "occurrences_with_event_time": timed, "occurrences_with_project": projects,
                    "matching_name_reason": matching_metadata, "vault_value_join_checked": False,
                    "inference_sent": False}
    return {"state": "no_eligible_pending_source", "metadata_rows_checked": len(rows), "inference_sent": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--max-source-bytes", type=int, default=4194304)
    args = parser.parse_args()
    if not 256 <= args.max_source_bytes <= 4194304:
        parser.error("--max-source-bytes must be 256..4194304")
    try:
        report = check(args.vault_root, args.archive_root, args.max_source_bytes)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["state"] == "verified" else 2
    except Exception as exc:
        print(json.dumps({"state": "failed", "error_category": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
