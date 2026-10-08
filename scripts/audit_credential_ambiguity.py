"""Read-only, value-free profile of archived credential parser uncertainty.

This deliberately bypasses scan receipts: a receipt means a snapshot was
previously scanned, not that its rejected assignments were adjudicated.
No candidate names, values, source paths, or excerpts are retained or printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from muninn.history.credential_discovery import (
    ExtractionStats,
    _verified_archive_chunks,
    iter_transcript_findings,
)
from muninn.history.credential_store import AmbiguousCandidate
from muninn.history.secure_archive import SecureHistoryArchive


def audit(archive: SecureHistoryArchive, *, offset: int, limit: int,
          metadata_only: bool = False) -> dict:
    manifest = archive._load_manifest()
    entries = [entry for _, versions in sorted(manifest["files"].items()) for entry in versions]
    if offset < 0 or limit < 1 or offset > len(entries):
        raise ValueError("Invalid ambiguity audit range")
    report = {
        "generation": manifest["generation"], "snapshots_total": len(entries),
        "offset": offset, "next_offset": min(len(entries), offset + limit),
        "attempted": 0, "succeeded": 0, "errors": 0,
        "candidates": 0, "ambiguous": 0, "ambiguous_reasons": {},
        "ambiguous_shapes": {},
        "range_size_mib": round(sum(entry["size"] for entry in entries[offset:min(len(entries), offset + limit)]) / 1048576, 1),
        "largest_snapshot_mib": round(max((entry["size"] for entry in entries[offset:min(len(entries), offset + limit)]), default=0) / 1048576, 1),
    }
    if metadata_only:
        report["metadata_only"] = True
        return report
    for entry in entries[offset:report["next_offset"]]:
        report["attempted"] += 1
        stats = ExtractionStats()
        try:
            for item in iter_transcript_findings(_verified_archive_chunks(archive, entry), stats,
                                                 include_ambiguous=True):
                if isinstance(item, AmbiguousCandidate):
                    candidate = item.candidate.strip()
                    if not candidate:
                        shape = "empty"
                    elif candidate.startswith("${") and candidate.endswith("}"):
                        shape = "variable_reference"
                    elif candidate.lower() in {"none", "null", "undefined", "false"}:
                        shape = "nullish"
                    elif candidate.startswith(("<", "*", "[REDACTED")):
                        shape = "masked_or_placeholder"
                    else:
                        shape = "other"
                    shapes = report["ambiguous_shapes"]
                    shapes[shape] = shapes.get(shape, 0) + 1
        except (OSError, ValueError, RuntimeError, UnicodeError, TypeError, KeyError):
            report["errors"] += 1
            continue
        report["succeeded"] += 1
        report["candidates"] += stats.accepted
        report["ambiguous"] += stats.ambiguous
        for reason, count in stats.ambiguous_reasons.items():
            reasons = report["ambiguous_reasons"]
            reasons[reason] = reasons.get(reason, 0) + count
    report["generation_at_end"] = archive._load_manifest()["generation"]
    report["changed_during_audit"] = report["generation"] != report["generation_at_end"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(SecureHistoryArchive(args.archive_root),
                       offset=args.offset, limit=args.limit,
                       metadata_only=args.metadata_only)
    except (OSError, ValueError, RuntimeError) as exc:
        # Do not include private paths or decoder excerpts from an exception.
        print(json.dumps({"error_category": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
