"""Verify a large encrypted-history snapshot without printing private content.

This diagnostic searches for a public structural word, confirms the returned
opaque reference belongs to the largest selected snapshot, then fetches one
bounded redacted span. It never prints source paths, transcript text or grants.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive

_STRUCTURAL_TERMS = ("mapping", "conversations", "conversation", "assistant", "message")


def verify(root: Path, *, min_bytes: int) -> dict[str, object]:
    index = SecureHistoryBlindIndex(SecureHistoryArchive(root))
    selected = [item for item in index._current() if item[2]["size"] >= min_bytes]
    if not selected:
        return {"status": "no_large_snapshot", "min_bytes": min_bytes}
    with index._connect() as db:
        upgraded = [item for item in selected
                    if index._completion(db, item[2]) is not None
                    and (prior := index._lookup_filter(db, item[2], item[1])) is not None
                    and prior[0] == b"O"]
    target = max(upgraded or selected, key=lambda item: item[2]["size"])
    target_class = "legacy_overflow_upgraded" if upgraded else "largest_large_snapshot"
    source, _version, entry, _latest, _versions = target
    target_ref = hmac.new(
        index.archive._key, b"history-metadata-ref-v1\0" + source.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    # Bound this diagnostic to the selected immutable blob. The public service
    # still searches the whole archive; focused tests cover global ordering.
    index._current = lambda: [target]
    for term in _STRUCTURAL_TERMS:
        report = index.search(term, limit=1, max_candidates=1)
        for match in report["matches"]:
            if match["ref"] != target_ref:
                continue
            span = index.fetch_span(match["fetch_capability"], max_chars=3000)
            return {
                "status": "ok", "query": term, "snapshot_bytes": entry["size"],
                "provider": entry["provider"], "kind": entry["kind"],
                "target_class": target_class,
                "scope": "single_selected_snapshot",
                "redacted_span_chars": len(span["redacted_text"]),
                "coverage_complete": report["missing"] == 0 and report["overflow"] == 0,
            }
    return {"status": "large_snapshot_not_found_by_structural_terms",
            "snapshot_bytes": entry["size"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-bytes", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()
    if args.min_bytes < 1:
        parser.error("--min-bytes must be positive")
    report = verify(args.root, min_bytes=args.min_bytes)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
