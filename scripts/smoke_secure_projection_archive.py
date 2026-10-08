"""Smoke-test real local encrypted-history continuation without printing text.

Runs the checked-out code against the configured archive while leaving the
shared HTTP service untouched. It may create a derived encrypted projection
for one matching snapshot. Do not pass a secret as the search query.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_access import ProjectionAccess
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import build_transcript_projection

_SAFE_PARSER_ERRORS = {
    "Invalid JSON Unicode surrogate pair", "Lone JSON low surrogate",
    "Invalid JSON scalar", "Invalid JSON Unicode escape", "Invalid JSON escape",
    "Invalid JSON string character", "Invalid JSON scalar length",
    "Invalid JSONL UTF-8", "Truncated JSON string", "Unexpected JSON value",
    "Multiple JSONL roots on one line", "Invalid JSON value completion",
    "Invalid JSON root completion", "JSON object key exceeds schema bound",
    "Unexpected JSON string fragment", "Unexpected JSON string end",
    "JSON nesting exceeds schema bound", "Unexpected JSON container end",
    "Unexpected JSON colon", "Unexpected JSON comma",
    "JSONL record crosses physical lines", "Truncated JSONL record",
    "Conflicting transcript metadata", "Conflicting transcript block type",
    "Unclosed transcript value", "Truncated transcript record",
    "Transcript passes have different message counts",
    "Transcript passes have different record counts",
}


def _default_archive() -> Path:
    configured = os.environ.get("MUNINN_HISTORY_ARCHIVE_DIR")
    if configured:
        return Path(configured)
    data = Path(os.environ.get("MUNINN_DATA_DIR", ".muninn_runtime"))
    return data / "history_secure_archive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=_default_archive())
    parser.add_argument("--query", required=True, help="Nonsecret search term; never printed")
    parser.add_argument("--min-size-kib", type=int, default=0)
    parser.add_argument("--max-size-kib", type=int, default=1024)
    parser.add_argument("--deadline-seconds", type=int, default=180)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--diagnose-parser", action="store_true",
                        help="On malformed JSON, retry once and report only an allowlisted parser error code")
    parser.add_argument("--inspect-candidates", action="store_true",
                        help="Report only provider and size buckets for candidate planning; build nothing")
    args = parser.parse_args()
    if not 1 <= args.max_size_kib <= 1024 * 1024:
        parser.error("--max-size-kib must be 1..1048576")
    if not 0 <= args.min_size_kib <= args.max_size_kib:
        parser.error("--min-size-kib must be between 0 and --max-size-kib")
    if not 1 <= args.pages <= 20:
        parser.error("--pages must be 1..20")
    if not 10 <= args.deadline_seconds <= 540:
        parser.error("--deadline-seconds must be 10..540")
    started = time.monotonic()
    access = None
    try:
        archive = SecureHistoryArchive(args.archive_root)
        result = SecureHistoryBlindIndex(archive).search(args.query, limit=10)
        if args.inspect_candidates:
            print(json.dumps({"state": "candidates", "search_complete": result["complete"],
                              "candidates": [{"provider": match["provider"],
                                              "size_bucket_kib": match["size_bucket_kib"]}
                                             for match in result["matches"]]}, sort_keys=True))
            return 0
        selected = next((match for match in result["matches"]
                         if match["kind"] == "transcript"
                         and match["provider"] in {"codex", "claude_code", "gemini_cli"}
                         and match["size_bucket_kib"] >= args.min_size_kib
                         and match["size_bucket_kib"] <= args.max_size_kib), None)
        if selected is None:
            print(json.dumps({"state": "no_eligible_match", "matches": len(result["matches"]),
                              "search_complete": result["complete"]}, sort_keys=True))
            return 2
        access = ProjectionAccess(archive)
        capability = selected["fetch_capability"]
        projected = access.start(capability)
        while projected["state"] == "pending" and time.monotonic() - started < args.deadline_seconds:
            time.sleep(1)
            projected = access.poll(capability)
        details: dict[str, object] = {
            "state": projected["state"], "provider": selected["provider"],
            "size_bucket_kib": selected["size_bucket_kib"],
            "elapsed_ms": round((time.monotonic() - started) * 1000),
            "search_complete": result["complete"],
        }
        if projected.get("reason"):
            details["reason"] = projected["reason"]
        if args.diagnose_parser and projected.get("reason") == "malformed_json":
            entry, version, _hit = access.index._entry_for_capability(capability)
            try:
                build_transcript_projection(access.store, entry, version)
            except StreamingJSONError as exc:
                details["parser_error"] = str(exc) if str(exc) in _SAFE_PARSER_ERRORS else "other"
        if isinstance(projected.get("coverage"), dict):
            details["coverage"] = projected["coverage"]
        if projected["state"] == "ready":
            cursor = projected["cursor"]
            checked = 0
            characters = 0
            while cursor and checked < args.pages:
                page = access.page(cursor)
                if not 1 <= len(page["redacted_text"]) <= 4000:
                    raise ValueError("invalid bounded page")
                checked += 1
                characters += len(page["redacted_text"])
                cursor = page["next_cursor"]
            details.update(pages_checked=checked, characters_checked=characters,
                           more=cursor is not None)
        print(json.dumps(details, sort_keys=True))
        return 0 if projected["state"] == "ready" and details.get("pages_checked", 0) else 2
    except Exception as exc:
        # Never print exception text: it may embed source paths or private data.
        print(json.dumps({"state": "probe_error", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    finally:
        if access is not None:
            access.close()


if __name__ == "__main__":
    raise SystemExit(main())
