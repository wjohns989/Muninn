"""Verify bounded real encrypted transcript samples; print no private contents.

Creates only rebuildable encrypted sidecars. No inference, credential unlock,
or transcript editing. Each provider sample is independently JSON-decoded
within the CLI byte bound and compared with the authenticated streaming units.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from muninn.history.credential_context import CredentialContextStore
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.structured_projector import UnsupportedTranscript


def check(root: Path, max_bytes: int) -> dict:
    archive = SecureHistoryArchive(root)
    manifest = archive._load_manifest()
    units, contexts = SourceEvidenceStore(archive), CredentialContextStore(archive)
    results = []
    for provider in ("codex", "claude_code", "gemini_cli"):
        candidates = [(source, len(versions) - 1, versions[-1])
                      for source, versions in manifest["files"].items() if versions
                      and versions[-1].get("provider") == provider
                      and versions[-1].get("kind") == "transcript"
                      and 256 <= versions[-1]["size"] <= max_bytes]
        candidates.sort(key=lambda item: item[2]["size"])
        errors = Counter()
        for source, version, entry in candidates[:20]:
            try:
                raw = archive.read_file(Path(source), version)
                decoded = raw.decode("utf-8")
                if provider == "gemini_cli":
                    root_record = json.loads(decoded)
                    records = root_record["messages"] if isinstance(root_record, dict) and "messages" in root_record else [root_record]
                else:
                    records = [json.loads(line) for line in decoded.splitlines() if line.strip()]
                attempt = units.build_snapshot(entry, version)
                parts = list(units.fragments(entry, version, attempt))
                final = [part.unit for part in parts if part.final]
                if len(final) != len(records):
                    raise ValueError("source unit count mismatch")
                context_attempt = contexts.build_snapshot(entry, version)
                raw_contexts = list(contexts.contexts(entry, version, context_attempt))
                physical = {unit.physical_line for unit in final if unit.physical_line is not None}
                results.append({"provider": provider, "state": "verified", "source_bytes": len(raw),
                                "units": len(final), "text_fragments": sum(bool(part.text) for part in parts),
                                "event_times": sum(part.event_at is not None for part in final),
                                "project_observations": sum(part.cwd is not None for part in final),
                                "ambiguity_occurrences": len(raw_contexts),
                                "line_joined": sum(item.source_line in physical for item in raw_contexts),
                                "independent_record_count_matches": True})
                break
            except (UnsupportedTranscript, StreamingJSONError, json.JSONDecodeError, KeyError, UnicodeError) as exc:
                errors[type(exc).__name__] += 1
        else:
            results.append({"provider": provider, "state": "no_eligible_sample",
                            "candidate_count": len(candidates), "error_categories": dict(errors)})
    return {"samples": results, "source_evidence": units.verify_all(),
            "credential_context": contexts.verify_all(), "inference_sent": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--max-bytes", default=65536, type=int)
    args = parser.parse_args()
    if not 256 <= args.max_bytes <= 131072:
        parser.error("--max-bytes must be 256..131072")
    try:
        report = check(args.archive_root, args.max_bytes)
        print(json.dumps(report, sort_keys=True))
        return 0 if all(row["state"] == "verified" for row in report["samples"]) else 2
    except Exception as exc:
        print(json.dumps({"state": "failed", "error_category": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
