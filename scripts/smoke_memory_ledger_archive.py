"""Bounded real-source ledger proof; prints counts/types, never source text.

No model is dispatched and no historical source is changed. --apply stores one
encrypted source observation. This is not automatic enrichment or full backfill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from muninn.history.memory_ledger import MemoryLedger
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore


def run(root: Path, *, apply=False, max_snapshots=8):
    if not 1 <= max_snapshots <= 8:
        raise ValueError("Invalid bounded source proof")
    archive = SecureHistoryArchive(root)
    units = SourceEvidenceStore(archive)
    manifest = archive._load_manifest()
    choices = [(version, entry) for versions in manifest["files"].values()
               for version, entry in enumerate(versions)
               if entry.get("provider") in {"claude_code", "codex", "gemini_cli"}
               and entry.get("kind") == "transcript" and 1024 <= entry["size"] <= 65536]
    # Previously completed sidecars first; no source path leaves the manifest.
    with units._connect() as db:
        completed = {(blob, sha, version) for blob, sha, version in db.execute(
            "SELECT blob,sha,version FROM attempts WHERE state='complete'")}
    choices.sort(key=lambda item: ((item[1]["blob"], item[1]["sha256"], item[0]) not in completed,
                                  {"claude_code": 0, "codex": 1, "gemini_cli": 2}[item[1]["provider"]],
                                  item[1]["size"], item[1]["blob"]))
    examined = 0
    counts = {"user_windows": 0, "bounded_windows": 0, "screen_denied": 0}
    for version, entry in choices:
        if examined >= max_snapshots:
            break
        examined += 1
        attempt = units.build_snapshot(entry, version)
        count = units.count_pages(entry, version, attempt)
        for page in range(count):
            data = json.loads(units.get_page(entry, version, attempt, page))
            if data["final"] or str(data["unit"].get("role") or "").casefold() != "user":
                continue
            counts["user_windows"] += 1
            if len(data["text"]) < 20 or data["text"].startswith("\n\n"):
                continue
            counts["bounded_windows"] += 1
            ledger = MemoryLedger(archive)
            if ledger.remote_input(entry, version, attempt, page) is None:
                counts["screen_denied"] += 1
                continue
            if not apply:
                return {"state": "eligible_real_source", "snapshots_examined": examined,
                        "provider": entry["provider"], "source_bytes": entry["size"],
                        "model_dispatched": False, "applied": False}
            before = ledger.verify_all()
            ident = ledger.record(entry, version, attempt, page,
                                  {"type": "observation", "text": data["text"][:1280],
                                   "quote": data["text"][:1280], "start": 0},
                                  model_identity=hashlib.sha256(b"local-source-observation-rule-v1").hexdigest())
            # Fresh object demonstrates durable reopen, not an in-memory result.
            reopened = MemoryLedger(archive)
            result = reopened.get(ident)
            after = reopened.verify_all()
            return {"state": "ok", "snapshots_examined": examined, "provider": entry["provider"],
                    "source_bytes": entry["size"], "memory_state": result["state"],
                    "epistemic_kind": result["epistemic_kind"], "truth_status": result["truth_status"],
                    "source_reference_present": bool(result["source_ref"]),
                    "event_time_present": result["event_at"] is not None,
                    "project_reference_present": bool(result["project_ref"]),
                    "candidate_delta": after["candidates"] - before["candidates"],
                    "events_verified": after["events"], "model_dispatched": False, "applied": True}
    return {"state": "no_eligible_bounded_source", "snapshots_examined": examined,
            "model_dispatched": False, "applied": False, **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.root, apply=args.apply, max_snapshots=args.max_snapshots)
    except Exception as exc:
        report = {"state": "failed", "error_category": type(exc).__name__}
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["state"] in {"ok", "eligible_real_source"} else 2)
