"""Bounded real-source ledger proof; prints counts/types, never source text.

Default checks dispatch no model and change no historical source. --apply stores
one encrypted source observation. Explicit inference-preview flags perform one
approved real-input call without publication. This is not historical backfill.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from muninn.history.memory_ledger import MemoryLedger
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.blind_index import SecureHistoryBlindIndex, _terms
from muninn.history.cited_analysis_source import CitedAnalysisSource


def preview_policy_root(policy_root: Path | None):
    """Never infer remote authority from the archive's configurable location."""
    if policy_root is None:
        raise ValueError("ZDR preview requires an explicit policy root")
    resolved = policy_root.resolve(strict=True)
    from muninn.history.auto_routing import remote_policy_snapshot
    policy = remote_policy_snapshot(resolved)
    if policy.source != "managed" or not policy.enabled:
        raise ValueError("ZDR preview requires an enabled managed policy")
    return resolved


def run(root: Path, *, apply=False, max_snapshots=8, cited_preview=False,
        local_analysis_preview=False, zdr_analysis_preview=False, policy_root=None):
    if not 1 <= max_snapshots <= 8:
        raise ValueError("Invalid bounded source proof")
    if sum(map(bool, (cited_preview, local_analysis_preview, zdr_analysis_preview))) > 1:
        raise ValueError("Select only one preview route")
    if (cited_preview or local_analysis_preview or zdr_analysis_preview) and apply:
        raise ValueError("Cited input preview does not publish memories")
    if zdr_analysis_preview:
        policy_root = preview_policy_root(policy_root)
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
            if cited_preview or local_analysis_preview or zdr_analysis_preview:
                terms = _terms(data["text"][:3000])
                if not terms:
                    continue
                term = max(terms, key=len)
                before = ledger.verify_all()
                cited = CitedAnalysisSource(archive)
                descriptor = cited.prepare(SecureHistoryBlindIndex(archive)._capability(entry, version, term))
                if descriptor is None:
                    continue
                window = CitedAnalysisSource(archive).reopen(descriptor)
                if local_analysis_preview or zdr_analysis_preview:
                    from muninn.history.secure_analysis import analyze_cited_window
                    # A one-call proof, not a second server or queued backfill.
                    class LocalContext:
                        data_dir = policy_root if zdr_analysis_preview else root.parent
                    result = asyncio.run(analyze_cited_window(LocalContext(), cited, descriptor,
                        allow_remote=zdr_analysis_preview, prefer_remote=zdr_analysis_preview))
                    return {"state": result["status"], "provider": result.get("provider"),
                            "model": result.get("model"), "reason": result.get("reason"),
                            "output_failure": result.get("output_failure"),
                            "source_provider": entry["provider"], "snapshots_examined": examined,
                            "source_bytes": entry["size"], "window_characters": len(window["text"]),
                            "validated_proposals": len(result.get("extraction", {}).get("proposals", [])),
                            "candidate_delta": ledger.verify_all()["candidates"] - before["candidates"],
                            "applied": False, "remote_allowed": zdr_analysis_preview}
                return {"state": "ok", "snapshots_examined": examined,
                        "provider": entry["provider"], "source_bytes": entry["size"],
                        "window_characters": len(window["text"]),
                        "query_term_retained": term in window["text"].casefold(),
                        "citation_ranges": len(window["citation_ranges"]),
                        "event_time_present": window["event_at"] is not None,
                        "project_reference_present": bool(window["project_ref"]),
                        "remote_eligible": cited.remote_input(descriptor) is not None,
                        "candidate_delta": ledger.verify_all()["candidates"] - before["candidates"],
                        "model_dispatched": False, "applied": False}
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
    parser.add_argument("--cited-preview", action="store_true",
                        help="Reopen one real cited model input; no inference or memory publication")
    parser.add_argument("--local-analysis-preview", action="store_true",
                        help="One explicitly requested local model call on real cited input; no publication")
    parser.add_argument("--zdr-analysis-preview", action="store_true",
                        help="One explicitly approved real-input ZDR call under persisted policy/budget; no publication")
    parser.add_argument("--policy-root", type=Path,
                        help="Explicit running installation's data directory; required for ZDR preview")
    args = parser.parse_args()
    try:
        report = run(args.root, apply=args.apply, max_snapshots=args.max_snapshots,
                     cited_preview=args.cited_preview, local_analysis_preview=args.local_analysis_preview,
                     zdr_analysis_preview=args.zdr_analysis_preview, policy_root=args.policy_root)
    except Exception as exc:
        report = {"state": "failed", "error_category": type(exc).__name__}
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["state"] in {"ok", "eligible_real_source"} else 2)
