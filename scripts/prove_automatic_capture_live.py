"""Read actual post-start capture ACKs without inference or private text output.

No synthetic events, retries, process changes or queue writes. Only aggregate
metadata is printed; existing authenticated read routes release no credentials.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import httpx
import psutil

from scripts.local_runtime_preflight import inspect_runtime
from muninn.history.auto_routing import _local_setting, inspect_ollama
from muninn.history.capture_journal import _ANALYSIS_RETRY_CODES, _ANALYSIS_TERMINAL_CODES
from muninn.history.private_acl import verify_private


def main():
    runtime = inspect_runtime(REPO, authenticated=True)
    owned = runtime["muninn_processes"]
    if len(owned) != 1 or runtime["listener_owners"] != [owned[0]["pid"]]:
        raise RuntimeError("Owned service is not unique")
    process = psutil.Process(owned[0]["pid"])
    environment = process.environ()  # Never display or persist environment values.
    data = Path(environment.get("MUNINN_DATA_DIR") or REPO / ".muninn_runtime")
    if not data.is_absolute():
        data = REPO / data
    archive = Path(environment.get("MUNINN_HISTORY_ARCHIVE_DIR") or data / "history_secure_archive")
    if not archive.is_absolute():
        archive = REPO / archive
    journal = archive / "capture-jobs.db"
    verify_private(archive)
    verify_private(journal)
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        cutoff = process.create_time()
        counts = dict(db.execute("SELECT state,COUNT(*) FROM history_analysis_jobs "
                                 "WHERE lane=1 AND created_at>=? GROUP BY state", (cutoff,)))
        errors = {}
        for code, count in db.execute("SELECT error_code,COUNT(*) FROM history_analysis_jobs "
                "WHERE lane=1 AND created_at>=? AND state='failed' GROUP BY error_code", (cutoff,)):
            safe_code = code if code in _ANALYSIS_RETRY_CODES | _ANALYSIS_TERMINAL_CODES else "other"
            errors[safe_code] = errors.get(safe_code, 0) + count
        remote = db.execute("SELECT COUNT(*) FROM history_analysis_jobs WHERE lane=1 "
                            "AND created_at>=? AND remote_dispatched!=0", (cutoff,)).fetchone()[0]
        jobs = [row[0] for row in db.execute("SELECT job_id FROM history_analysis_jobs WHERE lane=1 "
                "AND created_at>=? AND state='succeeded' ORDER BY updated_at DESC LIMIT 6", (cutoff,))]
    token = _local_setting("MUNINN_AUTH_TOKEN")
    if not token:
        raise RuntimeError("Authenticated local reads are unavailable")
    checked, ollama, refs_count = 0, 0, 0
    sample_refs = []
    with httpx.Client(base_url="http://127.0.0.1:42069", timeout=30, trust_env=False,
                      headers={"Authorization": "Bearer " + token}) as client:
        for ident in jobs:
            reply = client.get("/history/secure/analysis/jobs/" + ident)
            reply.raise_for_status()
            result = reply.json()["data"]
            checked += 1
            ollama += (result.get("result") or {}).get("provider") == "ollama"
            refs = result.get("memory_refs", [])
            refs_count += len(refs)
            for ref in refs:
                if ref not in sample_refs and len(sample_refs) < 12:
                    sample_refs.append(ref)
        memory_ok, source_ok, context_chars = False, False, 0
        withheld = 0
        for sample_ref in sample_refs:
            memory = client.post("/history/secure/memories/get", json={"memory_ref": sample_ref})
            memory_ok = memory.status_code == 200 and memory.json().get("success") is True
            source = client.post("/history/secure/memories/source",
                                 json={"memory_ref": sample_ref, "max_chars": 1200})
            source_ok = source.status_code == 200 and source.json().get("success") is True
            if source_ok:
                context_chars = len(source.json()["data"].get("context", ""))
                if context_chars:
                    break
                withheld += 1
    ollama_state = inspect_ollama("http://127.0.0.1:11434")
    print(json.dumps({"post_start_capture_jobs": counts, "failure_codes": errors,
        "remote_dispatches": remote, "recent_acks_checked": checked, "ollama_results": ollama,
        "memory_refs_in_checked_acks": refs_count, "cited_memory_read_ok": memory_ok,
        "redacted_source_read_ok": source_ok, "source_context_chars": context_chars,
        "withheld_source_samples": withheld,
        "resident_ollama_models_at_sample": (len(ollama_state.loaded_models) if ollama_state else None),
        "capture_mode": runtime["capture_enrichment"]},
        sort_keys=True))
    return 0 if checked and ollama == checked and remote == 0 and memory_ok and source_ok and context_chars else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, KeyError, ValueError, httpx.HTTPError) as exc:
        print(json.dumps({"state": "proof_unavailable", "error_category": type(exc).__name__}))
        raise SystemExit(2)
