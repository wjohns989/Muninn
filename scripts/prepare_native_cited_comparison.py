"""Explicit, read-only native-agent comparison using already-paid real windows.

Only screened bounded inputs can be emitted. No provider call, publication,
source export, queue mutation or credential reveal belongs in this helper.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from muninn.history.capture_journal import CaptureJournal
from muninn.history.cited_analysis_source import CitedAnalysisSource
from muninn.history.memory_ledger import MemoryLedger
from muninn.history.private_acl import verify_private
from muninn.history.remote_accounting import settled_response
from muninn.history.secure_analysis import _cited_prompt, _request_safe
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.source_evidence import SourceEvidenceStore


@contextmanager
def readonly(path):
    verify_private(path)
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        yield db
    finally:
        db.close()


class ReadOnlyUnits(SourceEvidenceStore):
    def __init__(self, archive):
        self.archive = archive
        self.root = archive.root / "source-evidence"
        self.db_path = self.root / "projections.sqlite3"

    def _connect(self):
        return readonly(self.db_path)


class ReadOnlyLedger(MemoryLedger):
    def __init__(self, archive):
        self.archive = archive
        self.root = archive.root / "memory-ledger"
        self.db_path = self.root / "ledger.sqlite3"
        self._key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                         info=b"muninn memory ledger key v1").derive(archive._key)
        self.units = ReadOnlyUnits(archive)
        self._screen_cache = OrderedDict()
        self._entries = {(entry["blob"], version): entry
                         for entries in archive._load_manifest()["files"].values()
                         for version, entry in enumerate(entries)}

    def _connect(self):
        return readonly(self.db_path)


def samples(root: Path, *, limit=10):
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("Comparison requires 1-10 bounded samples")
    archive = SecureHistoryArchive(root)
    # Deliberately avoid journal/source constructors that initialize schemas.
    journal = object.__new__(CaptureJournal)
    journal.archive = archive
    key = hmac.new(archive._key, b"muninn-capture-journal-key-v1", hashlib.sha256).digest()
    source = object.__new__(CitedAnalysisSource)
    source.archive, source.ledger = archive, ReadOnlyLedger(archive)
    result, seen = [], set()
    with readonly(root / "capture-jobs.db") as db:
        rows = db.execute("SELECT * FROM history_analysis_jobs WHERE lane=1 "
            "AND state='succeeded' AND provider='openrouter' "
            "AND model='openai/gpt-6-luna-pro' ORDER BY updated_at DESC LIMIT 200").fetchall()
    for row in rows:
        target = journal._open_search(row["sealed_target"], row["job_id"], "analysis-target")
        purpose = "analysis-window-v1:" + hashlib.sha256(journal._stage_json(target)).hexdigest()
        descriptor = journal._open_search(row["sealed_window"], row["job_id"], purpose)
        stage_purpose = ("analysis-extraction-v1:" + purpose.split(":", 1)[1] + ":"
            + hashlib.sha256(journal._stage_json(descriptor)).hexdigest() + ":" + row["extraction_id"])
        stage = journal._open_search(row["sealed_extraction"], row["job_id"], stage_purpose)
        if (stage["window"] != descriptor or stage["result"]["provider"] != "openrouter"
                or stage["result"]["model"] != row["model"]
                or hmac.new(key, b"analysis-stage-v1\0" + journal._stage_json(stage),
                            hashlib.sha256).hexdigest() != row["extraction_id"]
                or not settled_response(root.parent, stage["admission_id"], row["remote_policy_generation"])):
            raise ValueError("Comparison original is not authenticated and settled")
        window = source.remote_input(descriptor)
        if window is None or not _request_safe({"messages": _cited_prompt(window)}):
            continue
        content_id = hashlib.sha256(window["text"].encode("utf-8")).hexdigest()
        if content_id in seen or not stage["proposals"]:
            continue
        seen.add(content_id)
        result.append({"id": f"sample-{len(result)+1}",
            "input": {k: v for k, v in window.items() if k != "project_ref"},
            "openrouter_original": {**stage["result"]["analysis"], "proposals": stage["proposals"]}})
        if len(result) == limit:
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--emit-screened-samples", action="store_true")
    parser.add_argument("--include-originals-for-parent", action="store_true",
                        help="Explicitly include saved outputs for parent-only comparison")
    args = parser.parse_args()
    if args.include_originals_for_parent and not args.emit_screened_samples:
        parser.error("Original outputs require explicit screened-sample emission")
    selected = samples(args.archive_root.resolve(), limit=args.limit)
    if args.emit_screened_samples:
        if not args.include_originals_for_parent:
            selected = [{"id": item["id"], "input": item["input"]} for item in selected]
        print(json.dumps(selected, ensure_ascii=False))
    else:
        print(json.dumps({"samples": len(selected), "input_chars": sum(len(s["input"]["text"]) for s in selected)}))


if __name__ == "__main__":
    main()
