"""Synthetic manifest tests; no live archive or credential is opened."""

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from muninn.history.credential_store import source_fingerprint
from muninn.history.credential_provenance import archive_source_index
from scripts import audit_credential_provenance


class _Archive:
    vault_id = "synthetic-vault"

    def __init__(self, files):
        self.files = files

    def _load_manifest(self):
        return {"files": self.files}


def _entry(blob, sha, *, captured_at=10.0, mtime_ns=123):
    return {"blob": blob, "sha256": sha, "provider": "codex",
            "kind": "transcript", "captured_at": captured_at,
            "mtime_ns": mtime_ns}


def test_existing_credential_source_hash_joins_exact_snapshot():
    older = _entry("a" * 32, "1" * 64)
    newer = _entry("b" * 32, "2" * 64, captured_at=20.0, mtime_ns=456)
    index = archive_source_index(_Archive({"C:/private/session.jsonl": [older, newer]}))
    key = source_fingerprint("synthetic-vault:" + newer["blob"] + ":" + newer["sha256"])
    evidence = index[key]
    assert evidence is not None
    assert (evidence.source_path, evidence.version, evidence.captured_at,
            evidence.source_mtime_ns, evidence.event_at) == (
                "C:/private/session.jsonl", 1, 20.0, 456, None)
    assert source_fingerprint("synthetic-vault:" + newer["blob"] + ":" + older["sha256"]) not in index


def test_duplicate_blob_identity_fails_closed_instead_of_picking_a_path():
    shared = _entry("a" * 32, "1" * 64)
    index = archive_source_index(_Archive({
        "C:/private/one.jsonl": [shared], "C:/private/two.jsonl": [shared],
    }))
    key = source_fingerprint("synthetic-vault:" + shared["blob"] + ":" + shared["sha256"])
    assert index[key] is None


def test_invalid_manifest_entry_fails_closed():
    malformed = _entry("not-a-blob", "1" * 64)
    try:
        archive_source_index(_Archive({"C:/private/one.jsonl": [malformed]}))
    except ValueError as error:
        assert str(error) == "Invalid archive source provenance"
    else:
        raise AssertionError("Malformed archive source was accepted")


def test_invalid_manifest_shape_fails_closed():
    for files in (None, {"C:/private/one.jsonl": "not-versions"}):
        try:
            archive_source_index(_Archive(files))
        except ValueError as error:
            assert str(error) == "Invalid archive source provenance"
        else:
            raise AssertionError("Malformed archive structure was accepted")


def test_coverage_counts_only_and_opens_credential_db_readonly(monkeypatch):
    class FakeDB:
        def execute(self, sql):
            if sql.startswith("SELECT source_hash,origin,COUNT(*)"):
                return [
                    {"source_hash": "matched", "origin": "transcript", "n": 3},
                    {"source_hash": "missing", "origin": "transcript", "n": 2},
                    {"source_hash": "duplicate", "origin": "transcript", "n": 4},
                    {"source_hash": "project", "origin": "project", "n": 5},
                ]
            assert sql.startswith("SELECT id,group_digest,source_hash")
            return [
                {"id": "one", "group_digest": "v2:one", "source_hash": "matched"},
                {"id": "two", "group_digest": "v2:one", "source_hash": "missing"},
                {"id": "three", "group_digest": "v2:two", "source_hash": "duplicate"},
            ]

    class FakeStore:
        def __init__(self, _root):
            pass

        @contextmanager
        def _connect(self, *, readonly=False):
            assert readonly is True
            yield FakeDB()

    monkeypatch.setattr(audit_credential_provenance, "CredentialStore", FakeStore)
    monkeypatch.setattr(audit_credential_provenance, "SecureHistoryArchive",
                        lambda _root: object())
    monkeypatch.setattr(audit_credential_provenance, "archive_source_index",
                        lambda _archive: {"matched": SimpleNamespace(source_path="private"),
                                          "duplicate": None})
    report = audit_credential_provenance.coverage(Path("vault"), Path("archive"))
    assert report == {"transcript_rows": 9, "transcript_sources": 3,
                      "matched_rows": 3, "missing_rows": 2, "nonunique_rows": 4,
                      "project_rows": 5, "transcript_source_paths": 1,
                      "review_groups": 2, "cross_path_groups": 1}
    assert "private" not in str(report)


def test_audit_error_does_not_print_private_exception_text(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["audit_credential_provenance", "--root", "private",
                                  "--archive-root", "private-archive"])

    def fail(*_args):
        raise ValueError("private-source-path")

    monkeypatch.setattr(audit_credential_provenance, "coverage", fail)
    assert audit_credential_provenance.main() == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {"state": "unavailable", "error_category": "ValueError"}
    assert "private-source-path" not in output
