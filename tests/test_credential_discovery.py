"""No live secret or project file is read by these scanner tests."""

from __future__ import annotations

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.credential_discovery import (
    ExtractionStats,
    iter_env_findings,
    iter_transcript_findings,
    scan_archive,
    scan_project_env,
)
from muninn.history.credential_store import CredentialStore
from muninn.history.secure_archive import SecureHistoryArchive


class RecordingStore:
    def __init__(self):
        self.values = []

    def scan_source(self, *, findings, **_kwargs):
        # The production store commits only after the generator completes.
        proposed = list(findings)
        self.values.extend(proposed)
        return {"created": len(proposed), "rotated": 0, "staled": 0}


def test_boundary_spanning_value_and_ambiguous_examples():
    stats = ExtractionStats()
    chunks = [b"x" * 1000 + b"\nOPENROUTER_API_KEY=aaaabbbb", b"cccc11112222\nPASSWORD=${", b"NOT_SET}\n"]
    found = list(iter_env_findings(chunks, ".env.local", stats))
    assert found == [("OPENROUTER_API_KEY", "aaaabbbbcccc11112222", ".env.local")]
    assert stats.accepted == 1
    assert stats.ambiguous == 1


def test_large_noncredential_line_does_not_hide_later_assignment():
    stats = ExtractionStats()
    chunks = [b"x" * 65536, b"x" * 65536 + b"\nSERVICE_ACCESS_TOKEN=aaBBccDD12345678\n"]
    assert list(iter_env_findings(chunks, ".env", stats)) == [
        ("SERVICE_ACCESS_TOKEN", "aaBBccDD12345678", ".env")]
    assert stats.accepted == 1


def test_invalid_utf8_is_not_silently_skipped():
    with pytest.raises(UnicodeDecodeError):
        list(iter_env_findings([b"API_KEY=goodVALUE123\n", b"\xff"], ".env", ExtractionStats()))


def test_project_scan_excludes_templates_and_reports_bad_source(tmp_path):
    root = tmp_path / "my-project"
    root.mkdir()
    (root / ".env.example").write_text("SERVICE_API_KEY=exampleSECRET123\n")
    (root / ".env.local").write_text("SERVICE_API_KEY=realVALUE12345678\n")
    (root / ".env.bad").write_bytes(b"OTHER_API_KEY=validVALUE12345\n\xff")
    store = RecordingStore()
    report = scan_project_env(root, store, passphrase="test-only")
    assert report["files"] == 2
    assert report["succeeded"] == 1
    assert report["errors"] == 1
    assert report["complete"] is False
    assert len(store.values) == 1
    assert store.values[0][0] == "SERVICE_API_KEY"


def test_symlinked_env_is_rejected_without_following(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("SERVICE_API_KEY=realVALUE12345678\n")
    link = root / ".env"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    store = RecordingStore()
    report = scan_project_env(root, store, passphrase="test-only")
    assert report["errors"] == 1
    assert report["complete"] is False
    assert store.values == []


def test_transcript_assignment_scans_across_chunks_without_claiming_current_use():
    stats = ExtractionStats()
    found = list(iter_transcript_findings(
        [b'{"content":"SERVICE_API_KEY=aaaabbbb', b'cccc11112222"}\n'], stats))
    assert found == [("SERVICE_API_KEY", "aaaabbbbcccc11112222", "")]


def test_archive_late_integrity_failure_rolls_back_findings(tmp_path, monkeypatch):
    root = tmp_path / "archive"
    archive = SecureHistoryArchive.create(root, "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = RecordingStore()
    original = archive._verify_entry

    def fail_at_end(entry, *, collect, on_chunk=None):
        result = original(entry, collect=collect, on_chunk=on_chunk)
        if on_chunk is not None:
            raise VaultIntegrityError("synthetic late final digest failure")
        return result

    monkeypatch.setattr(archive, "_verify_entry", fail_at_end)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["attempted"] == 1
    assert report["succeeded"] == 0
    assert report["errors"] == 1
    assert report["complete"] is False
    assert store.values == []


def test_real_vault_archive_scan_is_idempotent_and_metadata_only(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    first = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert first["complete"] is True
    assert first["inserted"] == 1
    second = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert second["inserted"] == 0
    matches = store.search("SERVICE_API_KEY")
    assert len(matches) == 1
    assert matches[0]["origin"] == "transcript"
    assert "aaaabbbbcccc11112222" not in str(matches)
    with pytest.raises(ValueError, match="generation"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase", offset=1)
    with pytest.raises(ValueError, match="generation changed"):
        scan_archive(archive, store, passphrase="synthetic vault passphrase",
                     offset=1, expected_generation=first["generation"] + 1)


def test_real_vault_rolls_back_when_final_archive_verify_fails(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = CredentialStore.create(tmp_path / "vault", "synthetic vault passphrase")
    original = archive._verify_entry

    def fail_at_end(entry, *, collect, on_chunk=None):
        result = original(entry, collect=collect, on_chunk=on_chunk)
        if on_chunk is not None:
            raise VaultIntegrityError("synthetic late final digest failure")
        return result

    monkeypatch.setattr(archive, "_verify_entry", fail_at_end)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["complete"] is False
    assert store.search("SERVICE_API_KEY") == []


def test_new_capture_during_scan_is_reported_as_incomplete(tmp_path, monkeypatch):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic archive passphrase")
    source = tmp_path / "chat.jsonl"
    source.write_text('{"content":"SERVICE_API_KEY=aaaabbbbcccc11112222"}\n')
    archive.archive_file(source, "codex")
    store = RecordingStore()
    original = archive._load_manifest
    calls = 0

    def changed_manifest():
        nonlocal calls
        calls += 1
        value = original()
        return {**value, "generation": value["generation"] + (calls > 1)}

    monkeypatch.setattr(archive, "_load_manifest", changed_manifest)
    report = scan_archive(archive, store, passphrase="synthetic vault passphrase")
    assert report["succeeded"] == 1
    assert report["changed_during_scan"] is True
    assert report["complete"] is False
