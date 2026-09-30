import json

import pytest

from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError, SecureProjectionStore
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history.streaming_jsonl import StreamingJSONError
from muninn.history.credential_crypto import VaultIntegrityError


def _fixture(tmp_path, text="source observation"):
    source = tmp_path / "session.jsonl"
    source.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "C:/private/project"}}) + "\n" +
                      json.dumps({"type": "event_msg", "timestamp": "2026-09-30T12:00:00Z",
                                  "payload": {"type": "user_message", "message": text}}) + "\n")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic portable recovery phrase")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    return archive, entry


def test_evidence_is_encrypted_ordered_and_reuses_completed_snapshot(tmp_path):
    archive, entry = _fixture(tmp_path, "api_key=synthetic-sensitive-canary")
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    assert store.build_snapshot(entry, 0) == attempt
    parts = list(store.fragments(entry, 0, attempt))
    assert [part.unit.ordinal for part in parts if part.final] == [0, 1]
    assert parts[-1].unit.cwd == "C:/private/project"
    assert parts[-1].unit.event_at == 1790769600
    assert "synthetic-sensitive-canary" in "".join(part.text for part in parts)
    raw = store.db_path.read_bytes()
    assert b"synthetic-sensitive-canary" not in raw and b"C:/private/project" not in raw
    # A raw evidence DB cannot be treated as redacted public projection.
    public = SecureProjectionStore(archive, store.root)
    with pytest.raises(ProjectionIntegrityError):
        public.get_page(entry, 0, attempt, 0)


def test_oversized_source_has_contiguous_bounded_fragments(tmp_path):
    archive, entry = _fixture(tmp_path, "large source observation " * 16000)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    parts = list(store.fragments(entry, 0, attempt))
    assert max(len(part.text) for part in parts) <= 4096
    assert "".join(part.text for part in parts).count("large source observation") == 16000
    assert store.verify_all()["units"] == 2


def test_control_character_escaping_does_not_reject_a_bounded_fragment(tmp_path):
    archive, entry = _fixture(tmp_path, "\x01" * 4096)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    text = "".join(part.text for part in store.fragments(entry, 0, attempt))
    assert text.endswith("\x01" * 4096)
    assert text.count("\x01") == 4096


def test_late_source_error_publishes_nothing_and_cleans_staging(tmp_path, monkeypatch):
    archive, entry = _fixture(tmp_path, "staged observation " * 20000)
    store = SourceEvidenceStore(archive)
    original = archive._iter_verified_entry

    def fail_late(item):
        yield from original(item)
        raise StreamingJSONError("synthetic late failure")

    monkeypatch.setattr(archive, "_iter_verified_entry", fail_late)
    with pytest.raises(StreamingJSONError):
        store.build_snapshot(entry, 0)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0


def test_reordered_body_cannot_seal_against_original_snapshot(tmp_path, monkeypatch):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    original = archive._iter_verified_entry
    calls = 0

    def reordered(item):
        nonlocal calls
        calls += 1
        if calls == 1:  # Metadata pass: same count/schema, different sequence.
            raw = b"".join(original(item))
            yield b"\n".join(reversed(raw.splitlines())) + b"\n"
        else:
            yield from original(item)

    monkeypatch.setattr(archive, "_iter_verified_entry", reordered)
    with pytest.raises(VaultIntegrityError, match="authenticated snapshot"):
        store.build_snapshot(entry, 0)
    assert store.find_snapshot(entry, 0) is None


def test_tampered_fragment_or_completion_is_unavailable(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    with store._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=1", (attempt,))
    with pytest.raises(ProjectionIntegrityError):
        list(store.fragments(entry, 0, attempt))


def test_ciphertext_backup_is_recoverable_with_same_portable_archive(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    backup = tmp_path / "evidence-backup"
    store.backup_to(backup)
    reopened = SourceEvidenceStore(archive, backup)
    assert reopened.verify_all()["snapshots"] == 1
    assert list(reopened.fragments(entry, 0, attempt))[-1].unit.cwd == "C:/private/project"


def test_archive_restore_includes_source_evidence_and_portable_key(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "portable-restore", "synthetic portable recovery phrase")
    evidence = SourceEvidenceStore(restored)
    assert evidence.verify_all()["snapshots"] == 1
    assert list(evidence.fragments(entry, 0, attempt))[-1].unit.event_at == 1790769600


def test_abandoned_stage_is_invisible_and_recovered_without_touching_complete(tmp_path):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    abandoned = "a" * 32
    with store._connect() as db:
        db.execute("INSERT INTO attempts SELECT ?,vault,blob,sha,size,version,'building',count,digest,completion "
                   "FROM attempts WHERE attempt=?", (abandoned, attempt))
    with pytest.raises(ProjectionIntegrityError, match="incomplete"):
        list(store.fragments(entry, 0, abandoned))
    reopened = SourceEvidenceStore(archive)
    assert reopened.find_snapshot(entry, 0) == attempt
    with reopened._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='building'").fetchone()[0] == 0
