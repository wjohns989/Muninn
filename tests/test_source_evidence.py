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


def test_unit_fragment_seek_is_bounded_and_does_not_scan_preceding_units(tmp_path, monkeypatch):
    import math
    source = tmp_path / "many.jsonl"
    rows = [{"type": "session_meta", "payload": {"cwd": "C:/synthetic"}}]
    rows.extend({"type": "event_msg", "payload": {"type": "user_message",
                 "message": "ordinary prior context " * 1000}} for _ in range(100))
    rows.append({"type": "event_msg", "payload": {"type": "user_message", "message": "target unit"}})
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "synthetic portable recovery phrase")
    archive.archive_file(source, "codex")
    entry = archive._load_manifest()["files"][str(source.resolve())][0]
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    count = store.count_pages(entry, 0, attempt)
    decrypts, seals = [], []
    decrypt, seal = store._decrypt_page, store._authenticated_count
    def tracked_decrypt(*args, **kwargs):
        decrypts.append(args[2])
        return decrypt(*args, **kwargs)
    def tracked_seal(*args, **kwargs):
        seals.append(1)
        return seal(*args, **kwargs)
    monkeypatch.setattr(store, "_decrypt_page", tracked_decrypt)
    monkeypatch.setattr(store, "_authenticated_count", tracked_seal)
    selected = list(store.unit_fragments(entry, 0, attempt, 101))
    assert "".join(part.text for part in selected).endswith("target unit")
    assert selected[-1].final and all(part.unit.ordinal == 101 for part in selected)
    assert len(decrypts) <= 2 * math.ceil(math.log2(count)) + len(selected) + 2
    assert len(seals) == 1


@pytest.mark.parametrize("unit", [-1, True, 2])
def test_unit_fragment_seek_invalid_reference_fails_closed(tmp_path, unit):
    archive, entry = _fixture(tmp_path)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    with pytest.raises(ProjectionIntegrityError):
        list(store.unit_fragments(entry, 0, attempt, unit))


def test_unit_fragment_seek_preserves_omissions_and_fails_on_late_corruption(tmp_path):
    archive, entry = _fixture(tmp_path, "ordinary words " * 1000)
    store = SourceEvidenceStore(archive)
    attempt = store.build_snapshot(entry, 0)
    omitted = list(store.unit_fragments(entry, 0, attempt, 0))
    assert len(omitted) == 1 and omitted[0].final and not omitted[0].text
    total = store.count_pages(entry, 0, attempt)
    reader = store.unit_fragments(entry, 0, attempt, 1)
    assert next(reader).unit.ordinal == 1
    # Close the pinned reader before the isolated mutation; no models execute
    # during a unit screening pass. A fresh full drain must detect corruption.
    reader.close()
    with store._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                   "WHERE attempt=? AND ordinal=?", (attempt, total - 1))
    with pytest.raises(ProjectionIntegrityError):
        list(store.unit_fragments(entry, 0, attempt, 1))


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
