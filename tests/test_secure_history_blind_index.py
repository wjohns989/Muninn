"""An encrypted derived index must never become a plaintext history store."""

import json
import sqlite3
from pathlib import Path

import pytest

from muninn.history.blind_index import _STRUCTURED_PARSE_LIMIT, SecureHistoryBlindIndex
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive

PASSPHRASE = "correct horse battery archive recovery"


def _archive(tmp_path: Path) -> tuple[SecureHistoryArchive, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "private-project-name.jsonl"
    source.write_text(
        "work on the lunar-widget parser\ncredential CANARY-SECRET-91919\n",
        encoding="utf-8",
    )
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    return archive, source


def test_blind_index_returns_only_safe_metadata_and_no_plaintext(tmp_path: Path) -> None:
    archive, source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    built = index.build(max_snapshots=1)
    assert built["indexed"] == 1 and built["complete"] is True
    found = index.search("lunar-widget")
    assert len(found["matches"]) == 1
    assert set(found["matches"][0]) == {
        "ref", "provider", "kind", "captured_day_utc", "size_bucket_kib",
        "versions", "fetch_capability",
    }
    assert found["matches"][0]["provider"] == "codex"
    assert index.search("absent term")["matches"] == []
    db_bytes = (archive.root / "blind_index.db").read_bytes()
    for forbidden in (b"CANARY-SECRET", b"lunar-widget", b"private-project-name", b"credential"):
        assert forbidden not in db_bytes
    assert archive.read_file(source).startswith(b"work on")
    span = index.fetch_span(found["matches"][0]["fetch_capability"])
    assert "lunar-widget parser" in span["redacted_text"]
    assert "CANARY-SECRET-91919" not in str(span)
    assert "[REDACTED_SENSITIVE_LINE]" in span["redacted_text"]


def test_structured_fetch_keeps_message_when_jsonl_metadata_has_secret(tmp_path: Path) -> None:
    source = tmp_path / "codex-session.jsonl"
    source.write_text(json.dumps({
        "type": "event_msg", "timestamp": "2026-09-28T12:00:00Z",
        "payload": {"type": "user_message", "message": "Fix the lunar-widget parser",
                    "api_key": "CANARY-SECRET-91919"},
    }) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    span = index.fetch_span(capability)
    assert "Fix the lunar-widget parser" in span["redacted_text"]
    assert "CANARY-SECRET-91919" not in str(span)


def test_structured_parse_does_not_collect_oversize_snapshot(tmp_path: Path, monkeypatch) -> None:
    archive, _ = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("oversize snapshot must not be collected in memory")

    monkeypatch.setattr(archive, "_verify_entry", unexpected_collect)
    assert index._structured_span({
        "kind": "transcript", "provider": "codex", "size": _STRUCTURED_PARSE_LIMIT + 1,
    }, "lunar-widget", max_chars=3000) is None


def test_fetch_capability_rejects_tampering_and_expiry(tmp_path: Path, monkeypatch) -> None:
    archive, _source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    with pytest.raises(ValueError):
        index.fetch_span(capability[:-1] + ("A" if capability[-1] != "A" else "B"))
    import time
    now = time.time()
    monkeypatch.setattr("muninn.history.blind_index.time.time", lambda: now + 601)
    with pytest.raises(ValueError):
        index.fetch_span(capability)


def test_new_generation_preserves_unchanged_blob_and_resumes(tmp_path: Path) -> None:
    archive, source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    assert index.build(max_snapshots=1)["complete"] is True
    second = tmp_path / "second.jsonl"
    second.write_text("a different telescope plan", encoding="utf-8")
    archive.archive_file(second, "codex")
    partial = index.search("lunar-widget")
    assert len(partial["matches"]) == 1
    assert partial["complete"] is False and partial["missing"] == 1
    assert index.build(max_snapshots=1)["complete"] is True
    assert len(index.search("telescope")["matches"]) == 1
    assert archive.read_file(source).startswith(b"work on")


def test_filter_tamper_and_blob_tamper_fail_closed(tmp_path: Path) -> None:
    archive, _source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build(max_snapshots=1)
    with sqlite3.connect(index.path) as db:
        blob, sealed = db.execute("SELECT blob, sealed FROM filters").fetchone()
        altered = bytearray(sealed)
        altered[-1] ^= 1
        db.execute("UPDATE filters SET sealed=? WHERE blob=?", (bytes(altered), blob))
    with pytest.raises(VaultIntegrityError):
        index.search("lunar-widget")

    # Rebuild in a separate archive and corrupt a candidate ciphertext.
    other, _ = _archive(tmp_path / "other")
    other_index = SecureHistoryBlindIndex(other)
    other_index.build(max_snapshots=1)
    encrypted_blob = next((other.root / "blobs").glob("*.enc"))
    damaged = bytearray(encrypted_blob.read_bytes())
    damaged[-1] ^= 1
    encrypted_blob.write_bytes(damaged)
    with pytest.raises(VaultIntegrityError):
        other_index.search("lunar-widget")


def test_overflow_is_reported_not_silently_indexed(tmp_path: Path) -> None:
    source = tmp_path / "many-terms.txt"
    source.write_text(" ".join(f"word{n:06d}" for n in range(3000)), encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive, filter_bytes=128)
    report = index.build(max_snapshots=1)
    assert report["overflow"] == 1 and report["complete"] is False
    searched = index.search("word000001")
    assert searched["overflow"] == 1 and searched["complete"] is False
    assert searched["matches"] == []


def test_swapped_filter_cannot_be_replayed_on_another_blob(tmp_path: Path) -> None:
    archive, _source = _archive(tmp_path)
    second = tmp_path / "second.jsonl"
    second.write_text("independent nebula research", encoding="utf-8")
    archive.archive_file(second, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    with sqlite3.connect(index.path) as db:
        first, other = (row[0] for row in db.execute("SELECT blob FROM filters ORDER BY blob"))
        nonce, sealed = db.execute("SELECT nonce, sealed FROM filters WHERE blob=?", (first,)).fetchone()
        db.execute("UPDATE filters SET nonce=?, sealed=? WHERE blob=?", (nonce, sealed, other))
    with pytest.raises(VaultIntegrityError):
        index.search("nebula")


def test_token_crossing_encrypted_chunk_boundary_is_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("muninn.history.secure_archive._CHUNK", 16)
    source = tmp_path / "chunked.jsonl"
    source.write_text("prefix-xyz extraordinarytoken suffix", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    assert len(index.search("extraordinarytoken")["matches"]) == 1


def test_invalid_utf8_and_unbounded_candidate_budget_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b"topic \xff binary-like transcript")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    built = index.build()
    assert built["unsearchable"] == 1 and built["complete"] is False
    assert index.search("topic")["matches"] == []
    with pytest.raises(ValueError):
        index.search("topic", max_candidates=1001)


def test_repeated_tokens_are_positioned_once_per_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "repeated.jsonl"
    source.write_text("repeatword " * 10000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    original = index._positions
    calls = [0]

    def counted(term, filter_bytes=None):
        calls[0] += 1
        return original(term, filter_bytes)

    monkeypatch.setattr(index, "_positions", counted)
    assert index.build()["complete"] is True
    assert calls[0] == 1


def test_adaptive_large_filter_and_overflow_retry_keep_old_rows(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "large.jsonl"
    source.write_text("repeated-nebula " * 280000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index._large_filter_bytes({"size": 129 * 1024 * 1024}) == 8 * 1024 * 1024
    report = index.build(max_snapshots=1)
    assert report["complete"] is True
    assert len(index.search("nebula")["matches"]) == 1
    with sqlite3.connect(index.path) as db:
        assert db.execute("SELECT COUNT(*) FROM large_filters").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM filters").fetchone()[0] == 0

    # Simulate an already persisted v1 overflow and upgrade only that row.
    other = tmp_path / "other.jsonl"
    other.write_text(" ".join(f"word{n:06d}" for n in range(3000)), encoding="utf-8")
    second = SecureHistoryArchive.create(tmp_path / "second-archive", PASSPHRASE)
    second.archive_file(other, "codex")
    small = SecureHistoryBlindIndex(second, filter_bytes=128)
    assert small.build()["unsearchable"] == 1
    new_source = tmp_path / "not-yet-indexed.jsonl"
    new_source.write_text("new comet", encoding="utf-8")
    second.archive_file(new_source, "codex")
    monkeypatch.setattr(small, "_large_filter_bytes", lambda _entry: 4096)
    retried = small.build(max_snapshots=1, retry_unsearchable=True)
    assert retried["unsearchable"] == 0 and retried["missing"] == 1
    assert small.build(max_snapshots=1)["complete"] is True
    assert len(small.search("word000001")["matches"]) == 1
    with sqlite3.connect(small.path) as db:
        assert db.execute("SELECT COUNT(*) FROM filters").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM large_filters").fetchone()[0] == 2


def test_large_filter_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "large.jsonl"
    source.write_text("nebula " * 750000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    with sqlite3.connect(index.path) as db:
        blob, sealed = db.execute("SELECT blob, sealed FROM large_filters").fetchone()
        damaged = bytearray(sealed)
        damaged[-1] ^= 1
        db.execute("UPDATE large_filters SET sealed=? WHERE blob=?", (bytes(damaged), blob))
    with pytest.raises(VaultIntegrityError):
        index.search("nebula")
