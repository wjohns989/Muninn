"""An encrypted derived index must never become a plaintext history store."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from muninn.history.blind_index import SecureHistoryBlindIndex
from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.private_acl import create_private_file
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


def test_private_model_window_reads_original_but_public_fetch_stays_redacted(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    original = index._model_window(capability)
    assert "CANARY-SECRET-91919" in original
    assert len(original) <= 3000 and len(original.encode("utf-8")) <= 12000
    assert "CANARY-SECRET-91919" not in str(index.fetch_span(capability))


def test_private_model_window_authenticates_late_chunks(tmp_path: Path, monkeypatch) -> None:
    archive, _ = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    original = archive._verify_entry

    def tamper_after_hit(entry, *, collect, on_chunk=None):
        def late(chunk):
            if on_chunk:
                on_chunk(chunk)
            raise VaultIntegrityError("late tamper")
        return original(entry, collect=collect, on_chunk=late)

    monkeypatch.setattr(archive, "_verify_entry", tamper_after_hit)
    with pytest.raises(VaultIntegrityError):
        index._model_window(capability)


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


def test_structured_fetch_streams_oversize_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "huge-codex.jsonl"
    filler = json.dumps({"type": "event_msg", "payload": {
        "type": "agent_message", "message": "ordinary note " + "z" * 900}}) + "\n"
    target = json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "Fix the lunar-widget parser",
        "api_key": "CANARY-SECRET-91919"}}) + "\n"
    source.write_text(filler * 10000 + target, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    span = index.fetch_span(capability)
    assert "Fix the lunar-widget parser" in span["redacted_text"]
    assert "CANARY-SECRET-91919" not in str(span)
    assert '"payload"' not in span["redacted_text"]


def test_structured_fetch_parses_jsonl_across_archive_chunks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("muninn.history.secure_archive._CHUNK", 32)
    source = tmp_path / "chunked-codex.jsonl"
    source.write_text(json.dumps({"type": "event_msg", "payload": {
        "type": "user_message", "message": "Fix the lunar-widget parser",
        "api_key": "CANARY-SECRET-91919"}}) + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    span = index.fetch_span(capability)
    assert "Fix the lunar-widget parser" in span["redacted_text"]
    assert "CANARY-SECRET-91919" not in str(span)


def test_structured_fetch_late_tamper_fails_closed(tmp_path: Path, monkeypatch) -> None:
    archive, _ = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")["matches"][0]["fetch_capability"]
    original = archive._verify_entry
    def tamper_after_hit(entry, *, collect, on_chunk=None):
        def late(chunk):
            if on_chunk:
                on_chunk(chunk)
            raise VaultIntegrityError("late tamper")
        return original(entry, collect=collect, on_chunk=late)
    monkeypatch.setattr(archive, "_verify_entry", tamper_after_hit)
    with pytest.raises(VaultIntegrityError, match="authentication failed"):
        index.fetch_span(capability)


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


def test_analysis_target_survives_capability_expiry_without_changing_snapshot(tmp_path: Path,
                                                                              monkeypatch) -> None:
    archive, _ = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    index.build()
    capability = index.search("lunar-widget")['matches'][0]['fetch_capability']
    target = index._analysis_target(capability, ["lunar", "widget"])
    import time
    now = time.time()
    monkeypatch.setattr("muninn.history.blind_index.time.time", lambda: now + 601)
    with pytest.raises(ValueError):
        index.fetch_span(capability)
    renewed = index._analysis_capability(target)
    assert "lunar-widget" in index._model_window(renewed)
    assert renewed != capability
    with pytest.raises(ValueError):
        index._analysis_capability({**target, "sha256": "0" * 64})


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


def test_saturated_small_filter_is_segmented_and_searchable(tmp_path: Path) -> None:
    source = tmp_path / "many-terms.txt"
    source.write_text(" ".join(f"word{n:06d}" for n in range(3000)), encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive, filter_bytes=128)
    report = index.build(max_snapshots=1)
    assert report["overflow"] == 0 and report["complete"] is True
    searched = index.search("word000001")
    assert searched["overflow"] == 0
    assert len(searched["matches"]) == 1


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


def test_candidate_check_does_not_tokenize_every_word_in_large_snapshot(tmp_path: Path,
                                                                         monkeypatch) -> None:
    source = tmp_path / "large-chat.jsonl"
    source.write_text("ordinary filler " * 100000 + " distincttarget ", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    from muninn.history import blind_index

    original = blind_index._terms

    def bounded_terms(value: str) -> list[str]:
        assert len(value) < 1000, "search tokenized transcript content"
        return original(value)

    monkeypatch.setattr(blind_index, "_terms", bounded_terms)
    assert len(index.search("distincttarget")["matches"]) == 1


def test_candidate_check_rejects_substring_after_forced_filter_hit(tmp_path: Path,
                                                                   monkeypatch) -> None:
    source = tmp_path / "chat.jsonl"
    source.write_text("foobar", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    positions = index._positions("foobar")
    monkeypatch.setattr(index, "_positions", lambda _term, _bytes=None: positions)
    assert index.search("foo")["matches"] == []


@pytest.mark.parametrize("content, query", [
    ("prefix " + "a" * 64 + " suffix", "a" * 64),
    ("first STRAẞE last", "strasse"),
])
def test_candidate_check_handles_max_term_and_casefold_across_chunks(
    tmp_path: Path, monkeypatch, content: str, query: str,
) -> None:
    monkeypatch.setattr("muninn.history.secure_archive._CHUNK", 16)
    source = tmp_path / "chat.jsonl"
    source.write_text(content, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    assert len(index.search(query)["matches"]) == 1


def test_candidate_check_rejects_repeated_embedded_terms_across_chunks(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr("muninn.history.secure_archive._CHUNK", 16)
    source = tmp_path / "chat.jsonl"
    source.write_text("prefoopost " * 100, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    positions = index._positions("prefoopost")
    monkeypatch.setattr(index, "_positions", lambda _term, _bytes=None: positions)
    assert index.search("foo")["matches"] == []


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


def test_coverage_is_bound_to_the_manifest_generation(tmp_path: Path) -> None:
    archive, source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    first = index.build()
    assert first["archive_generation"] == archive.status()["generation"]
    source.write_text("a later transcript version", encoding="utf-8")
    archive.archive_file(source, "codex")
    later = index.coverage()
    assert later["archive_generation"] == archive.status()["generation"]
    assert later["archive_generation"] != first["archive_generation"]
    assert later["complete"] is False


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
        assert db.execute("SELECT COUNT(*) FROM chunk_completions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM chunk_filters").fetchone()[0] > 1
        assert db.execute("SELECT COUNT(*) FROM large_filters").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM filters").fetchone()[0] == 0

    # Simulate an already persisted v1 overflow and upgrade only that row.
    other = tmp_path / "other.jsonl"
    other.write_text(" ".join(f"word{n:06d}" for n in range(3000)), encoding="utf-8")
    second = SecureHistoryArchive.create(tmp_path / "second-archive", PASSPHRASE)
    second.archive_file(other, "codex")
    small = SecureHistoryBlindIndex(second, filter_bytes=128)
    assert small.build()["unsearchable"] == 0
    new_source = tmp_path / "not-yet-indexed.jsonl"
    new_source.write_text("new comet", encoding="utf-8")
    second.archive_file(new_source, "codex")
    monkeypatch.setattr(small, "_large_filter_bytes", lambda _entry: 4096)
    retried = small.build(max_snapshots=1, retry_unsearchable=True)
    assert retried["unsearchable"] == 0
    assert small.build(max_snapshots=1)["complete"] is True
    assert len(small.search("word000001")["matches"]) == 1
    with sqlite3.connect(small.path) as db:
        assert db.execute("SELECT COUNT(*) FROM filters").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM chunk_completions").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM chunk_filters").fetchone()[0] > 1


def test_large_filter_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "large.jsonl"
    source.write_text("nebula " * 750000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    index.build()
    with sqlite3.connect(index.path) as db:
        blob, sealed = db.execute("SELECT blob, sealed FROM chunk_filters LIMIT 1").fetchone()
        damaged = bytearray(sealed)
        damaged[-1] ^= 1
        db.execute("UPDATE chunk_filters SET sealed=? WHERE blob=?", (bytes(damaged), blob))
    with pytest.raises(VaultIntegrityError):
        index.search("nebula")
    with sqlite3.connect(index.path) as db:
        db.execute("UPDATE chunk_filters SET sealed=? WHERE blob=?", (sealed, blob))
    encrypted_blob = next((archive.root / "blobs").glob("*.enc"))
    damaged_blob = bytearray(encrypted_blob.read_bytes())
    damaged_blob[-1] ^= 1
    encrypted_blob.write_bytes(damaged_blob)
    with pytest.raises(VaultIntegrityError):
        index.search("nebula")


def test_build_cleans_only_abandoned_private_chunk_staging(tmp_path: Path) -> None:
    archive, _source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    abandoned = archive.root / "muninn-chunks-abandoned.db"
    create_private_file(abandoned)
    abandoned.write_bytes(b"rebuildable encrypted staging")
    unrelated = archive.root / "keep.txt"
    create_private_file(unrelated)
    unrelated.write_text("keep", encoding="utf-8")
    assert index.build()["complete"] is True
    assert not abandoned.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name != "nt", reason="Windows inherited ACL regression")
def test_build_recovers_owner_only_inherited_sqlite_journal(tmp_path: Path) -> None:
    archive, _source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive)
    journal = archive.root / "muninn-chunks-abandoned.db-journal"
    journal.write_bytes(b"rebuildable encrypted staging journal")
    from muninn.history.private_acl import VaultPermissionError, verify_private

    with pytest.raises(VaultPermissionError, match="inherits"):
        verify_private(journal)
    assert index.build()["complete"] is True
    assert not journal.exists()


def test_legacy_small_overflow_upgrades_without_rewriting_archive(tmp_path: Path) -> None:
    archive, source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive, filter_bytes=128)
    _source, version, entry, _latest, _versions = index._current()[0]
    nonce = os.urandom(12)
    sealed = index._cipher.encrypt(nonce, b"O", index._aad(entry, version))
    with sqlite3.connect(index.path) as db:
        db.execute("INSERT INTO filters VALUES (?, ?, ?)", (entry["blob"], nonce, sealed))
    assert index.coverage()["unsearchable"] == 1
    assert index.retry_plan()["retryable_snapshots"] == 1
    assert index.build(max_snapshots=1, retry_unsearchable=True)["complete"] is True
    assert index.retry_plan()["retryable_snapshots"] == 0
    assert len(index.search("lunar-widget")["matches"]) == 1
    assert archive.read_file(source).startswith(b"work on")


def test_legacy_large_overflow_upgrades_to_segments(tmp_path: Path) -> None:
    source = tmp_path / "large.jsonl"
    source.write_text("nebula " * 750000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    _source, version, entry, _latest, _versions = index._current()[0]
    desired = index._large_filter_bytes(entry)
    nonce = os.urandom(12)
    sealed = index._large_cipher.encrypt(nonce, b"O", index._large_aad(entry, version, desired))
    with sqlite3.connect(index.path) as db:
        db.execute("INSERT INTO large_filters VALUES (?, ?, ?, ?)",
                   (entry["blob"], desired, nonce, sealed))
    assert index.retry_plan()["retryable_snapshots"] == 1
    report = index.build(max_snapshots=1, retry_unsearchable=True)
    assert report["complete"] is True and report["unsearchable"] == 0
    assert index.retry_plan()["retryable_snapshots"] == 0
    assert len(index.search("nebula")["matches"]) == 1
    with sqlite3.connect(index.path) as db:
        assert db.execute("SELECT COUNT(*) FROM chunk_filters").fetchone()[0] > 1


def test_segmented_terms_in_different_chunks_and_boundary_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("muninn.history.secure_archive._CHUNK", 16)
    source = tmp_path / "chunked.jsonl"
    source.write_text("leftterm " + "x" * 32 + " extraordinarytoken " +
                      "y" * 32 + " rightterm", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive, filter_bytes=128)
    monkeypatch.setattr(index, "_large_filter_bytes", lambda _entry: 4096)
    assert index.build()["complete"] is True
    assert len(index.search("leftterm rightterm")["matches"]) == 1
    assert len(index.search("extraordinarytoken")["matches"]) == 1


def test_interrupted_segment_build_leaves_no_ready_or_plaintext_stage(tmp_path: Path,
                                                                     monkeypatch) -> None:
    archive, _source = _archive(tmp_path)
    index = SecureHistoryBlindIndex(archive, filter_bytes=128)
    monkeypatch.setattr(index, "_large_filter_bytes", lambda _entry: 4096)
    original = archive._verify_entry

    def interrupted(entry, *, collect, on_chunk=None):
        def stop_after_first(chunk):
            on_chunk(chunk)
            raise RuntimeError("interrupted")
        return original(entry, collect=collect, on_chunk=stop_after_first)

    monkeypatch.setattr(archive, "_verify_entry", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        index.build()
    assert index.coverage()["missing"] == 1
    assert list(archive.root.glob("muninn-chunks-*.db")) == []
    monkeypatch.setattr(archive, "_verify_entry", original)
    assert index.build()["complete"] is True


def test_completion_tamper_fails_closed_and_portable_restore_rebuilds(tmp_path: Path) -> None:
    source = tmp_path / "large.jsonl"
    source.write_text("telescope " * 600000, encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PASSPHRASE)
    archive.archive_file(source, "codex")
    index = SecureHistoryBlindIndex(archive)
    assert index.build()["complete"] is True
    restored = SecureHistoryArchive.restore_from_backup(
        archive.root, tmp_path / "restored", PASSPHRASE)
    restored_index = SecureHistoryBlindIndex(restored)
    assert restored_index.coverage()["missing"] == 1
    assert restored_index.build()["complete"] is True
    assert len(restored_index.search("telescope")["matches"]) == 1
    with sqlite3.connect(index.path) as db:
        db.execute("UPDATE chunk_completions SET parameters='{}'")
    with pytest.raises(VaultIntegrityError):
        index.coverage()
    with pytest.raises(VaultIntegrityError):
        index.search("telescope")
