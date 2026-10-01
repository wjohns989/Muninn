"""Certified append parsing: temporary encrypted archives, no providers/live data."""
import json
from contextlib import contextmanager

import pytest

from muninn.history.credential_crypto import VaultIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.source_evidence import SourceEvidenceStore
from muninn.history import transcript_units as units
from muninn.history.structured_projector import ProjectionCancelled


def fixture(tmp_path, provider="codex", *, newline=True, large=False):
    path = tmp_path / "chat.jsonl"
    if provider == "codex":
        records = [{"type": "session_meta", "payload": {"cwd": "C:/sample/one"}},
                   {"type": "event_msg", "payload": {"type": "user_message",
                    "message": "prior context " * (24000 if large else 1)}},
                   {"type": "turn_context", "payload": {"cwd": "C:/sample/two"}}]
        extra = {"type": "event_msg", "timestamp": "2026-10-01T10:00:00Z",
                 "payload": {"type": "user_message", "message": "new context"}}
    else:
        records = [{"type": "user", "cwd": "C:/sample/one", "uuid": "before",
                    "message": {"role": "user", "content": "prior context"}}]
        extra = {"type": "assistant", "uuid": "after",
                 "message": {"role": "assistant", "content": "new context"}}
    raw = "\n".join(json.dumps(row) for row in records).encode()
    raw += b"\n\n\n" if newline else b""
    suffix = (b"" if newline else b"\n") + json.dumps(extra).encode() + b"\n"
    path.write_bytes(raw)
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only recovery passphrase")
    archive.archive_file(path, provider)
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    store = SourceEvidenceStore(archive)
    parent_attempt = store.build_snapshot(entry, 0)
    path.write_bytes(raw + suffix)
    archive.archive_file(path, provider)
    child = archive._load_manifest()["files"][str(path.resolve())][1]
    return archive, store, entry, parent_attempt, child, raw, suffix


def instrument(monkeypatch, archive):
    parsed, exhausted = [], []
    metadata, events, original = units._metadata_units, units.events, archive._iter_verified_entry

    def watch(source, tag):
        count = 0
        for chunk in source:
            count += len(chunk)
            yield chunk
        parsed.append((tag, count))

    def labels(source, *args, **kwargs):
        yield from metadata(watch(source, "metadata"), *args, **kwargs)

    def body(source, *args, **kwargs):
        yield from events(watch(source, "body"), *args, **kwargs)

    def raw(entry):
        count = 0
        for chunk in original(entry):
            count += len(chunk)
            yield chunk
        exhausted.append((entry["blob"], count))

    monkeypatch.setattr(units, "_metadata_units", labels)
    monkeypatch.setattr(units, "events", body)
    monkeypatch.setattr(archive, "_iter_verified_entry", raw)
    return parsed, exhausted


@pytest.mark.parametrize("provider", ["codex", "claude_code"])
def test_append_matches_cold_build_but_parses_only_suffix(tmp_path, monkeypatch, provider):
    archive, store, parent, previous, child, raw, suffix = fixture(tmp_path, provider)
    cold = SourceEvidenceStore(archive, tmp_path / "cold")
    expected = list(cold.fragments(child, 1, cold.build_snapshot(child, 1)))
    parsed, exhausted = instrument(monkeypatch, archive)
    attempt = store.build_snapshot(child, 1)
    actual = list(store.fragments(child, 1, attempt))
    assert actual == expected
    assert sorted(parsed) == [("body", len(suffix)), ("metadata", len(suffix))]
    assert exhausted == [(child["blob"], len(raw + suffix))] * 2
    assert actual[-1].unit.physical_line == raw.count(b"\n")  # Existing zero-based coordinates.
    if provider == "codex":
        assert actual[-1].unit.cwd == "C:/sample/two"
        assert actual[-1].unit.project_basis == "turn_context"
    else:
        assert actual[-1].unit.cwd is None  # Claude cwd is record-local.
    parent_parts = list(store.fragments(parent, 0, previous))
    assert parent_parts == actual[:len(parent_parts)]
    assert b"new context" not in store.db_path.read_bytes()


def test_append_over_64_parent_pages_does_not_hold_reader_across_commits(tmp_path, monkeypatch):
    archive, store, parent, previous, child, raw, suffix = fixture(tmp_path, large=True)
    assert store.count_pages(parent, 0, previous) > 64
    parsed, _ = instrument(monkeypatch, archive)
    attempt = store.build_snapshot(child, 1)
    assert list(store.fragments(child, 1, attempt))[-1].text == ""
    assert sorted(parsed) == [("body", len(suffix)), ("metadata", len(suffix))]


def test_non_newline_boundary_uses_full_parser(tmp_path, monkeypatch):
    archive, store, _, _, child, raw, suffix = fixture(tmp_path, newline=False)
    parsed, _ = instrument(monkeypatch, archive)
    attempt = store.build_snapshot(child, 1)
    assert list(store.fragments(child, 1, attempt))[-1].unit.cwd == "C:/sample/two"
    assert sorted(parsed) == [("body", len(raw + suffix)), ("metadata", len(raw + suffix))]


def test_missing_parent_evidence_uses_full_parser(tmp_path, monkeypatch):
    archive, _, _, _, child, raw, suffix = fixture(tmp_path)
    store = SourceEvidenceStore(archive, tmp_path / "no-parent")
    parsed, _ = instrument(monkeypatch, archive)
    store.build_snapshot(child, 1)
    assert sorted(parsed) == [("body", len(raw + suffix)), ("metadata", len(raw + suffix))]


def test_prefix_boundary_inside_a_raw_chunk_and_new_cwd_reset_match_cold(tmp_path, monkeypatch):
    archive, store, _, _, child, raw, _ = fixture(tmp_path)
    path = tmp_path / "chat.jsonl"
    appended = json.dumps({"type": "turn_context", "payload": {}}).encode() + b"\n"
    appended += json.dumps({"type": "event_msg", "payload": {"type": "user_message",
                            "message": "caf\u00e9 after reset"}}).encode() + b"\n"
    path.write_bytes(raw + appended)
    archive.archive_file(path, "codex")
    # This rewritten version has no relation to v1. Use the certified v1 first
    # and then v2 cold fallback; tiny chunks exercise an intra-chunk boundary.
    original = archive._iter_verified_entry

    def tiny(entry):
        for chunk in original(entry):
            for offset in range(0, len(chunk), 7):
                yield chunk[offset:offset + 7]

    monkeypatch.setattr(archive, "_iter_verified_entry", tiny)
    attempt = store.build_snapshot(child, 1)
    assert list(store.fragments(child, 1, attempt))[-1].unit.cwd == "C:/sample/two"
    reset = archive._load_manifest()["files"][str(path.resolve())][2]
    reset_attempt = store.build_snapshot(reset, 2)
    assert list(store.fragments(reset, 2, reset_attempt))[-1].unit.cwd is None


def test_empty_parent_can_append_and_preserve_first_line(tmp_path, monkeypatch):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    archive = SecureHistoryArchive.create(tmp_path / "archive", "test-only recovery passphrase")
    archive.archive_file(path, "codex")
    store = SourceEvidenceStore(archive)
    parent = archive._load_manifest()["files"][str(path.resolve())][0]
    store.build_snapshot(parent, 0)
    suffix = b'{"type":"event_msg","payload":{"type":"user_message","message":"first"}}\n'
    path.write_bytes(suffix)
    archive.archive_file(path, "codex")
    child = archive._load_manifest()["files"][str(path.resolve())][1]
    parsed, exhausted = instrument(monkeypatch, archive)
    attempt = store.build_snapshot(child, 1)
    final = list(store.fragments(child, 1, attempt))[-1]
    assert final.unit.ordinal == 0 and final.unit.physical_line == 0
    assert sorted(parsed) == [("body", len(suffix)), ("metadata", len(suffix))]
    assert exhausted == [(child["blob"], len(suffix))] * 2


def test_cancellation_in_parent_fingerprint_tail_cleans_child_only(tmp_path, monkeypatch):
    archive, store, parent, previous, child, _, _ = fixture(tmp_path, large=True)
    original = store._connect
    state = {"final_scan": False, "cancelled": False}

    def trace(sql):
        if ("SELECT ordinal,length,CASE" in sql and "ORDER BY ordinal" in sql
                and "LIMIT" not in sql):
            state["final_scan"] = True

    @contextmanager
    def connection():
        with original() as db:
            db.set_trace_callback(trace)
            yield db

    def cancel():
        if state["final_scan"]:
            state["cancelled"] = True
            return True
        return False

    monkeypatch.setattr(store, "_connect", connection)
    with pytest.raises(ProjectionCancelled):
        store.build_snapshot(child, 1, should_cancel=cancel)
    assert state["cancelled"]
    assert store.find_snapshot(child, 1) is None
    assert store.find_snapshot(parent, 0) == previous
    assert store.verify_all()["snapshots"] == 1
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_legacy_certificate_absence_keeps_cold_parser(tmp_path, monkeypatch):
    archive, store, _, _, child, raw, suffix = fixture(tmp_path)
    with archive._write_lock():
        manifest = archive._load_manifest()
        manifest["files"][str((tmp_path / "chat.jsonl").resolve())][1].pop("prefix_of")
        manifest["generation"] += 1
        archive._save_manifest(manifest)
    child = archive._load_manifest()["files"][str((tmp_path / "chat.jsonl").resolve())][1]
    parsed, _ = instrument(monkeypatch, archive)
    store.build_snapshot(child, 1)
    assert sorted(parsed) == [("body", len(raw + suffix)), ("metadata", len(raw + suffix))]


def test_corrupt_parent_fragment_blocks_append_without_losing_parent_seal(tmp_path):
    archive, store, parent, previous, child, _, _ = fixture(tmp_path)
    with store._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=0",
                   (previous,))
    with pytest.raises(ProjectionIntegrityError):
        store.build_snapshot(child, 1)
    assert store.find_snapshot(child, 1) is None
    assert store.find_snapshot(parent, 0) == previous
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts WHERE state='building'").fetchone()[0] == 0


@pytest.mark.parametrize("pass_number", [1, 2])
def test_late_current_raw_failure_never_publishes_child(tmp_path, monkeypatch, pass_number):
    archive, store, parent, previous, child, _, _ = fixture(tmp_path)
    original = archive._iter_verified_entry
    calls = 0

    def fail(entry):
        nonlocal calls
        calls += 1
        mine = calls
        yield from original(entry)
        if mine == pass_number:
            raise VaultIntegrityError("test-only late authentication failure")

    monkeypatch.setattr(archive, "_iter_verified_entry", fail)
    with pytest.raises(VaultIntegrityError):
        store.build_snapshot(child, 1)
    assert store.find_snapshot(child, 1) is None
    assert store.find_snapshot(parent, 0) == previous
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
