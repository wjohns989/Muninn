"""Whole-unit privacy proofs survive worker boundaries, never contain text."""
import pytest
from dataclasses import replace

from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history import streaming_redaction
from tests.test_cited_analysis_source import fixture


def prepared(tmp_path, text="Keep SQLite for orbital-widget caching. " * 500):
    archive, source, cap = fixture(tmp_path, text=text)
    descriptor = source.prepare(cap)
    entry = source.ledger._entries[(descriptor["blob"], 0)]
    unit, _ = source.ledger._source(entry, 0, descriptor["attempt"], descriptor["page"])
    return archive, source.ledger, entry, descriptor, unit


def test_fresh_worker_reuses_complete_unit_screen_without_rescanning(tmp_path, monkeypatch):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    expected = ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    fresh = MemoryLedger(archive)
    def forbidden(*args, **kwargs):
        pytest.fail("Already authenticated immutable whole-unit privacy proof")
    monkeypatch.setattr(fresh.units, "unit_fragments", forbidden)
    assert fresh._unit_info(entry, 0, descriptor["attempt"], unit) == expected
    assert fresh.remote_input(entry, 0, descriptor["attempt"], descriptor["page"])


def test_late_unit_failure_publishes_no_screen_proof(tmp_path, monkeypatch):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    original = ledger.units.unit_fragments
    def corrupt_late(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise ProjectionIntegrityError("Late integrity failure")
    monkeypatch.setattr(ledger.units, "unit_fragments", corrupt_late)
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    with ledger.units._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM unit_screens").fetchone()[0] == 0


def test_tampered_screen_proof_fails_lookup_and_full_verification(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    with ledger.units._connect() as db:
        db.execute("UPDATE unit_screens SET ciphertext=zeroblob(length(ciphertext))")
    with pytest.raises(MemoryLedgerIntegrityError):
        MemoryLedger(archive)._unit_info(entry, 0, descriptor["attempt"], unit)
    with pytest.raises(ProjectionIntegrityError):
        ledger.units.verify_all()


def test_changed_unit_metadata_cannot_reuse_screen_proof(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    with pytest.raises(MemoryLedgerIntegrityError):
        MemoryLedger(archive)._unit_info(entry, 0, descriptor["attempt"], replace(unit, cwd="C:/elsewhere"))
    with ledger.units._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM unit_screens").fetchone()[0] == 1


def test_new_screen_policy_misses_without_deleting_older_valid_proof(tmp_path, monkeypatch):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    expected = ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    monkeypatch.setattr(streaming_redaction, "UNIT_SCREEN_VERSION", 2)
    fresh = MemoryLedger(archive)
    original = fresh.units.unit_fragments
    reads = []
    def counted(*args, **kwargs):
        reads.append(True)
        yield from original(*args, **kwargs)
    monkeypatch.setattr(fresh.units, "unit_fragments", counted)
    assert fresh._unit_info(entry, 0, descriptor["attempt"], unit) == expected
    assert reads == [True]
    with fresh.units._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM unit_screens").fetchone()[0] == 2
    assert fresh.units.verify_all()["snapshots"] == 1


def test_cache_hit_still_authenticates_selected_page(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    with ledger.units._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                   "WHERE attempt=? AND ordinal=?", (descriptor["attempt"], descriptor["page"]))
    with pytest.raises(MemoryLedgerIntegrityError):
        MemoryLedger(archive).remote_input(entry, 0, descriptor["attempt"], descriptor["page"])


def test_cache_hit_still_screens_actual_outgoing_request(tmp_path, monkeypatch):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    fresh = MemoryLedger(archive)
    seen = []
    def deny(value):
        seen.append(value)
        return False
    monkeypatch.setattr(fresh, "_screen", deny)
    assert fresh.remote_input(entry, 0, descriptor["attempt"], descriptor["page"]) is None
    assert len(seen) == 1 and seen[0]["text"]


def test_unsafe_whole_unit_stays_denied_in_fresh_worker(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path,
        text="Keep SQLite for orbital-widget caching. " * 500 + " SERVICE_API_KEY=synthetic$secret")
    assert ledger._unit_info(entry, 0, descriptor["attempt"], unit)[0] is False
    assert MemoryLedger(archive).remote_input(entry, 0, descriptor["attempt"], descriptor["page"]) is None


def test_ciphertext_cannot_be_transplanted_to_changed_binding(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    other = replace(unit, cwd="C:/elsewhere")
    ref = ledger.units._screen_ref(ledger.units._screen_binding(entry, 0, descriptor["attempt"], other))
    with ledger.units._connect() as db:
        db.execute("UPDATE unit_screens SET ref=?", (ref,))
    with pytest.raises(MemoryLedgerIntegrityError):
        MemoryLedger(archive)._unit_info(entry, 0, descriptor["attempt"], other)
    with pytest.raises(ProjectionIntegrityError):
        ledger.units.verify_all()


def test_backup_preserves_encrypted_cache_and_no_plaintext_body(tmp_path):
    import sqlite3
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    ledger.units.backup_to(tmp_path / "backup")
    raw = (tmp_path / "backup" / "projections.sqlite3").read_bytes()
    assert b"orbital-widget" not in raw and b"synthetic-project" not in raw
    with sqlite3.connect(tmp_path / "backup" / "projections.sqlite3") as db:
        row = db.execute("SELECT ref,ciphertext FROM unit_screens").fetchone()
        assert ledger.units._decode_screen(*row)["binding"]["unit"] == ledger.units._screen_binding(
            entry, 0, descriptor["attempt"], unit)["unit"]


def test_cache_hit_requires_authenticated_parent_completion(tmp_path):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    with ledger.units._connect() as db:
        db.execute("UPDATE attempts SET completion=zeroblob(length(completion)) WHERE attempt=?",
                   (descriptor["attempt"],))
    with pytest.raises(MemoryLedgerIntegrityError):
        MemoryLedger(archive)._unit_info(entry, 0, descriptor["attempt"], unit)


def test_new_source_version_and_attempt_do_not_alias_existing_screen(tmp_path, monkeypatch):
    archive, ledger, entry, descriptor, unit = prepared(tmp_path)
    ledger._unit_info(entry, 0, descriptor["attempt"], unit)
    path = tmp_path / "chat.jsonl"
    path.write_bytes(path.read_bytes() + b'\n')
    archive.archive_file(path, "codex")
    new_entry = archive._load_manifest()["files"][str(path.resolve())][1]
    fresh = MemoryLedger(archive)
    attempt = fresh.units.build_snapshot(new_entry, 1)
    parts = fresh.units.fragments(new_entry, 1, attempt)
    try:
        new_unit = next(parts).unit
    finally:
        parts.close()
    original = fresh.units.unit_fragments
    reads = []
    def counted(*args, **kwargs):
        reads.append(True)
        yield from original(*args, **kwargs)
    monkeypatch.setattr(fresh.units, "unit_fragments", counted)
    fresh._unit_info(new_entry, 1, attempt, new_unit)
    assert reads == [True]
    with fresh.units._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM unit_screens").fetchone()[0] == 2
