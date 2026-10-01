"""Query-independent coverage plans; isolated encrypted archives, no inference."""
import importlib
import json

import pytest

from muninn.history.cited_analysis_source import CitedSourceError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError
from muninn.history.structured_projector import ProjectionCancelled
from tests.test_cited_analysis_source import fixture, PHRASE


def store(archive):
    return importlib.import_module("muninn.history.cited_windows").CitedWindowPlanStore(archive)


def entry_for(source):
    return next(entry for (blob, version), entry in source.ledger._entries.items() if version == 0)


def test_plan_covers_arbitrary_message_without_query_or_source_size_cutoff(tmp_path):
    text = "Window coverage marker. " * 5000 + "end-of-message"
    archive, source, _cap = fixture(tmp_path, text=text)
    plans = store(archive)
    entry = entry_for(source)
    attempt = plans.build_snapshot(entry, 0)
    count = plans.count_pages(entry, 0, attempt)
    recovered = []
    coordinates = []
    for ordinal in range(count):
        descriptor = plans.window_at(entry, 0, attempt, ordinal)
        window = source.reopen(descriptor)
        assert 1 <= len(window["text"]) <= 3000
        assert window["role"] == "user" and window["time_basis"] == "provider_record"
        assert "complete" not in descriptor and "text" not in descriptor
        coordinates.append((descriptor["page"], descriptor["offset"], descriptor["length"]))
        recovered.append(window["text"])
    assert "".join(recovered) == text + "Later harmless conversation."
    assert len(set(coordinates)) == count and count > 40
    assert plans.verify_all() == {"snapshots": 1, "windows": count}
    assert text[:40].encode() not in plans.db_path.read_bytes()


def test_reuse_sealed_plan_without_rereading_raw_source(tmp_path, monkeypatch):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    attempt = plans.build_snapshot(entry, 0)
    def forbidden(*args, **kwargs):
        pytest.fail("A reusable plan must not rescan the raw archive")
    monkeypatch.setattr(archive, "_iter_verified_entry", forbidden)
    assert plans.build_snapshot(entry, 0) == attempt
    assert plans.window_at(entry, 0, attempt, 0)["offset"] == 0


def test_late_source_failure_never_publishes_partial_window_plan(tmp_path, monkeypatch):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    original = plans.source.ledger.units.fragments
    def late(*args, **kwargs):
        yield from original(*args, **kwargs)
        raise ProjectionIntegrityError("isolated late integrity failure")
    monkeypatch.setattr(plans.source.ledger.units, "fragments", late)
    with pytest.raises(ProjectionIntegrityError):
        plans.build_snapshot(entry, 0)
    with plans._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0


def test_cancelled_plan_is_not_coverage(tmp_path):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    with pytest.raises(ProjectionCancelled):
        plans.build_snapshot(entry, 0, should_cancel=lambda: True)
    assert plans.find_snapshot(entry, 0) is None


def test_omitted_records_have_authenticated_zero_window_plan(tmp_path):
    path = tmp_path / "empty-context.jsonl"
    path.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "C:/sample"}})
                    + "\n", encoding="utf-8")
    archive = SecureHistoryArchive.create(tmp_path / "archive", PHRASE)
    archive.archive_file(path, "codex")
    entry = archive._load_manifest()["files"][str(path.resolve())][0]
    plans = store(archive)
    attempt = plans.build_snapshot(entry, 0)
    assert plans.projection_info(entry, 0, attempt) == (0, {
        "source_units": 1, "conversational_units": 0, "omitted_units": 1})
    assert plans.verify_all() == {"snapshots": 1, "windows": 0}


def test_portable_restore_includes_and_authenticates_window_plans(tmp_path):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    attempt = plans.build_snapshot(entry, 0)
    restored = SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restored", PHRASE)
    recovered = store(restored)
    assert recovered.find_snapshot(entry, 0) == attempt
    descriptor = recovered.window_at(entry, 0, attempt, 0)
    assert recovered.source.reopen(descriptor)["text"].startswith("Keep SQLite")
    assert recovered.verify_all() == plans.verify_all()


def test_tampered_plan_or_source_cannot_return_a_window(tmp_path):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    attempt = plans.build_snapshot(entry, 0)
    descriptor = plans.window_at(entry, 0, attempt, 0)
    with plans.source.ledger.units._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) WHERE attempt=? AND ordinal=?",
                   (descriptor["attempt"], descriptor["page"]))
    with pytest.raises(CitedSourceError):
        plans.window_at(entry, 0, attempt, 0)
    with pytest.raises(ProjectionIntegrityError):
        plans.verify_all()


def test_rebuilt_source_attempt_requires_new_plan_instead_of_stranding_old_windows(tmp_path):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    old_plan = plans.build_snapshot(entry, 0)
    old_window = plans.window_at(entry, 0, old_plan, 0)
    # An explicitly regenerated evidence sidecar has the same raw snapshot,
    # but a new authenticated source attempt; cached descriptors cannot switch.
    original = plans.source.ledger.units.find_snapshot
    plans.source.ledger.units.find_snapshot = lambda *args: None
    new_source = plans.source.ledger.units.build_snapshot(entry, 0)
    plans.source.ledger.units.find_snapshot = original
    assert new_source != old_window["attempt"]
    assert plans.find_snapshot(entry, 0) is None
    new_plan = plans.build_snapshot(entry, 0)
    assert new_plan != old_plan
    assert plans.window_at(entry, 0, new_plan, 0)["attempt"] == new_source
    with pytest.raises(ProjectionIntegrityError):
        plans.window_at(entry, 0, old_plan, 0)
    assert plans.verify_all()["snapshots"] == 2


def test_interrupted_plan_after_staging_commit_can_retry_without_partial_coverage(tmp_path, monkeypatch):
    archive, source, _cap = fixture(tmp_path, text="bounded context " * 14000)
    plans, entry = store(archive), entry_for(source)
    original = plans._descriptors
    def interrupted(*args, **kwargs):
        for ordinal, descriptor in enumerate(original(*args, **kwargs)):
            if ordinal == 65:
                raise ProjectionCancelled("isolated interrupted window build")
            yield descriptor
    monkeypatch.setattr(plans, "_descriptors", interrupted)
    with pytest.raises(ProjectionCancelled):
        plans.build_snapshot(entry, 0)
    with plans._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
    monkeypatch.setattr(plans, "_descriptors", original)
    attempt = plans.build_snapshot(entry, 0)
    assert plans.count_pages(entry, 0, attempt) > 65
    assert plans.verify_all()["snapshots"] == 1


def test_corrupted_late_window_plan_blocks_verification_and_portable_restore(tmp_path):
    archive, source, _cap = fixture(tmp_path)
    plans, entry = store(archive), entry_for(source)
    attempt = plans.build_snapshot(entry, 0)
    with plans._connect() as db:
        db.execute("UPDATE pages SET ciphertext=zeroblob(length(ciphertext)) "
                   "WHERE attempt=? AND ordinal=(SELECT MAX(ordinal) FROM pages WHERE attempt=?)",
                   (attempt, attempt))
    with pytest.raises(ProjectionIntegrityError):
        plans.verify_all()
    with pytest.raises(ProjectionIntegrityError):
        SecureHistoryArchive.restore_from_backup(archive.root, tmp_path / "restore-bad", PHRASE)


def test_changed_geometry_builds_distinct_plan_and_preserves_verifiable_old_plan(tmp_path, monkeypatch):
    archive, source, _cap = fixture(tmp_path, text="ordinary source words " * 500)
    plans, entry = store(archive), entry_for(source)
    old = plans.build_snapshot(entry, 0)
    monkeypatch.setattr(plans, "window_chars", 1500)
    assert plans.find_snapshot(entry, 0) is None
    new = plans.build_snapshot(entry, 0)
    assert new != old
    assert plans.window_at(entry, 0, new, 0)["length"] <= 1500
    with pytest.raises(ProjectionIntegrityError):
        plans.window_at(entry, 0, old, 0)
    assert plans.verify_all()["snapshots"] == 2
