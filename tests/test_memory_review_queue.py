"""Authenticated, nonpersisting review pages on isolated encrypted evidence."""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from muninn.history.memory_ledger import MemoryLedger, MemoryLedgerIntegrityError
from muninn.history.secure_archive import SecureHistoryArchive
from muninn.history.secure_projection_store import ProjectionIntegrityError
from tests.test_memory_ledger import fixture, record


def queue_fixture(tmp_path):
    archive, entry, attempt, page = fixture(tmp_path, role="assistant")
    ledger = MemoryLedger(archive)
    ids = [record(ledger, entry, attempt, page, type=kind) for kind in ("observation", "fact", "preference")]
    return ledger, ids, (entry, attempt, page)


def test_pages_anchor_candidates_but_apply_current_reviews_and_never_write(tmp_path, monkeypatch):
    ledger, ids, binding = queue_fixture(tmp_path)
    before = ledger.verify_all()
    monkeypatch.setattr(ledger.units, "_store_screen_info", lambda *a, **kw: pytest.fail("read must not persist"))
    first = ledger.review_page(limit=1)
    assert [item["id"] for item in first["matches"]] == ids[:1]
    assert first["has_more"] and first["next_cursor"]
    assert ledger.verify_all() == before
    # Decisions after the anchor apply immediately, but new candidates do not join this traversal.
    ledger.resolve_review(ids[1], state="filed", expected_state="provisional", reason="source_supported")
    # Restore the normal writer only for this isolated fixture insertion.
    monkeypatch.undo()
    later = record(ledger, *binding, type="task")
    monkeypatch.setattr(ledger.units, "_store_screen_info", lambda *a, **kw: pytest.fail("read must not persist"))
    second = ledger.review_page(limit=1, cursor=first["next_cursor"])
    assert [item["id"] for item in second["matches"]] == ids[2:]
    assert not second["has_more"] and second["next_cursor"] is None
    assert second["current_events"] > second["snapshot_events"]
    assert later in {item["id"] for item in ledger.review_page()["matches"]}
    assert all(item["truth_status"] != "verified" for item in first["matches"] + second["matches"])


def test_cursor_is_bound_to_archive_limit_and_intact_prefix(tmp_path):
    ledger, _ids, _binding = queue_fixture(tmp_path)
    cursor = ledger.review_page(limit=1)["next_cursor"]
    with pytest.raises(ValueError):
        ledger.review_page(limit=2, cursor=cursor)
    with pytest.raises(ValueError):
        ledger.review_page(limit=1, cursor=cursor[:-2] + "AA")
    other = tmp_path / "other"
    other.mkdir()
    different, _ids, _binding = queue_fixture(other)
    with pytest.raises(ValueError):
        different.review_page(limit=1, cursor=cursor)
    # Tail verification is required even after enough results have been found.
    with sqlite3.connect(ledger.db_path) as db:
        db.execute("UPDATE events SET ciphertext=? WHERE seq=(SELECT max(seq) FROM events)", (b"damaged",))
    with pytest.raises(MemoryLedgerIntegrityError):
        ledger.review_page(limit=1)


@pytest.mark.parametrize("limit", [0, 21, True, 1.5])
def test_review_page_limit_is_strictly_bounded(tmp_path, limit):
    ledger, _ids, _binding = queue_fixture(tmp_path)
    with pytest.raises(ValueError):
        ledger.review_page(limit=limit)


def test_withheld_and_credential_candidates_never_enter_the_queue(tmp_path, monkeypatch):
    ledger, ids, binding = queue_fixture(tmp_path)
    private = record(ledger, *binding, type="possible_credential")
    original = ledger._public_text_safe

    def hidden(candidate, **kwargs):
        return candidate["type"] != "fact" and original(candidate, **kwargs)

    monkeypatch.setattr(ledger, "_public_text_safe", hidden)
    result = ledger.review_page()
    assert {item["id"] for item in result["matches"]} == {ids[0], ids[2]}
    assert private not in str(result) and ids[1] not in str(result)
    assert all("text" in item and "quote" in item for item in result["matches"])


def files_digest(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_read_only_open_supports_relative_configured_archive_root(tmp_path, monkeypatch):
    ledger, ids, _binding = queue_fixture(tmp_path)
    monkeypatch.chdir(ledger.archive.root.parent)
    ledger.archive.root = Path(ledger.archive.root.name)
    readonly = MemoryLedger(ledger.archive, read_only=True)
    assert [row["id"] for row in readonly.review_page()["matches"]] == ids


def test_read_only_open_never_initializes_missing_stores(tmp_path):
    archive = SecureHistoryArchive.create(tmp_path / "archive", "isolated recovery passphrase")
    before = files_digest(archive.root)
    with pytest.raises(OSError):
        MemoryLedger(archive, read_only=True).review_page(cursor="invalid")
    assert files_digest(archive.root) == before
    assert not (archive.root / "memory-ledger").exists()
    assert not (archive.root / "source-evidence").exists()


@pytest.mark.parametrize("store", ["ledger", "source"])
def test_read_only_open_rejects_incomplete_schema_without_repair(tmp_path, store):
    ledger, _ids, _binding = queue_fixture(tmp_path)
    if store == "ledger":
        # A writer can never be invoked by this read to repair a missing head.
        with sqlite3.connect(ledger.db_path) as db:
            db.execute("DROP TABLE head")
    else:
        with sqlite3.connect(ledger.units.db_path) as db:
            db.execute("DROP TABLE unit_screens")
    before = files_digest(ledger.archive.root)
    with pytest.raises((MemoryLedgerIntegrityError, ProjectionIntegrityError)):
        MemoryLedger(ledger.archive, read_only=True).review_page(cursor="invalid")
    assert files_digest(ledger.archive.root) == before


def test_read_only_open_does_not_recover_abandoned_source_work(tmp_path):
    ledger, ids, _binding = queue_fixture(tmp_path)
    with sqlite3.connect(ledger.units.db_path) as db:
        db.execute(
            "INSERT INTO attempts SELECT 'abandoned',vault,blob,sha,size,version,'building',"
            "count,digest,completion FROM attempts LIMIT 1"
        )
    before = files_digest(ledger.archive.root)
    readonly = MemoryLedger(ledger.archive, read_only=True)
    assert [row["id"] for row in readonly.review_page()["matches"]] == ids
    with readonly._connect() as db, pytest.raises(sqlite3.OperationalError):
        db.execute("DELETE FROM events")
    assert files_digest(ledger.archive.root) == before
